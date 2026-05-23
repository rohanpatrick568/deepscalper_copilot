"""
execution/strategy.py — Core Lumibot Strategy: CryptoDeepScalper.

V2 CHANGE: Migrated from S&P 100 equities to BTC/USD cryptocurrency.

A Lumibot Strategy subclass that:
  • Loads a pre-trained DeepScalper BDQ model (N_DIR=2, FLAT/LONG) at startup.
  • On every 1-minute bar, runs inference for BTC/USD.
  • Routes LONG / exit-to-FLAT signals to Alpaca Crypto Trading via Lumibot.
  • Enforces CryptoCitruitBreaker (24/7 conditions — no EOD session guards).
  • Publishes all state updates to a DataBridge for the PyQt5 dashboard.
  • Uses real Alpaca LOB snapshots (top 3 bid/ask levels) for inference.

No short selling: Alpaca crypto does not support short positions.
Action space: Discrete(2) — 0=FLAT, 1=LONG.

Threading note:
    Lumibot runs this strategy in its own internal thread.  All interactions
    with the PyQt5 dashboard must go through DataBridge (thread-safe locks).
    Never update Qt widgets directly from this class.
"""

import logging
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import (
    N_DIR,
    N_SIZE,
    GRU_HIDDEN,
    MACRO_EMBED_DIM,
    FC_HIDDEN,
    MACRO_DIM,
    LOB_DIM,
    PRIV_DIM,
    ATR_PERIOD,
    KELLY_FRACTION,
    LOOKBACK_BARS,
    MAX_DAILY_LOSS_PCT,
    MAX_POSITION_PCT,
    SLEEP_TIME,
    CRYPTO_PAIRS,         # V2 CHANGE: was SP100_TICKERS
    STARTING_CAPITAL,
    WEIGHTS_DIR,
    VOLATILITY_HALT_MULTIPLIER,   # V2 CHANGE: crypto circuit breaker
    CONSECUTIVE_LOSS_HALT,        # V2 CHANGE: crypto circuit breaker
    TRANSACTION_COST_LAMBDA,      # V2 CHANGE: 25 bps
)
from dashboard.data_bridge import (
    DataBridge,
    ModelSignal,
    PositionSnapshot,
    TradeEvent,
)
from execution.risk import calculate_atr_stop, kelly_position_size
from execution.state_builder import build_observation

# V2 CHANGE: Crypto data clients instead of stock clients
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.live import CryptoDataStream
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestOrderbookRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Import architecture — add colab directory to sys.path if needed
_COLAB_PATH = Path(__file__).parent.parent / "colab"
if str(_COLAB_PATH) not in sys.path:
    sys.path.insert(0, str(_COLAB_PATH))

from deepscalper.architecture import DeepScalperNet  # noqa: E402
from deepscalper.utils import compute_micro_features   # V2: dual-mode LOB features

from lumibot.entities import Asset
from lumibot.strategies import Strategy

import pytz
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# V2 CHANGE: Binary action semantics (no short selling)
ACTION_FLAT = 0
ACTION_LONG = 1
ACTION_NAMES = {ACTION_FLAT: "FLAT", ACTION_LONG: "LONG"}


class CryptoDeepScalper(Strategy):
    """V2: 24/7 crypto intraday strategy using DeepScalper BDQ model.

    V2 CHANGES vs. MultiStockDeepScalper:
      - Universe: BTC/USD (was S&P 100 equities)
      - Actions: FLAT/LONG binary (no short selling — Alpaca crypto limitation)
      - Session guards: removed (crypto is 24/7)
      - Circuit breaker: CryptoCitruitBreaker (ATR spike + rolling loss + streak)
      - LOB features: real Alpaca orderbook (top 3 bid/ask) with proxy fallback
      - Weights file: BTC_USD.pth

    Args:
        data_bridge: Shared DataBridge instance for dashboard state updates.
        alpaca_api_key: Alpaca API key for data client (not broker auth).
        alpaca_secret_key: Alpaca secret key for data client.
        **kwargs: Passed through to the Lumibot Strategy base class.
    """

    # Minimum number of historical trades before using Kelly sizing
    _MIN_TRADES_FOR_KELLY: int = 5

    def __init__(
        self,
        data_bridge: DataBridge,
        alpaca_api_key: str,
        alpaca_secret_key: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._data_bridge: DataBridge = data_bridge
        self._models: Dict[str, DeepScalperNet] = {}
        self._crypto_circuit_breaker: Optional["CryptoCitruitBreaker"] = None

        # Alpaca crypto data client for LOB snapshots
        self._data_client = CryptoHistoricalDataClient(
            api_key=alpaca_api_key,
            secret_key=alpaca_secret_key,
        )

        # Per-pair win/loss tracking for Kelly sizing
        self._trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # Track entry prices and stops for open positions
        self._entry_prices: Dict[str, float] = {}
        self._stop_prices:  Dict[str, float] = {}
        self._tp_prices:    Dict[str, float] = {}

        # 24-hour rolling returns for circuit breaker
        self._hourly_returns: deque = deque(maxlen=24)
        self._consecutive_losses: int = 0

    # ------------------------------------------------------------------
    # Lumibot lifecycle hooks
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Called once at strategy startup.

        Loads model weights, instantiates CryptoCitruitBreaker, validates
        the Alpaca connection.
        """
        self.sleeptime = SLEEP_TIME

        logger.info("=" * 60)
        logger.info("CryptoDeepScalper — initialising (V2 crypto)")
        logger.info("Capital: $%.2f | Universe: %s", STARTING_CAPITAL, CRYPTO_PAIRS)
        logger.info("=" * 60)

        # Load model weights
        self._load_models()

        # V2 CHANGE: CryptoCitruitBreaker (not equity CircuitBreaker)
        self._crypto_circuit_breaker = CryptoCitruitBreaker(
            max_24h_loss_pct          = MAX_DAILY_LOSS_PCT,
            volatility_halt_multiplier = VOLATILITY_HALT_MULTIPLIER,
            consecutive_loss_halt      = CONSECUTIVE_LOSS_HALT,
            starting_capital           = STARTING_CAPITAL,
        )

        self._data_bridge.portfolio_value = STARTING_CAPITAL

        logger.info(
            "Initialisation complete. %d/%d models loaded.",
            len(self._models),
            len(CRYPTO_PAIRS),
        )

    def before_market_opens(self) -> None:
        """V2 CHANGE: Crypto is 24/7 — no meaningful session open.
        Kept for Lumibot compatibility; resets circuit breaker daily counters.
        """
        if self._crypto_circuit_breaker:
            self._crypto_circuit_breaker.reset_for_new_utc_day()
        self._data_bridge.is_halted = False
        self._data_bridge.halt_reason = ""
        logger.info("UTC day boundary: circuit breaker daily counters reset.")

    def after_market_closes(self) -> None:
        """V2 CHANGE: No-op for crypto (no session close).
        Logs daily summary for monitoring purposes only.
        """
        portfolio = self.get_portfolio_value()
        pnl = portfolio - STARTING_CAPITAL
        logger.info(
            "UTC day ended. Portfolio: $%.2f | Daily P&L: $%.2f",
            portfolio,
            pnl,
        )

    def on_trading_iteration(self) -> None:
        """Called every 1 minute, 24/7.

        V2 CHANGE: Removed EOD session guards (crypto has no market close).
        Checks CryptoCitruitBreaker, then runs inference for CRYPTO_PAIRS.
        """
        # ----------------------------------------------------------------
        # Step 1: CryptoCitruitBreaker check (V2 CHANGE: replaces CircuitBreaker)
        # ----------------------------------------------------------------
        if self._crypto_circuit_breaker:
            halted, reason = self._crypto_circuit_breaker.is_trading_halted()
            if halted:
                logger.info("Crypto trading halted: %s", reason)
                self._data_bridge.is_halted = True
                self._data_bridge.halt_reason = reason
                self._push_event("HALT", "ALL", 0, 0.0, reason)
                return

        self._data_bridge.is_halted = False
        self._data_bridge.halt_reason = ""

        # V2 CHANGE: No EOD close guard (crypto 24/7)

        # ----------------------------------------------------------------
        # Step 2: Run inference for BTC/USD
        # ----------------------------------------------------------------
        portfolio_value   = self.get_portfolio_value()
        current_positions = {str(p.asset.symbol): p for p in self.get_positions()}

        for pair in CRYPTO_PAIRS:
            if pair not in self._models:
                continue
            self._process_pair(pair, portfolio_value, current_positions)

        # ----------------------------------------------------------------
        # Step 3: Push portfolio snapshot to dashboard
        # ----------------------------------------------------------------
        self._push_portfolio_snapshot(portfolio_value)

    def on_filled_order(self, position, order, price, quantity, multiplier) -> None:
        """Called by Lumibot on every order fill.

        Updates circuit breaker P&L, logs fill, records win/loss for Kelly.
        """
        symbol = str(order.asset.symbol)  # 'BTCUSD' or 'BTC'
        side   = order.side               # 'buy' or 'sell'

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        logger.info(
            "FILL: %s %.6f %s @ $%.2f",
            side.upper(),
            float(quantity),
            symbol,
            price,
        )

        event_type = "FILL"
        self._push_event(event_type, symbol, float(quantity), price, side.upper())

        # V2 CHANGE: No short selling — only LONG exits (sell-side)
        if side == "sell" and symbol in self._entry_prices:
            entry    = self._entry_prices.pop(symbol, price)
            trade_pnl = (price - entry) * float(quantity)
            is_win    = trade_pnl > 0
            self._trade_history[symbol].append({"pnl": trade_pnl, "is_win": is_win})

            if self._crypto_circuit_breaker:
                self._crypto_circuit_breaker.record_trade(trade_pnl, is_win)

            logger.info(
                "CLOSED %s: P&L $%.2f (%s)",
                symbol,
                trade_pnl,
                "WIN" if is_win else "LOSS",
            )
        elif side == "buy":
            self._entry_prices[symbol] = float(price)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load DeepScalperNet weights for each crypto pair from WEIGHTS_DIR."""
        weights_path = Path(WEIGHTS_DIR)
        loaded = 0
        missing = []

        for pair in CRYPTO_PAIRS:
            safe_name = pair.replace('/', '_')   # 'BTC/USD' → 'BTC_USD'
            pth_file  = weights_path / f"{safe_name}.pth"
            if not pth_file.exists():
                missing.append(pair)
                continue

            model = DeepScalperNet(
                macro_dim   = MACRO_DIM,
                lob_dim     = LOB_DIM,
                priv_dim    = PRIV_DIM,
                gru_hidden  = GRU_HIDDEN,
                macro_embed = MACRO_EMBED_DIM,
                fc_hidden   = FC_HIDDEN,
                n_dir       = N_DIR,
                n_size      = N_SIZE,
            )
            try:
                ckpt = torch.load(str(pth_file), map_location="cpu", weights_only=True)
                state_dict = ckpt.get("online_net", ckpt)
                model.load_state_dict(state_dict)
                model.eval()
                self._models[pair] = model
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load weights for %s: %s", pair, exc)
                missing.append(pair)

        if missing:
            logger.warning("Could not load weights for: %s", missing)

        logger.info("Loaded %d/%d model weights.", loaded, len(CRYPTO_PAIRS))

    @staticmethod
    def _get_crypto_asset(pair: str) -> Asset:
        """V2 CHANGE: Build Lumibot Asset for a crypto pair.

        Alpaca symbol formats:
          Data API   : 'BTC/USD' (with slash)
          Trading API: 'BTCUSD'  (no slash)
          Lumibot    : Asset(symbol='BTC', asset_type=Asset.AssetType.CRYPTO)
        """
        base = pair.split('/')[0]   # 'BTC/USD' → 'BTC'
        return Asset(symbol=base, asset_type=Asset.AssetType.CRYPTO)

    def _get_live_lob_features(self, pair: str) -> np.ndarray:
        """V2 CHANGE: Fetch real Alpaca orderbook and compute 4 LOB features.

        Extracts top 3 bid/ask levels from CryptoLatestOrderbookRequest and
        computes microstructure features in real-LOB mode.  Falls back to
        proxy (OHLCV-based) on any exception.

        Args:
            pair: Alpaca data-API symbol string e.g. 'BTC/USD'.

        Returns:
            float32 array of shape (1, 4) — single-bar LOB features.
        """
        try:
            req = CryptoLatestOrderbookRequest(symbol_or_symbols=pair)
            ob  = self._data_client.get_crypto_latest_orderbook(req)
            book = ob[pair]

            # Build a one-row DataFrame matching compute_micro_features(use_proxy=False)
            lob_row = {}
            for level, bid, ask in zip([1, 2, 3], book.bids[:3], book.asks[:3]):
                lob_row[f'bid_price_{level}'] = float(bid.p)
                lob_row[f'bid_size_{level}']  = float(bid.s)
                lob_row[f'ask_price_{level}'] = float(ask.p)
                lob_row[f'ask_size_{level}']  = float(ask.s)

            lob_snap = pd.DataFrame([lob_row])

            # Placeholder OHLCV for the wrapper signature
            dummy_bars = pd.DataFrame({
                'open': [lob_snap['bid_price_1'].iloc[0]],
                'high': [lob_snap['ask_price_1'].iloc[0]],
                'low':  [lob_snap['bid_price_1'].iloc[0]],
                'close':[lob_snap['ask_price_1'].iloc[0]],
                'volume': [1.0],
            })

            features = compute_micro_features(
                bars=dummy_bars, lob_snapshots=lob_snap, use_proxy=False
            )  # (1, 4)
            return features.astype(np.float32)

        except Exception as exc:
            logger.warning("LOB fetch failed for %s (%s); using proxy.", pair, exc)
            return None   # Caller falls back to OHLCV proxy

    def _process_pair(
        self,
        pair: str,
        portfolio_value: float,
        current_positions: dict,
    ) -> None:
        """V2 CHANGE: Run inference for a single crypto pair and submit orders.

        Action 0=FLAT: If currently long → exit. If flat → hold.
        Action 1=LONG: If currently flat → enter. If long → hold.
        No short selling.

        Args:
            pair: Alpaca data-API symbol string e.g. 'BTC/USD'.
            portfolio_value: Current portfolio value in USD.
            current_positions: Dict mapping symbol key → Position for open positions.
        """
        asset = self._get_crypto_asset(pair)  # Asset(symbol='BTC', asset_type=CRYPTO)

        # Fetch historical bars
        try:
            bars_obj = self.get_historical_prices(asset, LOOKBACK_BARS + 5, "minute")
        except Exception as exc:
            logger.debug("Could not fetch bars for %s: %s", pair, exc)
            return

        if bars_obj is None:
            return

        bars = bars_obj.df
        if bars is None or len(bars) < LOOKBACK_BARS:
            logger.debug(
                "%s: insufficient bars (%d < %d), skipping",
                pair,
                len(bars) if bars is not None else 0,
                LOOKBACK_BARS,
            )
            return

        # V2 CHANGE: Track position flag (0=flat, 1=long — no short)
        trading_symbol = pair.replace('/', '')   # 'BTC/USD' → 'BTCUSD'
        pos_obj = current_positions.get(trading_symbol) or current_positions.get('BTC')
        position_flag  = 0
        unrealized_pnl_pct = 0.0
        is_long = False

        if pos_obj is not None and float(pos_obj.quantity) > 0:
            position_flag = 1
            is_long = True
            entry = self._entry_prices.get(trading_symbol, 0.0)
            last_price = self._get_last_price_crypto(pair) or 0.0
            if entry > 0 and last_price > 0:
                unrealized_pnl_pct = (last_price - entry) / entry

        # V2 CHANGE: Try real LOB features; fall back to proxy on failure
        lob_snap = self._get_live_lob_features(pair)   # (1, 4) or None
        if lob_snap is None:
            lob_snap = compute_micro_features(bars, use_proxy=True)  # (n, 4)

        obs = build_observation(
            bars,
            position      = position_flag,
            unrealized_pnl_pct = unrealized_pnl_pct,
            lob_override  = lob_snap,   # inject real/proxy LOB into obs builder
        )

        # Run BDQ inference
        model = self._models[pair]
        with torch.no_grad():
            q_dir, _q_size, _ = model(obs['lob'], obs['priv'], obs['macro'])

        q_np   = q_dir.squeeze(0).cpu().numpy()   # (N_DIR=2,)
        action = int(q_np.argmax())                # 0=FLAT, 1=LONG
        confidence = float(F.softmax(torch.from_numpy(q_np), dim=0).max())

        self._publish_signal(pair, action, q_np.tolist(), confidence)

        current_price = float(bars["close"].iloc[-1])

        # V2 CHANGE: Binary FLAT/LONG logic — no short selling
        if action == ACTION_LONG and not is_long:
            # Enter long
            self._submit_buy(trading_symbol, asset, bars, current_price, portfolio_value)
        elif action == ACTION_FLAT and is_long:
            # Exit long (do NOT enter short)
            self._submit_sell(trading_symbol, asset, current_price)
        # else: already in desired state, hold

    def _submit_buy(
        self,
        symbol: str,
        asset: Asset,
        bars,
        current_price: float,
        portfolio_value: float,
    ) -> None:
        """V2 CHANGE: Calculate fractional crypto position size and submit BUY.

        Alpaca crypto supports fractional quantities, so we size in notional
        USD then convert to fractional BTC.
        """
        # Kelly position sizing using recent win/loss history
        history = list(self._trade_history[symbol])
        if len(history) >= self._MIN_TRADES_FOR_KELLY:
            wins   = [h for h in history if h["is_win"]]
            losses = [h for h in history if not h["is_win"]]
            win_rate = len(wins) / len(history)
            avg_win  = abs(sum(h["pnl"] for h in wins)  / len(wins))  if wins   else 1.0
            avg_loss = abs(sum(h["pnl"] for h in losses) / len(losses)) if losses else 1.0
        else:
            win_rate = 0.50
            avg_win  = current_price * 0.005
            avg_loss = current_price * 0.0025

        notional = kelly_position_size(
            win_rate         = win_rate,
            avg_win          = avg_win,
            avg_loss         = avg_loss,
            portfolio_value  = portfolio_value,
            price            = current_price,
            kelly_fraction   = KELLY_FRACTION,
            max_position_pct = MAX_POSITION_PCT,
        ) * current_price   # convert shares → notional USD

        # V2 CHANGE: crypto fractional quantity
        qty = round(notional / current_price, 6)
        if qty < 0.000001:
            logger.debug("Skipping BUY %s: position size too small (%.8f BTC)", symbol, qty)
            return

        stop_price, tp_price = calculate_atr_stop(bars, current_price, "buy")
        self._stop_prices[symbol] = stop_price
        self._tp_prices[symbol]   = tp_price

        try:
            order = self.create_order(
                asset,
                qty,
                "buy",
                order_class="bracket",
                stop_loss_price=stop_price,
                take_profit_price=tp_price,
            )
            self.submit_order(order)
            logger.info(
                "BUY %.6f %s @ ~$%.2f | stop=$%.2f tp=$%.2f",
                qty, symbol, current_price, stop_price, tp_price,
            )
        except Exception as exc:
            logger.error("Failed to submit BUY for %s: %s", symbol, exc)

    def _submit_sell(self, symbol: str, asset: Asset, current_price: float) -> None:
        """V2 CHANGE: Submit market SELL to exit long position (no short entry).

        Alpaca crypto supports fractional sell quantities.
        """
        try:
            position = self.get_position(asset)
            if position and float(position.quantity) > 0:
                order = self.create_order(asset, float(position.quantity), "sell")
                self.submit_order(order)
                logger.info(
                    "SELL %.6f %s @ ~$%.2f (model FLAT signal)",
                    float(position.quantity), symbol, current_price,
                )
        except Exception as exc:
            logger.error("Failed to submit SELL for %s: %s", symbol, exc)

    def _close_all_positions(self, reason: str = "CIRCUIT_BREAKER") -> None:
        """Flatten all open crypto positions.

        V2 CHANGE: 'reason' default is CIRCUIT_BREAKER (not EOD_CLOSE).
        """
        logger.info("Closing all positions: %s", reason)
        positions = self.get_positions()
        if not positions:
            return

        try:
            self.sell_all()
        except Exception as exc:
            logger.error("sell_all failed: %s — trying per-position close", exc)
            for pos in self.get_positions():
                try:
                    order = self.create_order(pos.asset, float(pos.quantity), "sell")
                    self.submit_order(order)
                except Exception as e2:
                    logger.error("Could not close %s: %s", pos.asset.symbol, e2)

        self._push_event(reason, "ALL", 0, 0.0, reason)

    def _publish_signal(
        self,
        pair: str,
        action: int,
        q_values: List[float],
        confidence: float,
    ) -> None:
        """Push the latest model signal to DataBridge for the dashboard."""
        signal = ModelSignal(
            symbol     = pair,
            action     = ACTION_NAMES[action],
            q_values   = q_values,
            confidence = confidence,
            timestamp  = datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        )
        self._data_bridge.update_signal(pair, signal)

    def _push_portfolio_snapshot(self, portfolio_value: float) -> None:
        """Update DataBridge with current portfolio value and open positions."""
        self._data_bridge.portfolio_value = portfolio_value
        daily_pnl = (
            self._crypto_circuit_breaker.daily_pnl
            if self._crypto_circuit_breaker
            else 0.0
        )
        self._data_bridge.daily_pnl = daily_pnl

        positions  = self.get_positions()
        snapshots: Dict[str, PositionSnapshot] = {}
        for pos in positions:
            symbol        = str(pos.asset.symbol)
            current_price = self._get_last_price_crypto(
                f"{symbol}/USD"
            ) or 0.0
            entry = self._entry_prices.get(symbol, current_price)
            qty   = float(pos.quantity)
            upnl  = (current_price - entry) * qty

            snapshots[symbol] = PositionSnapshot(
                symbol         = symbol,
                side           = "LONG" if qty > 0 else "FLAT",
                qty            = abs(qty),
                entry_price    = entry,
                current_price  = current_price,
                unrealized_pnl = upnl,
                atr_stop       = self._stop_prices.get(symbol, 0.0),
                atr_tp         = self._tp_prices.get(symbol, 0.0),
            )

        self._data_bridge.update_positions(snapshots)

    def _push_event(
        self,
        event_type: str,
        symbol: str,
        qty: float,
        price: float,
        detail: str,
    ) -> None:
        """Append an event to the DataBridge trade log."""
        event = TradeEvent(
            timestamp  = datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            symbol     = symbol,
            side       = detail,
            qty        = qty,
            price      = price,
            event_type = event_type,
        )
        self._data_bridge.append_trade_event(event)

    def _get_last_price_crypto(self, pair: str) -> Optional[float]:
        """V2 CHANGE: Fetch last price for a crypto pair from Alpaca data client.

        Args:
            pair: Symbol in data-API format e.g. 'BTC/USD'.

        Returns:
            Last trade price as float, or None if unavailable.
        """
        try:
            from alpaca.data.requests import CryptoLatestBarRequest
            req  = CryptoLatestBarRequest(symbol_or_symbols=pair)
            bars = self._data_client.get_crypto_latest_bar(req)
            return float(bars[pair].close)
        except Exception:
            return None

    def _get_last_price(self, symbol: str) -> Optional[float]:
        """Legacy equity price lookup — kept for DataBridge compat."""
        return self._get_last_price_crypto(f"{symbol}/USD")


# =============================================================================
# V2 CHANGE: CryptoCitruitBreaker — replaces equity CircuitBreaker
# =============================================================================

class CryptoCitruitBreaker:
    """24/7 crypto-aware circuit breaker with three independent halt conditions.

    V2 CHANGE: Three halt conditions (no market-hours dependency):

    1. 24-hour rolling loss gate:
       Halt if cumulative P&L over the last 24 hours exceeds
       -max_24h_loss_pct × starting_capital.

    2. ATR-spike volatility gate:
       Halt if current 1-min |return| > volatility_halt_multiplier × 72-hour
       baseline ATR.  Prevents entering positions during flash crashes or pumps.

    3. Consecutive-loss streak gate:
       Halt for 30 minutes after consecutive_loss_halt consecutive losing trades.
       Prevents revenge-trading during adverse market conditions.

    All halts are temporary: once the cooldown period ends and the condition
    clears, trading resumes automatically.

    Args:
        max_24h_loss_pct           : Maximum 24-hour portfolio loss before halt (0.05 = 5%).
        volatility_halt_multiplier : ATR spike multiplier before halt (4.0 = 4× baseline).
        consecutive_loss_halt      : Number of consecutive losses before streak halt (8).
        starting_capital           : Portfolio starting value (for loss % calculation).
        cooldown_minutes           : Minutes to hold the halt after a streak trigger (30).
    """

    def __init__(
        self,
        max_24h_loss_pct:          float = 0.05,
        volatility_halt_multiplier: float = 4.0,
        consecutive_loss_halt:     int   = 8,
        starting_capital:          float = 10_000.0,
        cooldown_minutes:          int   = 30,
    ) -> None:
        self.max_24h_loss_pct           = max_24h_loss_pct
        self.volatility_halt_multiplier = volatility_halt_multiplier
        self.consecutive_loss_halt      = consecutive_loss_halt
        self.starting_capital           = starting_capital
        self.cooldown_minutes           = cooldown_minutes

        # State
        self.daily_pnl:       float = 0.0
        self._trade_pnls:     deque = deque(maxlen=1000)   # timestamped P&Ls
        self._trade_times:    deque = deque(maxlen=1000)   # corresponding UTC timestamps
        self._consecutive:    int   = 0
        self._streak_halted_until: Optional[datetime] = None
        self._minute_returns: deque = deque(maxlen=72 * 60)  # 72-hour baseline window

    def reset_for_new_utc_day(self) -> None:
        """Reset daily P&L counter at UTC midnight (called by before_market_opens)."""
        self.daily_pnl = 0.0
        logger.info("CryptoCitruitBreaker: daily P&L counter reset.")

    def record_trade(self, pnl: float, is_win: bool) -> None:
        """Record a completed trade for rolling P&L and streak tracking.

        Args:
            pnl:    Realised P&L in USD.
            is_win: True if the trade was profitable.
        """
        now = datetime.now(timezone.utc)
        self._trade_pnls.append(pnl)
        self._trade_times.append(now)
        self.daily_pnl += pnl

        if is_win:
            self._consecutive = 0
        else:
            self._consecutive += 1
            if self._consecutive >= self.consecutive_loss_halt:
                self._streak_halted_until = now + timedelta(minutes=self.cooldown_minutes)
                logger.warning(
                    "CryptoCitruitBreaker: %d consecutive losses — halting for %d min.",
                    self._consecutive,
                    self.cooldown_minutes,
                )

    def record_bar_return(self, log_return: float) -> None:
        """Record a per-bar log return for ATR-spike detection.

        Call this every on_trading_iteration with the latest BTC bar return.
        """
        self._minute_returns.append(abs(log_return))

    def is_trading_halted(self) -> tuple:
        """Check all three halt conditions.

        Returns:
            (halted: bool, reason: str)  — reason is empty string if not halted.
        """
        now = datetime.now(timezone.utc)

        # 1. Streak cooldown
        if self._streak_halted_until and now < self._streak_halted_until:
            remaining = int((self._streak_halted_until - now).total_seconds() / 60)
            return True, f"Consecutive loss streak — cooldown {remaining} min remaining"
        elif self._streak_halted_until and now >= self._streak_halted_until:
            self._streak_halted_until = None
            self._consecutive = 0   # Reset streak after cooldown

        # 2. 24-hour rolling loss gate
        cutoff = now - timedelta(hours=24)
        rolling_pnl = sum(
            pnl
            for pnl, ts in zip(self._trade_pnls, self._trade_times)
            if ts >= cutoff
        )
        max_loss_usd = -self.max_24h_loss_pct * self.starting_capital
        if rolling_pnl < max_loss_usd:
            return True, (
                f"24h rolling loss ${rolling_pnl:.2f} exceeds "
                f"limit ${max_loss_usd:.2f} ({self.max_24h_loss_pct*100:.0f}%)"
            )

        # 3. ATR-spike gate
        if len(self._minute_returns) >= 72:
            baseline_atr = float(np.mean(list(self._minute_returns)))
            if self._minute_returns and self._minute_returns[-1] > (
                self.volatility_halt_multiplier * baseline_atr
            ):
                return True, (
                    f"ATR spike {self._minute_returns[-1]:.4f} > "
                    f"{self.volatility_halt_multiplier}× baseline {baseline_atr:.4f}"
                )

        return False, ""
