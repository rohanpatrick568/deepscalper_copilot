"""
execution/strategy.py — Core Lumibot Strategy: MultiStockDeepScalper.

A single Lumibot Strategy subclass that:
  • Loads 100 pre-trained DeepScalper DuelingQNetwork models at startup.
  • On every 1-minute bar, runs inference for all 100 S&P 100 tickers.
  • Routes BUY / SELL signals (with Kelly sizing and ATR stops) to Alpaca
    Paper Trading via Lumibot's order API.
  • Enforces circuit breakers and EOD position flattening.
  • Publishes all state updates to a DataBridge for the PyQt5 dashboard.

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
    CLOSE_ALL_EOD,
    EOD_CLOSE_BUFFER_MIN,
    KELLY_FRACTION,
    LOOKBACK_BARS,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_TIMEZONE,
    MAX_DAILY_LOSS_PCT,
    MAX_POSITION_PCT,
    SLEEP_TIME,
    SP100_TICKERS,
    STARTING_CAPITAL,
    WEIGHTS_DIR,
)
from dashboard.data_bridge import (
    DataBridge,
    ModelSignal,
    PositionSnapshot,
    TradeEvent,
)
from execution.circuit_breakers import CircuitBreaker
from execution.risk import calculate_atr_stop, kelly_position_size
from execution.state_builder import build_observation

# Import architecture — add colab directory to sys.path if needed
_COLAB_PATH = Path(__file__).parent.parent / "colab"
if str(_COLAB_PATH) not in sys.path:
    sys.path.insert(0, str(_COLAB_PATH))

from deepscalper.architecture import DeepScalperNet  # noqa: E402

from lumibot.entities import Asset
from lumibot.strategies import Strategy

import pytz
from datetime import datetime

logger = logging.getLogger(__name__)

_ET = pytz.timezone(MARKET_TIMEZONE)

# Action index constants (must match environment.py and training)
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_NAMES = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}


class MultiStockDeepScalper(Strategy):
    """Unified multi-stock intraday strategy using DeepScalper DRL models.

    Loads one pre-trained DuelingQNetwork per S&P 100 ticker and runs inference
    every 1-minute bar.  Routes all orders to Alpaca Paper Trading API via
    Lumibot's built-in order management.

    Args:
        data_bridge: Shared DataBridge instance for dashboard state updates.
        **kwargs: Passed through to the Lumibot Strategy base class.
    """

    # Minimum number of historical trades per ticker before using Kelly sizing
    _MIN_TRADES_FOR_KELLY: int = 5

    def __init__(self, data_bridge: DataBridge, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data_bridge: DataBridge = data_bridge
        self._models: Dict[str, DeepScalperNet] = {}
        self._circuit_breaker: Optional[CircuitBreaker] = None

        # Per-ticker win/loss tracking for Kelly sizing
        # deque stores tuples of (pnl_dollars: float, is_win: bool)
        self._trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # Track entry prices and stops for open positions
        self._entry_prices: Dict[str, float] = {}
        self._stop_prices: Dict[str, float] = {}
        self._tp_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lumibot lifecycle hooks
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Called once at strategy startup.

        Loads model weights, configures the circuit breaker, and validates
        the Alpaca connection.
        """
        self.sleeptime = SLEEP_TIME

        logger.info("=" * 60)
        logger.info("MultiStockDeepScalper — initialising")
        logger.info("Capital: $%.2f | Universe: %d tickers", STARTING_CAPITAL, len(SP100_TICKERS))
        logger.info("=" * 60)

        # Load all model weights
        self._load_models()

        # Instantiate circuit breaker
        self._circuit_breaker = CircuitBreaker(
            max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
            starting_capital=STARTING_CAPITAL,
        )

        # Publish initial dashboard state
        self._data_bridge.portfolio_value = STARTING_CAPITAL

        logger.info(
            "Initialisation complete. %d/%d models loaded.",
            len(self._models),
            len(SP100_TICKERS),
        )

    def before_market_opens(self) -> None:
        """Reset daily circuit breaker state at the start of each session."""
        if self._circuit_breaker:
            self._circuit_breaker.reset_for_new_day()
        self._data_bridge.is_halted = False
        self._data_bridge.halt_reason = ""
        logger.info("Market open: circuit breaker reset.")

    def after_market_closes(self) -> None:
        """Log daily summary statistics after session close."""
        portfolio = self.get_portfolio_value()
        pnl = portfolio - STARTING_CAPITAL
        logger.info(
            "Session ended. Portfolio: $%.2f | Daily P&L: $%.2f",
            portfolio,
            pnl,
        )

    def on_trading_iteration(self) -> None:
        """Called every 1 minute during market hours.

        Checks circuit breakers, runs model inference for all 100 tickers,
        and routes qualifying signals to Alpaca via Lumibot.
        """
        # ----------------------------------------------------------------
        # Step 1: Circuit breaker check
        # ----------------------------------------------------------------
        halted, reason = self._circuit_breaker.is_trading_halted()
        if halted:
            logger.info("Trading halted: %s", reason)
            self._data_bridge.is_halted = True
            self._data_bridge.halt_reason = reason
            self._push_event("HALT", "ALL", 0, 0.0, reason)
            return

        self._data_bridge.is_halted = False
        self._data_bridge.halt_reason = ""

        # ----------------------------------------------------------------
        # Step 2: EOD position flattening
        # ----------------------------------------------------------------
        if CLOSE_ALL_EOD and self._is_eod_close_time():
            self._close_all_positions(reason="EOD_CLOSE")
            return

        # ----------------------------------------------------------------
        # Step 3: Iterate over all tickers and run inference
        # ----------------------------------------------------------------
        portfolio_value = self.get_portfolio_value()
        current_positions = {p.asset.symbol: p for p in self.get_positions()}

        for ticker in SP100_TICKERS:
            if ticker not in self._models:
                continue  # Weight file was missing at startup; skip

            self._process_ticker(ticker, portfolio_value, current_positions)

        # ----------------------------------------------------------------
        # Step 4: Push portfolio snapshot to dashboard
        # ----------------------------------------------------------------
        self._push_portfolio_snapshot(portfolio_value)

    def on_filled_order(self, position, order, price, quantity, multiplier) -> None:
        """Called by Lumibot on every order fill.

        Updates circuit breaker daily P&L, logs the fill to the trade log,
        and records win/loss stats for Kelly sizing.

        Args:
            position: Lumibot Position object after the fill.
            order: The filled Order object.
            price: Fill price.
            quantity: Number of shares filled.
            multiplier: Contract multiplier (1 for equities).
        """
        symbol = order.asset.symbol
        side = order.side  # "buy" or "sell"

        timestamp = datetime.now(_ET).strftime("%H:%M:%S")
        logger.info(
            "FILL: %s %d %s @ $%.2f",
            side.upper(),
            int(quantity),
            symbol,
            price,
        )

        # Log to DataBridge trade log
        event_type = "FILL"
        self._push_event(event_type, symbol, int(quantity), price, side.upper())

        # On a closing sell, calculate P&L and update Kelly stats
        if side == "sell" and symbol in self._entry_prices:
            entry = self._entry_prices.pop(symbol, price)
            trade_pnl = (price - entry) * quantity
            is_win = trade_pnl > 0

            self._trade_history[symbol].append(
                {"pnl": trade_pnl, "is_win": is_win}
            )
            self._circuit_breaker.update_daily_pnl(trade_pnl)

            logger.info(
                "CLOSED %s: P&L $%.2f (%s)",
                symbol,
                trade_pnl,
                "WIN" if is_win else "LOSS",
            )
        elif side == "buy":
            self._entry_prices[symbol] = price

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load all DeepScalperNet weights from WEIGHTS_DIR into self._models."""
        weights_path = Path(WEIGHTS_DIR)
        loaded = 0
        missing = []

        for ticker in SP100_TICKERS:
            pth_file = weights_path / f"{ticker}.pth"
            if not pth_file.exists():
                missing.append(ticker)
                continue

            model = DeepScalperNet(
                macro_dim=MACRO_DIM,
                lob_dim=LOB_DIM,
                priv_dim=PRIV_DIM,
                gru_hidden=GRU_HIDDEN,
                macro_embed=MACRO_EMBED_DIM,
                fc_hidden=FC_HIDDEN,
                n_dir=N_DIR,
                n_size=N_SIZE,
            )
            try:
                ckpt = torch.load(str(pth_file), map_location="cpu", weights_only=True)
                # Support both raw state_dict and agent checkpoint
                state_dict = ckpt.get("online_net", ckpt)
                model.load_state_dict(state_dict)
                model.eval()
                self._models[ticker] = model
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load weights for %s: %s", ticker, exc)
                missing.append(ticker)

        if missing:
            logger.warning(
                "Could not load weights for %d tickers: %s",
                len(missing),
                missing,
            )

        logger.info("Loaded %d/%d model weights.", loaded, len(SP100_TICKERS))

    def _process_ticker(
        self,
        ticker: str,
        portfolio_value: float,
        current_positions: dict,
    ) -> None:
        """Run inference for a single ticker and submit orders if warranted.

        Args:
            ticker: Stock symbol.
            portfolio_value: Current portfolio value in USD.
            current_positions: Dict mapping symbol → Position for open positions.
        """
        asset = Asset(ticker, asset_type="stock")

        # Fetch historical bars
        try:
            bars_obj = self.get_historical_prices(asset, LOOKBACK_BARS + 5, "minute")
        except Exception as exc:
            logger.debug("Could not fetch bars for %s: %s", ticker, exc)
            return

        if bars_obj is None:
            return

        bars = bars_obj.df
        if bars is None or len(bars) < LOOKBACK_BARS:
            logger.debug(
                "%s: insufficient bars (%d < %d), skipping",
                ticker,
                len(bars) if bars is not None else 0,
                LOOKBACK_BARS,
            )
            return

        # Determine current position flag and unrealized P&L for private state
        pos_obj = current_positions.get(ticker)
        position_flag = 0
        unrealized_pnl_pct = 0.0
        if pos_obj is not None and pos_obj.quantity != 0:
            position_flag = 1 if pos_obj.quantity > 0 else -1
            entry = self._entry_prices.get(ticker, 0.0)
            last_price = self._get_last_price(ticker) or 0.0
            if entry > 0 and last_price > 0:
                unrealized_pnl_pct = (last_price - entry) / entry * position_flag

        # Build dict observation
        obs = build_observation(bars, position=position_flag, unrealized_pnl_pct=unrealized_pnl_pct)

        # Run BDQ inference (no_grad prevents gradient accumulation)
        model = self._models[ticker]
        with torch.no_grad():
            q_dir, q_size, _ = model(
                obs['lob'], obs['priv'], obs['macro']
            )  # q_dir: (1, N_DIR), q_size: (1, N_SIZE)

        q_np = q_dir.squeeze(0).cpu().numpy()  # (N_DIR,) — use direction branch
        action = int(q_np.argmax())
        confidence = float(F.softmax(torch.from_numpy(q_np), dim=0).max())

        # Push signal to dashboard regardless of whether we trade
        self._publish_signal(ticker, action, q_np.tolist(), confidence)

        # Resolve whether we currently hold this ticker
        is_long = ticker in current_positions

        current_price = float(bars["close"].iloc[-1])

        # ---- BUY logic ----
        if action == ACTION_BUY and not is_long:
            self._submit_buy(ticker, asset, bars, current_price, portfolio_value)

        # ---- SELL logic ----
        elif action == ACTION_SELL and is_long:
            self._submit_sell(ticker, asset, current_price)

    def _submit_buy(
        self,
        ticker: str,
        asset: Asset,
        bars,
        current_price: float,
        portfolio_value: float,
    ) -> None:
        """Calculate position size and submit a bracket BUY order.

        Args:
            ticker: Stock symbol.
            asset: Lumibot Asset object.
            bars: Historical OHLCV DataFrame.
            current_price: Latest close price.
            portfolio_value: Current total portfolio value.
        """
        # Kelly position sizing using recent win/loss history
        history = list(self._trade_history[ticker])
        if len(history) >= self._MIN_TRADES_FOR_KELLY:
            wins = [h for h in history if h["is_win"]]
            losses = [h for h in history if not h["is_win"]]
            win_rate = len(wins) / len(history)
            avg_win = abs(sum(h["pnl"] for h in wins) / len(wins)) if wins else 1.0
            avg_loss = abs(sum(h["pnl"] for h in losses) / len(losses)) if losses else 1.0
        else:
            # Insufficient history: use conservative defaults
            win_rate = 0.5
            avg_win = current_price * 0.005   # 0.5 % average win
            avg_loss = current_price * 0.0025  # 0.25 % average loss

        shares = kelly_position_size(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            portfolio_value=portfolio_value,
            price=current_price,
            kelly_fraction=KELLY_FRACTION,
            max_position_pct=MAX_POSITION_PCT,
        )

        # ATR-based stops
        stop_price, tp_price = calculate_atr_stop(bars, current_price, "buy")
        self._stop_prices[ticker] = stop_price
        self._tp_prices[ticker] = tp_price

        try:
            order = self.create_order(
                asset,
                shares,
                "buy",
                order_class="bracket",
                stop_loss_price=stop_price,
                take_profit_price=tp_price,
            )
            self.submit_order(order)
            logger.info(
                "BUY %d %s @ ~$%.2f | stop=$%.2f tp=$%.2f",
                shares,
                ticker,
                current_price,
                stop_price,
                tp_price,
            )
        except Exception as exc:
            logger.error("Failed to submit BUY for %s: %s", ticker, exc)

    def _submit_sell(self, ticker: str, asset: Asset, current_price: float) -> None:
        """Submit a market SELL order to close an existing long position.

        Args:
            ticker: Stock symbol.
            asset: Lumibot Asset object.
            current_price: Latest close price (for logging only).
        """
        try:
            position = self.get_position(asset)
            if position and position.quantity > 0:
                order = self.create_order(asset, position.quantity, "sell")
                self.submit_order(order)
                logger.info(
                    "SELL %d %s @ ~$%.2f (model signal)",
                    int(position.quantity),
                    ticker,
                    current_price,
                )
        except Exception as exc:
            logger.error("Failed to submit SELL for %s: %s", ticker, exc)

    def _close_all_positions(self, reason: str = "EOD_CLOSE") -> None:
        """Flatten all open positions (EOD or circuit-breaker triggered).

        Args:
            reason: Human-readable reason string for the dashboard log.
        """
        logger.info("Closing all positions: %s", reason)
        positions = self.get_positions()
        if not positions:
            return

        for position in positions:
            try:
                self.sell_all()
                break  # sell_all covers everything; no need to loop further
            except Exception as exc:
                logger.error("sell_all failed: %s — trying per-position close", exc)
                for pos in self.get_positions():
                    try:
                        order = self.create_order(pos.asset, pos.quantity, "sell")
                        self.submit_order(order)
                    except Exception as e2:
                        logger.error("Could not close %s: %s", pos.asset.symbol, e2)
                break

        self._push_event(reason, "ALL", 0, 0.0, reason)

    def _is_eod_close_time(self) -> bool:
        """Return True if we are within EOD_CLOSE_BUFFER_MIN minutes of market close."""
        now = datetime.now(_ET)
        close_minute = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE
        now_minute = now.hour * 60 + now.minute
        return now_minute >= close_minute - EOD_CLOSE_BUFFER_MIN

    def _publish_signal(
        self,
        ticker: str,
        action: int,
        q_values: List[float],
        confidence: float,
    ) -> None:
        """Push the latest model signal to DataBridge for the dashboard.

        Args:
            ticker: Stock symbol.
            action: Predicted action index (0=HOLD, 1=BUY, 2=SELL).
            q_values: Raw Q-values as a 3-element list.
            confidence: Softmax-derived confidence score in [0, 1].
        """
        signal = ModelSignal(
            symbol=ticker,
            action=ACTION_NAMES[action],
            q_values=q_values,
            confidence=confidence,
            timestamp=datetime.now(_ET).strftime("%H:%M:%S"),
        )
        self._data_bridge.update_signal(ticker, signal)

    def _push_portfolio_snapshot(self, portfolio_value: float) -> None:
        """Update DataBridge with current portfolio value and open positions.

        Args:
            portfolio_value: Total portfolio value in USD.
        """
        self._data_bridge.portfolio_value = portfolio_value
        daily_pnl = self._circuit_breaker.daily_pnl if self._circuit_breaker else 0.0
        self._data_bridge.daily_pnl = daily_pnl

        # Push position snapshots
        positions = self.get_positions()
        snapshots: Dict[str, PositionSnapshot] = {}
        for pos in positions:
            symbol = pos.asset.symbol
            current_price = self._get_last_price(symbol) or 0.0
            entry = self._entry_prices.get(symbol, current_price)
            qty = int(pos.quantity)
            upnl = (current_price - entry) * qty

            snapshots[symbol] = PositionSnapshot(
                symbol=symbol,
                side="LONG" if qty > 0 else "SHORT",
                qty=abs(qty),
                entry_price=entry,
                current_price=current_price,
                unrealized_pnl=upnl,
                atr_stop=self._stop_prices.get(symbol, 0.0),
                atr_tp=self._tp_prices.get(symbol, 0.0),
            )

        self._data_bridge.update_positions(snapshots)

    def _push_event(
        self,
        event_type: str,
        symbol: str,
        qty: int,
        price: float,
        detail: str,
    ) -> None:
        """Append an event to the DataBridge trade log.

        Args:
            event_type: One of "FILL", "HALT", "EOD_CLOSE", etc.
            symbol: Stock symbol or "ALL".
            qty: Share quantity (0 for non-trade events).
            price: Fill price (0.0 for non-trade events).
            detail: Human-readable description or side string.
        """
        event = TradeEvent(
            timestamp=datetime.now(_ET).strftime("%H:%M:%S"),
            symbol=symbol,
            side=detail,
            qty=qty,
            price=price,
            event_type=event_type,
        )
        self._data_bridge.append_trade_event(event)

    def _get_last_price(self, symbol: str) -> Optional[float]:
        """Fetch the last available price for a symbol via Lumibot.

        Args:
            symbol: Stock ticker string.

        Returns:
            Last trade price as float, or None if unavailable.
        """
        try:
            asset = Asset(symbol, asset_type="stock")
            price = self.get_last_price(asset)
            return float(price) if price else None
        except Exception:
            return None
