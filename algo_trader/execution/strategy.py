"""
execution/strategy.py — Core Lumibot Strategy for equities intraday trading.

This strategy runs DeepScalper with 3-action semantics:
  0 = SHORT
  1 = FLAT
  2 = LONG

Operational policy:
- Trade only during regular US session hours.
- Respect open/close no-trade buffers via CircuitBreaker.
- Force-flat near market close and after market close.
"""

import logging
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
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
    KELLY_FRACTION,
    LOOKBACK_BARS,
    MAX_DAILY_LOSS_PCT,
    MAX_POSITION_PCT,
    SLEEP_TIME,
    TRADING_UNIVERSE,
    STARTING_CAPITAL,
    WEIGHTS_DIR,
    MIN_HOLD_BARS,
    ENTRY_COOLDOWN_BARS,
    USE_TRAILING_STOP,
    TRAILING_ATR_MULTIPLIER,
    TRAILING_STOP_FLOOR_PCT,
    USE_VOLATILITY_SIZING,
    TARGET_ENTRY_RISK_PCT,
    MAX_ORDER_NOTIONAL,
    MAX_SYMBOL_NOTIONAL,
    MAX_TOTAL_NOTIONAL,
    RUN_MODE,
    DATA_FEED,
    LONG_ONLY,
    ALLOW_PYRAMIDING,
    MIN_POSITION_SCALE,
    MAX_POSITION_SCALE,
    CLOSE_ALL_EOD,
    MARKET_TIMEZONE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    EOD_CLOSE_BUFFER_MIN,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    DATA_ADJUSTMENT,
    EXECUTION_STATE_PATH,
    NO_ORDER_MODES,
    PAPER_ORDER_MODES,
    RUN_MODES,
)
from dashboard.data_bridge import DataBridge, ModelSignal, PositionSnapshot, TradeEvent
from execution.circuit_breakers import CircuitBreaker
from execution.risk import (
    calculate_atr_stop,
    capped_entry_quantity,
    kelly_position_size,
    weighted_fill_price,
)
from execution.state_builder import build_observation
from execution.market_data import AlpacaBarSource
from execution.state_store import ExecutionStateStore
from execution.validation import validate_bars

_COLAB_PATH = Path(__file__).parent.parent / "colab"
if str(_COLAB_PATH) not in sys.path:
    sys.path.insert(0, str(_COLAB_PATH))

from deepscalper.architecture import DeepScalperNet  # noqa: E402
from deepscalper.utils import compute_micro_features  # noqa: E402
from colab.deepscalper.policy import (  # noqa: E402
    PolicyConfig,
    decide_target_position,
)

from lumibot.entities import Asset
from lumibot.strategies import Strategy

logger = logging.getLogger(__name__)

ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2
ACTION_NAMES = {
    ACTION_SHORT: "SHORT",
    ACTION_FLAT: "FLAT",
    ACTION_LONG: "LONG",
}


class EquityDeepScalper(Strategy):
    """Intraday equity strategy using 3-action DeepScalper policy."""

    _MIN_TRADES_FOR_KELLY: int = 5
    _ENTRY_Q_EDGE_MIN: float = 0.02
    _ENTRY_CONFIDENCE_MIN: float = 0.58

    @property
    def execution_policy(self) -> PolicyConfig:
        """Single source of truth for the decision rules.

        Shared verbatim with the training/approval evaluator so validation
        reflects the filters, holding period, and cooldown that paper
        execution actually applies.
        """
        return PolicyConfig(
            long_only=LONG_ONLY,
            allow_pyramiding=ALLOW_PYRAMIDING,
            entry_q_edge_min=self._ENTRY_Q_EDGE_MIN,
            entry_confidence_min=self._ENTRY_CONFIDENCE_MIN,
            min_hold_bars=MIN_HOLD_BARS,
            entry_cooldown_bars=ENTRY_COOLDOWN_BARS,
        )

    def __init__(
        self,
        data_bridge: DataBridge,
        alpaca_api_key: str,
        alpaca_secret_key: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._data_bridge = data_bridge
        self._models: Dict[str, DeepScalperNet] = {}
        self._circuit_breaker: Optional[CircuitBreaker] = None

        self._trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._entry_prices: Dict[str, float] = {}
        self._entry_side: Dict[str, str] = {}
        self._stop_prices: Dict[str, float] = {}
        self._tp_prices: Dict[str, float] = {}
        self._peak_prices: Dict[str, float] = {}
        self._trough_prices: Dict[str, float] = {}

        self._iteration_index: int = 0
        self._entry_iteration: Dict[str, int] = {}
        self._last_exit_iteration: Dict[str, int] = {}
        self._exit_iteration: Dict[str, int] = {}
        self._order_state: Dict[str, dict] = {}
        self._private_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=LOOKBACK_BARS)
        )
        self._filled_entry_qty: Dict[str, float] = {}
        self._last_bar_timestamp: Dict[str, pd.Timestamp] = {}
        self._state_store = ExecutionStateStore(EXECUTION_STATE_PATH)
        self._entries_enabled = False
        self._shutdown_requested = False
        self._calendar_cache: Dict[date, datetime] = {}
        self._bar_source: Optional[AlpacaBarSource] = None
        self._exit_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)

        self._tz = pytz.timezone(MARKET_TIMEZONE)

    def initialize(self) -> None:
        if RUN_MODE not in RUN_MODES:
            raise ValueError(f"Unsupported run mode: {RUN_MODE}")
        self.sleeptime = SLEEP_TIME

        logger.info("=" * 60)
        logger.info("EquityDeepScalper — initialising")
        logger.info("Capital: $%.2f | Universe: %s", STARTING_CAPITAL, TRADING_UNIVERSE)
        logger.info("=" * 60)

        self._load_models()
        self._circuit_breaker = CircuitBreaker(MAX_DAILY_LOSS_PCT, STARTING_CAPITAL)
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            self._bar_source = AlpacaBarSource(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                feed=DATA_FEED,
                adjustment=DATA_ADJUSTMENT,
            )
        self._restore_persisted_state()
        self._reconcile_broker_state()
        portfolio = float(self.get_portfolio_value())
        self._restore_session_loss(portfolio)
        self._data_bridge.portfolio_value = portfolio
        self._entries_enabled = True

        logger.info(
            "Initialisation complete. %d/%d models loaded.",
            len(self._models),
            len(TRADING_UNIVERSE),
        )

    def before_market_opens(self) -> None:
        if self._circuit_breaker:
            self._circuit_breaker.reset_for_new_day()
        portfolio = float(self.get_portfolio_value())
        self._state_store.set_session(
            date=datetime.now(self._tz).date().isoformat(),
            baseline=portfolio,
            halted=False,
        )
        self._data_bridge.is_halted = False
        self._data_bridge.halt_reason = ""

    def after_market_closes(self) -> None:
        if CLOSE_ALL_EOD:
            self._close_all_positions(reason="EOD_CLOSE")

        portfolio = self.get_portfolio_value()
        pnl = portfolio - STARTING_CAPITAL
        logger.info("Market closed. Portfolio: $%.2f | Daily P&L: $%.2f", portfolio, pnl)

    def on_trading_iteration(self) -> None:
        if self._shutdown_requested:
            return
        now_et = datetime.now(self._tz)
        if not self._is_regular_session(now_et):
            return

        self._reconcile_broker_state()
        self._iteration_index += 1

        if self._is_eod_close_window(now_et) and CLOSE_ALL_EOD:
            self._close_all_positions(reason="EOD_CLOSE_WINDOW")
            return

        entry_halted = not self._entries_enabled
        if self._circuit_breaker:
            self._update_session_loss(float(self.get_portfolio_value()))
            halted, reason = self._circuit_breaker.is_trading_halted()
            if halted:
                entry_halted = True
                self._data_bridge.is_halted = True
                self._data_bridge.halt_reason = reason
                self._push_event("HALT", "ALL", 0.0, 0.0, reason)
                logger.info("Trading halted: %s", reason)
            else:
                self._data_bridge.is_halted = False
                self._data_bridge.halt_reason = ""

        portfolio_value = self.get_portfolio_value()
        current_positions = {str(p.asset.symbol): p for p in self.get_positions()}

        for symbol in TRADING_UNIVERSE:
            if symbol not in self._models:
                continue
            self._process_symbol(
                symbol,
                portfolio_value,
                current_positions,
                entry_allowed=not entry_halted,
            )

        self._push_portfolio_snapshot(portfolio_value)

    def on_filled_order(self, position, order, price, quantity, multiplier) -> None:
        symbol = str(order.asset.symbol)
        side = str(order.side).lower()
        qty = abs(float(quantity))
        fill_price = float(price)

        self._push_event("FILL", symbol, qty, float(price), side.upper())

        existing_side = self._entry_side.get(symbol)
        trade_pnl = None

        client_id = self._client_order_id(order)
        pending = self._order_state.get(client_id)
        if pending:
            filled = float(pending.get("filled_qty", 0.0)) + qty
            remaining = max(0.0, float(pending.get("qty", 0.0)) - filled)
            pending.update(
                filled_qty=filled,
                remaining_qty=remaining,
                status="filled" if remaining <= 0 else "partially_filled",
            )
            self._state_store.set_order(client_id, pending)

        if existing_side == "buy" and side == "sell":
            entry = self._entry_prices.get(symbol, fill_price)
            trade_pnl = (float(price) - entry) * qty
            if not position or abs(float(getattr(position, "quantity", 0.0))) <= 0:
                self._clear_symbol_state(symbol)
        elif existing_side == "sell" and side == "buy":
            entry = self._entry_prices.get(symbol, fill_price)
            trade_pnl = (entry - float(price)) * qty
            if not position or abs(float(getattr(position, "quantity", 0.0))) <= 0:
                self._clear_symbol_state(symbol)
        elif existing_side is None or existing_side == side:
            self._entry_side[symbol] = side
            old_qty = self._filled_entry_qty.get(symbol, 0.0)
            old_price = self._entry_prices.get(symbol, fill_price)
            self._entry_prices[symbol] = (
                weighted_fill_price(old_qty, old_price, qty, fill_price)
                if old_qty > 0 else fill_price
            )
            self._filled_entry_qty[symbol] = old_qty + qty
            self._persist_symbol_state(symbol, position)

        if trade_pnl is not None:
            is_win = trade_pnl > 0
            self._trade_history[symbol].append({"pnl": trade_pnl, "is_win": is_win})
            if self._circuit_breaker:
                self._circuit_breaker.update_daily_pnl(trade_pnl)
            logger.info("CLOSED %s: P&L $%.2f (%s)", symbol, trade_pnl, "WIN" if is_win else "LOSS")

    def on_partially_filled_order(self, position, order, price, quantity, multiplier) -> None:
        self.on_filled_order(position, order, price, quantity, multiplier)

    def on_canceled_order(self, order) -> None:
        client_id = self._client_order_id(order)
        pending = self._order_state.get(client_id)
        if pending:
            pending["status"] = "canceled"
            self._state_store.set_order(client_id, pending)

    def _load_models(self) -> None:
        weights_path = Path(WEIGHTS_DIR)
        loaded = 0
        missing = []

        for symbol in TRADING_UNIVERSE:
            pth_file = weights_path / f"{symbol.replace('/', '_')}.pth"
            if not pth_file.exists():
                missing.append(symbol)
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
                state_dict = ckpt.get("online_net", ckpt)
                model.load_state_dict(state_dict)
                model.eval()
                self._models[symbol] = model
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load weights for %s: %s", symbol, exc)
                missing.append(symbol)

        if missing:
            raise RuntimeError(f"Required model checkpoint(s) could not be loaded: {missing}")
        logger.info("Loaded %d/%d model weights.", loaded, len(TRADING_UNIVERSE))

    @staticmethod
    def _get_equity_asset(symbol: str) -> Asset:
        return Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)

    def _proxy_lob_features(self, bars: pd.DataFrame) -> np.ndarray:
        micro = compute_micro_features(bars, use_proxy=True)
        return micro[-LOOKBACK_BARS:].astype(np.float32)

    def _process_symbol(
        self,
        symbol: str,
        portfolio_value: float,
        current_positions: dict,
        *,
        entry_allowed: bool = True,
    ) -> None:
        asset = self._get_equity_asset(symbol)

        try:
            if self._bar_source is not None:
                bars = self._bar_source.completed_minute_bars(
                    symbol, LOOKBACK_BARS + 5
                )
            else:
                bars_obj = self.get_historical_prices(
                    asset, LOOKBACK_BARS + 5, "minute"
                )
                bars = bars_obj.df if bars_obj is not None else None
        except Exception as exc:
            logger.warning("Skipping %s entry: bar request failed: %s", symbol, exc)
            self._manage_position_without_signal(symbol, asset, current_positions.get(symbol))
            return

        if bars is None:
            self._manage_position_without_signal(symbol, asset, current_positions.get(symbol))
            return

        if len(bars) < LOOKBACK_BARS:
            logger.info("Skipping %s: only %d/%d bars", symbol, len(bars), LOOKBACK_BARS)
            self._manage_position_without_signal(symbol, asset, current_positions.get(symbol))
            return
        valid, reason = validate_bars(bars, now=pd.Timestamp.now(tz="UTC"))
        if not valid:
            logger.warning("Skipping %s entry: %s", symbol, reason)
            self._push_event("DATA_SKIP", symbol, 0, 0, reason)
            self._manage_position_without_signal(symbol, asset, current_positions.get(symbol))
            return
        bar_timestamp = bars.index[-1]
        if self._last_bar_timestamp.get(symbol) == bar_timestamp:
            logger.info("Skipping %s: bar %s already processed", symbol, bar_timestamp)
            self._manage_position_without_signal(symbol, asset, current_positions.get(symbol))
            return
        self._last_bar_timestamp[symbol] = bar_timestamp
        bar_age = (pd.Timestamp.now(tz="UTC") - bar_timestamp).total_seconds()
        logger.info(
            "Processing %s bar=%s age=%.1fs feed=%s adjustment=%s",
            symbol,
            bar_timestamp,
            bar_age,
            DATA_FEED,
            DATA_ADJUSTMENT,
        )

        pos_obj = current_positions.get(symbol)
        position_flag = 0
        unrealized_pnl_pct = 0.0
        qty = 0.0

        if pos_obj is not None:
            qty = float(pos_obj.quantity)
            if qty > 0:
                position_flag = 1
            elif qty < 0:
                position_flag = -1

        current_price = float(bars["close"].iloc[-1])
        entry = self._entry_prices.get(symbol, current_price)
        if position_flag == 1 and entry > 0:
            unrealized_pnl_pct = (current_price - entry) / entry
        elif position_flag == -1 and entry > 0:
            unrealized_pnl_pct = (entry - current_price) / entry
        self._private_history[symbol].append(
            (float(position_flag), float(unrealized_pnl_pct))
        )

        obs = build_observation(
            bars,
            position=position_flag,
            unrealized_pnl_pct=unrealized_pnl_pct,
            private_history=list(self._private_history[symbol]),
            lob_override=self._proxy_lob_features(bars),
        )

        model = self._models[symbol]
        with torch.no_grad():
            q_dir, _q_size = model(obs["lob"], obs["priv"], obs["macro"])

        q_np = q_dir.squeeze(0).cpu().numpy()
        action = int(q_np.argmax())
        confidence = float(F.softmax(torch.from_numpy(q_np), dim=0).max())

        risk_exit_submitted = False
        if position_flag != 0:
            if position_flag == 1:
                self._peak_prices[symbol] = max(self._peak_prices.get(symbol, current_price), current_price)
                risk_exit_submitted = self._risk_exit_check(symbol, asset, bars, current_price, side="buy")
            else:
                self._trough_prices[symbol] = min(self._trough_prices.get(symbol, current_price), current_price)
                risk_exit_submitted = self._risk_exit_check(symbol, asset, bars, current_price, side="sell")

        self._publish_signal(symbol, action, q_np.tolist(), confidence)
        if risk_exit_submitted:
            # A protective exit suppresses any model-driven change this iteration.
            logger.info("%s decision suppressed by protective exit", symbol)
            return

        decision = decide_target_position(
            q_np.tolist(),
            position_flag,
            bars_held=self._holding_bars(symbol),
            bars_since_exit=self._bars_since_exit(symbol),
            config=self.execution_policy,
            entries_enabled=entry_allowed,
        )
        logger.info(
            "%s decision=%s target=%s confidence=%.4f position=%s",
            symbol,
            decision.reason,
            decision.target_position,
            decision.confidence,
            position_flag,
        )
        if not decision.changes_position(position_flag):
            return

        if decision.target_position == 0:
            self._submit_exit_flat(symbol, asset, current_price, reason=decision.reason)
        elif position_flag != 0:
            # Reversals close and confirm before any opposite entry is submitted.
            self._submit_exit_flat(symbol, asset, current_price, reason=decision.reason)
        else:
            self._submit_entry(
                symbol,
                asset,
                bars,
                current_price,
                portfolio_value,
                side="buy" if decision.target_position > 0 else "sell",
            )

    def _submit_entry(
        self,
        symbol: str,
        asset: Asset,
        bars: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        side: str,
    ) -> None:
        if self._shutdown_requested or not self._entries_enabled:
            logger.info("Skipping %s entry: execution is stopping or recovery is incomplete", symbol)
            return
        history = list(self._trade_history[symbol])
        if len(history) >= self._MIN_TRADES_FOR_KELLY:
            wins = [h for h in history if h["is_win"]]
            losses = [h for h in history if not h["is_win"]]
            win_rate = len(wins) / len(history)
            avg_win = abs(sum(h["pnl"] for h in wins) / len(wins)) if wins else 1.0
            avg_loss = abs(sum(h["pnl"] for h in losses) / len(losses)) if losses else 1.0
        else:
            win_rate = 0.50
            avg_win = current_price * 0.005
            avg_loss = current_price * 0.0025

        qty = kelly_position_size(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            portfolio_value=portfolio_value,
            price=current_price,
            kelly_fraction=KELLY_FRACTION,
            max_position_pct=MAX_POSITION_PCT,
        )

        stop_price, tp_price = calculate_atr_stop(bars, current_price, side)

        if USE_VOLATILITY_SIZING:
            if side == "buy":
                stop_risk_pct = max((current_price - stop_price) / max(current_price, 1e-9), 1e-9)
            else:
                stop_risk_pct = max((stop_price - current_price) / max(current_price, 1e-9), 1e-9)
            raw_scale = TARGET_ENTRY_RISK_PCT / stop_risk_pct
            scale = float(np.clip(raw_scale, MIN_POSITION_SCALE, MAX_POSITION_SCALE))
            qty = int(qty * scale)

        symbol_notional, aggregate_notional, pending_notional = self._exposure_snapshot(
            symbol, current_price
        )
        pending_buy_notional = sum(
            float(state.get("remaining_qty", state.get("qty", 0.0)))
            * float(state.get("estimated_price", 0.0))
            for state in self._order_state.values()
            if state.get("status") not in {"filled", "canceled", "rejected"}
            and state.get("side") == "buy"
        )
        buying_power = max(0.0, float(self.get_cash()) - pending_buy_notional)
        qty = min(qty, int(buying_power / current_price))
        qty = capped_entry_quantity(
            qty,
            current_price,
            symbol_notional=symbol_notional,
            aggregate_notional=aggregate_notional,
            pending_notional=pending_notional,
            order_cap=MAX_ORDER_NOTIONAL,
            symbol_cap=MAX_SYMBOL_NOTIONAL,
            aggregate_cap=MAX_TOTAL_NOTIONAL,
        )
        notional = qty * current_price
        if qty <= 0 or notional > MAX_ORDER_NOTIONAL:
            logger.info("Skipping %s entry: quantity/notional outside cap", symbol)
            return

        try:
            client_id = self._state_store.next_client_order_id(symbol, "entry")
            order = self.create_order(
                asset,
                qty,
                side,
                order_class="bracket",
                secondary_limit_price=tp_price,
                secondary_stop_price=stop_price,
                time_in_force="day",
            )
            order.custom_params = {"client_order_id": client_id}
            if RUN_MODE in {"offline", "dry-run", "preflight"}:
                logger.info("NO ORDER (%s): %s %d %s", RUN_MODE, side, qty, symbol)
                return
            state = {
                "symbol": symbol,
                "intent": "entry",
                "side": side,
                "qty": qty,
                "filled_qty": 0.0,
                "remaining_qty": qty,
                "estimated_price": current_price,
                "status": "submitting",
            }
            self._order_state[client_id] = state
            self._state_store.set_order(client_id, state)
            submitted = self.submit_order(order)
            status = str(getattr(submitted, "status", "submitted")).lower()
            if getattr(submitted, "error_message", None):
                status = "rejected"
            state["status"] = status
            state["broker_order_id"] = str(getattr(submitted, "identifier", ""))
            self._state_store.set_order(client_id, state)
            if status in {"error", "rejected"}:
                logger.error("Entry rejected for %s: %s", symbol, getattr(submitted, "error_message", ""))
                return
            self._entry_iteration[symbol] = self._iteration_index
            self._entry_side[symbol] = side
            self._stop_prices[symbol] = stop_price
            self._tp_prices[symbol] = tp_price
            if side == "buy":
                self._peak_prices[symbol] = current_price
            else:
                self._trough_prices[symbol] = current_price
            logger.info(
                "%s %d %s @ ~$%.2f | stop=$%.2f tp=$%.2f",
                side.upper(),
                qty,
                symbol,
                current_price,
                stop_price,
                tp_price,
            )
        except Exception as exc:
            logger.error("Failed to submit %s for %s: %s", side.upper(), symbol, exc)

    def _submit_exit_flat(self, symbol: str, asset: Asset, current_price: float, reason: str) -> None:
        if not hasattr(self, "_exit_locks"):
            self._exit_locks = defaultdict(threading.RLock)
        with self._exit_locks[symbol]:
            if self._exit_iteration.get(symbol) == self._iteration_index:
                logger.info("Skipping duplicate exit for %s in iteration %d", symbol, self._iteration_index)
                return
            if self._has_pending_exit(symbol):
                logger.info("Exit already pending for %s", symbol)
                return
            try:
                if RUN_MODE not in NO_ORDER_MODES:
                    self._wait_for_symbol_orders(symbol)
                # Re-read only after cancellations and late fills are reconciled.
                position = self.get_position(asset)
                if not position:
                    return
                qty = float(position.quantity)
                if qty == 0:
                    return
                side = "sell" if qty > 0 else "buy"
                order = self.create_order(asset, abs(qty), side)
                client_id = self._state_store.next_client_order_id(symbol, "exit")
                order.custom_params = {"client_order_id": client_id}
                if RUN_MODE in {"offline", "dry-run", "preflight"}:
                    logger.info("NO ORDER (%s): EXIT %.2f %s", RUN_MODE, abs(qty), symbol)
                    return
                state = {
                    "symbol": symbol,
                    "intent": "exit",
                    "side": side,
                    "qty": abs(qty),
                    "filled_qty": 0.0,
                    "remaining_qty": abs(qty),
                    "estimated_price": current_price,
                    "status": "submitting",
                    "reason": reason,
                }
                self._order_state[client_id] = state
                self._state_store.set_order(client_id, state)
                submitted = self.submit_order(order)
                status = str(getattr(submitted, "status", "submitted")).lower()
                if getattr(submitted, "error_message", None):
                    status = "rejected"
                state["status"] = status
                state["broker_order_id"] = str(getattr(submitted, "identifier", ""))
                self._state_store.set_order(client_id, state)
                if status in {"error", "rejected"}:
                    logger.error("Exit rejected for %s; protection remains active", symbol)
                    return
                self._exit_iteration[symbol] = self._iteration_index
                self._last_exit_iteration[symbol] = self._iteration_index
                logger.info("EXIT %s %.2f %s @ ~$%.2f (%s)", side.upper(), abs(qty), symbol, current_price, reason)
            except Exception as exc:
                logger.error("Failed to submit EXIT for %s: %s", symbol, exc)

    def _risk_exit_check(self, symbol: str, asset: Asset, bars: pd.DataFrame, current_price: float, side: str) -> bool:
        active_stop = self._stop_prices.get(symbol, 0.0)
        active_tp = self._tp_prices.get(symbol, 0.0)

        if USE_TRAILING_STOP:
            trailing = self._compute_trailing_stop(symbol, bars, current_price, side)
            if side == "buy" and trailing is not None and trailing > active_stop:
                self._stop_prices[symbol] = trailing
                active_stop = trailing
            elif side == "sell" and trailing is not None and (active_stop <= 0 or trailing < active_stop):
                self._stop_prices[symbol] = trailing
                active_stop = trailing

        if side == "buy":
            if active_stop > 0 and current_price <= active_stop:
                before = self._exit_iteration.get(symbol)
                self._submit_exit_flat(symbol, asset, current_price, reason="RISK_STOP")
                return before != self._exit_iteration.get(symbol)
            if active_tp > 0 and current_price >= active_tp:
                before = self._exit_iteration.get(symbol)
                self._submit_exit_flat(symbol, asset, current_price, reason="TAKE_PROFIT")
                return before != self._exit_iteration.get(symbol)
        else:
            if active_stop > 0 and current_price >= active_stop:
                before = self._exit_iteration.get(symbol)
                self._submit_exit_flat(symbol, asset, current_price, reason="RISK_STOP")
                return before != self._exit_iteration.get(symbol)
            if active_tp > 0 and current_price <= active_tp:
                before = self._exit_iteration.get(symbol)
                self._submit_exit_flat(symbol, asset, current_price, reason="TAKE_PROFIT")
                return before != self._exit_iteration.get(symbol)
        return False

    def _holding_bars(self, symbol: str) -> int:
        entry_iter = self._entry_iteration.get(symbol)
        if entry_iter is None:
            return MIN_HOLD_BARS
        return max(0, self._iteration_index - entry_iter)

    def _bars_since_exit(self, symbol: str) -> Optional[int]:
        exit_iter = self._last_exit_iteration.get(symbol)
        if exit_iter is None:
            return None
        return max(0, self._iteration_index - exit_iter)

    def _in_entry_cooldown(self, symbol: str) -> bool:
        bars_since_exit = self._bars_since_exit(symbol)
        if bars_since_exit is None:
            return False
        return bars_since_exit < ENTRY_COOLDOWN_BARS

    def _compute_trailing_stop(self, symbol: str, bars: pd.DataFrame, current_price: float, side: str) -> Optional[float]:
        if len(bars) < 2:
            return None

        close = bars["close"].astype(float)
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(tr.tail(14).mean()) if len(tr) >= 14 else float(tr.mean())

        atr_dist = TRAILING_ATR_MULTIPLIER * atr
        floor_dist = TRAILING_STOP_FLOOR_PCT * current_price
        trail_dist = max(atr_dist, floor_dist)

        if side == "buy":
            peak = max(self._peak_prices.get(symbol, current_price), current_price)
            return float(peak - trail_dist)

        trough = min(self._trough_prices.get(symbol, current_price), current_price)
        return float(trough + trail_dist)

    def _has_pending_exit(self, symbol: str) -> bool:
        return any(
            state.get("symbol") == symbol
            and state.get("intent") == "exit"
            and state.get("status") not in {"filled", "canceled", "rejected", "error"}
            and float(state.get("remaining_qty", state.get("qty", 0.0))) > 0
            for state in self._order_state.values()
        )

    def _exposure_snapshot(self, symbol: str, current_price: float) -> tuple[float, float, float]:
        symbol_notional = 0.0
        aggregate_notional = 0.0
        for position in self.get_positions():
            position_symbol = str(position.asset.symbol)
            qty = abs(float(position.quantity))
            price = (
                current_price
                if position_symbol == symbol
                else self._get_last_price_equity(position_symbol)
            )
            if not price:
                price = float(getattr(position, "avg_fill_price", 0.0) or 0.0)
            notional = qty * float(price)
            aggregate_notional += notional
            if position_symbol == symbol:
                symbol_notional += notional

        pending_states = [
            state
            for state in self._order_state.values()
            if state.get("intent") == "entry"
            and state.get("status") not in {"filled", "canceled", "rejected", "error"}
        ]
        pending_notional = sum(
            float(state.get("remaining_qty", state.get("qty", 0.0)))
            * float(state.get("estimated_price", 0.0))
            for state in pending_states
        )
        symbol_notional += sum(
            float(state.get("remaining_qty", state.get("qty", 0.0)))
            * float(state.get("estimated_price", current_price))
            for state in pending_states
            if state.get("symbol") == symbol
        )
        return symbol_notional, aggregate_notional, pending_notional

    def _client_order_id(self, order) -> str:
        custom = getattr(order, "custom_params", {}) or {}
        if custom.get("client_order_id"):
            return str(custom["client_order_id"])
        raw = getattr(order, "_raw", None)
        raw_client_id = getattr(raw, "client_order_id", None)
        if raw_client_id:
            return str(raw_client_id)
        broker_id = str(getattr(order, "identifier", ""))
        for client_id, state in self._order_state.items():
            if str(state.get("broker_order_id", "")) == broker_id:
                return client_id
        return broker_id

    def _persist_symbol_state(self, symbol: str, position=None) -> None:
        quantity = float(getattr(position, "quantity", 0.0)) if position else 0.0
        self._state_store.set_symbol(
            symbol,
            {
                "quantity": quantity,
                "entry_side": self._entry_side.get(symbol),
                "entry_price": self._entry_prices.get(symbol),
                "filled_entry_qty": self._filled_entry_qty.get(symbol, abs(quantity)),
                "stop_price": self._stop_prices.get(symbol),
                "target_price": self._tp_prices.get(symbol),
                "peak_price": self._peak_prices.get(symbol),
                "trough_price": self._trough_prices.get(symbol),
            },
        )

    def _restore_persisted_state(self) -> None:
        snapshot = self._state_store.snapshot()
        self._order_state = snapshot["orders"]
        for symbol, state in snapshot["symbols"].items():
            if state.get("entry_side"):
                self._entry_side[symbol] = state["entry_side"]
            for target, key in (
                (self._entry_prices, "entry_price"),
                (self._filled_entry_qty, "filled_entry_qty"),
                (self._stop_prices, "stop_price"),
                (self._tp_prices, "target_price"),
                (self._peak_prices, "peak_price"),
                (self._trough_prices, "trough_price"),
            ):
                if state.get(key) is not None:
                    target[symbol] = float(state[key])

    def _reconcile_broker_state(self) -> None:
        positions = {str(position.asset.symbol): position for position in self.get_positions()}
        broker_orders = list(self.get_orders())
        seen_client_ids: set[str] = set()
        for order in broker_orders:
            client_id = self._client_order_id(order)
            if not client_id:
                continue
            seen_client_ids.add(client_id)
            state = self._order_state.get(client_id)
            if state is None:
                continue
            status = str(getattr(order, "status", state.get("status", "unknown"))).lower()
            state["status"] = status
            state["broker_order_id"] = str(getattr(order, "identifier", ""))
            self._state_store.set_order(client_id, state)

        uncertain = False
        terminal = {"filled", "fill", "canceled", "cancelled", "rejected", "error", "expired"}
        for client_id, state in self._order_state.items():
            if state.get("status") in terminal or client_id in seen_client_ids:
                continue
            api = getattr(getattr(self, "broker", None), "api", None)
            lookup = getattr(api, "get_order_by_client_id", None)
            if lookup is None:
                if state.get("status") in {"submitting", "unknown"}:
                    uncertain = True
                continue
            try:
                broker_order = lookup(client_id)
            except Exception:
                uncertain = True
                state["status"] = "unknown"
            else:
                state["status"] = str(getattr(broker_order, "status", "unknown")).lower()
                state["broker_order_id"] = str(getattr(broker_order, "id", ""))
            self._state_store.set_order(client_id, state)

        for symbol, position in positions.items():
            quantity = float(position.quantity)
            if quantity == 0:
                continue
            self._entry_side[symbol] = "buy" if quantity > 0 else "sell"
            broker_average = getattr(position, "avg_fill_price", None)
            if broker_average:
                self._entry_prices[symbol] = float(broker_average)
            self._filled_entry_qty[symbol] = abs(quantity)
            self._persist_symbol_state(symbol, position)

        for symbol in list(self._entry_side):
            if symbol not in positions and not self._has_pending_exit(symbol):
                has_pending_entry = any(
                    state.get("symbol") == symbol
                    and state.get("intent") == "entry"
                    and state.get("status") not in terminal
                    for state in self._order_state.values()
                )
                if not has_pending_entry:
                    self._clear_symbol_state(symbol)
        self._entries_enabled = not uncertain and not self._shutdown_requested
        if uncertain:
            logger.error("Entry disabled: one or more persisted submissions are unreconciled")

    def _restore_session_loss(self, portfolio_value: float) -> None:
        session = self._state_store.snapshot()["session"]
        today = datetime.now(self._tz).date().isoformat()
        if session.get("date") != today:
            session = {"date": today, "baseline": portfolio_value, "halted": False}
            self._state_store.set_session(**session)
        pnl = portfolio_value - float(session.get("baseline", portfolio_value))
        if self._circuit_breaker:
            self._circuit_breaker.set_daily_pnl(pnl)
            if session.get("halted"):
                self._circuit_breaker.restore_latched_loss(pnl)

    def _update_session_loss(self, portfolio_value: float) -> None:
        session = self._state_store.snapshot()["session"]
        baseline = float(session.get("baseline", portfolio_value))
        pnl = portfolio_value - baseline
        if self._circuit_breaker:
            self._circuit_breaker.set_daily_pnl(pnl)
            self._state_store.set_session(
                date=datetime.now(self._tz).date().isoformat(),
                baseline=baseline,
                pnl=pnl,
                halted=self._circuit_breaker.is_halted_by_loss,
            )

    def _manage_position_without_signal(self, symbol: str, asset: Asset, position) -> None:
        if position is None or float(position.quantity) == 0:
            return
        current_price = self._get_last_price_equity(symbol)
        if current_price is None:
            logger.error("Cannot manage %s protection: no fresh trade price", symbol)
            return
        side = "buy" if float(position.quantity) > 0 else "sell"
        stop = float(self._stop_prices.get(symbol, 0.0))
        target = float(self._tp_prices.get(symbol, 0.0))
        breached = (
            side == "buy" and ((stop > 0 and current_price <= stop) or (target > 0 and current_price >= target))
        ) or (
            side == "sell" and ((stop > 0 and current_price >= stop) or (target > 0 and current_price <= target))
        )
        if breached:
            self._submit_exit_flat(symbol, asset, current_price, reason="PROTECTION_WITH_BAD_BARS")

    def _cancel_active_orders_for_symbol(self, symbol: str) -> None:
        terminal = {"filled", "fill", "canceled", "cancelled", "rejected", "expired"}
        for order in self.get_orders():
            if str(order.asset.symbol) != symbol:
                continue
            if str(getattr(order, "status", "")).lower() in terminal:
                continue
            self.cancel_order(order)

    def _wait_for_symbol_orders(self, symbol: str, timeout: float = 5.0) -> None:
        """Cancel and reconcile before calculating the close quantity."""
        terminal = {"filled", "fill", "canceled", "cancelled", "rejected", "expired"}
        self._cancel_active_orders_for_symbol(symbol)
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            active = [
                order for order in self.get_orders()
                if str(order.asset.symbol) == symbol
                and str(getattr(order, "status", "")).lower() not in terminal
            ]
            if not active:
                return
            self._reconcile_broker_state()
            self.sleep(0.05)
        self._reconcile_broker_state()
        active = [
            order for order in self.get_orders()
            if str(order.asset.symbol) == symbol
            and str(getattr(order, "status", "")).lower() not in terminal
        ]
        if active:
            raise TimeoutError(f"Timed out cancelling active orders for {symbol}")

    def request_close_position(self, symbol: str, reason: str = "MANUAL_CLOSE") -> bool:
        if symbol not in TRADING_UNIVERSE:
            raise ValueError(f"Symbol {symbol!r} is outside the configured universe")
        asset = self._get_equity_asset(symbol)
        position = self.get_position(asset)
        if not position or float(position.quantity) == 0:
            return True
        self._submit_exit_flat(
            symbol,
            asset,
            self._get_last_price_equity(symbol) or self._entry_prices.get(symbol, 0.0),
            reason=reason,
        )
        reconciled = self.get_position(asset)
        return not reconciled or float(reconciled.quantity) == 0

    def emergency_cancel_and_flatten(self) -> bool:
        if RUN_MODE in NO_ORDER_MODES:
            logger.warning("Emergency broker mutations blocked in %s mode", RUN_MODE)
            return not self.get_positions()
        self._shutdown_requested = True
        self._entries_enabled = False
        for order in self.get_orders():
            status = str(getattr(order, "status", "")).lower()
            if status not in {"filled", "canceled", "cancelled", "rejected", "expired"}:
                self.cancel_order(order)
        self._close_all_positions(reason="EMERGENCY_FLATTEN")
        return False

    def graceful_shutdown(self, timeout_seconds: float = 30.0) -> bool:
        self._shutdown_requested = True
        self._entries_enabled = False
        if RUN_MODE in NO_ORDER_MODES:
            return not bool(self.get_positions())
        self.emergency_cancel_and_flatten()
        deadline = datetime.now().timestamp() + max(0.0, timeout_seconds)
        while datetime.now().timestamp() < deadline:
            self._reconcile_broker_state()
            active = [
                order
                for order in self.get_orders()
                if str(getattr(order, "status", "")).lower()
                not in {"filled", "canceled", "cancelled", "rejected", "expired"}
            ]
            if not self.get_positions() and not active:
                logger.info("Broker confirmed all strategy positions flat and orders terminal")
                return True
            self.sleep(1)
        logger.error("Shutdown timed out before broker confirmed flat/terminal state")
        return False

    def run_live(self) -> None:
        """Run through a retained Trader so shutdown can stop the engine."""
        from lumibot.traders import Trader

        self._trader = Trader()
        self._trader.add_strategy(self)
        self._trader.run_all()

    def stop_live_engine(self) -> None:
        trader = getattr(self, "_trader", None)
        if trader is not None:
            trader.stop_all()

    def _close_all_positions(self, reason: str = "CIRCUIT_BREAKER") -> None:
        positions = self.get_positions()
        if not positions:
            return

        logger.info("Closing all positions: %s", reason)
        for pos in positions:
            symbol = str(pos.asset.symbol)
            current_price = self._get_last_price_equity(symbol) or float(
                self._entry_prices.get(symbol, 0.0)
            )
            self._submit_exit_flat(
                symbol,
                pos.asset,
                current_price,
                reason=reason,
            )

        self._push_event(reason, "ALL", 0.0, 0.0, reason)

    def _publish_signal(self, symbol: str, action: int, q_values: List[float], confidence: float) -> None:
        signal = ModelSignal(
            symbol=symbol,
            action=ACTION_NAMES[action],
            q_values=q_values,
            confidence=confidence,
            timestamp=datetime.utcnow().strftime("%H:%M:%S UTC"),
        )
        self._data_bridge.update_signal(symbol, signal)

    def _push_portfolio_snapshot(self, portfolio_value: float) -> None:
        self._data_bridge.portfolio_value = portfolio_value
        self._data_bridge.daily_pnl = self._circuit_breaker.daily_pnl if self._circuit_breaker else 0.0

        snapshots: Dict[str, PositionSnapshot] = {}
        for pos in self.get_positions():
            symbol = str(pos.asset.symbol)
            qty = float(pos.quantity)
            if qty > 0:
                side = "LONG"
            elif qty < 0:
                side = "SHORT"
            else:
                side = "FLAT"

            current_price = self._get_last_price_equity(symbol) or 0.0
            entry = self._entry_prices.get(symbol, current_price)
            if qty > 0:
                upnl = (current_price - entry) * abs(qty)
            elif qty < 0:
                upnl = (entry - current_price) * abs(qty)
            else:
                upnl = 0.0

            snapshots[symbol] = PositionSnapshot(
                symbol=symbol,
                side=side,
                qty=int(abs(qty)),
                entry_price=float(entry),
                current_price=float(current_price),
                unrealized_pnl=float(upnl),
                atr_stop=float(self._stop_prices.get(symbol, 0.0)),
                atr_tp=float(self._tp_prices.get(symbol, 0.0)),
            )

        self._data_bridge.update_positions(snapshots)

    def _push_event(self, event_type: str, symbol: str, qty: float, price: float, detail: str) -> None:
        event = TradeEvent(
            timestamp=datetime.utcnow().strftime("%H:%M:%S UTC"),
            symbol=symbol,
            side=detail,
            qty=int(qty),
            price=float(price),
            event_type=event_type,
        )
        self._data_bridge.append_trade_event(event)

    def _get_last_price_equity(self, symbol: str) -> Optional[float]:
        try:
            asset = self._get_equity_asset(symbol)
            price = self.get_last_price(asset)
            return float(price) if price is not None else None
        except Exception:
            return None

    def _is_regular_session(self, now_et: datetime) -> bool:
        open_dt = now_et.replace(
            hour=9,
            minute=30,
            second=0,
            microsecond=0,
        )
        close_dt = self._session_close(now_et)
        return open_dt <= now_et < close_dt

    def _is_eod_close_window(self, now_et: datetime) -> bool:
        close_dt = self._session_close(now_et)
        window_start = close_dt - timedelta(minutes=EOD_CLOSE_BUFFER_MIN)
        return window_start <= now_et < close_dt

    def _session_close(self, now_et: datetime) -> datetime:
        session_date = now_et.date()
        cached = self._calendar_cache.get(session_date)
        if cached is not None:
            return cached
        fallback = now_et.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        api = getattr(getattr(self, "broker", None), "api", None)
        get_calendar = getattr(api, "get_calendar", None)
        if get_calendar is None:
            return fallback
        try:
            from alpaca.trading.requests import GetCalendarRequest

            sessions = get_calendar(
                GetCalendarRequest(start=session_date, end=session_date)
            )
            if not sessions:
                return fallback
            close_value = sessions[0].close
            if isinstance(close_value, datetime):
                close_dt = close_value
            else:
                close_dt = datetime.combine(session_date, close_value)
            if close_dt.tzinfo is None:
                close_dt = self._tz.localize(close_dt)
            else:
                close_dt = close_dt.astimezone(self._tz)
            self._calendar_cache[session_date] = close_dt
            return close_dt
        except Exception as exc:
            logger.error("Broker calendar lookup failed for %s: %s", session_date, exc)
            return fallback

    def _clear_symbol_state(self, symbol: str) -> None:
        self._entry_prices.pop(symbol, None)
        self._entry_side.pop(symbol, None)
        self._stop_prices.pop(symbol, None)
        self._tp_prices.pop(symbol, None)
        self._peak_prices.pop(symbol, None)
        self._trough_prices.pop(symbol, None)
        self._filled_entry_qty.pop(symbol, None)
        self._state_store.clear_symbol(symbol)


# Backward-compatible alias for legacy imports
CryptoDeepScalper = EquityDeepScalper
