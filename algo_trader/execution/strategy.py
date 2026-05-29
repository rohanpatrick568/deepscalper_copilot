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
from collections import defaultdict, deque
from datetime import datetime, timedelta
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
    MIN_POSITION_SCALE,
    MAX_POSITION_SCALE,
    CLOSE_ALL_EOD,
    MARKET_TIMEZONE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    EOD_CLOSE_BUFFER_MIN,
)
from dashboard.data_bridge import DataBridge, ModelSignal, PositionSnapshot, TradeEvent
from execution.circuit_breakers import CircuitBreaker
from execution.risk import calculate_atr_stop, kelly_position_size
from execution.state_builder import build_observation

_COLAB_PATH = Path(__file__).parent.parent / "colab"
if str(_COLAB_PATH) not in sys.path:
    sys.path.insert(0, str(_COLAB_PATH))

from deepscalper.architecture import DeepScalperNet  # noqa: E402
from deepscalper.utils import compute_micro_features  # noqa: E402

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

        self._tz = pytz.timezone(MARKET_TIMEZONE)

    def initialize(self) -> None:
        self.sleeptime = SLEEP_TIME

        logger.info("=" * 60)
        logger.info("EquityDeepScalper — initialising")
        logger.info("Capital: $%.2f | Universe: %s", STARTING_CAPITAL, TRADING_UNIVERSE)
        logger.info("=" * 60)

        self._load_models()
        self._circuit_breaker = CircuitBreaker(MAX_DAILY_LOSS_PCT, STARTING_CAPITAL)
        self._data_bridge.portfolio_value = STARTING_CAPITAL

        logger.info(
            "Initialisation complete. %d/%d models loaded.",
            len(self._models),
            len(TRADING_UNIVERSE),
        )

    def before_market_opens(self) -> None:
        if self._circuit_breaker:
            self._circuit_breaker.reset_for_new_day()
        self._data_bridge.is_halted = False
        self._data_bridge.halt_reason = ""

    def after_market_closes(self) -> None:
        if CLOSE_ALL_EOD:
            self._close_all_positions(reason="EOD_CLOSE")

        portfolio = self.get_portfolio_value()
        pnl = portfolio - STARTING_CAPITAL
        logger.info("Market closed. Portfolio: $%.2f | Daily P&L: $%.2f", portfolio, pnl)

    def on_trading_iteration(self) -> None:
        now_et = datetime.now(self._tz)
        if not self._is_regular_session(now_et):
            return

        self._iteration_index += 1

        if self._is_eod_close_window(now_et) and CLOSE_ALL_EOD:
            self._close_all_positions(reason="EOD_CLOSE_WINDOW")
            return

        if self._circuit_breaker:
            halted, reason = self._circuit_breaker.is_trading_halted()
            if halted:
                self._data_bridge.is_halted = True
                self._data_bridge.halt_reason = reason
                self._push_event("HALT", "ALL", 0.0, 0.0, reason)
                logger.info("Trading halted: %s", reason)
                return

        self._data_bridge.is_halted = False
        self._data_bridge.halt_reason = ""

        portfolio_value = self.get_portfolio_value()
        current_positions = {str(p.asset.symbol): p for p in self.get_positions()}

        for symbol in TRADING_UNIVERSE:
            if symbol not in self._models:
                continue
            self._process_symbol(symbol, portfolio_value, current_positions)

        self._push_portfolio_snapshot(portfolio_value)

    def on_filled_order(self, position, order, price, quantity, multiplier) -> None:
        symbol = str(order.asset.symbol)
        side = str(order.side).lower()
        qty = abs(float(quantity))

        self._push_event("FILL", symbol, qty, float(price), side.upper())

        existing_side = self._entry_side.get(symbol)
        trade_pnl = None

        if existing_side == "buy" and side == "sell":
            entry = self._entry_prices.get(symbol, float(price))
            trade_pnl = (float(price) - entry) * qty
            self._clear_symbol_state(symbol)
        elif existing_side == "sell" and side == "buy":
            entry = self._entry_prices.get(symbol, float(price))
            trade_pnl = (entry - float(price)) * qty
            self._clear_symbol_state(symbol)
        elif existing_side is None:
            self._entry_side[symbol] = side
            self._entry_prices[symbol] = float(price)

        if trade_pnl is not None:
            is_win = trade_pnl > 0
            self._trade_history[symbol].append({"pnl": trade_pnl, "is_win": is_win})
            if self._circuit_breaker:
                self._circuit_breaker.update_daily_pnl(trade_pnl)
            logger.info("CLOSED %s: P&L $%.2f (%s)", symbol, trade_pnl, "WIN" if is_win else "LOSS")

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
            logger.warning("Could not load weights for: %s", missing)
        logger.info("Loaded %d/%d model weights.", loaded, len(TRADING_UNIVERSE))

    @staticmethod
    def _get_equity_asset(symbol: str) -> Asset:
        return Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)

    def _proxy_lob_features(self, bars: pd.DataFrame) -> np.ndarray:
        micro = compute_micro_features(bars, use_proxy=True)
        return micro[-1:].astype(np.float32)

    def _process_symbol(self, symbol: str, portfolio_value: float, current_positions: dict) -> None:
        asset = self._get_equity_asset(symbol)

        try:
            bars_obj = self.get_historical_prices(asset, LOOKBACK_BARS + 5, "minute")
        except Exception as exc:
            logger.debug("Could not fetch bars for %s: %s", symbol, exc)
            return

        if bars_obj is None or bars_obj.df is None:
            return

        bars = bars_obj.df
        if len(bars) < LOOKBACK_BARS:
            return

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

        obs = build_observation(
            bars,
            position=position_flag,
            unrealized_pnl_pct=unrealized_pnl_pct,
            lob_override=self._proxy_lob_features(bars),
        )

        model = self._models[symbol]
        with torch.no_grad():
            q_dir, _q_size = model(obs["lob"], obs["priv"], obs["macro"])

        q_np = q_dir.squeeze(0).cpu().numpy()
        action = int(q_np.argmax())
        confidence = float(F.softmax(torch.from_numpy(q_np), dim=0).max())

        q_short = float(q_np[ACTION_SHORT])
        q_flat = float(q_np[ACTION_FLAT])
        q_long = float(q_np[ACTION_LONG])

        long_entry_allowed = (q_long - q_flat) > self._ENTRY_Q_EDGE_MIN and confidence >= self._ENTRY_CONFIDENCE_MIN
        short_entry_allowed = (q_short - q_flat) > self._ENTRY_Q_EDGE_MIN and confidence >= self._ENTRY_CONFIDENCE_MIN

        if position_flag != 0:
            if position_flag == 1:
                self._peak_prices[symbol] = max(self._peak_prices.get(symbol, current_price), current_price)
                self._risk_exit_check(symbol, asset, bars, current_price, side="buy")
            else:
                self._trough_prices[symbol] = min(self._trough_prices.get(symbol, current_price), current_price)
                self._risk_exit_check(symbol, asset, bars, current_price, side="sell")

        self._publish_signal(symbol, action, q_np.tolist(), confidence)

        hold_bars = self._holding_bars(symbol)
        in_cooldown = self._in_entry_cooldown(symbol)

        if action == ACTION_LONG:
            if position_flag == 1:
                return
            if position_flag == -1:
                if hold_bars >= MIN_HOLD_BARS:
                    self._submit_exit_flat(symbol, asset, current_price, reason="REVERSE_TO_LONG")
                return
            if not in_cooldown and long_entry_allowed:
                self._submit_entry(symbol, asset, bars, current_price, portfolio_value, side="buy")

        elif action == ACTION_SHORT:
            if position_flag == -1:
                return
            if position_flag == 1:
                if hold_bars >= MIN_HOLD_BARS:
                    self._submit_exit_flat(symbol, asset, current_price, reason="REVERSE_TO_SHORT")
                return
            if not in_cooldown and short_entry_allowed:
                self._submit_entry(symbol, asset, bars, current_price, portfolio_value, side="sell")

        else:  # ACTION_FLAT
            if position_flag != 0 and hold_bars >= MIN_HOLD_BARS:
                self._submit_exit_flat(symbol, asset, current_price, reason="MODEL_FLAT")

    def _submit_entry(
        self,
        symbol: str,
        asset: Asset,
        bars: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        side: str,
    ) -> None:
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
            qty = max(1, int(qty * scale))

        if qty <= 0:
            return

        try:
            order = self.create_order(asset, qty, side)
            self.submit_order(order)
            self._entry_iteration[symbol] = self._iteration_index
            self._entry_side[symbol] = side
            self._entry_prices[symbol] = current_price
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
        try:
            position = self.get_position(asset)
            if not position:
                return
            qty = float(position.quantity)
            if qty == 0:
                return
            side = "sell" if qty > 0 else "buy"
            order = self.create_order(asset, abs(qty), side)
            self.submit_order(order)
            self._last_exit_iteration[symbol] = self._iteration_index
            self._entry_iteration.pop(symbol, None)
            self._peak_prices.pop(symbol, None)
            self._trough_prices.pop(symbol, None)
            self._stop_prices.pop(symbol, None)
            self._tp_prices.pop(symbol, None)
            logger.info("EXIT %s %.2f %s @ ~$%.2f (%s)", side.upper(), abs(qty), symbol, current_price, reason)
        except Exception as exc:
            logger.error("Failed to submit EXIT for %s: %s", symbol, exc)

    def _risk_exit_check(self, symbol: str, asset: Asset, bars: pd.DataFrame, current_price: float, side: str) -> None:
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
                self._submit_exit_flat(symbol, asset, current_price, reason="RISK_STOP")
                return
            if active_tp > 0 and current_price >= active_tp:
                self._submit_exit_flat(symbol, asset, current_price, reason="TAKE_PROFIT")
                return
        else:
            if active_stop > 0 and current_price >= active_stop:
                self._submit_exit_flat(symbol, asset, current_price, reason="RISK_STOP")
                return
            if active_tp > 0 and current_price <= active_tp:
                self._submit_exit_flat(symbol, asset, current_price, reason="TAKE_PROFIT")
                return

    def _holding_bars(self, symbol: str) -> int:
        entry_iter = self._entry_iteration.get(symbol)
        if entry_iter is None:
            return MIN_HOLD_BARS
        return max(0, self._iteration_index - entry_iter)

    def _in_entry_cooldown(self, symbol: str) -> bool:
        exit_iter = self._last_exit_iteration.get(symbol)
        if exit_iter is None:
            return False
        return (self._iteration_index - exit_iter) < ENTRY_COOLDOWN_BARS

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

    def _close_all_positions(self, reason: str = "CIRCUIT_BREAKER") -> None:
        positions = self.get_positions()
        if not positions:
            return

        logger.info("Closing all positions: %s", reason)
        for pos in positions:
            try:
                qty = float(pos.quantity)
                if qty == 0:
                    continue
                side = "sell" if qty > 0 else "buy"
                order = self.create_order(pos.asset, abs(qty), side)
                self.submit_order(order)
            except Exception as exc:
                logger.error("Could not close %s: %s", pos.asset.symbol, exc)

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
        close_dt = now_et.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        return open_dt <= now_et < close_dt

    def _is_eod_close_window(self, now_et: datetime) -> bool:
        close_dt = now_et.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        window_start = close_dt - timedelta(minutes=EOD_CLOSE_BUFFER_MIN)
        return window_start <= now_et < close_dt

    def _clear_symbol_state(self, symbol: str) -> None:
        self._entry_prices.pop(symbol, None)
        self._entry_side.pop(symbol, None)
        self._stop_prices.pop(symbol, None)
        self._tp_prices.pop(symbol, None)
        self._peak_prices.pop(symbol, None)
        self._trough_prices.pop(symbol, None)


# Backward-compatible alias for legacy imports
CryptoDeepScalper = EquityDeepScalper
