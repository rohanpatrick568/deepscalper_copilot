"""
tests/test_data_bridge.py — Unit tests for dashboard/data_bridge.py.

Covers:
    DataBridge.update_signal / get_all_signals
    DataBridge.update_positions / get_all_positions
    DataBridge.append_trade_event / get_trade_log
    DataBridge.set_halted / is_halted
    DataBridge.update_portfolio_metrics
    Thread-safety: concurrent reads and writes
    ModelSignal, PositionSnapshot, TradeEvent dataclass field validation
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.data_bridge import DataBridge, ModelSignal, PositionSnapshot, TradeEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal(symbol: str = "BTC/USD") -> ModelSignal:
    return ModelSignal(
        symbol=symbol,
        action="BUY",
        q_values=[0.1, 0.9],
        confidence=0.9,
        timestamp="10:00:00",
    )


def _position(symbol: str = "BTC/USD") -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side="LONG",
        qty=0.1,
        entry_price=50_000.0,
        current_price=51_000.0,
        unrealized_pnl=100.0,
        atr_stop=49_000.0,
        atr_tp=53_000.0,
    )


def _event(symbol: str = "BTC/USD", event_type: str = "FILL") -> TradeEvent:
    return TradeEvent(
        timestamp="10:01:00",
        symbol=symbol,
        side="BUY",
        qty=1,
        price=50_000.0,
        event_type=event_type,
    )


# ===========================================================================
# Dataclass field validation
# ===========================================================================

class TestDataclassFields:
    def test_model_signal_fields(self):
        s = _signal()
        assert s.symbol == "BTC/USD"
        assert s.action == "BUY"
        assert isinstance(s.q_values, list)
        assert 0.0 <= s.confidence <= 1.0

    def test_position_snapshot_fields(self):
        p = _position()
        assert p.side in ("LONG", "SHORT")
        assert p.qty >= 0
        assert p.entry_price > 0
        assert p.current_price > 0

    def test_trade_event_fields(self):
        e = _event()
        assert e.symbol == "BTC/USD"
        assert isinstance(e.qty, int)
        assert isinstance(e.price, float)
        assert isinstance(e.event_type, str)

    def test_position_default_entry_time(self):
        """entry_time must default to a non-empty ISO string."""
        p = _position()
        assert isinstance(p.entry_time, str)
        assert len(p.entry_time) > 0


# ===========================================================================
# DataBridge — signals
# ===========================================================================

class TestSignals:
    def test_get_signals_empty_initially(self):
        bridge = DataBridge()
        assert bridge.get_all_signals() == {}

    def test_update_signal_stores_signal(self):
        bridge = DataBridge()
        bridge.update_signal("BTC/USD", _signal())
        sigs = bridge.get_all_signals()
        assert "BTC/USD" in sigs

    def test_update_signal_returns_latest(self):
        bridge = DataBridge()
        bridge.update_signal("BTC/USD", _signal())
        s2 = ModelSignal("BTC/USD", "SELL", [0.9, 0.1], 0.9, "10:02:00")
        bridge.update_signal("BTC/USD", s2)
        sigs = bridge.get_all_signals()
        assert sigs["BTC/USD"].action == "SELL"

    def test_multiple_symbols(self):
        bridge = DataBridge()
        bridge.update_signal("BTC/USD", _signal("BTC/USD"))
        bridge.update_signal("ETH/USD", _signal("ETH/USD"))
        sigs = bridge.get_all_signals()
        assert "BTC/USD" in sigs
        assert "ETH/USD" in sigs

    def test_get_signals_returns_copy(self):
        """Mutating the returned dict must not affect internal state."""
        bridge = DataBridge()
        bridge.update_signal("BTC/USD", _signal())
        copy = bridge.get_all_signals()
        copy["EXTRA"] = _signal("EXTRA")
        assert "EXTRA" not in bridge.get_all_signals()


# ===========================================================================
# DataBridge — positions
# ===========================================================================

class TestPositions:
    def test_get_positions_empty_initially(self):
        bridge = DataBridge()
        assert bridge.get_all_positions() == {}

    def test_update_positions_stores_all(self):
        bridge = DataBridge()
        bridge.update_positions({"BTC/USD": _position()})
        pos = bridge.get_all_positions()
        assert "BTC/USD" in pos

    def test_update_positions_replaces_entirely(self):
        bridge = DataBridge()
        bridge.update_positions({"BTC/USD": _position("BTC/USD")})
        bridge.update_positions({"ETH/USD": _position("ETH/USD")})
        pos = bridge.get_all_positions()
        assert "BTC/USD" not in pos
        assert "ETH/USD" in pos

    def test_empty_dict_clears_positions(self):
        bridge = DataBridge()
        bridge.update_positions({"BTC/USD": _position()})
        bridge.update_positions({})
        assert bridge.get_all_positions() == {}


# ===========================================================================
# DataBridge — trade log
# ===========================================================================

class TestTradeLog:
    def test_trade_log_empty_initially(self):
        bridge = DataBridge()
        assert bridge.get_trade_log() == []

    def test_append_trade_event(self):
        bridge = DataBridge()
        bridge.append_trade_event(_event())
        log = bridge.get_trade_log()
        assert len(log) == 1

    def test_trade_log_maintains_order(self):
        bridge = DataBridge()
        for i in range(5):
            bridge.append_trade_event(TradeEvent(
                timestamp=f"10:0{i}:00",
                symbol="BTC/USD",
                side="BUY",
                qty=1,
                price=float(50_000 + i),
                event_type="FILL",
            ))
        log = bridge.get_trade_log()
        assert len(log) == 5
        prices = [e.price for e in log]
        assert prices == sorted(prices)

    def test_trade_log_capped_at_max(self):
        """Log must not grow beyond MAX_TRADE_LOG_ENTRIES (500)."""
        from config import MAX_TRADE_LOG_ENTRIES
        bridge = DataBridge()
        for _ in range(MAX_TRADE_LOG_ENTRIES + 100):
            bridge.append_trade_event(_event())
        assert len(bridge.get_trade_log()) <= MAX_TRADE_LOG_ENTRIES

    def test_get_trade_log_returns_copy(self):
        bridge = DataBridge()
        bridge.append_trade_event(_event())
        copy = bridge.get_trade_log()
        copy.append(_event("EXTRA"))
        assert len(bridge.get_trade_log()) == 1


# ===========================================================================
# DataBridge — halt flags
# ===========================================================================

class TestHaltFlags:
    def test_not_halted_initially(self):
        bridge = DataBridge()
        assert not bridge.is_halted
        assert bridge.halt_reason == ""

    def test_set_halted(self):
        bridge = DataBridge()
        bridge.is_halted = True
        bridge.halt_reason = "test reason"
        assert bridge.is_halted
        assert "test" in bridge.halt_reason.lower()

    def test_clear_halt(self):
        bridge = DataBridge()
        bridge.is_halted = True
        bridge.halt_reason = "some reason"
        bridge.is_halted = False
        bridge.halt_reason = ""
        assert not bridge.is_halted


# ===========================================================================
# DataBridge — portfolio metrics
# ===========================================================================

class TestPortfolioMetrics:
    def test_initial_portfolio_value(self):
        bridge = DataBridge()
        assert bridge.portfolio_value == pytest.approx(0.0)

    def test_update_portfolio_value(self):
        bridge = DataBridge()
        bridge.portfolio_value = 100_000.0
        assert bridge.portfolio_value == pytest.approx(100_000.0)

    def test_update_daily_pnl(self):
        bridge = DataBridge()
        bridge.daily_pnl = 500.0
        assert bridge.daily_pnl == pytest.approx(500.0)


# ===========================================================================
# Thread-safety
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_writes_do_not_raise(self):
        bridge = DataBridge()
        errors = []

        def writer():
            try:
                for i in range(100):
                    bridge.update_signal("BTC/USD", _signal())
                    bridge.append_trade_event(_event())
                    bridge.update_positions({"BTC/USD": _position()})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_reads_and_writes(self):
        bridge = DataBridge()
        bridge.update_signal("BTC/USD", _signal())
        errors = []

        def reader():
            try:
                for _ in range(200):
                    bridge.get_all_signals()
                    bridge.get_all_positions()
                    bridge.get_trade_log()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def writer():
            try:
                for i in range(200):
                    bridge.update_signal("BTC/USD", _signal())
                    bridge.append_trade_event(_event())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads  = [threading.Thread(target=reader) for _ in range(3)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not errors, f"Thread errors: {errors}"
