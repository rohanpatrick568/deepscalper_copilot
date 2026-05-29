"""
tests/test_circuit_breakers.py — Unit tests for execution/circuit_breakers.py.

Tests:
    - Opening / closing no-trade buffers.
    - Daily loss halt trigger and sticky behaviour.
    - Daily PnL reset for new trading day.
"""

import sys
from pathlib import Path
from datetime import time as dtime
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.circuit_breakers import CircuitBreaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cb(max_daily_loss_pct: float = 0.03, starting_capital: float = 10_000.0) -> CircuitBreaker:
    return CircuitBreaker(
        max_daily_loss_pct=max_daily_loss_pct,
        starting_capital=starting_capital,
    )


def _patch_time(cb: CircuitBreaker, hour: int, minute: int):
    """Context manager: patch _current_et_time to return a fixed time."""
    return patch("execution.circuit_breakers._current_et_time", return_value=dtime(hour, minute))


# ---------------------------------------------------------------------------
# Opening buffer (9:30 – 9:45)
# ---------------------------------------------------------------------------

class TestOpeningBuffer:
    def test_halted_at_market_open(self):
        cb = _make_cb()
        with _patch_time(cb, 9, 30):
            halted, reason = cb.is_trading_halted()
        assert halted
        assert "opening" in reason.lower()

    def test_halted_during_opening_buffer(self):
        cb = _make_cb()
        with _patch_time(cb, 9, 44):
            halted, reason = cb.is_trading_halted()
        assert halted

    def test_not_halted_after_opening_buffer(self):
        cb = _make_cb()
        with _patch_time(cb, 9, 45):
            halted, _ = cb.is_trading_halted()
        assert not halted

    def test_not_halted_midday(self):
        cb = _make_cb()
        with _patch_time(cb, 12, 0):
            halted, _ = cb.is_trading_halted()
        assert not halted


# ---------------------------------------------------------------------------
# Closing buffer (15:45 – 16:00)
# ---------------------------------------------------------------------------

class TestClosingBuffer:
    def test_halted_in_closing_buffer(self):
        cb = _make_cb()
        with _patch_time(cb, 15, 45):
            halted, reason = cb.is_trading_halted()
        assert halted
        assert "closing" in reason.lower()

    def test_halted_at_market_close(self):
        cb = _make_cb()
        with _patch_time(cb, 15, 59):
            halted, _ = cb.is_trading_halted()
        assert halted

    def test_not_halted_before_closing_buffer(self):
        cb = _make_cb()
        with _patch_time(cb, 15, 44):
            halted, _ = cb.is_trading_halted()
        assert not halted


# ---------------------------------------------------------------------------
# Daily loss halt
# ---------------------------------------------------------------------------

class TestDailyLossHalt:
    def test_halt_triggered_by_loss(self):
        cb = _make_cb(max_daily_loss_pct=0.03, starting_capital=10_000)
        # Loss of $301 exceeds 3% of $10,000 = $300
        with _patch_time(cb, 11, 0):
            cb.update_daily_pnl(-301.0)
            halted, reason = cb.is_trading_halted()
        assert halted
        assert "daily loss" in reason.lower()

    def test_not_halted_below_threshold(self):
        cb = _make_cb(max_daily_loss_pct=0.03, starting_capital=10_000)
        with _patch_time(cb, 11, 0):
            cb.update_daily_pnl(-299.0)
            halted, _ = cb.is_trading_halted()
        assert not halted

    def test_halt_is_sticky(self):
        """Once halted by daily loss, remains halted even with positive PnL updates."""
        cb = _make_cb(max_daily_loss_pct=0.03, starting_capital=10_000)
        with _patch_time(cb, 11, 0):
            cb.update_daily_pnl(-400.0)   # Trigger halt
            cb.update_daily_pnl(+200.0)   # Try to recover
            halted, _ = cb.is_trading_halted()
        assert halted, "Halt should persist after daily loss threshold is breached"

    def test_reset_clears_halt(self):
        cb = _make_cb(max_daily_loss_pct=0.03, starting_capital=10_000)
        with _patch_time(cb, 11, 0):
            cb.update_daily_pnl(-400.0)
            cb.reset_for_new_day()
            halted, _ = cb.is_trading_halted()
        assert not halted

    def test_set_absolute_pnl_triggers_halt(self):
        cb = _make_cb(max_daily_loss_pct=0.03, starting_capital=10_000)
        with _patch_time(cb, 11, 0):
            cb.set_daily_pnl(-350.0)
            halted, _ = cb.is_trading_halted()
        assert halted


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_pnl_accumulator(self):
        cb = _make_cb(starting_capital=10_000)
        cb.update_daily_pnl(-100.0)
        cb.reset_for_new_day()
        # After reset, small loss should not trigger halt
        with _patch_time(cb, 11, 0):
            cb.update_daily_pnl(-100.0)
            halted, _ = cb.is_trading_halted()
        assert not halted
