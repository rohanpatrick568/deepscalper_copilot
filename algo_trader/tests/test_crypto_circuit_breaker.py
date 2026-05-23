"""
tests/test_crypto_circuit_breaker.py — Unit tests for CryptoCitruitBreaker.

The class is in execution/strategy.py.  Three halt conditions:
  1. 24-hour rolling loss > max_24h_loss_pct × starting_capital
  2. ATR spike > volatility_halt_multiplier × 72-h baseline
  3. Consecutive losing trades ≥ consecutive_loss_halt (30-min cooldown)
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Stub out heavy dependencies BEFORE importing execution.strategy ──────────
# execution/strategy.py imports lumibot and alpaca at module level, which both
# try to connect to Alpaca at import time (reading credentials).  We provide
# lightweight MagicMock stubs so the file can be imported in a test environment
# without valid API keys.
_STUB_MODS = [
    "lumibot",
    "lumibot.entities",
    "lumibot.strategies",
    "lumibot.strategies.strategy",
    "lumibot.credentials",
    "lumibot.brokers",
    "lumibot.brokers.alpaca",
    "lumibot.data_sources",
    "lumibot.data_sources.alpaca_data",
    "alpaca",
    "alpaca.data",
    "alpaca.data.historical",
    "alpaca.data.live",
    "alpaca.data.requests",
    "alpaca.data.timeframe",
]
for _mod in _STUB_MODS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Asset / Strategy must be importable names
sys.modules["lumibot.entities"].Asset = MagicMock
sys.modules["lumibot.strategies"].Strategy = MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.strategy import CryptoCitruitBreaker  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cb(
    max_24h_loss_pct:          float = 0.05,
    volatility_halt_multiplier: float = 4.0,
    consecutive_loss_halt:     int   = 8,
    starting_capital:          float = 10_000.0,
    cooldown_minutes:          int   = 30,
) -> CryptoCitruitBreaker:
    return CryptoCitruitBreaker(
        max_24h_loss_pct=max_24h_loss_pct,
        volatility_halt_multiplier=volatility_halt_multiplier,
        consecutive_loss_halt=consecutive_loss_halt,
        starting_capital=starting_capital,
        cooldown_minutes=cooldown_minutes,
    )


# ===========================================================================
# Default state — not halted
# ===========================================================================

class TestDefaultState:
    def test_not_halted_initially(self):
        cb    = _make_cb()
        halted, reason = cb.is_trading_halted()
        assert not halted
        assert reason == ""

    def test_daily_pnl_starts_at_zero(self):
        cb = _make_cb()
        assert cb.daily_pnl == pytest.approx(0.0)

    def test_consecutive_starts_at_zero(self):
        cb = _make_cb()
        assert cb._consecutive == 0

    def test_streak_halt_until_is_none(self):
        cb = _make_cb()
        assert cb._streak_halted_until is None


# ===========================================================================
# reset_for_new_utc_day
# ===========================================================================

class TestResetForNewDay:
    def test_resets_daily_pnl(self):
        cb = _make_cb()
        cb.daily_pnl = -999.0
        cb.reset_for_new_utc_day()
        assert cb.daily_pnl == pytest.approx(0.0)

    def test_does_not_clear_streak(self):
        """Daily reset should NOT clear the consecutive loss streak."""
        cb = _make_cb(consecutive_loss_halt=3)
        for _ in range(2):
            cb.record_trade(pnl=-100.0, is_win=False)
        cb.reset_for_new_utc_day()
        assert cb._consecutive == 2


# ===========================================================================
# Halt condition 1: 24-hour rolling loss gate
# ===========================================================================

class TestRollingLossHalt:
    def test_halt_when_loss_exceeds_threshold(self):
        cb = _make_cb(max_24h_loss_pct=0.05, starting_capital=10_000)
        # Limit is 0.05 × 10000 = $500
        cb.record_trade(pnl=-501.0, is_win=False)
        halted, reason = cb.is_trading_halted()
        assert halted
        assert "24h" in reason.lower() or "loss" in reason.lower()

    def test_not_halted_below_threshold(self):
        cb = _make_cb(max_24h_loss_pct=0.05, starting_capital=10_000)
        cb.record_trade(pnl=-499.0, is_win=False)
        halted, _ = cb.is_trading_halted()
        assert not halted

    def test_exact_threshold_not_halted(self):
        """Exactly at the limit should not halt (strict < check)."""
        cb = _make_cb(max_24h_loss_pct=0.05, starting_capital=10_000)
        cb.record_trade(pnl=-500.0, is_win=False)
        halted, _ = cb.is_trading_halted()
        # rolling_pnl = -500; max_loss_usd = -500; -500 < -500 is False → not halted
        assert not halted

    def test_old_trades_excluded_from_rolling_window(self):
        """Trades older than 24 hours must not count toward the rolling loss."""
        cb = _make_cb(max_24h_loss_pct=0.05, starting_capital=10_000)
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        # Manually inject an old trade entry
        cb._trade_pnls.append(-600.0)
        cb._trade_times.append(old_time)
        halted, _ = cb.is_trading_halted()
        assert not halted

    def test_positive_trades_offset_losses(self):
        cb = _make_cb(max_24h_loss_pct=0.05, starting_capital=10_000)
        cb.record_trade(pnl=-400.0, is_win=False)
        cb.record_trade(pnl=+300.0, is_win=True)
        # Net = -100, well below 500 limit
        halted, _ = cb.is_trading_halted()
        assert not halted

    def test_accumulated_losses_halt(self):
        cb = _make_cb(max_24h_loss_pct=0.05, starting_capital=10_000)
        for _ in range(6):
            cb.record_trade(pnl=-100.0, is_win=False)
        # Net = -600 > 500 threshold
        halted, _ = cb.is_trading_halted()
        assert halted


# ===========================================================================
# Halt condition 2: ATR spike gate
# ===========================================================================

class TestAtrSpikeHalt:
    def test_not_halted_without_history(self):
        """Fewer than 72 bars → baseline not established → no halt."""
        cb = _make_cb()
        cb.record_bar_return(0.10)  # huge return but not enough history
        halted, _ = cb.is_trading_halted()
        assert not halted

    def test_not_halted_with_normal_returns(self):
        cb = _make_cb(volatility_halt_multiplier=4.0)
        for _ in range(72):
            cb.record_bar_return(0.001)   # 0.1% per bar — normal
        halted, _ = cb.is_trading_halted()
        assert not halted

    def test_halted_on_spike(self):
        """After 72+ bars at ~0.001 baseline, a spike of 4× triggers halt."""
        cb = _make_cb(volatility_halt_multiplier=4.0)
        # Build baseline: 72 bars at 0.001
        for _ in range(72):
            cb.record_bar_return(0.001)
        # Add extreme spike (>> 4 × 0.001 = 0.004)
        cb.record_bar_return(0.10)
        halted, reason = cb.is_trading_halted()
        assert halted
        assert "atr" in reason.lower() or "spike" in reason.lower()

    def test_returns_below_multiplier_not_halted(self):
        cb = _make_cb(volatility_halt_multiplier=4.0)
        for _ in range(72):
            cb.record_bar_return(0.001)
        # Add only 3× the baseline (below 4× threshold)
        cb.record_bar_return(0.003)
        halted, _ = cb.is_trading_halted()
        assert not halted

    def test_record_bar_return_appends_to_window(self):
        cb = _make_cb()
        initial = len(cb._minute_returns)
        cb.record_bar_return(0.001)
        assert len(cb._minute_returns) == initial + 1

    def test_minute_returns_window_capped(self):
        cb = _make_cb()
        for _ in range(72 * 60 + 100):
            cb.record_bar_return(0.001)
        assert len(cb._minute_returns) <= 72 * 60


# ===========================================================================
# Halt condition 3: Consecutive loss streak gate
# ===========================================================================

class TestConsecutiveLossStreak:
    def test_not_halted_before_streak(self):
        cb = _make_cb(consecutive_loss_halt=8)
        for _ in range(7):
            cb.record_trade(pnl=-50.0, is_win=False)
        halted, _ = cb.is_trading_halted()
        assert not halted

    def test_halted_after_streak(self):
        cb = _make_cb(consecutive_loss_halt=8, cooldown_minutes=30)
        for _ in range(8):
            cb.record_trade(pnl=-50.0, is_win=False)
        halted, reason = cb.is_trading_halted()
        assert halted
        assert "consecutive" in reason.lower() or "streak" in reason.lower() or "cooldown" in reason.lower()

    def test_win_resets_consecutive(self):
        cb = _make_cb(consecutive_loss_halt=8)
        for _ in range(5):
            cb.record_trade(pnl=-50.0, is_win=False)
        cb.record_trade(pnl=100.0, is_win=True)
        assert cb._consecutive == 0

    def test_halted_until_set_after_streak(self):
        cb = _make_cb(consecutive_loss_halt=4, cooldown_minutes=30)
        for _ in range(4):
            cb.record_trade(pnl=-50.0, is_win=False)
        assert cb._streak_halted_until is not None

    def test_cooldown_expires_clears_halt(self):
        """After the cooldown time passes, is_trading_halted must return False."""
        cb = _make_cb(consecutive_loss_halt=4, cooldown_minutes=30)
        for _ in range(4):
            cb.record_trade(pnl=-50.0, is_win=False)

        # Simulate time advancing past the cooldown
        past_time = datetime.now(timezone.utc) - timedelta(minutes=31)
        cb._streak_halted_until = past_time

        halted, _ = cb.is_trading_halted()
        assert not halted

    def test_streak_cleared_after_cooldown_expires(self):
        """Consecutive counter should reset once the cooldown expires."""
        cb = _make_cb(consecutive_loss_halt=4, cooldown_minutes=30)
        for _ in range(4):
            cb.record_trade(pnl=-50.0, is_win=False)
        # Expire cooldown
        cb._streak_halted_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        cb.is_trading_halted()  # trigger check
        assert cb._consecutive == 0
        assert cb._streak_halted_until is None

    def test_additional_losses_after_first_streak_retriggered(self):
        """Hitting the streak threshold multiple times each triggers a halt."""
        cb = _make_cb(consecutive_loss_halt=4, cooldown_minutes=30)
        for _ in range(4):
            cb.record_trade(pnl=-10.0, is_win=False)
        # Expire cooldown manually
        cb._streak_halted_until = datetime.now(timezone.utc) - timedelta(minutes=31)
        cb.is_trading_halted()  # clears halt
        # Trigger again
        for _ in range(4):
            cb.record_trade(pnl=-10.0, is_win=False)
        halted, _ = cb.is_trading_halted()
        assert halted


# ===========================================================================
# record_trade — side effects
# ===========================================================================

class TestRecordTrade:
    def test_increments_daily_pnl(self):
        cb = _make_cb()
        cb.record_trade(pnl=200.0, is_win=True)
        assert cb.daily_pnl == pytest.approx(200.0)

    def test_decrements_daily_pnl_on_loss(self):
        cb = _make_cb()
        cb.record_trade(pnl=-100.0, is_win=False)
        assert cb.daily_pnl == pytest.approx(-100.0)

    def test_appends_to_trade_pnls(self):
        cb = _make_cb()
        initial = len(cb._trade_pnls)
        cb.record_trade(pnl=50.0, is_win=True)
        assert len(cb._trade_pnls) == initial + 1

    def test_appends_to_trade_times(self):
        cb = _make_cb()
        initial = len(cb._trade_times)
        cb.record_trade(pnl=50.0, is_win=True)
        assert len(cb._trade_times) == initial + 1

    def test_trade_times_are_utc(self):
        cb = _make_cb()
        before = datetime.now(timezone.utc)
        cb.record_trade(pnl=10.0, is_win=True)
        after  = datetime.now(timezone.utc)
        ts = cb._trade_times[-1]
        assert before <= ts <= after

    def test_deque_bounded_at_1000(self):
        cb = _make_cb()
        for _ in range(1500):
            cb.record_trade(pnl=1.0, is_win=True)
        assert len(cb._trade_pnls)  <= 1000
        assert len(cb._trade_times) <= 1000
