"""
tests/test_risk.py — Unit tests for execution/risk.py.

Tests:
    - kelly_position_size: valid inputs, edge cases, cap enforcement.
    - calculate_atr_stop: long and short sides, fallback when ATR is NaN.
"""

import math
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add project root to path so imports resolve without an installed package
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.risk import kelly_position_size, calculate_atr_stop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high  = close + rng.uniform(0.1, 0.5, n)
    low   = close - rng.uniform(0.1, 0.5, n)
    vol   = rng.integers(100_000, 500_000, n).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol})


# ---------------------------------------------------------------------------
# kelly_position_size
# ---------------------------------------------------------------------------

class TestKellyPositionSize:
    def test_returns_positive_int(self):
        qty = kelly_position_size(
            win_rate=0.55, avg_win=0.02, avg_loss=0.01,
            portfolio_value=10_000, price=100.0,
        )
        assert isinstance(qty, int)
        assert qty >= 1

    def test_cap_at_max_position_pct(self):
        """Kelly fraction must never result in more than MAX_POSITION_PCT of portfolio."""
        qty = kelly_position_size(
            win_rate=0.99, avg_win=0.50, avg_loss=0.01,   # Extremely bullish
            portfolio_value=100_000, price=10.0,
            max_position_pct=0.03,
        )
        max_allowed_notional = 100_000 * 0.03
        assert qty * 10.0 <= max_allowed_notional + 10.0   # Allow one share rounding

    def test_minimum_one_share(self):
        """Even with tiny portfolio or high price, at least 1 share is returned."""
        qty = kelly_position_size(
            win_rate=0.51, avg_win=0.001, avg_loss=0.001,
            portfolio_value=100, price=5_000.0,
        )
        assert qty == 1

    def test_negative_kelly_returns_one(self):
        """Negative Kelly (edge < 0) should not return 0 or negative."""
        qty = kelly_position_size(
            win_rate=0.1, avg_win=0.01, avg_loss=0.50,   # Terrible edge
            portfolio_value=10_000, price=50.0,
        )
        assert qty == 1

    def test_kelly_fraction_scales_down(self):
        """Applying a 0.5 Kelly fraction should produce fewer shares than full Kelly."""
        kwargs = dict(
            win_rate=0.6, avg_win=0.03, avg_loss=0.01,
            portfolio_value=50_000, price=100.0, max_position_pct=1.0,
        )
        qty_half = kelly_position_size(**kwargs, kelly_fraction=0.5)
        qty_full = kelly_position_size(**kwargs, kelly_fraction=1.0)
        assert qty_half <= qty_full


# ---------------------------------------------------------------------------
# calculate_atr_stop
# ---------------------------------------------------------------------------

class TestCalculateAtrStop:
    def test_long_stop_below_entry(self):
        bars = _make_bars()
        stop, tp = calculate_atr_stop(bars, entry_price=105.0, side="buy")
        assert stop < 105.0, "Long stop must be below entry price"
        assert tp   > 105.0, "Long take-profit must be above entry price"

    def test_short_stop_above_entry(self):
        bars = _make_bars()
        stop, tp = calculate_atr_stop(bars, entry_price=105.0, side="sell")
        assert stop > 105.0, "Short stop must be above entry price"
        assert tp   < 105.0, "Short take-profit must be below entry price"

    def test_tp_farther_than_stop(self):
        """Take-profit distance should be larger than stop distance."""
        bars = _make_bars()
        entry = 100.0
        stop, tp = calculate_atr_stop(bars, entry_price=entry, side="buy")
        stop_dist = entry - stop
        tp_dist   = tp - entry
        assert tp_dist > stop_dist, "TP:Stop ratio must be > 1"

    def test_fallback_on_nan_atr(self):
        """If ATR is NaN (e.g. single-row DataFrame), fallback to percentage-based levels."""
        single_bar = pd.DataFrame({
            "open": [100], "high": [101], "low": [99], "close": [100], "volume": [1_000],
        })
        stop, tp = calculate_atr_stop(single_bar, entry_price=100.0, side="buy")
        # Fallback should still produce valid levels (not NaN)
        assert math.isfinite(stop)
        assert math.isfinite(tp)
        assert stop < 100.0
        assert tp   > 100.0

    def test_stop_price_positive(self):
        bars = _make_bars()
        stop, tp = calculate_atr_stop(bars, entry_price=50.0, side="buy")
        assert stop > 0.0, "Stop price must be positive"
