"""
tests/test_utils.py — Unit tests for colab/deepscalper/utils.py.

Covers:
    compute_macro_features   — shape, dtype, clip bounds, NaN-free
    compute_micro_features   — proxy mode, real-LOB mode, shape, clip bounds
    compute_day_starts       — UTC boundary detection
    _compute_time_features   — sin/cos encoding range
    compute_sharpe           — formula correctness, edge cases
    compute_win_rate         — formula correctness, edge cases
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "colab"))

from deepscalper.utils import (
    compute_macro_features,
    compute_micro_features,
    compute_day_starts,
    _compute_time_features,
    compute_sharpe,
    compute_win_rate,
)
from config import LOB_DIM


# ---------------------------------------------------------------------------
# Helpers (local, no conftest dependency to keep this file self-contained)
# ---------------------------------------------------------------------------

def _bars(n: int = 60, seed: int = 7, base: float = 50_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close  = base + np.cumsum(rng.normal(0, 50, n))
    close  = np.maximum(close, 1.0)
    high   = close + rng.uniform(5, 50, n)
    low    = np.maximum(close - rng.uniform(5, 50, n), 1.0)
    volume = rng.integers(1, 100, n).astype(float)
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": close, "high": high, "low": low,
                          "close": close, "volume": volume}, index=idx)


def _lob_snap(n: int = 1, base: float = 50_000.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        mid  = base + i * 10
        sprd = mid * 0.0002
        rows.append({
            "bid_price_1": mid - sprd,     "bid_size_1": 1.0,
            "bid_price_2": mid - sprd * 2, "bid_size_2": 0.8,
            "bid_price_3": mid - sprd * 3, "bid_size_3": 0.5,
            "ask_price_1": mid + sprd,     "ask_size_1": 1.0,
            "ask_price_2": mid + sprd * 2, "ask_size_2": 0.8,
            "ask_price_3": mid + sprd * 3, "ask_size_3": 0.5,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# compute_macro_features
# ===========================================================================

class TestComputeMacroFeatures:
    def test_shape(self):
        bars  = _bars(60)
        feats = compute_macro_features(bars)
        assert feats.shape == (60, 11)

    def test_dtype_float32(self):
        feats = compute_macro_features(_bars(30))
        assert feats.dtype == np.float32

    def test_no_nan(self):
        feats = compute_macro_features(_bars(60))
        assert not np.isnan(feats).any(), "NaN in macro features"

    def test_no_inf(self):
        feats = compute_macro_features(_bars(60))
        assert np.isfinite(feats).all(), "Inf in macro features"

    def test_z_close_clip(self):
        """Column 3 (z_close) must be clipped to [-0.1, 0.1]."""
        feats = compute_macro_features(_bars(60))
        assert feats[:, 3].max() <= 0.1 + 1e-6
        assert feats[:, 3].min() >= -0.1 - 1e-6

    def test_ma_spread_clip(self):
        """MA-spread features (cols 5–10) clipped to [-0.05, 0.05]."""
        feats = compute_macro_features(_bars(60))
        for col in range(5, 11):
            assert feats[:, col].max() <= 0.05 + 1e-6
            assert feats[:, col].min() >= -0.05 - 1e-6

    def test_column_names_case_insensitive(self):
        """compute_macro_features should work with uppercase column names."""
        bars = _bars(30)
        bars.columns = [c.upper() for c in bars.columns]
        feats = compute_macro_features(bars)
        assert feats.shape == (30, 11)
        assert not np.isnan(feats).any()

    def test_single_row(self):
        """Should not crash on a one-row DataFrame."""
        feats = compute_macro_features(_bars(1))
        assert feats.shape == (1, 11)

    def test_large_price_crash(self):
        """Clipping prevents overflow when price moves 50% in one bar."""
        bars = _bars(30)
        bars["close"].iloc[15] = bars["close"].iloc[14] * 1.5
        feats = compute_macro_features(bars)
        assert np.isfinite(feats).all()


# ===========================================================================
# compute_micro_features — proxy mode
# ===========================================================================

class TestComputeMicroFeaturesProxy:
    def test_shape(self):
        feats = compute_micro_features(_bars(60))
        assert feats.shape == (60, LOB_DIM)

    def test_dtype(self):
        feats = compute_micro_features(_bars(60))
        assert feats.dtype == np.float32

    def test_no_nan(self):
        feats = compute_micro_features(_bars(60))
        assert not np.isnan(feats).any()

    def test_no_inf(self):
        feats = compute_micro_features(_bars(60))
        assert np.isfinite(feats).all()

    def test_spread_pct_nonnegative(self):
        """Spread proxy (col 0) must be ≥ 0."""
        feats = compute_micro_features(_bars(60))
        assert feats[:, 0].min() >= -1e-6

    def test_order_imbalance_range(self):
        """Order imbalance (col 1) clipped to [-1, 1]."""
        feats = compute_micro_features(_bars(60))
        assert feats[:, 1].max() <= 1.0 + 1e-6
        assert feats[:, 1].min() >= -1.0 - 1e-6

    def test_mid_move_range(self):
        """Mid-move (col 3) clipped to [-0.05, 0.05]."""
        feats = compute_micro_features(_bars(60))
        assert feats[:, 3].max() <= 0.05 + 1e-6
        assert feats[:, 3].min() >= -0.05 - 1e-6

    def test_default_is_proxy(self):
        """Default call (no lob_snapshots) should silently use proxy."""
        bars = _bars(20)
        feats = compute_micro_features(bars)
        assert feats.shape == (20, LOB_DIM)


# ===========================================================================
# compute_micro_features — real LOB mode
# ===========================================================================

class TestComputeMicroFeaturesRealLOB:
    def test_shape_single_row(self):
        feats = compute_micro_features(
            _bars(1), lob_snapshots=_lob_snap(1), use_proxy=False
        )
        assert feats.shape == (1, LOB_DIM)

    def test_shape_multi_row(self):
        feats = compute_micro_features(
            _bars(30), lob_snapshots=_lob_snap(30), use_proxy=False
        )
        assert feats.shape == (30, LOB_DIM)

    def test_dtype(self):
        feats = compute_micro_features(
            _bars(10), lob_snapshots=_lob_snap(10), use_proxy=False
        )
        assert feats.dtype == np.float32

    def test_no_nan(self):
        feats = compute_micro_features(
            _bars(10), lob_snapshots=_lob_snap(10), use_proxy=False
        )
        assert not np.isnan(feats).any()

    def test_spread_positive_with_valid_book(self):
        """With a normal uncrossed book, spread_pct (col 0) should be > 0."""
        feats = compute_micro_features(
            _bars(5), lob_snapshots=_lob_snap(5), use_proxy=False
        )
        assert feats[:, 0].min() >= 0.0

    def test_same_feature_count_as_proxy(self):
        """Real-LOB and proxy must produce the same feature dimensionality."""
        proxy   = compute_micro_features(_bars(10))
        real_lob = compute_micro_features(
            _bars(10), lob_snapshots=_lob_snap(10), use_proxy=False
        )
        assert proxy.shape[1] == real_lob.shape[1]


# ===========================================================================
# compute_day_starts
# ===========================================================================

class TestComputeDayStarts:
    def _make_index(self, days: int) -> pd.DatetimeIndex:
        """Generate a 1-minute UTC index spanning `days` full days."""
        return pd.date_range("2024-01-01 00:00", periods=days * 1440, freq="1min", tz="UTC")

    def test_single_day(self):
        idx = self._make_index(1)
        starts = compute_day_starts(idx)
        assert starts == [0]

    def test_two_days(self):
        idx = self._make_index(2)
        starts = compute_day_starts(idx)
        assert len(starts) == 2
        assert starts[0] == 0
        assert starts[1] == 1440  # second day starts at bar 1440

    def test_five_days(self):
        idx = self._make_index(5)
        starts = compute_day_starts(idx)
        assert len(starts) == 5

    def test_returns_sorted(self):
        idx = self._make_index(3)
        starts = compute_day_starts(idx)
        assert starts == sorted(starts)

    def test_first_element_always_zero(self):
        for n in [1, 2, 7]:
            idx = self._make_index(n)
            starts = compute_day_starts(idx)
            assert starts[0] == 0

    def test_naive_index_handled(self):
        """Naive (no tz) index should be treated as UTC without raising."""
        idx = pd.date_range("2024-01-01", periods=2880, freq="1min")  # no tz
        starts = compute_day_starts(idx)
        assert len(starts) == 2


# ===========================================================================
# _compute_time_features
# ===========================================================================

class TestComputeTimeFeatures:
    def test_shape(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
        out = _compute_time_features(idx)
        assert out.shape == (100, 2)

    def test_dtype(self):
        idx = pd.date_range("2024-01-01", periods=24, freq="1h", tz="UTC")
        out = _compute_time_features(idx)
        assert out.dtype == np.float32

    def test_sin_cos_range(self):
        """Both sin and cos columns must be in [-1, 1]."""
        idx = pd.date_range("2024-01-01", periods=1440, freq="1min", tz="UTC")
        out = _compute_time_features(idx)
        assert out[:, 0].min() >= -1.0 - 1e-6
        assert out[:, 0].max() <= 1.0 + 1e-6
        assert out[:, 1].min() >= -1.0 - 1e-6
        assert out[:, 1].max() <= 1.0 + 1e-6

    def test_midnight_is_zero_sin(self):
        """At UTC 00:00, sin_hour should be 0 (2π·0/24 = 0)."""
        idx = pd.DatetimeIndex(["2024-01-01 00:00:00+00:00"])
        out = _compute_time_features(idx)
        assert abs(out[0, 0]) < 1e-5    # sin(0) = 0

    def test_midnight_is_one_cos(self):
        """At UTC 00:00, cos_hour should be 1 (cos(0) = 1)."""
        idx = pd.DatetimeIndex(["2024-01-01 00:00:00+00:00"])
        out = _compute_time_features(idx)
        assert abs(out[0, 1] - 1.0) < 1e-5  # cos(0) = 1

    def test_period_is_24h(self):
        """Features at t=0 and t=24h must be identical (cyclic)."""
        idx = pd.DatetimeIndex(
            ["2024-01-01 00:00:00+00:00", "2024-01-02 00:00:00+00:00"]
        )
        out = _compute_time_features(idx)
        np.testing.assert_allclose(out[0], out[1], atol=1e-5)


# ===========================================================================
# compute_sharpe
# ===========================================================================

class TestComputeSharpe:
    def test_positive_sharpe(self):
        # Alternating values give non-zero std and a positive mean → Sharpe > 0
        rewards = [0.01, 0.02] * 50   # mean=0.015, std > 0
        sharpe  = compute_sharpe(rewards)
        assert sharpe > 0

    def test_negative_sharpe(self):
        # Negative mean, non-zero std → Sharpe < 0
        rewards = [-0.02, -0.01] * 50  # mean=-0.015, std > 0
        sharpe  = compute_sharpe(rewards)
        assert sharpe < 0

    def test_zero_std_returns_zero(self):
        """If all rewards are identical the std is 0 — must not raise."""
        sharpe = compute_sharpe([1.0] * 50)
        assert sharpe == 0.0

    def test_empty_returns_zero(self):
        sharpe = compute_sharpe([])
        assert sharpe == 0.0

    def test_annualised_magnitude(self):
        """Random walk should produce a Sharpe near zero over long sequences."""
        rng = np.random.default_rng(1)
        rewards = rng.normal(0, 0.001, 10_000).tolist()
        sharpe  = compute_sharpe(rewards)
        assert abs(sharpe) < 3.0   # Not extreme


# ===========================================================================
# compute_win_rate
# ===========================================================================

class TestComputeWinRate:
    def test_all_positive(self):
        assert compute_win_rate([1.0, 0.5, 0.1]) == pytest.approx(1.0)

    def test_all_negative(self):
        assert compute_win_rate([-1.0, -0.5]) == pytest.approx(0.0)

    def test_mixed(self):
        rate = compute_win_rate([1.0, -1.0, 1.0, -1.0])
        assert rate == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        assert compute_win_rate([]) == 0.0

    def test_range(self):
        rng = np.random.default_rng(0)
        rewards = rng.normal(0, 1, 500).tolist()
        rate = compute_win_rate(rewards)
        assert 0.0 <= rate <= 1.0
