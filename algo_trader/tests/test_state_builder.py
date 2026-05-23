"""
tests/test_state_builder.py — Unit tests for execution/state_builder.py.

Tests:
    - build_observation: returns dict with keys 'lob', 'priv', 'macro'
    - Correct tensor shapes: (1, 60, 5), (1, 60, 2), (1, 11)
    - No NaN or Inf in any output tensor
    - Handles various column-name casings
    - Position flag and P&L are reflected in private state
    - Deprecated build_state_tensor still works (backward compat)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.state_builder import build_observation, build_state_tensor
from config import LOOKBACK_BARS, MACRO_DIM, LOB_DIM, PRIV_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with a proper DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close  = 150.0 + np.cumsum(rng.normal(0, 0.3, n))
    high   = close + rng.uniform(0.05, 0.3, n)
    low    = close - rng.uniform(0.05, 0.3, n)
    volume = rng.integers(50_000, 300_000, n).astype(float)
    idx = pd.date_range(
        "2024-01-02 09:30", periods=n, freq="1min", tz="America/New_York"
    )
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests — build_observation (primary API)
# ---------------------------------------------------------------------------

class TestBuildObservation:
    def test_returns_dict(self):
        bars = _make_bars(LOOKBACK_BARS + 20)
        obs = build_observation(bars)
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {"lob", "priv", "macro"}

    def test_lob_shape(self):
        bars = _make_bars(LOOKBACK_BARS + 20)
        obs = build_observation(bars)
        assert obs["lob"].shape == (1, LOOKBACK_BARS, LOB_DIM), (
            f"Expected (1, {LOOKBACK_BARS}, {LOB_DIM}), got {obs['lob'].shape}"
        )

    def test_priv_shape(self):
        bars = _make_bars(LOOKBACK_BARS + 20)
        obs = build_observation(bars)
        assert obs["priv"].shape == (1, LOOKBACK_BARS, PRIV_DIM), (
            f"Expected (1, {LOOKBACK_BARS}, {PRIV_DIM}), got {obs['priv'].shape}"
        )

    def test_macro_shape(self):
        bars = _make_bars(LOOKBACK_BARS + 20)
        obs = build_observation(bars)
        assert obs["macro"].shape == (1, MACRO_DIM), (
            f"Expected (1, {MACRO_DIM}), got {obs['macro'].shape}"
        )

    def test_all_tensors(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        obs = build_observation(bars)
        for key in ("lob", "priv", "macro"):
            assert isinstance(obs[key], torch.Tensor), f"obs['{key}'] is not a Tensor"

    def test_float32_dtype(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        obs = build_observation(bars)
        for key in ("lob", "priv", "macro"):
            assert obs[key].dtype == torch.float32, (
                f"obs['{key}'] has dtype {obs[key].dtype}, expected float32"
            )

    def test_no_nan_or_inf(self):
        bars = _make_bars(LOOKBACK_BARS + 10)
        obs = build_observation(bars)
        for key in ("lob", "priv", "macro"):
            assert not torch.isnan(obs[key]).any(), f"NaN in obs['{key}']"
            assert not torch.isinf(obs[key]).any(), f"Inf in obs['{key}']"

    def test_fewer_bars_than_lookback_pads_correctly(self):
        """When fewer than LOOKBACK_BARS are available, zeros are prepended."""
        bars = _make_bars(LOOKBACK_BARS - 10)
        obs = build_observation(bars)
        assert obs["lob"].shape  == (1, LOOKBACK_BARS, LOB_DIM)
        assert obs["priv"].shape == (1, LOOKBACK_BARS, PRIV_DIM)

    def test_uppercase_column_names(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        bars.columns = [c.upper() for c in bars.columns]
        obs = build_observation(bars)
        assert obs["lob"].shape == (1, LOOKBACK_BARS, LOB_DIM)

    def test_mixed_case_column_names(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        bars.columns = ["Open", "High", "Low", "Close", "Volume"]
        obs = build_observation(bars)
        assert obs["macro"].shape == (1, MACRO_DIM)

    def test_device_cpu(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        obs = build_observation(bars, device="cpu")
        for key in ("lob", "priv", "macro"):
            assert obs[key].device.type == "cpu"

    def test_position_flag_in_priv(self):
        """Position flag should be 1.0 when position != 0."""
        bars = _make_bars(LOOKBACK_BARS + 5)
        obs_flat  = build_observation(bars, position=0)
        obs_long  = build_observation(bars, position=1)
        # priv[:, :, 0] is position flag — should differ
        assert obs_flat["priv"][0, 0, 0].item() == pytest.approx(0.0)
        assert obs_long["priv"][0, 0, 0].item() == pytest.approx(1.0)

    def test_pnl_clamped(self):
        """Unrealized P&L should be clamped to [-0.5, 0.5]."""
        bars = _make_bars(LOOKBACK_BARS + 5)
        obs = build_observation(bars, position=1, unrealized_pnl_pct=999.0)
        pnl_val = obs["priv"][0, 0, 1].item()
        assert pnl_val <= 0.5 + 1e-6

    def test_macro_values_bounded(self):
        """All macro features should be in a reasonable normalised range."""
        bars = _make_bars(LOOKBACK_BARS + 50)
        obs = build_observation(bars)
        arr = obs["macro"].numpy()
        assert arr.max() <= 1.1, f"macro max {arr.max()} out of expected range"
        assert arr.min() >= -1.1, f"macro min {arr.min()} out of expected range"


# ---------------------------------------------------------------------------
# Tests — build_state_tensor (deprecated backward-compat shim)
# ---------------------------------------------------------------------------

class TestBuildStateTensorBackcompat:
    def test_returns_tensor(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tensor = build_state_tensor(bars)
        assert isinstance(tensor, torch.Tensor)

    def test_shape(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tensor = build_state_tensor(bars)
        assert tensor.shape == (1, LOOKBACK_BARS, MACRO_DIM)

    def test_emits_deprecation_warning(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        with pytest.warns(DeprecationWarning):
            build_state_tensor(bars)
