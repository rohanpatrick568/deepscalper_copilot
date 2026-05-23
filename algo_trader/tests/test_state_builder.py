"""
tests/test_state_builder.py — Unit tests for execution/state_builder.py.

Tests:
    - build_state_tensor: correct output shape, None on insufficient data,
      no NaN/Inf in output, handles various column-name casings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.state_builder import build_state_tensor
from config import LOOKBACK_BARS, INPUT_DIM


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
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildStateTensor:
    def test_output_shape(self):
        bars = _make_bars(LOOKBACK_BARS + 20)
        tensor = build_state_tensor(bars)
        assert tensor is not None
        assert tensor.shape == (1, LOOKBACK_BARS, INPUT_DIM)

    def test_returns_tensor_type(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        tensor = build_state_tensor(bars)
        assert isinstance(tensor, torch.Tensor)

    def test_no_nan_or_inf(self):
        bars = _make_bars(LOOKBACK_BARS + 10)
        tensor = build_state_tensor(bars)
        assert tensor is not None
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
        assert not torch.isinf(tensor).any(), "Tensor contains Inf values"

    def test_insufficient_data_returns_none(self):
        bars = _make_bars(LOOKBACK_BARS - 1)
        result = build_state_tensor(bars)
        assert result is None

    def test_exactly_lookback_bars(self):
        """Exactly LOOKBACK_BARS rows is the minimum valid input."""
        bars = _make_bars(LOOKBACK_BARS)
        tensor = build_state_tensor(bars)
        assert tensor is not None
        assert tensor.shape == (1, LOOKBACK_BARS, INPUT_DIM)

    def test_uppercase_column_names(self):
        """Column names should be normalised to lower-case internally."""
        bars = _make_bars(LOOKBACK_BARS + 5)
        bars.columns = [c.upper() for c in bars.columns]
        tensor = build_state_tensor(bars)
        assert tensor is not None

    def test_mixed_case_column_names(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        bars.columns = ["Open", "High", "Low", "Close", "Volume"]
        tensor = build_state_tensor(bars)
        assert tensor is not None

    def test_device_parameter(self):
        bars = _make_bars(LOOKBACK_BARS + 5)
        tensor = build_state_tensor(bars, device="cpu")
        assert tensor is not None
        assert tensor.device.type == "cpu"

    def test_value_range(self):
        """All feature values should be in [-5, 5] after normalisation."""
        bars = _make_bars(LOOKBACK_BARS + 50)
        tensor = build_state_tensor(bars)
        assert tensor is not None
        arr = tensor.numpy()
        assert arr.max() <= 5.1, f"Max value {arr.max()} exceeds expected range"
        assert arr.min() >= -5.1, f"Min value {arr.min()} below expected range"
