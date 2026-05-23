"""
tests/conftest.py — Shared pytest fixtures for the DeepScalper test suite.

All fixtures here are available to every test module without imports.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so all module imports resolve.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Colab sub-package path (contains deepscalper/)
COLAB = ROOT / "colab"
if str(COLAB) not in sys.path:
    sys.path.insert(0, str(COLAB))


# ---------------------------------------------------------------------------
# Synthetic market data helpers
# ---------------------------------------------------------------------------

def make_bars(n: int = 60, seed: int = 42, base_price: float = 50_000.0) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with UTC DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close = base_price + np.cumsum(rng.normal(0, 50, n))
    close = np.maximum(close, 1.0)   # keep prices positive
    high  = close + rng.uniform(10, 100, n)
    low   = close - rng.uniform(10, 100, n)
    low   = np.maximum(low, 1.0)
    volume = rng.integers(1, 100, n).astype(float)
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_lob_snap(n: int = 1, base_price: float = 50_000.0, seed: int = 0) -> pd.DataFrame:
    """Return synthetic LOB snapshot DataFrame with top-3 bid/ask levels."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        mid = base_price + rng.normal(0, 50)
        spread = mid * 0.0002  # 2 bps
        row = {
            "bid_price_1": mid - spread,   "bid_size_1": rng.uniform(0.1, 2.0),
            "bid_price_2": mid - spread * 2, "bid_size_2": rng.uniform(0.1, 2.0),
            "bid_price_3": mid - spread * 3, "bid_size_3": rng.uniform(0.1, 2.0),
            "ask_price_1": mid + spread,   "ask_size_1": rng.uniform(0.1, 2.0),
            "ask_price_2": mid + spread * 2, "ask_size_2": rng.uniform(0.1, 2.0),
            "ask_price_3": mid + spread * 3, "ask_size_3": rng.uniform(0.1, 2.0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bars_60():
    return make_bars(60)


@pytest.fixture
def bars_200():
    return make_bars(200)


@pytest.fixture
def env_data():
    """Pre-computed feature arrays and day_starts for ScalperEnv construction."""
    from deepscalper.utils import compute_macro_features, compute_micro_features, compute_day_starts

    bars = make_bars(200)
    lob   = compute_micro_features(bars)
    macro = compute_macro_features(bars)
    close = bars["close"].values.astype(np.float64)
    days  = compute_day_starts(bars.index)
    return dict(lob=lob, macro=macro, close=close, days=days, bars=bars)


@pytest.fixture
def scalper_env(env_data):
    """Minimal ScalperEnv with 200 bars of synthetic BTC data."""
    from deepscalper.environment import ScalperEnv
    return ScalperEnv(
        lob_features   = env_data["lob"],
        macro_features = env_data["macro"],
        close_prices   = env_data["close"],
        day_starts     = env_data["days"],
        lookback_bars  = 10,
    )


@pytest.fixture
def net_v2():
    """DeepScalperNet configured for V2 (N_DIR=2, LOB_DIM=4)."""
    from deepscalper.architecture import DeepScalperNet
    return DeepScalperNet(
        macro_dim=11, lob_dim=4, priv_dim=2,
        gru_hidden=32, macro_embed=16, fc_hidden=32,
        n_dir=2, n_size=1,
    )
