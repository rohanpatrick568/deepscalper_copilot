"""
colab/deepscalper/utils.py — Feature Engineering for DeepScalper (paper-faithful).

Implements the exact feature sets described in the DeepScalper paper (CIKM '22):

  compute_macro_features(bars_df) → ndarray (n, MACRO_DIM=11)
      11 macro features from Table 2:
        z_open, z_high, z_low, z_close, z_adj_close,
        z_d_5, z_d_10, z_d_15, z_d_20, z_d_25, z_d_30

  compute_micro_features(bars_df) → ndarray (n, LOB_DIM=5)
      5 intrabar microstructure features (LOB proxy — no real LOB available):
        range_norm, body, lower_wick, upper_wick, vol_norm

  compute_features(bars_df) → ndarray (n, 11)
      Alias for compute_macro_features (backward compatibility).

  compute_day_starts(index) → list[int]
  compute_sharpe(rewards)   → float
  compute_win_rate(rewards) → float
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Macro features — Table 2 of the paper
# ---------------------------------------------------------------------------

def compute_macro_features(bars: pd.DataFrame) -> np.ndarray:
    """Compute 11 macro features matching DeepScalper paper Table 2.

    Features:
        0  z_open      = open_t  / close_t  - 1
        1  z_high      = high_t  / close_t  - 1
        2  z_low       = low_t   / close_t  - 1
        3  z_close     = close_t / close_{t-1} - 1
        4  z_adj_close = z_close  (intraday — no dividend adjustment)
        5  z_d_5       = close_t / SMA(5,  close) - 1
        6  z_d_10      = close_t / SMA(10, close) - 1
        7  z_d_15      = close_t / SMA(15, close) - 1
        8  z_d_20      = close_t / SMA(20, close) - 1
        9  z_d_25      = close_t / SMA(25, close) - 1
        10 z_d_30      = close_t / SMA(30, close) - 1

    Args:
        bars: DataFrame with OHLCV columns and a DatetimeIndex.

    Returns:
        float32 array of shape (len(bars), 11).
    """
    df = bars.copy()
    df.columns = [c.lower() for c in df.columns]

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    z_open      = (o / (c + 1e-10) - 1.0).clip(-0.1, 0.1)
    z_high      = (h / (c + 1e-10) - 1.0).clip(-0.1, 0.1)
    z_low       = (l / (c + 1e-10) - 1.0).clip(-0.1, 0.1)
    z_close     = c.pct_change().fillna(0.0).clip(-0.1, 0.1)
    z_adj_close = z_close.copy()  # Same as z_close for intraday data

    def _ma_spread(k: int) -> pd.Series:
        sma = c.rolling(k, min_periods=1).mean()
        return (c / (sma + 1e-10) - 1.0).clip(-0.05, 0.05)

    features = np.column_stack([
        z_open.values,
        z_high.values,
        z_low.values,
        z_close.values,
        z_adj_close.values,
        _ma_spread(5).values,
        _ma_spread(10).values,
        _ma_spread(15).values,
        _ma_spread(20).values,
        _ma_spread(25).values,
        _ma_spread(30).values,
    ]).astype(np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.05, neginf=-0.05)


# ---------------------------------------------------------------------------
# Micro features — LOB proxy (5 intrabar microstructure features)
# ---------------------------------------------------------------------------

def compute_micro_features(bars: pd.DataFrame) -> np.ndarray:
    """Compute 5 intrabar microstructure features (LOB proxy).

    The paper uses actual Limit Order Book (LOB) data; since Alpaca free-tier
    does not provide LOB for historical bars, we substitute candlestick
    microstructure signals that capture the same buying/selling pressure.

    Features:
        0  range_norm   = (H - L) / C                    — intrabar volatility
        1  body         = (C - O) / C                    — candle direction
        2  lower_wick   = (O - L) / (H - L + ε)         — buying pressure
        3  upper_wick   = (H - C) / (H - L + ε)         — selling pressure
        4  vol_norm     = z-score(volume, window=60)     — relative volume

    Args:
        bars: DataFrame with OHLCV columns and a DatetimeIndex.

    Returns:
        float32 array of shape (len(bars), 5).
    """
    df = bars.copy()
    df.columns = [c.lower() for c in df.columns]

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    hl = (h - l).replace(0.0, 1e-8)

    range_norm  = ((h - l) / (c + 1e-10)).clip(0.0, 0.05) / 0.05
    body        = ((c - o) / (c + 1e-10)).clip(-0.05, 0.05) / 0.05
    lower_wick  = ((o - l) / hl).clip(0.0, 1.0)
    upper_wick  = ((h - c) / hl).clip(0.0, 1.0)

    vol_mean = v.rolling(60, min_periods=1).mean()
    vol_std  = v.rolling(60, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
    vol_norm = ((v - vol_mean) / vol_std).clip(-3.0, 3.0) / 3.0

    features = np.column_stack([
        range_norm.values,
        body.values,
        lower_wick.values,
        upper_wick.values,
        vol_norm.values,
    ]).astype(np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

def compute_features(bars: pd.DataFrame) -> np.ndarray:
    """Alias for compute_macro_features (backward compatibility)."""
    return compute_macro_features(bars)


def _session_time_feature(index: pd.DatetimeIndex) -> np.ndarray:
    """Session-time fraction (kept for any legacy references)."""
    _MARKET_OPEN = 9 * 60 + 30
    _SESSION_DURATION = 390
    out = np.zeros(len(index))
    for i, ts in enumerate(index):
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts).astimezone(_ET)
        else:
            ts = ts.astimezone(_ET)
        mins = ts.hour * 60 + ts.minute - _MARKET_OPEN
        out[i] = np.clip(mins, 0, _SESSION_DURATION) / _SESSION_DURATION
    return out




# ---------------------------------------------------------------------------
# Day boundary detection
# ---------------------------------------------------------------------------

def compute_day_starts(index: pd.DatetimeIndex) -> List[int]:
    """Find integer indices where each new trading day begins.

    Args:
        index: DatetimeIndex of the full feature dataset (may span many days).

    Returns:
        Sorted list of integer positions where each day's first bar is located.
    """
    if index.tzinfo is None:
        index = index.tz_localize("UTC").tz_convert(_ET)
    else:
        index = index.tz_convert(_ET)

    dates = []
    current_date = None
    for i, ts in enumerate(index):
        day = ts.date()
        if day != current_date:
            current_date = day
            dates.append(i)
    return dates


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_sharpe(episode_rewards: List[float], risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio from per-episode reward values.

    Args:
        episode_rewards: List of total reward per episode.
        risk_free_rate: Daily risk-free rate (default 0.0).

    Returns:
        Sharpe ratio as float.  Returns 0.0 if std is zero or <2 episodes.
    """
    if len(episode_rewards) < 2:
        return 0.0
    arr = np.array(episode_rewards, dtype=np.float64)
    excess = arr - risk_free_rate
    std = excess.std()
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def compute_win_rate(episode_rewards: List[float]) -> float:
    """Fraction of episodes with positive total reward."""
    if not episode_rewards:
        return 0.0
    return float(np.mean([r > 0 for r in episode_rewards]))

