"""
colab/deepscalper/utils.py — Tensor Formatting and Feature Engineering Helpers.

Shared utility functions used by both the Colab training notebooks and the
live state_builder.  Keeping the feature computation logic here (and importing
it from state_builder.py) ensures training and inference pipelines are identical.

Key exports:
    compute_features(bars_df) → np.ndarray  shape (n_bars, INPUT_DIM)
    compute_sharpe(returns)   → float
    compute_day_starts(index) → list[int]
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")
_MARKET_OPEN_MINUTES = 9 * 60 + 30
_SESSION_DURATION = 390


# ---------------------------------------------------------------------------
# Feature helpers — identical to execution/state_builder.py
# ---------------------------------------------------------------------------

def _returns(close: pd.Series) -> pd.Series:
    ret = close.pct_change().fillna(0.0)
    return ret.clip(-0.1, 0.1)


def _norm_volume(volume: pd.Series, window: int = 60) -> pd.Series:
    mean = volume.rolling(window, min_periods=1).mean()
    std  = volume.rolling(window, min_periods=1).std().fillna(1.0).replace(0, 1.0)
    return ((volume - mean) / std).clip(-3.0, 3.0) / 3.0


def _atr_norm(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return (atr / (close + 1e-10)).clip(0.0, 0.1) / 0.1


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = ag / (al + 1e-10)
    return ((100.0 - 100.0 / (1.0 + rs)) / 100.0).fillna(0.5)


def _vwap_dev(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    tp   = (high + low + close) / 3.0
    vwap = (tp * volume).cumsum() / (volume.cumsum() + 1e-10)
    dev  = (close - vwap) / (vwap + 1e-10)
    return dev.clip(-0.05, 0.05) / 0.05


def _price_range_pos(close: pd.Series, window: int = 60) -> pd.Series:
    hi60 = close.rolling(window, min_periods=1).max()
    lo60 = close.rolling(window, min_periods=1).min()
    rng  = (hi60 - lo60).replace(0.0, 1e-10)
    return ((close - lo60) / rng).clip(0.0, 1.0)


def _vol_trend(volume: pd.Series) -> pd.Series:
    sma5  = volume.rolling(5,  min_periods=1).mean()
    sma20 = volume.rolling(20, min_periods=1).mean()
    return ((sma5 / (sma20 + 1e-10)) - 1.0).clip(-2.0, 2.0) / 2.0


def _spread(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return ((high - low) / (close + 1e-10)).clip(0.0, 0.05) / 0.05


def _momentum(close: pd.Series, window: int = 5) -> pd.Series:
    return close.pct_change(window).fillna(0.0).clip(-0.1, 0.1) / 0.1


def _session_time_feature(index: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros(len(index))
    for i, ts in enumerate(index):
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts).astimezone(_ET)
        else:
            ts = ts.astimezone(_ET)
        mins = ts.hour * 60 + ts.minute - _MARKET_OPEN_MINUTES
        out[i] = np.clip(mins, 0, _SESSION_DURATION) / _SESSION_DURATION
    return out


def _dow_feature(index: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros(len(index))
    for i, ts in enumerate(index):
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts).astimezone(_ET)
        else:
            ts = ts.astimezone(_ET)
        out[i] = ts.weekday() / 4.0
    return out


# ---------------------------------------------------------------------------
# Primary public function: compute all 11 features
# ---------------------------------------------------------------------------

def compute_features(bars: pd.DataFrame) -> np.ndarray:
    """Compute the 11-feature state matrix from a raw OHLCV DataFrame.

    This function is the canonical feature pipeline shared between training
    (Colab) and inference (live).  Any change here must be reflected in both.

    Args:
        bars: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
              and a DatetimeIndex.

    Returns:
        Numpy float32 array of shape (len(bars), 11).
        NaN and Inf values are replaced with 0.0 / ±1.0.
    """
    bars = bars.copy()
    bars.columns = [c.lower() for c in bars.columns]

    o = bars["open"].astype(float)
    h = bars["high"].astype(float)
    l = bars["low"].astype(float)
    c = bars["close"].astype(float)
    v = bars["volume"].astype(float)
    idx = bars.index

    features = np.column_stack([
        _returns(c).values,            # 0: bar return
        _norm_volume(v).values,        # 1: z-score volume
        _atr_norm(h, l, c).values,     # 2: ATR/close
        _rsi(c).values,                # 3: RSI (14)
        _vwap_dev(h, l, c, v).values,  # 4: VWAP deviation
        _price_range_pos(c).values,    # 5: price range position
        _vol_trend(v).values,          # 6: volume trend
        _spread(h, l, c).values,       # 7: spread proxy
        _momentum(c).values,           # 8: 5-bar momentum
        _session_time_feature(idx),    # 9: session time
        _dow_feature(idx),             # 10: day of week
    ]).astype(np.float32)

    # Replace non-finite values
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features


# ---------------------------------------------------------------------------
# Day boundary detection
# ---------------------------------------------------------------------------

def compute_day_starts(index: pd.DatetimeIndex) -> List[int]:
    """Find the integer indices where each new trading day begins.

    Args:
        index: DatetimeIndex of the full feature dataset (may span many days).

    Returns:
        Sorted list of integer positions where each day's first bar is located.
    """
    dates = []
    if index.tzinfo is None:
        index = index.tz_localize("UTC").tz_convert(_ET)
    else:
        index = index.tz_convert(_ET)

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
    """Annualised Sharpe ratio from a list of per-episode reward values.

    Uses 252 trading days per year as the annualisation factor.

    Args:
        episode_rewards: List of total reward per episode.
        risk_free_rate: Daily risk-free rate (default 0.0).

    Returns:
        Sharpe ratio as float.  Returns 0.0 if std is zero or fewer than 2
        episodes provided.
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
    """Fraction of episodes with positive total reward.

    Args:
        episode_rewards: List of total reward per episode.

    Returns:
        Win rate in [0, 1].
    """
    if not episode_rewards:
        return 0.0
    return float(np.mean([r > 0 for r in episode_rewards]))
