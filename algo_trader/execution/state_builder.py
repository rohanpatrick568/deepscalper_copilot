"""
execution/state_builder.py — Live Bar Data → DeepScalper State Tensor.

This module is the critical bridge between Lumibot's live bar DataFrame and
the DeepScalper model's expected input format.

The feature engineering pipeline implemented here MUST be identical to the
one in `colab/02_feature_engineering.ipynb`.  Any divergence will cause silent
mismatch between training and inference distributions.

Feature vector (11 features per bar, matching INPUT_DIM = 11):
    0.  Return            — (close_t − close_{t-1}) / close_{t-1}
    1.  Norm volume       — (volume − vol_mean) / (vol_std + ε), window=60
    2.  ATR(14)           — mean(high − low, 14) / close, normalised to [0, 1]
    3.  RSI(14)           — Wilder RSI / 100, range [0, 1]
    4.  VWAP deviation    — (close − vwap) / (vwap + ε)
    5.  Price range pos.  — (close − low_60) / (high_60 − low_60 + ε), [0, 1]
    6.  Volume trend      — vol_SMA5 / (vol_SMA20 + ε)
    7.  Spread proxy      — (high − low) / (close + ε)
    8.  Momentum (5-bar)  — (close_t − close_{t-5}) / (close_{t-5} + ε)
    9.  Session time      — minutes_since_open / 390, range [0, 1]
    10. Day of week       — ordinal 0 (Monday) – 4 (Friday) / 4, range [0, 1]

Usage:
    from execution.state_builder import build_state_tensor
    tensor = build_state_tensor(bars_df)   # shape (1, 60, 11)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import torch

from config import INPUT_DIM, LOOKBACK_BARS, MARKET_TIMEZONE

logger = logging.getLogger(__name__)

_ET = pytz.timezone(MARKET_TIMEZONE)
_MARKET_OPEN_MINUTES = 9 * 60 + 30   # 9:30 ET in minutes from midnight
_SESSION_DURATION_MINUTES = 390       # 6.5 hour regular session


# ---------------------------------------------------------------------------
# Feature computation helpers (stateless pure functions)
# ---------------------------------------------------------------------------

def _returns(close: pd.Series) -> pd.Series:
    """Compute bar-over-bar returns. First element is 0."""
    ret = close.pct_change()
    ret.iloc[0] = 0.0
    return ret.clip(-0.1, 0.1)  # Cap at ±10 % to suppress outliers


def _norm_volume(volume: pd.Series, window: int = 60) -> pd.Series:
    """Z-score normalised volume with a rolling window."""
    mean = volume.rolling(window, min_periods=1).mean()
    std = volume.rolling(window, min_periods=1).std().fillna(1.0)
    std = std.replace(0, 1.0)
    return ((volume - mean) / std).clip(-3.0, 3.0) / 3.0  # Rescale to ~[-1, 1]


def _atr_normalised(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR(period) divided by close price → scale-free, normalised to [0, ~0.05]."""
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return (atr / (close + 1e-10)).clip(0.0, 0.1) / 0.1  # Normalise to [0, 1]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI normalised to [0, 1].  Uses SMA-based initialisation."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return (rsi / 100.0).fillna(0.5)  # [0, 1]; default 0.5 (neutral) for NaN


def _vwap_deviation(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """VWAP deviation: (close − VWAP) / VWAP.  VWAP computed over the available bars."""
    typical = (high + low + close) / 3.0
    vwap = (typical * volume).cumsum() / (volume.cumsum() + 1e-10)
    deviation = (close - vwap) / (vwap + 1e-10)
    return deviation.clip(-0.05, 0.05) / 0.05  # Normalise to [-1, 1]


def _price_range_position(close: pd.Series, window: int = 60) -> pd.Series:
    """Fractional position of close within its rolling high-low range → [0, 1]."""
    high_60 = close.rolling(window, min_periods=1).max()
    low_60 = close.rolling(window, min_periods=1).min()
    rng = high_60 - low_60
    rng = rng.replace(0.0, 1e-10)
    return ((close - low_60) / rng).clip(0.0, 1.0)


def _volume_trend(volume: pd.Series) -> pd.Series:
    """5-bar volume SMA / 20-bar volume SMA — ratio of short to long-term volume."""
    sma5 = volume.rolling(5, min_periods=1).mean()
    sma20 = volume.rolling(20, min_periods=1).mean()
    ratio = sma5 / (sma20 + 1e-10)
    return (ratio - 1.0).clip(-2.0, 2.0) / 2.0  # Centre at 0, normalise


def _spread_proxy(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Intra-bar spread proxy: (high − low) / close → [0, 1]."""
    spread = (high - low) / (close + 1e-10)
    return spread.clip(0.0, 0.05) / 0.05  # Normalise to [0, 1]


def _momentum(close: pd.Series, window: int = 5) -> pd.Series:
    """N-bar momentum: (close_t − close_{t-N}) / close_{t-N} → [-1, 1]."""
    mom = close.pct_change(window).fillna(0.0)
    return mom.clip(-0.1, 0.1) / 0.1


def _session_time(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since market open (9:30 ET) normalised to [0, 1]."""
    result = np.zeros(len(index))
    for i, ts in enumerate(index):
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts).astimezone(_ET)
        else:
            ts = ts.astimezone(_ET)
        minutes_from_midnight = ts.hour * 60 + ts.minute
        minutes_since_open = minutes_from_midnight - _MARKET_OPEN_MINUTES
        result[i] = np.clip(minutes_since_open, 0, _SESSION_DURATION_MINUTES) / _SESSION_DURATION_MINUTES
    return result


def _day_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of week ordinal (Mon=0, Fri=4) normalised to [0, 1]."""
    result = np.zeros(len(index))
    for i, ts in enumerate(index):
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts).astimezone(_ET)
        else:
            ts = ts.astimezone(_ET)
        result[i] = ts.weekday() / 4.0  # Normalise Monday=0, Friday=1
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_state_tensor(
    bars: pd.DataFrame,
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """Convert a Lumibot historical bars DataFrame into a DeepScalper state tensor.

    Applies the exact same normalisation pipeline used during training in
    `colab/02_feature_engineering.ipynb`.  If the DataFrame has fewer rows than
    LOOKBACK_BARS the function returns None and logs a warning (the calling
    strategy should skip inference for this bar).

    Args:
        bars: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
              and a DatetimeIndex.  Should have at least LOOKBACK_BARS rows.
        device: PyTorch device string, e.g. "cpu" or "cuda:0".

    Returns:
        Tensor of shape (1, LOOKBACK_BARS, INPUT_DIM) on the requested device,
        or None if the DataFrame is too short or contains critical NaN values.
    """
    # --- Validate input ---
    required_cols = {"open", "high", "low", "close", "volume"}
    missing_cols = required_cols - set(bars.columns)
    if missing_cols:
        logger.warning("build_state_tensor: DataFrame missing columns %s", missing_cols)
        return None

    if len(bars) < LOOKBACK_BARS:
        logger.debug(
            "build_state_tensor: only %d bars available, need %d — skipping",
            len(bars),
            LOOKBACK_BARS,
        )
        return None

    # Use only the most recent LOOKBACK_BARS rows (important for consistency)
    bars = bars.tail(LOOKBACK_BARS).copy()

    # Normalise column names to lower-case (Lumibot may return title-case)
    bars.columns = [c.lower() for c in bars.columns]

    open_ = bars["open"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    index = bars.index

    # --- Compute all 11 features ---
    # Feature 0: return
    f0 = _returns(close)
    # Feature 1: normalised volume
    f1 = _norm_volume(volume)
    # Feature 2: ATR (normalised)
    f2 = _atr_normalised(high, low, close)
    # Feature 3: RSI (14)
    f3 = _rsi(close)
    # Feature 4: VWAP deviation
    f4 = _vwap_deviation(high, low, close, volume)
    # Feature 5: price range position (60-bar)
    f5 = _price_range_position(close)
    # Feature 6: volume trend (SMA5/SMA20)
    f6 = _volume_trend(volume)
    # Feature 7: spread proxy
    f7 = _spread_proxy(high, low, close)
    # Feature 8: 5-bar momentum
    f8 = _momentum(close)
    # Feature 9: session time
    f9 = pd.Series(_session_time(index), index=index)
    # Feature 10: day of week
    f10 = pd.Series(_day_of_week(index), index=index)

    # --- Stack into (LOOKBACK_BARS, INPUT_DIM) array ---
    feature_matrix = np.column_stack([
        f0.values,
        f1.values,
        f2.values,
        f3.values,
        f4.values,
        f5.values,
        f6.values,
        f7.values,
        f8.values,
        f9.values,
        f10.values,
    ]).astype(np.float32)

    assert feature_matrix.shape == (LOOKBACK_BARS, INPUT_DIM), (
        f"Feature matrix shape mismatch: expected ({LOOKBACK_BARS}, {INPUT_DIM}), "
        f"got {feature_matrix.shape}"
    )

    # --- Check for NaN / inf ---
    if not np.isfinite(feature_matrix).all():
        nan_count = np.sum(~np.isfinite(feature_matrix))
        logger.warning(
            "build_state_tensor: %d non-finite values in feature matrix — replacing with 0",
            nan_count,
        )
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    # --- Convert to tensor and add batch dimension ---
    tensor = torch.from_numpy(feature_matrix).unsqueeze(0).to(device)
    # Shape: (1, LOOKBACK_BARS, INPUT_DIM)
    return tensor
