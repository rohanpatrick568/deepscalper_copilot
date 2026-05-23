"""\ncolab/deepscalper/utils.py — Feature Engineering for DeepScalper (paper-faithful).\n\nImplements the exact feature sets described in the DeepScalper paper (CIKM '22):\n\n  compute_macro_features(bars_df) → ndarray (n, MACRO_DIM=11)\n      11 macro features from Table 2:\n        z_open, z_high, z_low, z_close, z_adj_close,\n        z_d_5, z_d_10, z_d_15, z_d_20, z_d_25, z_d_30\n\n  compute_micro_features(bars_df, lob_snapshots=None, use_proxy=True) → ndarray (n, LOB_DIM=4)\n      V2 CHANGE: Dual-mode (proxy for training, real LOB for inference).\n      4 microstructure features: spread_pct, order_imbalance, depth_ratio, mid_move_1min\n\n  compute_features(bars_df) → ndarray (n, 11)\n      Alias for compute_macro_features (backward compatibility).\n\n  compute_day_starts(index) → list[int]  # V2 CHANGE: UTC midnight boundaries (crypto 24/7)\n  _compute_time_features(index) → ndarray (n, 2)  # V2 CHANGE: UTC sin/cos hour encoding\n  compute_sharpe(rewards)   → float\n  compute_win_rate(rewards) → float\n"""\n\nimport logging\nfrom typing import List, Optional\n\nimport numpy as np\nimport pandas as pd\nimport pytz\n\nlogger = logging.getLogger(__name__)\n\n_ET = pytz.timezone("US/Eastern")\n

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
# Micro features — Dual-mode (V2 CHANGE: proxy OHLCV or real Alpaca LOB)
# ---------------------------------------------------------------------------

def compute_micro_features(
    bars: pd.DataFrame,
    lob_snapshots: Optional[pd.DataFrame] = None,
    use_proxy: bool = True,
) -> np.ndarray:
    """Compute 4 microstructure features — dual-mode for training vs. inference.

    TRAINING MODE (use_proxy=True, default):
        Reconstructs LOB microstructure from OHLCV candlestick data using
        proven proxy formulas (Corwin-Schultz, Kyle volume model).

    INFERENCE MODE (use_proxy=False, lob_snapshots provided):
        Uses real Alpaca orderbook data (top 3 bid/ask levels).
        lob_snapshots must have columns:
            bid_price_1, bid_size_1, bid_price_2, bid_size_2, bid_price_3, bid_size_3
            ask_price_1, ask_size_1, ask_price_2, ask_size_2, ask_price_3, ask_size_3

    V2 CHANGE: 5 features → 4 features (matches TradeMaster micro feature count):
        0  spread_pct:      Bid-ask spread as % of mid price
        1  order_imbalance: (buy_vol - sell_vol) / total_vol in [-1, 1]
        2  depth_ratio:     bid_depth / ask_depth (log-normalised)
        3  mid_move_1min:   Close pct change clipped to [-5%, +5%]

    IMPORTANT: Feature names and ORDER are identical in both modes so that
    a model trained on proxies can be used directly with real LOB features.

    Args:
        bars: DataFrame with OHLCV columns and DatetimeIndex.
        lob_snapshots: Optional real orderbook snapshot DataFrame.
        use_proxy: If True, use OHLCV proxy; if False, use lob_snapshots.

    Returns:
        float32 array of shape (len(bars), 4).
    """
    df = bars.copy()
    df.columns = [c.lower() for c in df.columns]

    if use_proxy or lob_snapshots is None:
        # ----------------------------------------------------------------
        # PROXY MODE: reconstruct from OHLCV
        # ----------------------------------------------------------------
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["volume"].astype(float)

        # 1. Spread proxy (Corwin-Schultz simplified for 1-min bars)
        spread_pct = ((h - l) / ((h + l) / 2 + 1e-10)).clip(0.0, 0.05)

        # 2. Order imbalance proxy (Kyle 1985: up-bars = buy-driven)
        price_direction = np.sign(c - o)
        buy_vol  = v * (price_direction > 0).astype(float)
        sell_vol = v * (price_direction < 0).astype(float)
        total_vol = buy_vol + sell_vol
        order_imbalance = np.where(
            total_vol > 0,
            (buy_vol - sell_vol) / total_vol,
            0.0,
        ).clip(-1.0, 1.0)

        # 3. Depth ratio proxy: rolling volume ratio as bid/ask depth stand-in
        vol_5   = v.rolling(5, min_periods=1).sum()
        vol_p5  = v.shift(5).rolling(5, min_periods=1).sum().fillna(vol_5)
        depth_ratio = (np.log((vol_5 / (vol_p5 + 1e-8)).clip(0.1, 10.0))).values / 3.0

        # 4. Mid-price 1-min return
        mid_move = c.pct_change().fillna(0.0).clip(-0.05, 0.05)

        features = np.column_stack([
            spread_pct.values,
            order_imbalance,
            depth_ratio,
            mid_move.values,
        ]).astype(np.float32)

    else:
        # ----------------------------------------------------------------
        # REAL LOB MODE: compute from Alpaca orderbook snapshots
        # ----------------------------------------------------------------
        mid_price = (
            lob_snapshots["bid_price_1"] + lob_snapshots["ask_price_1"]
        ) / 2

        # 1. Real spread
        spread_pct = (
            (lob_snapshots["ask_price_1"] - lob_snapshots["bid_price_1"])
            / (mid_price + 1e-10)
        ).clip(0.0, 0.05)

        # 2. Real order imbalance (top 3 levels)
        bid_vol = (
            lob_snapshots["bid_size_1"]
            + lob_snapshots["bid_size_2"]
            + lob_snapshots["bid_size_3"]
        )
        ask_vol = (
            lob_snapshots["ask_size_1"]
            + lob_snapshots["ask_size_2"]
            + lob_snapshots["ask_size_3"]
        )
        order_imbalance = (
            (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        ).clip(-1.0, 1.0)

        # 3. Real depth ratio (log-normalised)
        depth_ratio = np.log(
            (bid_vol / (ask_vol + 1e-8)).clip(0.01, 100.0)
        ).clip(-3.0, 3.0) / 3.0

        # 4. Mid-price 1-min return
        mid_move = mid_price.pct_change().fillna(0.0).clip(-0.05, 0.05)

        features = np.column_stack([
            spread_pct.values,
            order_imbalance.values,
            depth_ratio.values,
            mid_move.values,
        ]).astype(np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.05, neginf=-0.05)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

def compute_features(bars: pd.DataFrame) -> np.ndarray:
    """Alias for compute_macro_features (backward compatibility)."""
    return compute_macro_features(bars)


def _session_time_feature(index: pd.DatetimeIndex) -> np.ndarray:
    """Legacy ET session-time feature — kept for backward compatibility only.

    V2 CHANGE: Replaced by _compute_time_features() for 24/7 crypto markets.
    """
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


def _compute_time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """V2 CHANGE: UTC cyclical hour-of-day features for 24/7 crypto markets.

    Rationale: Crypto has real time-of-day patterns even without formal sessions
    (Asia 00:00-08:00 UTC, Europe 07:00-16:00 UTC, US 13:00-21:00 UTC).
    Sin/cos encoding prevents the model from treating 23:59 and 00:01 as
    maximally different — they are adjacent, not opposite.

    Matches TradeMaster's time feature implementation convention.

    Args:
        index: DatetimeIndex of the bar series.

    Returns:
        float32 array of shape (N, 2): columns are [sin_hour, cos_hour].
    """
    if index.tzinfo is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    hour_of_day = index.hour + index.minute / 60.0   # e.g. 14:30 → 14.5
    sin_hour = np.sin(2 * np.pi * hour_of_day / 24.0).astype(np.float32)
    cos_hour = np.cos(2 * np.pi * hour_of_day / 24.0).astype(np.float32)
    return np.stack([sin_hour, cos_hour], axis=1)  # (N, 2)




# ---------------------------------------------------------------------------
# Day boundary detection
# ---------------------------------------------------------------------------

def compute_day_starts(index: pd.DatetimeIndex) -> List[int]:
    """Find integer indices where each new UTC calendar day begins.

    V2 CHANGE: Uses UTC midnight boundaries instead of US/Eastern session starts.
    For 24/7 crypto assets, a 'day' is defined as UTC 00:00 to 23:59.
    TradeMaster uses this same convention for all continuous markets.

    Args:
        index: DatetimeIndex of the full feature dataset (may span many days).

    Returns:
        Sorted list of integer positions where each UTC day's first bar is located.
    """
    # V2 CHANGE: Convert to UTC (was US/Eastern)
    if index.tzinfo is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    dates = index.date
    day_starts = [0]  # First bar is always a day start
    for i in range(1, len(dates)):
        if dates[i] != dates[i - 1]:
            day_starts.append(i)
    return day_starts


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

