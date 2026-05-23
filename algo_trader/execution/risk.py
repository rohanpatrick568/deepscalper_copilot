"""
execution/risk.py — Kelly Criterion Position Sizing and ATR-based Stop Calculation.

Provides two pure functions used by the execution strategy:

1. kelly_position_size  — fractional Kelly formula → integer share count
2. calculate_atr_stop   — ATR-based stop-loss and take-profit prices

All numeric constants default to values from config.py so that every
parameter can be tuned in one place.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    ATR_PERIOD,
    ATR_STOP_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    KELLY_FRACTION,
    MAX_POSITION_PCT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_atr(bars: pd.DataFrame, period: int) -> float:
    """Compute Average True Range from a OHLCV DataFrame.

    Uses the Wilder/standard True Range definition:
        TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
        ATR = rolling mean of TR over `period` bars

    Args:
        bars: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
              Must contain at least `period + 1` rows.
        period: Number of bars over which to average the True Range.

    Returns:
        ATR value as a float.  Returns NaN if insufficient data.
    """
    if len(bars) < period + 1:
        logger.warning(
            "ATR calculation requires at least %d bars, got %d", period + 1, len(bars)
        )
        return float("nan")

    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    portfolio_value: float,
    price: float,
    kelly_fraction: float = KELLY_FRACTION,
    max_position_pct: float = MAX_POSITION_PCT,
) -> int:
    """Calculate fractional Kelly Criterion position size in integer shares.

    Applies the Kelly formula:
        f* = (p × b − q) / b
    where:
        p  = win_rate  (fraction of trades that are profitable)
        q  = 1 − p
        b  = avg_win / avg_loss  (payoff ratio)

    The raw Kelly fraction f* is then scaled by `kelly_fraction` (e.g. 0.5 for
    half-Kelly) and capped at `max_position_pct` of portfolio_value.

    Args:
        win_rate: Historical win rate in range [0, 1].
        avg_win: Average dollar gain on winning trades (must be > 0).
        avg_loss: Average dollar loss on losing trades (must be > 0).
        portfolio_value: Total portfolio value in USD.
        price: Current ask price of the asset in USD.
        kelly_fraction: Scaling factor applied to raw Kelly (default KELLY_FRACTION).
        max_position_pct: Maximum fraction of portfolio for this position (default MAX_POSITION_PCT).

    Returns:
        Integer number of shares to purchase (minimum 1).
    """
    if avg_loss <= 0 or avg_win <= 0 or portfolio_value <= 0 or price <= 0:
        logger.warning(
            "kelly_position_size received invalid inputs: "
            "avg_win=%.4f avg_loss=%.4f portfolio=%.2f price=%.2f — returning 1 share",
            avg_win,
            avg_loss,
            portfolio_value,
            price,
        )
        return 1

    # Clamp win_rate to a sensible range to avoid degenerate Kelly fractions
    win_rate = float(np.clip(win_rate, 0.01, 0.99))
    lose_rate = 1.0 - win_rate
    payoff_ratio = avg_win / avg_loss  # b

    # Kelly formula: f* = (p*b - q) / b
    raw_kelly = (win_rate * payoff_ratio - lose_rate) / payoff_ratio

    if raw_kelly <= 0:
        # Negative Kelly → edge is too thin; take minimum position
        logger.debug("Kelly fraction is negative (%.4f) — sizing to 1 share", raw_kelly)
        return 1

    # Apply fractional Kelly and max-position cap
    kelly_dollars = raw_kelly * kelly_fraction * portfolio_value
    max_dollars = max_position_pct * portfolio_value
    position_dollars = min(kelly_dollars, max_dollars)

    shares = int(position_dollars / price)
    shares = max(1, shares)  # Enforce minimum of 1 share

    logger.debug(
        "Kelly sizing: win_rate=%.2f payoff=%.2f raw_f=%.4f "
        "scaled=$%.2f → %d shares @ $%.2f",
        win_rate,
        payoff_ratio,
        raw_kelly,
        position_dollars,
        shares,
        price,
    )
    return shares


def calculate_atr_stop(
    bars: pd.DataFrame,
    entry_price: float,
    side: str,
    atr_period: int = ATR_PERIOD,
    stop_multiplier: float = ATR_STOP_MULTIPLIER,
    tp_multiplier: float = ATR_TP_MULTIPLIER,
) -> Tuple[float, float]:
    """Calculate ATR-based stop-loss and take-profit prices.

    ATR (Average True Range) is computed over the last `atr_period` bars of the
    supplied OHLCV DataFrame.  Stop and take-profit distances are then:
        distance_stop = ATR × stop_multiplier
        distance_tp   = ATR × tp_multiplier

    For long positions:
        stop_loss   = entry_price − distance_stop
        take_profit = entry_price + distance_tp

    For short positions (reversed):
        stop_loss   = entry_price + distance_stop
        take_profit = entry_price − distance_tp

    Args:
        bars: DataFrame with columns ['high', 'low', 'close'] and at least
              `atr_period + 1` rows.
        entry_price: The fill / entry price for the trade.
        side: "buy" for long positions, "sell" for short positions.
        atr_period: Look-back period for ATR (default ATR_PERIOD from config).
        stop_multiplier: ATR multiple for stop distance (default ATR_STOP_MULTIPLIER).
        tp_multiplier: ATR multiple for take-profit distance (default ATR_TP_MULTIPLIER).

    Returns:
        Tuple of (stop_loss_price, take_profit_price) as floats.
        Both are rounded to 2 decimal places.

    Raises:
        ValueError: If `side` is not "buy" or "sell".
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")

    atr = _compute_atr(bars, atr_period)

    if np.isnan(atr) or atr <= 0:
        # Fallback: use 0.5 % of entry price as a minimal distance
        logger.warning(
            "ATR calculation returned invalid value (%.4f). "
            "Falling back to 0.5%% of entry price.",
            atr,
        )
        atr = entry_price * 0.005

    stop_distance = atr * stop_multiplier
    tp_distance = atr * tp_multiplier

    if side == "buy":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + tp_distance
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - tp_distance

    stop_loss = round(max(stop_loss, 0.01), 2)   # Price can't go below $0.01
    take_profit = round(take_profit, 2)

    logger.debug(
        "ATR stops for %s @ $%.2f: ATR=%.4f stop=$%.2f tp=$%.2f",
        side,
        entry_price,
        atr,
        stop_loss,
        take_profit,
    )
    return stop_loss, take_profit
