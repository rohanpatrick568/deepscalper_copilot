"""
execution/circuit_breakers.py — Safety Guards for the DeepScalper Trading System.

The CircuitBreaker class enforces two categories of safety rules:

1. Session time guards: no trades in the first or last N minutes of the market
   session.  Time is evaluated in US/Eastern timezone using pytz, independent of
   Lumibot's internal clock.

2. Daily loss halt: if the cumulative daily P&L (realised + unrealised) falls
   below MAX_DAILY_LOSS_PCT × starting_capital the entire strategy is halted for
   the remainder of the session.

Usage:
    cb = CircuitBreaker(MAX_DAILY_LOSS_PCT, STARTING_CAPITAL)
    halted, reason = cb.is_trading_halted()
    if halted:
        logger.warning(reason)
        return
"""

import logging
from datetime import datetime, time
from typing import Tuple

import pytz

from config import (
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_TIMEZONE,
    NO_TRADE_CLOSE_BUFFER_MIN,
    NO_TRADE_OPEN_BUFFER_MIN,
)

logger = logging.getLogger(__name__)

# US/Eastern timezone object (reused to avoid repeated construction)
_ET = pytz.timezone(MARKET_TIMEZONE)


def _current_et_time() -> time:
    """Return the current wall-clock time in US/Eastern timezone.

    Returns:
        datetime.time object in Eastern Time.
    """
    return datetime.now(_ET).time()


class CircuitBreaker:
    """Stateful safety guard enforcing session-time and daily-loss rules.

    Args:
        max_daily_loss_pct: Maximum allowed daily loss as a fraction of capital
                            (e.g. 0.03 for 3 %).
        starting_capital: Portfolio starting value in USD used to compute the
                          absolute dollar loss threshold.

    Attributes:
        _daily_pnl: Running realised + unrealised daily P&L in USD.
        _halted_by_loss: Whether the daily loss limit has been triggered.
    """

    def __init__(self, max_daily_loss_pct: float, starting_capital: float) -> None:
        self._max_daily_loss_pct: float = max_daily_loss_pct
        self._starting_capital: float = starting_capital
        self._daily_pnl: float = 0.0
        self._halted_by_loss: bool = False

        # Pre-compute session boundary times for fast comparison
        self._session_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
        self._no_trade_start = time(
            MARKET_OPEN_HOUR,
            MARKET_OPEN_MINUTE + NO_TRADE_OPEN_BUFFER_MIN,
        )
        # Close buffer: last N minutes of the session
        total_close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE
        buffer_start_minutes = total_close_minutes - NO_TRADE_CLOSE_BUFFER_MIN
        self._no_trade_end = time(
            buffer_start_minutes // 60,
            buffer_start_minutes % 60,
        )
        self._session_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)

        logger.info(
            "CircuitBreaker initialised: max_loss=%.1f%% ($%.2f), "
            "no-trade window=%s–%s and %s–%s ET",
            max_daily_loss_pct * 100,
            starting_capital * max_daily_loss_pct,
            self._session_open.strftime("%H:%M"),
            self._no_trade_start.strftime("%H:%M"),
            self._no_trade_end.strftime("%H:%M"),
            self._session_close.strftime("%H:%M"),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_trading_halted(self) -> Tuple[bool, str]:
        """Determine whether trading should be suspended at this moment.

        Checks are evaluated in priority order:
        1. Daily loss breach (permanent for remainder of session).
        2. Opening buffer (first N minutes after market open).
        3. Closing buffer (last N minutes before market close).

        Returns:
            Tuple of (is_halted: bool, reason: str).
            If not halted, reason is an empty string.
        """
        # Check 1: Daily loss limit (sticky — persists once triggered)
        if self._halted_by_loss:
            loss_pct = abs(self._daily_pnl) / self._starting_capital * 100
            return (
                True,
                f"CIRCUIT BREAKER: Daily loss limit reached "
                f"(P&L: ${self._daily_pnl:.2f}, {loss_pct:.2f}%)",
            )

        now = _current_et_time()

        # Check 2: Opening no-trade buffer
        if self._session_open <= now < self._no_trade_start:
            return (
                True,
                f"OPENING BUFFER: No trades until {self._no_trade_start.strftime('%H:%M')} ET "
                f"(currently {now.strftime('%H:%M:%S')} ET)",
            )

        # Check 3: Closing no-trade buffer
        if self._no_trade_end <= now < self._session_close:
            return (
                True,
                f"CLOSING BUFFER: No trades after {self._no_trade_end.strftime('%H:%M')} ET "
                f"(currently {now.strftime('%H:%M:%S')} ET)",
            )

        return False, ""

    def update_daily_pnl(self, pnl_delta: float) -> None:
        """Update the running daily P&L and trigger a halt if the limit is breached.

        Should be called on every fill event and whenever unrealised P&L is refreshed.

        Args:
            pnl_delta: Dollar amount to add to the running daily P&L total.
                       Use negative values for losses.
        """
        self._daily_pnl += pnl_delta

        loss_threshold = -self._max_daily_loss_pct * self._starting_capital
        if self._daily_pnl <= loss_threshold and not self._halted_by_loss:
            self._halted_by_loss = True
            logger.critical(
                "CIRCUIT BREAKER TRIGGERED: Daily P&L $%.2f breached "
                "%.1f%% loss threshold ($%.2f). Trading halted.",
                self._daily_pnl,
                self._max_daily_loss_pct * 100,
                loss_threshold,
            )

    def set_daily_pnl(self, absolute_pnl: float) -> None:
        """Replace the daily P&L with an absolute value from a portfolio snapshot.

        Use this when the execution engine has the total daily P&L directly
        (e.g. from Alpaca account data) rather than incremental deltas.

        Args:
            absolute_pnl: Absolute daily P&L in USD (negative for loss).
        """
        delta = absolute_pnl - self._daily_pnl
        self.update_daily_pnl(delta)

    def reset_for_new_day(self) -> None:
        """Reset all daily-session state at market open.

        Should be called from the strategy's before_market_opens() hook.
        """
        logger.info(
            "CircuitBreaker: resetting for new day (previous P&L: $%.2f).",
            self._daily_pnl,
        )
        self._daily_pnl = 0.0
        self._halted_by_loss = False

    @property
    def daily_pnl(self) -> float:
        """Current running daily P&L in USD (read-only)."""
        return self._daily_pnl

    @property
    def is_halted_by_loss(self) -> bool:
        """True if the daily loss limit has been permanently triggered today."""
        return self._halted_by_loss
