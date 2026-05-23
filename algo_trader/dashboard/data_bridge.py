"""
dashboard/data_bridge.py — Thread-Safe Shared State Between Lumibot and PyQt5.

The DataBridge acts as the single source of truth for all dashboard data.  The
Lumibot execution thread writes to it; the PyQt5 QTimer slot reads from it.
All mutation goes through a threading.Lock to prevent race conditions.

Design principle: the Lumibot thread should never block the GUI thread, and
the GUI thread should never call Lumibot APIs.  DataBridge is the membrane
between these two worlds.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config import MAX_TRADE_LOG_ENTRIES


# ---------------------------------------------------------------------------
# Data classes — lightweight value objects, not thread-safe on their own
# ---------------------------------------------------------------------------

@dataclass
class PositionSnapshot:
    """Immutable snapshot of a single open position at a point in time.

    Attributes:
        symbol: Trading symbol (e.g., "BTC/USD").
        side: "LONG" or "FLAT".
        qty: Absolute position quantity.
        entry_price: Average fill price of the opening order.
        current_price: Most recent market price.
        unrealized_pnl: Unrealised P&L in USD = (current_price − entry_price) × qty.
        atr_stop: ATR-derived stop-loss price for this position.
        atr_tp: ATR-derived take-profit price for this position.
        entry_time: ISO timestamp when the position was opened (optional).
    """

    symbol: str
    side: str
    qty: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    atr_stop: float
    atr_tp: float
    entry_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ModelSignal:
    """Latest inference output for a single ticker from the DeepScalper model.

    Attributes:
        symbol: Trading symbol (e.g., "BTC/USD").
        action: "LONG" or "FLAT".
        q_values: Raw Q-values as a 2-element list [Q_FLAT, Q_LONG].
        confidence: Softmax-max score in [0, 1] indicating action certainty.
        timestamp: HH:MM:SS Eastern Time string of the inference.
    """

    symbol: str
    action: str
    q_values: List[float]
    confidence: float
    timestamp: str


@dataclass
class TradeEvent:
    """A single timestamped event in the trade log.

    Attributes:
        timestamp: HH:MM:SS Eastern Time string.
        symbol: Affected symbol, or "ALL" for portfolio-wide events.
        side: "BUY", "SELL", or descriptive string for non-trade events.
        qty: Trade quantity (0 for non-trade events).
        price: Fill price (0.0 for non-trade events).
        event_type: Category label: "FILL", "HALT", "EOD_CLOSE", "STOP_HIT", etc.
    """

    timestamp: str
    symbol: str
    side: str
    qty: int
    price: float
    event_type: str


# ---------------------------------------------------------------------------
# DataBridge
# ---------------------------------------------------------------------------

class DataBridge:
    """Thread-safe shared state between the Lumibot execution thread and
    the PyQt5 dashboard thread.

    All public methods acquire self._lock before reading or writing state to
    ensure the dashboard always sees a consistent snapshot even when Lumibot
    is mid-iteration.

    Usage (Lumibot thread):
        bridge.update_signal("BTC/USD", ModelSignal(...))
        bridge.append_trade_event(TradeEvent(...))

    Usage (PyQt5 thread — inside QTimer slot):
        signals = bridge.get_all_signals()
        positions = bridge.get_all_positions()
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()

        # State containers
        self._positions: Dict[str, PositionSnapshot] = {}
        self._signals: Dict[str, ModelSignal] = {}
        self._trade_log: List[TradeEvent] = []

        # Portfolio-level metrics
        self._portfolio_value: float = 0.0
        self._daily_pnl: float = 0.0
        self._max_drawdown: float = 0.0
        self._peak_value: float = 0.0

        # Status flags
        self._is_halted: bool = False
        self._halt_reason: str = ""

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def update_positions(self, positions: Dict[str, PositionSnapshot]) -> None:
        """Replace the entire positions snapshot atomically.

        Args:
            positions: Dict mapping symbol → PositionSnapshot.
        """
        with self._lock:
            self._positions = dict(positions)

    def get_all_positions(self) -> Dict[str, PositionSnapshot]:
        """Return a copy of all current open position snapshots.

        Returns:
            Dict mapping symbol → PositionSnapshot.  Safe to iterate without
            holding the lock.
        """
        with self._lock:
            return dict(self._positions)

    # ------------------------------------------------------------------
    # Model signal management
    # ------------------------------------------------------------------

    def update_signal(self, symbol: str, signal: ModelSignal) -> None:
        """Set the latest model signal for a single ticker.

        Args:
            symbol: Stock ticker symbol.
            signal: ModelSignal instance from the current iteration.
        """
        with self._lock:
            self._signals[symbol] = signal

    def get_all_signals(self) -> Dict[str, ModelSignal]:
        """Return a copy of all latest model signals.

        Returns:
            Dict mapping symbol → ModelSignal.
        """
        with self._lock:
            return dict(self._signals)

    # ------------------------------------------------------------------
    # Trade log management
    # ------------------------------------------------------------------

    def append_trade_event(self, event: TradeEvent) -> None:
        """Append a trade event and enforce the maximum log size (FIFO).

        Args:
            event: TradeEvent to add to the log.
        """
        with self._lock:
            self._trade_log.append(event)
            # Evict oldest entries when over the limit
            if len(self._trade_log) > MAX_TRADE_LOG_ENTRIES:
                self._trade_log = self._trade_log[-MAX_TRADE_LOG_ENTRIES:]

    def get_trade_log(self) -> List[TradeEvent]:
        """Return a copy of the trade log.

        Returns:
            List of TradeEvent objects, oldest first.
        """
        with self._lock:
            return list(self._trade_log)

    # ------------------------------------------------------------------
    # Portfolio metrics — property-based for ergonomics
    # ------------------------------------------------------------------

    @property
    def portfolio_value(self) -> float:
        """Current total portfolio value in USD."""
        with self._lock:
            return self._portfolio_value

    @portfolio_value.setter
    def portfolio_value(self, value: float) -> None:
        with self._lock:
            self._portfolio_value = value
            # Track peak for drawdown calculation
            if value > self._peak_value:
                self._peak_value = value
            if self._peak_value > 0:
                dd = (value - self._peak_value) / self._peak_value
                self._max_drawdown = min(self._max_drawdown, dd)

    @property
    def daily_pnl(self) -> float:
        """Running daily P&L in USD."""
        with self._lock:
            return self._daily_pnl

    @daily_pnl.setter
    def daily_pnl(self, value: float) -> None:
        with self._lock:
            self._daily_pnl = value

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown fraction since strategy start (negative value)."""
        with self._lock:
            return self._max_drawdown

    @property
    def is_halted(self) -> bool:
        """True when a circuit breaker has suspended trading."""
        with self._lock:
            return self._is_halted

    @is_halted.setter
    def is_halted(self, value: bool) -> None:
        with self._lock:
            self._is_halted = value

    @property
    def halt_reason(self) -> str:
        """Human-readable reason for the current trading halt (empty if not halted)."""
        with self._lock:
            return self._halt_reason

    @halt_reason.setter
    def halt_reason(self, value: str) -> None:
        with self._lock:
            self._halt_reason = value

    # ------------------------------------------------------------------
    # Convenience snapshot
    # ------------------------------------------------------------------

    def get_portfolio_summary(self) -> dict:
        """Return a snapshot of all portfolio-level metrics in one lock acquisition.

        Returns:
            Dict with keys: portfolio_value, daily_pnl, max_drawdown,
            is_halted, halt_reason, open_positions_count.
        """
        with self._lock:
            return {
                "portfolio_value": self._portfolio_value,
                "daily_pnl": self._daily_pnl,
                "max_drawdown": self._max_drawdown,
                "is_halted": self._is_halted,
                "halt_reason": self._halt_reason,
                "open_positions_count": len(self._positions),
            }
