"""
dashboard/main_window.py — Root PyQt5 QMainWindow for the AlgoTrader Dashboard.

Layout (dark-themed, 1400 × 900 minimum):
    ┌─────────────────────────────────────────────────────────┐
    │  EquityBar  (full width, 44px tall)                     │
    ├──────────────────┬────────────────┬──────────────────── ┤
    │  PositionsTable  │ ConfidencePanel│  TradeLog           │
    │  (50 %)          │ (30 %)         │  (20 %)             │
    └──────────────────┴────────────────┴─────────────────────┘

A QTimer fires every DASHBOARD_REFRESH_MS milliseconds and calls
refresh_all(), which propagates to each sub-widget in sequence.

Threading note:
    The QTimer runs in the Qt main thread.  No Lumibot calls are made here —
    all data is pulled from DataBridge.
"""

import logging

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import DASHBOARD_REFRESH_MS
from dashboard.confidence_panel import ConfidencePanel
from dashboard.data_bridge import DataBridge
from dashboard.equity_bar import EquityBar
from dashboard.positions_table import PositionsTable
from dashboard.trade_log import TradeLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global dark-theme stylesheet applied to QApplication
# ---------------------------------------------------------------------------
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: "Segoe UI";
}
QSplitter::handle {
    background-color: #30363d;
}
QScrollBar:vertical {
    background: #161b22;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #484f58;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    height: 0;
}
"""


class MainWindow(QMainWindow):
    """Root application window for the AlgoTrader real-time dashboard.

    Args:
        data_bridge: Shared DataBridge instance (written by Lumibot thread,
                     read here in the Qt main thread).
        close_position_callback: Optional callable invoked when the user
            requests a manual position close via the positions table context
            menu.  Signature: (symbol: str) -> None.
        parent: Parent QWidget (None for a top-level window).
    """

    def __init__(
        self,
        data_bridge: DataBridge,
        close_position_callback=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bridge = data_bridge
        self._close_callback = close_position_callback

        self._configure_window()
        self._build_ui()
        self._start_timer()

        logger.info("Dashboard MainWindow initialised.")

    # ------------------------------------------------------------------
    # Window configuration
    # ------------------------------------------------------------------

    def _configure_window(self) -> None:
        """Set window title, size, and dark theme stylesheet."""
        self.setWindowTitle("AlgoTrader — DeepScalper × Alpaca Paper Trading")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        self.setStyleSheet(DARK_STYLESHEET)
        self.setFont(QFont("Segoe UI", 10))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the full widget hierarchy."""
        # Central widget is a simple container for the vertical layout
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ---- Top equity bar (full width) ----
        self._equity_bar = EquityBar(data_bridge=self._bridge, parent=self)
        root_layout.addWidget(self._equity_bar)

        # ---- Three-panel horizontal splitter ----
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(2)

        # Left panel: open positions table
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._positions_table = PositionsTable(
            data_bridge=self._bridge,
            close_position_callback=self._close_callback,
            parent=left_container,
        )
        left_layout.addWidget(self._positions_table)
        splitter.addWidget(left_container)

        # Centre panel: model confidence / signal panel
        self._confidence_panel = ConfidencePanel(
            data_bridge=self._bridge, parent=self
        )
        splitter.addWidget(self._confidence_panel)

        # Right panel: trade log
        self._trade_log = TradeLog(data_bridge=self._bridge, parent=self)
        splitter.addWidget(self._trade_log)

        # Set proportional widths: 50 % / 30 % / 20 %
        # Values are relative sizes, not pixels
        splitter.setSizes([700, 420, 280])

        root_layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        """Start the 1-second refresh QTimer."""
        self._timer = QTimer(self)
        self._timer.setInterval(DASHBOARD_REFRESH_MS)
        self._timer.timeout.connect(self._refresh_all)
        self._timer.start()
        logger.debug("Dashboard QTimer started (%d ms interval).", DASHBOARD_REFRESH_MS)

    # ------------------------------------------------------------------
    # Refresh slot (called by QTimer in Qt main thread)
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        """Atomically refresh all dashboard panels from the latest DataBridge state.

        This slot is the only place where Qt widgets are written.  It runs
        exclusively in the Qt main thread, so no locking beyond DataBridge's
        internal lock is required here.
        """
        try:
            self._equity_bar.refresh()
            self._positions_table.refresh()
            self._confidence_panel.refresh()
            self._trade_log.refresh()
        except Exception as exc:
            # Log but never crash the GUI loop
            logger.error("Dashboard refresh error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Window close event
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop the refresh timer cleanly when the window is closed.

        Args:
            event: QCloseEvent.
        """
        self._timer.stop()
        logger.info("Dashboard window closed — timer stopped.")
        super().closeEvent(event)
