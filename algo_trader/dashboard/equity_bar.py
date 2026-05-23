"""
dashboard/equity_bar.py — Top-of-window Portfolio Equity Summary Bar.

A QWidget designed to sit at the top of the main window and display
key portfolio metrics that refresh every second via the QTimer in MainWindow.

Displayed metrics:
    Portfolio value | Daily P&L (%) | Drawdown | Open positions | Status | ET time
"""

from datetime import datetime

import pytz
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget

from config import MARKET_TIMEZONE
from dashboard.data_bridge import DataBridge

_ET = pytz.timezone(MARKET_TIMEZONE)


# ---------------------------------------------------------------------------
# Palette (matches main_window.py dark theme)
# ---------------------------------------------------------------------------
_BG = "#0d1117"
_FG = "#e6edf3"
_TEAL = "#00d4aa"
_RED = "#ff4757"
_GREY = "#8b949e"
_PANEL_BG = "#161b22"


class _MetricLabel(QWidget):
    """Compact labelled metric widget used inside the equity bar.

    Args:
        label: Short heading text (e.g. "Portfolio").
        parent: Parent QWidget.
    """

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        heading_font = QFont("Segoe UI", 8)
        value_font = QFont("Consolas", 11)
        value_font.setBold(True)

        self._heading = QLabel(f"{label}:")
        self._heading.setFont(heading_font)
        self._heading.setStyleSheet(f"color: {_GREY};")

        self._value = QLabel("—")
        self._value.setFont(value_font)
        self._value.setStyleSheet(f"color: {_FG};")

        layout.addWidget(self._heading)
        layout.addWidget(self._value)

    def set_value(self, text: str, color: str = _FG) -> None:
        """Update the displayed value and optional colour.

        Args:
            text: New value text.
            color: CSS colour string (e.g. "#00d4aa").
        """
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color};")


class _StatusIndicator(QWidget):
    """Animated dot + text indicator showing TRADING / HALTED status.

    Args:
        parent: Parent QWidget.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        heading_font = QFont("Segoe UI", 8)
        self._heading = QLabel("Status:")
        self._heading.setFont(heading_font)
        self._heading.setStyleSheet(f"color: {_GREY};")

        value_font = QFont("Segoe UI", 11)
        value_font.setBold(True)
        self._dot = QLabel("●")
        self._dot.setFont(value_font)
        self._dot.setStyleSheet(f"color: {_TEAL};")

        self._text = QLabel("TRADING")
        self._text.setFont(value_font)
        self._text.setStyleSheet(f"color: {_TEAL};")

        layout.addWidget(self._heading)
        layout.addWidget(self._dot)
        layout.addWidget(self._text)

    def set_status(self, halted: bool, reason: str = "") -> None:
        """Update the status indicator.

        Args:
            halted: True if trading is currently suspended.
            reason: Short halt reason (used in tooltip).
        """
        if halted:
            self._dot.setStyleSheet(f"color: {_RED};")
            self._text.setStyleSheet(f"color: {_RED};")
            self._text.setText("HALTED")
            self.setToolTip(reason)
        else:
            self._dot.setStyleSheet(f"color: {_TEAL};")
            self._text.setStyleSheet(f"color: {_TEAL};")
            self._text.setText("TRADING")
            self.setToolTip("")


class EquityBar(QWidget):
    """Full-width portfolio summary bar displayed at the top of the main window.

    Refreshed by calling :meth:`refresh` from the dashboard's QTimer slot.

    Args:
        data_bridge: Shared DataBridge instance.
        parent: Parent QWidget.
    """

    def __init__(self, data_bridge: DataBridge, parent=None) -> None:
        super().__init__(parent)
        self._bridge = data_bridge
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the widget layout."""
        self.setStyleSheet(f"background-color: {_PANEL_BG}; border-bottom: 1px solid #30363d;")
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(20)

        self._portfolio = _MetricLabel("Portfolio")
        self._daily_pnl = _MetricLabel("Daily P&L")
        self._drawdown = _MetricLabel("Drawdown")
        self._open_pos = _MetricLabel("Positions")
        self._status = _StatusIndicator()
        self._clock = _MetricLabel("Time (ET)")

        for widget in (
            self._portfolio,
            self._daily_pnl,
            self._drawdown,
            self._open_pos,
            self._status,
            self._clock,
        ):
            layout.addWidget(widget)

        layout.addStretch()

    def refresh(self) -> None:
        """Pull the latest metrics from DataBridge and update all labels.

        Should be called once per QTimer tick (every DASHBOARD_REFRESH_MS ms).
        """
        summary = self._bridge.get_portfolio_summary()
        value = summary["portfolio_value"]
        daily = summary["daily_pnl"]
        dd = summary["max_drawdown"]
        open_count = summary["open_positions_count"]
        halted = summary["is_halted"]
        halt_reason = summary["halt_reason"]

        # Portfolio value
        self._portfolio.set_value(f"${value:,.2f}")

        # Daily P&L with colour coding
        if value > 0:
            pct = (daily / value) * 100
        else:
            pct = 0.0
        sign = "+" if daily >= 0 else ""
        daily_color = _TEAL if daily >= 0 else _RED
        self._daily_pnl.set_value(f"{sign}${daily:,.2f} ({sign}{pct:.2f}%)", daily_color)

        # Max drawdown
        dd_color = _RED if dd < -0.01 else _GREY
        self._drawdown.set_value(f"{dd * 100:.2f}%", dd_color)

        # Open positions count
        self._open_pos.set_value(str(open_count))

        # Status indicator
        self._status.set_status(halted, halt_reason)

        # Clock (Eastern Time)
        now_et = datetime.now(_ET).strftime("%H:%M:%S")
        self._clock.set_value(now_et)
