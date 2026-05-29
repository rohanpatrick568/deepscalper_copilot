"""
dashboard/trade_log.py — Scrolling Trade History Feed.

A read-only QTextEdit displaying timestamped trade events with colour-coded
formatting:
  • FILL   — green for buys, red for sells
  • HALT   — orange warning
  • EOD_CLOSE — yellow notification
  • Other  — grey info line

Automatically scrolls to the newest entry.  Retains at most
MAX_TRADE_LOG_ENTRIES lines (FIFO eviction).
"""

from PyQt5.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import MAX_TRADE_LOG_ENTRIES
from dashboard.data_bridge import DataBridge, TradeEvent

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
_BG = "#0d1117"
_PANEL_BG = "#161b22"
_FG = "#e6edf3"
_TEAL = "#00d4aa"
_RED = "#ff4757"
_ORANGE = "#ffa502"
_YELLOW = "#f7c948"
_GREY = "#8b949e"
_DIM = "#484f58"

_MONO = QFont("Consolas", 9)


def _format_event(event: TradeEvent) -> tuple[str, str]:
    """Convert a TradeEvent into a display line and CSS colour.

    Args:
        event: TradeEvent from DataBridge.

    Returns:
        Tuple of (line_text, css_hex_colour).
    """
    ts = event.timestamp
    sym = event.symbol
    qty = event.qty
    price = event.price
    side = event.side
    etype = event.event_type

    if etype == "FILL":
        if side.upper() in ("BUY", "BUY "):
            icon = "✅"
            color = _TEAL
            line = f"[{ts}] {icon} BOUGHT {qty} {sym} @ ${price:.2f}"
        else:
            icon = "🔴"
            color = _RED
            line = f"[{ts}] {icon} SOLD {qty} {sym} @ ${price:.2f}"

    elif etype == "HALT":
        line = f"[{ts}] ⚠️  CIRCUIT BREAKER: {side}"
        color = _ORANGE

    elif etype in ("EOD_CLOSE", "EOD"):
        line = f"[{ts}] 🔔 EOD CLOSE: All positions flattened"
        color = _YELLOW

    elif etype == "STOP_HIT":
        line = f"[{ts}] 🛑 STOP HIT: {sym} @ ${price:.2f}"
        color = _RED

    else:
        line = f"[{ts}] ℹ️  {etype}: {sym} — {side}"
        color = _GREY

    return line, color


class TradeLog(QWidget):
    """Scrolling, colour-coded trade history log widget.

    Args:
        data_bridge: Shared DataBridge instance.
        parent: Parent QWidget.
    """

    def __init__(self, data_bridge: DataBridge, parent=None) -> None:
        super().__init__(parent)
        self._bridge = data_bridge
        self._rendered_count: int = 0  # Track how many events have been rendered
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the text editor and clear button."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(24)
        title_bar.setStyleSheet(f"background: {_PANEL_BG};")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(8, 0, 4, 0)

        title_label = QLabel("Trade Log")
        title_label.setFont(QFont("Segoe UI", 9))
        title_label.setStyleSheet(f"color: {_GREY}; border: none;")

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedSize(52, 18)
        self._clear_btn.setFont(QFont("Segoe UI", 8))
        self._clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: #21262d;
                color: {_GREY};
                border: 1px solid #30363d;
                border-radius: 3px;
            }}
            QPushButton:hover {{ background: #2d333b; color: {_FG}; }}
        """)
        self._clear_btn.clicked.connect(self._clear_log)

        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self._clear_btn)
        layout.addWidget(title_bar)

        # Text editor (read-only)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_MONO)
        self._log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {_BG};
                color: {_FG};
                border: none;
                padding: 4px;
            }}
        """)
        self._log.document().setDefaultStyleSheet(
            f"body {{ color: {_FG}; font-family: Consolas; font-size: 9pt; }}"
        )
        layout.addWidget(self._log)

    def refresh(self) -> None:
        """Append any new trade events from DataBridge since the last refresh.

        Only appends the delta (new events) rather than re-rendering the full
        log each second, which avoids flicker and keeps scroll position stable.
        """
        all_events = self._bridge.get_trade_log()
        new_events = all_events[self._rendered_count:]

        if not new_events:
            return

        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.End)

        for event in new_events:
            line, color = _format_event(event)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            cursor.insertText(line + "\n", fmt)

        self._rendered_count = len(all_events)

        # Enforce maximum line count — trim oldest lines
        doc = self._log.document()
        if doc.blockCount() > MAX_TRADE_LOG_ENTRIES:
            trim_cursor = self._log.textCursor()
            trim_cursor.movePosition(QTextCursor.Start)
            excess = doc.blockCount() - MAX_TRADE_LOG_ENTRIES
            for _ in range(excess):
                trim_cursor.select(QTextCursor.LineUnderCursor)
                trim_cursor.removeSelectedText()
                trim_cursor.deleteChar()  # Remove the newline

        # Auto-scroll to bottom
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _clear_log(self) -> None:
        """Clear all displayed log lines (does not clear DataBridge history)."""
        self._log.clear()
        # Reset rendered count so we don't re-render old events from DataBridge
        self._rendered_count = len(self._bridge.get_trade_log())
