"""
dashboard/positions_table.py — Open Positions QTableWidget.

Displays all current open positions with real-time P&L colouring, ATR stop
and take-profit levels, and an entry-duration counter.  Rows are sorted by
absolute unrealised P&L descending (biggest movers at top).

Columns:
    Symbol | Side | Qty | Entry Price | Current Price | Unrealised P&L |
    ATR Stop | ATR TP | Duration

Features:
    • Green / red colouring for P&L cells
    • 300 ms background flash on P&L changes > $0.10
    • Alternating row backgrounds
    • Right-click "Close Position" context menu
"""

import time
from typing import Callable, Dict, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QHeaderView,
    QLabel,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dashboard.data_bridge import DataBridge, PositionSnapshot

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
_BG_ODD = "#161b22"
_BG_EVEN = "#1c2128"
_BG_HOVER = "#2d333b"
_FG = "#e6edf3"
_TEAL = "#00d4aa"
_RED = "#ff4757"
_ORANGE = "#ffa502"
_GREY = "#8b949e"
_FLASH_COLOR = "#2d4a3e"     # Green tinge for positive flash
_FLASH_RED_COLOR = "#4a1e2e"  # Red tinge for negative flash
_PANEL_BG = "#161b22"

_COLS = [
    "Symbol", "Side", "Qty", "Entry Price",
    "Current Price", "Unrealised P&L", "ATR Stop", "ATR TP", "Duration",
]

_MONO_FONT = QFont("Consolas", 10)
_SANS_FONT = QFont("Segoe UI", 10)
_BOLD_FONT = QFont("Consolas", 10)
_BOLD_FONT.setBold(True)


def _make_item(text: str, align=Qt.AlignRight | Qt.AlignVCenter) -> QTableWidgetItem:
    """Create a non-editable, right-aligned table cell item."""
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    item.setTextAlignment(align)
    item.setFont(_MONO_FONT)
    return item


class PositionsTable(QWidget):
    """Open positions table widget.

    Args:
        data_bridge: Shared DataBridge instance.
        close_position_callback: Callable invoked with a ticker symbol when the
            user selects "Close Position" from the context menu.  The strategy
            layer is responsible for routing the close order.
        parent: Parent QWidget.
    """

    def __init__(
        self,
        data_bridge: DataBridge,
        close_position_callback: Optional[Callable[[str], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bridge = data_bridge
        self._close_callback = close_position_callback
        self._prev_pnl: Dict[str, float] = {}   # Previous P&L for flash detection
        self._flash_timers: Dict[str, QTimer] = {}
        self._entry_times: Dict[str, float] = {}  # symbol → UNIX timestamp of first appearance

        self._build_ui()

    def _build_ui(self) -> None:
        """Set up the table widget and title label."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("  Open Positions")
        title.setFont(QFont("Segoe UI", 9))
        title.setFixedHeight(24)
        title.setStyleSheet(f"color: {_GREY}; background: {_PANEL_BG}; padding-left: 8px;")
        layout.addWidget(title)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(False)  # We handle this manually
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setFont(QFont("Segoe UI", 9))
        header.setStyleSheet(f"color: {_GREY}; background: {_BG_ODD};")

        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {_BG_ODD};
                color: {_FG};
                border: none;
                gridline-color: #30363d;
            }}
            QTableWidget::item:selected {{
                background-color: {_BG_HOVER};
            }}
            QHeaderView::section {{
                background-color: {_BG_ODD};
                color: {_GREY};
                border: none;
                padding: 4px;
                font-size: 9px;
            }}
        """)

        layout.addWidget(self._table)

    def refresh(self) -> None:
        """Re-populate the table from the latest DataBridge positions snapshot.

        Called every 1 second by the MainWindow QTimer.
        """
        positions: Dict[str, PositionSnapshot] = self._bridge.get_all_positions()

        # Track first-seen time for duration calculation
        for sym in positions:
            if sym not in self._entry_times:
                self._entry_times[sym] = time.time()
        # Remove departed symbols
        for sym in list(self._entry_times.keys()):
            if sym not in positions:
                del self._entry_times[sym]

        # Sort by |unrealised_pnl| descending
        sorted_syms = sorted(
            positions.keys(),
            key=lambda s: abs(positions[s].unrealized_pnl),
            reverse=True,
        )

        self._table.setRowCount(len(sorted_syms))

        for row, sym in enumerate(sorted_syms):
            snap = positions[sym]
            row_bg = _BG_ODD if row % 2 == 0 else _BG_EVEN

            # Col 0: Symbol (bold, left-aligned)
            sym_item = _make_item(sym, Qt.AlignLeft | Qt.AlignVCenter)
            sym_item.setFont(_BOLD_FONT)
            sym_item.setForeground(QBrush(QColor(_FG)))
            self._set_cell(row, 0, sym_item, row_bg)

            # Col 1: Side
            side_color = _TEAL if snap.side == "LONG" else _ORANGE
            side_item = _make_item(snap.side, Qt.AlignCenter | Qt.AlignVCenter)
            side_item.setForeground(QBrush(QColor(side_color)))
            self._set_cell(row, 1, side_item, row_bg)

            # Col 2: Qty
            self._set_cell(row, 2, _make_item(str(snap.qty)), row_bg)

            # Col 3: Entry Price
            self._set_cell(row, 3, _make_item(f"${snap.entry_price:.2f}"), row_bg)

            # Col 4: Current Price
            cur_item = _make_item(f"${snap.current_price:.2f}")
            price_color = _TEAL if snap.current_price >= snap.entry_price else _RED
            cur_item.setForeground(QBrush(QColor(price_color)))
            self._set_cell(row, 4, cur_item, row_bg)

            # Col 5: Unrealised P&L
            upnl = snap.unrealized_pnl
            pnl_color = _TEAL if upnl >= 0 else _RED
            sign = "+" if upnl >= 0 else ""
            if snap.entry_price > 0 and snap.qty > 0:
                pnl_pct = upnl / (snap.entry_price * snap.qty) * 100
                pnl_text = f"{sign}${upnl:.2f} ({sign}{pnl_pct:.2f}%)"
            else:
                pnl_text = f"{sign}${upnl:.2f}"

            pnl_item = _make_item(pnl_text)
            pnl_item.setForeground(QBrush(QColor(pnl_color)))
            self._set_cell(row, 5, pnl_item, row_bg)

            # Flash on significant P&L change
            prev = self._prev_pnl.get(sym, upnl)
            if abs(upnl - prev) >= 0.10:
                self._flash_row(row, upnl >= prev)
            self._prev_pnl[sym] = upnl

            # Col 6: ATR Stop (muted grey)
            stop_item = _make_item(f"${snap.atr_stop:.2f}")
            stop_item.setForeground(QBrush(QColor(_GREY)))
            self._set_cell(row, 6, stop_item, row_bg)

            # Col 7: ATR TP (muted teal)
            tp_item = _make_item(f"${snap.atr_tp:.2f}")
            tp_item.setForeground(QBrush(QColor("#00a88c")))  # muted teal
            self._set_cell(row, 7, tp_item, row_bg)

            # Col 8: Duration
            elapsed_s = int(time.time() - self._entry_times.get(sym, time.time()))
            mins, secs = divmod(elapsed_s, 60)
            dur_item = _make_item(f"{mins}m {secs:02d}s", Qt.AlignCenter | Qt.AlignVCenter)
            dur_item.setForeground(QBrush(QColor(_GREY)))
            self._set_cell(row, 8, dur_item, row_bg)

    def _set_cell(self, row: int, col: int, item: QTableWidgetItem, bg_color: str) -> None:
        """Set a cell item and apply the row background colour."""
        item.setBackground(QBrush(QColor(bg_color)))
        self._table.setItem(row, col, item)

    def _flash_row(self, row: int, positive: bool) -> None:
        """Briefly tint the P&L cell to indicate a significant change.

        Args:
            row: Row index to flash.
            positive: True for a positive change (green tint), False for red.
        """
        flash_color = _FLASH_COLOR if positive else _FLASH_RED_COLOR
        pnl_item = self._table.item(row, 5)
        if pnl_item:
            pnl_item.setBackground(QBrush(QColor(flash_color)))

        # Cancel any existing flash timer for this row
        row_key = f"flash_{row}"
        if row_key in self._flash_timers:
            self._flash_timers[row_key].stop()

        timer = QTimer(self)
        timer.setSingleShot(True)

        def _restore_bg():
            item = self._table.item(row, 5)
            if item:
                bg = _BG_ODD if row % 2 == 0 else _BG_EVEN
                item.setBackground(QBrush(QColor(bg)))

        timer.timeout.connect(_restore_bg)
        timer.start(300)
        self._flash_timers[row_key] = timer

    def _show_context_menu(self, pos) -> None:
        """Display a right-click context menu with a 'Close Position' option.

        Args:
            pos: Cursor position in widget coordinates.
        """
        index = self._table.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        sym_item = self._table.item(row, 0)
        if not sym_item:
            return
        symbol = sym_item.text()

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: #161b22;
                color: {_FG};
                border: 1px solid #30363d;
            }}
            QMenu::item:selected {{ background-color: #2d333b; }}
        """)

        close_action = QAction(f"Close Position: {symbol}", self)
        close_action.triggered.connect(lambda: self._request_close(symbol))
        menu.addAction(close_action)
        menu.exec_(self._table.viewport().mapToGlobal(pos))

    def _request_close(self, symbol: str) -> None:
        """Invoke the close-position callback with the selected symbol.

        Args:
            symbol: Stock ticker to close.
        """
        if self._close_callback:
            self._close_callback(symbol)
