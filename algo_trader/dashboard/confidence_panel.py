"""
dashboard/confidence_panel.py — Per-Stock Model Signal Panel.

Displays a scrollable grid of cards, one per S&P 100 ticker.
Each card shows the model's latest action (BUY / SELL / HOLD),
Q-values, and confidence score.

Cards are:
  • Sorted by confidence descending (highest conviction at top).
  • Only shown when confidence > CONFIDENCE_THRESHOLD OR the ticker
    is currently held as an open position.
  • Colour-coded: BUY = teal, SELL = red, HOLD = grey.
"""

from typing import Dict, Set

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from config import CONFIDENCE_THRESHOLD
from dashboard.data_bridge import DataBridge, ModelSignal

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
_BG = "#0d1117"
_CARD_BG = "#161b22"
_CARD_BORDER = "#30363d"
_FG = "#e6edf3"
_TEAL = "#00d4aa"
_RED = "#ff4757"
_GREY = "#8b949e"
_PANEL_BG = "#161b22"


def _action_color(action: str) -> str:
    """Map action string to display colour.

    Args:
        action: "BUY", "SELL", or "HOLD".

    Returns:
        CSS hex colour string.
    """
    return {
        "BUY": _TEAL,
        "SELL": _RED,
        "HOLD": _GREY,
    }.get(action, _GREY)


def _action_arrow(action: str) -> str:
    """Map action to a directional arrow character."""
    return {"BUY": "↑", "SELL": "↓", "HOLD": "—"}.get(action, "—")


class _SignalCard(QFrame):
    """A compact display card for a single ticker's model signal.

    Args:
        symbol: Stock ticker symbol.
        parent: Parent widget.
    """

    def __init__(self, symbol: str, parent=None) -> None:
        super().__init__(parent)
        self._symbol = symbol
        self._build_ui()

    def _build_ui(self) -> None:
        """Set up card layout."""
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {_CARD_BG};
                border: 1px solid {_CARD_BORDER};
                border-radius: 6px;
            }}
        """)
        self.setFixedHeight(72)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(10)

        # --- Symbol block ---
        sym_label = QLabel(self._symbol)
        sym_font = QFont("Consolas", 10)
        sym_font.setBold(True)
        sym_label.setFont(sym_font)
        sym_label.setStyleSheet(f"color: {_FG}; border: none;")
        sym_label.setFixedWidth(54)
        outer.addWidget(sym_label)

        # --- Right block: action + Q-values + timestamp ---
        right = QVBoxLayout()
        right.setSpacing(2)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(4)

        self._action_label = QLabel("—")
        action_font = QFont("Segoe UI", 10)
        action_font.setBold(True)
        self._action_label.setFont(action_font)
        self._action_label.setStyleSheet(f"color: {_GREY}; border: none;")

        self._conf_label = QLabel("Conf: —")
        conf_font = QFont("Segoe UI", 9)
        self._conf_label.setFont(conf_font)
        self._conf_label.setStyleSheet(f"color: {_GREY}; border: none;")

        action_layout.addWidget(self._action_label)
        action_layout.addStretch()
        action_layout.addWidget(self._conf_label)

        self._q_label = QLabel("Q: H=— B=— S=—")
        q_font = QFont("Consolas", 8)
        self._q_label.setFont(q_font)
        self._q_label.setStyleSheet(f"color: {_GREY}; border: none;")

        self._ts_label = QLabel("—")
        ts_font = QFont("Segoe UI", 8)
        self._ts_label.setFont(ts_font)
        self._ts_label.setStyleSheet(f"color: {_GREY}; border: none;")

        right.addLayout(action_layout)
        right.addWidget(self._q_label)
        right.addWidget(self._ts_label)

        outer.addLayout(right)

    def update_signal(self, signal: ModelSignal, is_held: bool) -> None:
        """Refresh card with the latest model signal.

        Args:
            signal: ModelSignal from the DataBridge.
            is_held: True if this ticker is currently in an open position.
        """
        color = _action_color(signal.action)
        arrow = _action_arrow(signal.action)

        self._action_label.setText(f"{signal.action} {arrow}")
        self._action_label.setStyleSheet(f"color: {color}; border: none; font-weight: bold;")

        conf_pct = signal.confidence * 100
        self._conf_label.setText(f"Conf: {conf_pct:.1f}%")

        if len(signal.q_values) >= 3:
            self._q_label.setText(
                f"Q: H={signal.q_values[0]:.2f}  "
                f"B={signal.q_values[1]:.2f}  "
                f"S={signal.q_values[2]:.2f}"
            )

        ts_suffix = " [HELD]" if is_held else ""
        self._ts_label.setText(f"Updated: {signal.timestamp}{ts_suffix}")

        if is_held:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #1a2332;
                    border: 1px solid {_TEAL};
                    border-radius: 6px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {_CARD_BG};
                    border: 1px solid {_CARD_BORDER};
                    border-radius: 6px;
                }}
            """)


class ConfidencePanel(QWidget):
    """Scrollable panel of per-ticker model signal cards.

    Args:
        data_bridge: Shared DataBridge instance.
        parent: Parent QWidget.
    """

    def __init__(self, data_bridge: DataBridge, parent=None) -> None:
        super().__init__(parent)
        self._bridge = data_bridge
        self._cards: Dict[str, _SignalCard] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the scrollable card container."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        title = QLabel("  Model Signals")
        title.setFont(QFont("Segoe UI", 9))
        title.setFixedHeight(24)
        title.setStyleSheet(f"color: {_GREY}; background: {_PANEL_BG}; padding-left: 8px;")
        root_layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {_BG}; }}")

        self._container = QWidget()
        self._container.setStyleSheet(f"background: {_BG};")
        self._card_layout = QVBoxLayout(self._container)
        self._card_layout.setContentsMargins(6, 6, 6, 6)
        self._card_layout.setSpacing(4)
        self._card_layout.addStretch()

        scroll.setWidget(self._container)
        root_layout.addWidget(scroll)

    def refresh(self) -> None:
        """Update all visible cards from the latest DataBridge signals.

        Called every 1 second by MainWindow's QTimer.
        """
        signals: Dict[str, ModelSignal] = self._bridge.get_all_signals()
        positions: Set[str] = set(self._bridge.get_all_positions().keys())

        # Determine which tickers should be visible
        visible = {
            sym: sig
            for sym, sig in signals.items()
            if sig.confidence >= CONFIDENCE_THRESHOLD or sym in positions
        }

        # Sort by confidence descending
        sorted_syms = sorted(visible.keys(), key=lambda s: visible[s].confidence, reverse=True)

        # Remove cards that are no longer visible
        for sym in list(self._cards.keys()):
            if sym not in visible:
                card = self._cards.pop(sym)
                self._card_layout.removeWidget(card)
                card.deleteLater()

        # Insert / update cards in sorted order
        # Re-build card order by clearing and re-adding (simple but effective at 1Hz)
        # Remove the stretch first
        stretch_item = self._card_layout.itemAt(self._card_layout.count() - 1)

        for i, sym in enumerate(sorted_syms):
            if sym not in self._cards:
                card = _SignalCard(sym, self._container)
                self._cards[sym] = card

            card = self._cards[sym]
            sig = visible[sym]
            is_held = sym in positions
            card.update_signal(sig, is_held)

            # Ensure card is at the correct position in the layout
            current_index = self._card_layout.indexOf(card)
            if current_index != i:
                self._card_layout.removeWidget(card)
                self._card_layout.insertWidget(i, card)

            if not card.isVisible():
                card.show()
