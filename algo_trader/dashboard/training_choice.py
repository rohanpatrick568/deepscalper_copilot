"""Safe dashboard selector for the shared local/Colab training entry point."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QWidget


class TrainingChoice(QWidget):
    """Display copyable launch guidance without starting compute from the GUI."""

    choiceChanged = pyqtSignal(str)

    COMMANDS = {
        "Local": (
            "python -m colab.deepscalper.training train --data DATA.npz "
            "--output-dir runs/AAPL --symbol AAPL --location local"
        ),
        "Google Colab": (
            "python -m colab.deepscalper.training train --data /content/drive/DATA.npz "
            "--output-dir /content/drive/MyDrive/deepscalper/AAPL --symbol AAPL "
            "--location colab"
        ),
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.addWidget(QLabel("Training:"))
        self.selector = QComboBox(self)
        self.selector.addItems(self.COMMANDS)
        self.command = QLineEdit(self)
        self.command.setReadOnly(True)
        self.command.setToolTip("Copy this command into the selected runtime.")
        layout.addWidget(self.selector)
        layout.addWidget(self.command, stretch=1)
        self.selector.currentTextChanged.connect(self._set_choice)
        self._set_choice(self.selector.currentText())

    def _set_choice(self, choice: str) -> None:
        self.command.setText(self.COMMANDS[choice])
        self.choiceChanged.emit(choice)
