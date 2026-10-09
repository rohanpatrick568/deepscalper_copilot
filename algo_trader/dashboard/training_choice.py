"""Dashboard selector that launches the shared local/Colab training entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import QProcess, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from workflow import COLAB_NOTEBOOK_URL, training_launch_plan

LOCAL = "Local"
COLAB = "Google Colab"


class TrainingChoice(QWidget):
    """Expose the user's Local/Colab choice and start the selected route.

    Local training runs the shared training CLI as a child process so the GUI
    stays responsive and the run can be cancelled. Colab compute is started by
    the user inside their own authorized session: this widget only opens the
    launcher notebook, it never claims to start remote compute.
    """

    choiceChanged = pyqtSignal(str)
    started = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, parent=None, *, project_root: Path | None = None) -> None:
        super().__init__(parent)
        self._root = Path(project_root or Path(__file__).resolve().parents[1])
        self._process: QProcess | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.addWidget(QLabel("Training:"))
        self.selector = QComboBox(self)
        self.selector.addItems([LOCAL, COLAB])
        self.command = QLineEdit(self)
        self.command.setReadOnly(True)
        self.command.setToolTip("The exact command this choice runs.")
        self.data_path = QLineEdit(self)
        self.data_path.setPlaceholderText("path to SYMBOL_features.npz")
        self.action = QPushButton(self)
        self.status = QLabel("", self)

        layout.addWidget(self.selector)
        layout.addWidget(self.data_path, stretch=1)
        layout.addWidget(self.command, stretch=2)
        layout.addWidget(self.action)
        layout.addWidget(self.status)

        self.selector.currentTextChanged.connect(self._set_choice)
        self.data_path.textChanged.connect(lambda _: self._refresh_command())
        self.action.clicked.connect(self.activate)
        self._set_choice(self.selector.currentText())

    @property
    def choice(self) -> str:
        return self.selector.currentText()

    def plan(self) -> dict:
        text = self.data_path.text().strip()
        return training_launch_plan(
            "local" if self.choice == LOCAL else "colab",
            data=Path(text) if text else None,
            output_dir=self._root / "runs" / "AAPL",
            device="cpu",
            resume=(self._root / "runs" / "AAPL" / "latest.pth").exists(),
        )

    def _set_choice(self, choice: str) -> None:
        self.data_path.setEnabled(choice == LOCAL)
        self._refresh_command()
        self.choiceChanged.emit(choice)

    def _refresh_command(self) -> None:
        plan = self.plan()
        self.command.setText(plan["command"])
        running = self._process is not None
        if self.choice == COLAB:
            self.action.setText("Open Colab launcher")
        else:
            self.action.setText("Cancel training" if running else "Start local training")

    def activate(self) -> None:
        if self.choice == COLAB:
            QDesktopServices.openUrl(QUrl(COLAB_NOTEBOOK_URL))
            self.status.setText("Opened the Colab launcher notebook.")
            return
        if self._process is not None:
            self.cancel()
            return
        self.start_local()

    def start_local(self) -> None:
        data = self.data_path.text().strip()
        if not data or not Path(data).is_file():
            self.status.setText("Select an existing feature file first.")
            return
        plan = self.plan()
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", plan["module"], *plan["argv"]])
        process.setWorkingDirectory(str(self._root))
        process.finished.connect(self._on_finished)
        self._process = process
        process.start()
        self.status.setText("Training locally on CPU...")
        self._refresh_command()
        self.started.emit(plan["command"])

    def cancel(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        if not self._process.waitForFinished(5000):
            self._process.kill()
        self.status.setText("Training cancelled; the last checkpoint is kept.")

    def _on_finished(self, exit_code: int, _status=None) -> None:
        self._process = None
        if exit_code == 0:
            self.status.setText("Training finished and the model was accepted.")
        elif exit_code == 2:
            self.status.setText("Training finished but the model was REJECTED.")
        else:
            self.status.setText(f"Training stopped with status {exit_code}.")
        self._refresh_command()
        self.finished.emit(exit_code)
