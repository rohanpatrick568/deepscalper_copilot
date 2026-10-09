"""
main.py — AlgoTrader System Entry Point.

Starts the complete DeepScalper × Alpaca paper trading system:
  1. Validates environment (.env) and credentials.
        2. Verifies all configured equity weight files exist in ./weights/.
  3. Starts Lumibot's asynchronous pool from the main thread.
  4. Starts PyQt5 dashboard in the main thread (required by Qt).

Usage:
    python main.py

Shutdown:
    Close the dashboard window or press Ctrl+C.  Both methods trigger a graceful
    shutdown of the Lumibot thread before the process exits.
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Load .env before importing any project modules that read config
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# Configure logging early so all module-level loggers inherit this config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "algo_trader.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _validate_environment() -> None:
    """Validate credentials and weight files before starting any threads.

    Raises:
        SystemExit: On any validation failure.
    """
    from config import (
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        PAPER_ORDER_MODES,
        RUN_MODE,
        RUN_MODES,
        TRADING_UNIVERSE,
        WEIGHTS_DIR,
    )
    from execution.validation import validate_startup_configuration

    if RUN_MODE not in RUN_MODES:
        raise RuntimeError(f"Unsupported ALGO_TRADER_RUN_MODE: {RUN_MODE}")
    if RUN_MODE != "paper":
        raise RuntimeError(
            f"main.py runs model-driven paper mode only; use "
            f"'python -m workflow {RUN_MODE}' for {RUN_MODE!r}"
        )
    if RUN_MODE not in PAPER_ORDER_MODES:
        raise RuntimeError("Broker mutations were not explicitly enabled")
    try:
        validate_startup_configuration(TRADING_UNIVERSE, WEIGHTS_DIR)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Configuration/checkpoint validation failed: {exc}") from exc

    # 1. Credentials check
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
        )

    # 2. Live API reachability check
    try:
        from alpaca.trading.client import TradingClient

        tc = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        account = tc.get_account()
        logger.info(
            "Alpaca paper account verified — equity: $%s  buying power: $%s",
            account.equity,
            account.buying_power,
        )
    except Exception as exc:
        raise RuntimeError(f"Alpaca paper verification failed: {exc}") from exc

    logger.info("All %d symbol checkpoint(s) verified.", len(TRADING_UNIVERSE))


class EngineController:
    """Thread-safe dashboard facade for reconciled strategy controls."""

    def __init__(self, bridge) -> None:
        self.bridge = bridge
        self.strategy = None
        self.failure: Exception | None = None
        self.shutdown_confirmed: bool | None = None
        self._lock = threading.RLock()
        self.trader = None

    def attach(self, strategy) -> None:
        with self._lock:
            self.strategy = strategy

    def fail(self, exc: Exception) -> None:
        with self._lock:
            self.failure = exc
            self.bridge.engine_error = f"{type(exc).__name__}: {exc}"

    def close_position(self, symbol: str) -> bool:
        with self._lock:
            if self.strategy is None:
                self.bridge.engine_error = "Execution engine is not ready"
                return False
            return self.strategy.request_close_position(symbol)

    def emergency(self, timeout_seconds: float = 30.0) -> bool:
        with self._lock:
            if self.strategy is None:
                self.bridge.engine_error = "Execution engine is not ready"
                return False
            confirmed = self.strategy.graceful_shutdown(timeout_seconds)
            self.strategy.stop_live_engine()
            self.shutdown_confirmed = confirmed
            return confirmed

    def shutdown(self, timeout_seconds: float = 30.0) -> bool:
        with self._lock:
            if self.strategy is None:
                self.shutdown_confirmed = False
                return False
            confirmed = self.strategy.graceful_shutdown(timeout_seconds)
            self.strategy.stop_live_engine()
            self.shutdown_confirmed = confirmed
            return confirmed


def _start_lumibot(bridge, controller: EngineController) -> None:
    """Start Lumibot from the process main thread.

    ``Trader.run_all`` installs SIGINT handlers, so it must not be called from
    a worker.  Its asynchronous pool keeps the Qt event loop responsive while
    retaining Lumibot's real strategy executor and shutdown lifecycle.
    """
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
    from execution.broker import get_broker
    from execution.strategy import EquityDeepScalper
    from lumibot.traders import Trader

    broker = get_broker()
    strategy = EquityDeepScalper(
        broker=broker,
        data_bridge=bridge,
        alpaca_api_key=ALPACA_API_KEY,
        alpaca_secret_key=ALPACA_SECRET_KEY,
    )
    trader = Trader()
    trader.add_strategy(strategy)
    strategy._trader = trader
    controller.attach(strategy)
    controller.trader = trader
    logger.info("Starting Lumibot trading engine from the main thread…")
    trader.run_all(
        async_=True,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
        show_indicators=False,
    )


def _run_dashboard(bridge, controller: EngineController) -> int:
    """Start the PyQt5 dashboard in the main thread.

    Args:
        bridge: Shared DataBridge instance to read state from.
    """
    from PyQt5.QtWidgets import QApplication
    from dashboard.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow(
        data_bridge=bridge,
        close_position_callback=controller.close_position,
        emergency_callback=controller.emergency,
        shutdown_callback=controller.shutdown,
    )
    window.show()
    logger.info("Dashboard window opened.")
    exit_code = app.exec_()
    logger.info("Dashboard closed (exit code %d).", exit_code)
    return exit_code


def run_training_setup() -> int:
    """Open the Local/Colab training chooser without a model or credentials.

    This path deliberately skips environment validation so a new user can
    train a first model before any approved weights or broker keys exist.
    """
    from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

    from dashboard.training_choice import TrainingChoice

    logger.info("Opening the training setup window (no model or broker required).")
    app = QApplication.instance() or QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("DeepScalper — Training setup")
    container = QWidget(window)
    layout = QVBoxLayout(container)
    layout.addWidget(
        QLabel(
            "Choose where to train. Local runs on this machine's CPU; Google "
            "Colab opens the launcher notebook you authorize yourself.\n"
            "Neither option needs Alpaca credentials or an approved model."
        )
    )
    layout.addWidget(TrainingChoice(parent=container))
    layout.addStretch(1)
    window.setCentralWidget(container)
    window.resize(1100, 180)
    window.show()
    return app.exec_()


def main(argv: list[str] | None = None) -> int:
    """Main entry point — validates, starts threads, runs Qt event loop."""
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__)
    parser.add_argument(
        "--training-setup",
        action="store_true",
        help="open the Local/Colab training chooser without a model or credentials",
    )
    args = parser.parse_args(argv)
    if args.training_setup:
        return run_training_setup()

    logger.info("AlgoTrader starting up…")

    # Validate before doing anything else
    try:
        _validate_environment()
    except RuntimeError as exc:
        logger.critical("%s", exc)
        return 1

    # Instantiate the shared DataBridge (single source of truth for UI)
    from dashboard.data_bridge import DataBridge
    from config import STARTING_CAPITAL

    bridge = DataBridge()
    bridge.portfolio_value = STARTING_CAPITAL
    logger.info("DataBridge initialised with starting capital $%.2f.", STARTING_CAPITAL)

    # Warm up PyTorch on the main thread so the Lumibot worker can reuse the
    # already-loaded DLLs instead of initializing them inside the thread.
    import torch  # noqa: F401

    # Lumibot must start in the main thread because Trader.run_all registers
    # SIGINT. Its asynchronous executor then runs alongside Qt.
    controller = EngineController(bridge)
    try:
        _start_lumibot(bridge, controller)
    except Exception as exc:
        controller.fail(exc)
        logger.exception("Lumibot startup failed:")
        return 1

    # Give Lumibot a moment to connect before the dashboard appears
    time.sleep(2)

    headless_mode = os.getenv("ALGO_TRADER_HEADLESS", "0").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

    # Qt event loop runs in main thread (required by PyQt5).
    # In headless mode we keep the process alive while Lumibot runs.
    try:
        if headless_mode:
            logger.info("Headless mode enabled (ALGO_TRADER_HEADLESS=1); dashboard is disabled.")
            while controller.trader and any(
                getattr(worker, "is_alive", lambda: False)()
                for worker in getattr(controller.trader, "_pool", [])
            ):
                time.sleep(1)
            exit_code = 0
        else:
            exit_code = _run_dashboard(bridge, controller)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down.")
        exit_code = 0 if controller.shutdown() else 2

    if controller.shutdown_confirmed is None:
        if not controller.shutdown():
            exit_code = max(exit_code, 2)
    workers = getattr(controller.trader, "_pool", []) if controller.trader else []
    if any(getattr(worker, "is_alive", lambda: False)() for worker in workers):
        logger.error("Lumibot engine did not stop within the bounded shutdown timeout")
        exit_code = max(exit_code, 3)
    if controller.failure is not None:
        exit_code = max(exit_code, 1)

    logger.info("AlgoTrader shutdown complete (exit code %d).", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
