"""
main.py — AlgoTrader System Entry Point.

Starts the complete DeepScalper × Alpaca paper trading system:
  1. Validates environment (.env) and credentials.
  2. Verifies all 100 weight files exist in ./weights/.
  3. Starts Lumibot trading engine in a background daemon thread.
  4. Starts PyQt5 dashboard in the main thread (required by Qt).

Usage:
    python main.py

Shutdown:
    Close the dashboard window or press Ctrl+C.  Both methods trigger a graceful
    shutdown of the Lumibot thread before the process exits.
"""

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
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, SP100_TICKERS, WEIGHTS_DIR

    # 1. Credentials check
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.critical(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env — aborting."
        )
        sys.exit(1)

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
        logger.critical("Alpaca credential verification failed: %s — aborting.", exc)
        sys.exit(1)

    # 3. Weight files check
    missing = [
        ticker
        for ticker in SP100_TICKERS
        if not (WEIGHTS_DIR / f"{ticker}.pth").exists()
    ]
    if missing:
        logger.critical(
            "%d weight file(s) missing from %s:\n  %s\n\n"
            "Run the Colab training pipeline first (notebooks 01→04).",
            len(missing),
            WEIGHTS_DIR,
            ", ".join(missing),
        )
        sys.exit(1)

    logger.info("All %d weight files verified ✓", len(SP100_TICKERS))


def _run_lumibot(bridge) -> None:
    """Target function for the Lumibot daemon thread.

    Args:
        bridge: Shared DataBridge instance passed to the strategy.
    """
    try:
        from execution.broker import get_broker
        from execution.strategy import MultiStockDeepScalper

        broker = get_broker()
        strategy = MultiStockDeepScalper(
            broker=broker,
            data_bridge=bridge,
        )
        logger.info("Starting Lumibot trading engine…")
        strategy.run_all()
    except Exception:
        logger.exception("Lumibot thread encountered an unhandled exception:")


def _run_dashboard(bridge) -> None:
    """Start the PyQt5 dashboard in the main thread.

    Args:
        bridge: Shared DataBridge instance to read state from.
    """
    from PyQt5.QtWidgets import QApplication
    from dashboard.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow(bridge=bridge)
    window.show()
    logger.info("Dashboard window opened.")
    exit_code = app.exec_()
    logger.info("Dashboard closed (exit code %d).", exit_code)
    return exit_code


def main() -> None:
    """Main entry point — validates, starts threads, runs Qt event loop."""
    logger.info("AlgoTrader starting up…")

    # Validate before doing anything else
    _validate_environment()

    # Instantiate the shared DataBridge (single source of truth for UI)
    from dashboard.data_bridge import DataBridge
    from config import STARTING_CAPITAL

    bridge = DataBridge()
    bridge.portfolio_value = STARTING_CAPITAL
    logger.info("DataBridge initialised with starting capital $%.2f.", STARTING_CAPITAL)

    # Start Lumibot in a daemon background thread
    lumibot_thread = threading.Thread(
        target=_run_lumibot,
        args=(bridge,),
        name="LumibotEngine",
        daemon=True,   # Dies automatically when main thread exits
    )
    lumibot_thread.start()
    logger.info("Lumibot engine thread started (daemon=True).")

    # Give Lumibot a moment to connect before the dashboard appears
    time.sleep(2)

    # Qt event loop runs in main thread (required by PyQt5)
    try:
        exit_code = _run_dashboard(bridge)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down.")
        exit_code = 0

    logger.info("AlgoTrader shutdown complete.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
