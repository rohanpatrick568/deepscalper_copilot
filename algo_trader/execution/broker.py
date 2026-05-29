"""
execution/broker.py — Alpaca Paper Trading Broker Configuration.

Constructs and returns a Lumibot-compatible Alpaca broker instance configured
for paper trading.  The API credentials are always sourced from config.py
(which reads them from the .env file) — they are never hard-coded here.

Usage:
    from execution.broker import get_broker
    broker = get_broker()
"""

import logging

from lumibot.brokers import Alpaca

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alpaca broker configuration dictionary expected by Lumibot.
# PAPER: True is enforced here and must never be set to False in this file.
# ---------------------------------------------------------------------------
ALPACA_CONFIG: dict = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_SECRET_KEY,
    "MARKET": "NYSE",
    "PAPER": True,   # Always True — this system is paper-trading only
}


def get_broker() -> Alpaca:
    """Create and return a Lumibot Alpaca broker configured for paper trading.

    Reads API credentials from config.py (sourced from the .env file).
    Raises a RuntimeError if credentials are missing.

    Returns:
        Alpaca: A Lumibot Alpaca broker instance ready for paper trading.

    Raises:
        RuntimeError: If ALPACA_API_KEY or ALPACA_SECRET_KEY are empty strings.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError(
            "Alpaca API credentials are missing. "
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
        )

    logger.info(
        "Creating Alpaca paper trading broker (key: %s...)",
        ALPACA_API_KEY[:6] if len(ALPACA_API_KEY) >= 6 else "***",
    )
    return Alpaca(ALPACA_CONFIG)
