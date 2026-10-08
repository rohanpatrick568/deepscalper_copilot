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
import threading
from typing import Any

from lumibot.brokers import Alpaca

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_DATA_URL,
    DATA_FEED,
    DATA_ADJUSTMENT,
    NO_ORDER_MODES,
    PAPER_ONLY,
    PAPER_ORDER_MODES,
    RUN_MODE,
    RUN_MODES,
)

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
    "DATA_FEED": DATA_FEED,
    "ADJUSTMENT": DATA_ADJUSTMENT,
    "DATA_URL": ALPACA_DATA_URL,
}


class BrokerMutationBlocked(RuntimeError):
    """Raised when a no-order run mode reaches a broker mutation."""


class _ClientOrderIdProxy:
    def __init__(self, client: Any, client_order_id: str) -> None:
        self._client = client
        self._client_order_id = client_order_id

    def submit_order(self, order_data):
        order_data.client_order_id = self._client_order_id
        return self._client.submit_order(order_data)

    def __getattr__(self, name: str):
        return getattr(self._client, name)


class GuardedAlpaca(Alpaca):
    """Alpaca adapter that enforces run mode at every mutation boundary."""

    def __init__(self, config, *, run_mode: str, **kwargs) -> None:
        if run_mode not in RUN_MODES:
            raise ValueError(f"Unsupported run mode: {run_mode}")
        self.run_mode = run_mode
        self._submission_lock = threading.Lock()
        super().__init__(config, **kwargs)

    def _require_mutations(self, operation: str) -> None:
        if self.run_mode in NO_ORDER_MODES:
            raise BrokerMutationBlocked(
                f"Broker mutation {operation!r} is disabled in {self.run_mode!r} mode"
            )
        if self.run_mode not in PAPER_ORDER_MODES or not self.is_paper:
            raise BrokerMutationBlocked("Only explicitly enabled Alpaca paper mutations are allowed")

    def _submit_order(self, order):
        self._require_mutations("submit_order")
        client_order_id = str(
            getattr(order, "custom_params", {}).get("client_order_id", "")
        ).strip()
        if not client_order_id:
            raise ValueError("Every paper order requires a stable client_order_id")
        with self._submission_lock:
            original_api = self.api
            self.api = _ClientOrderIdProxy(original_api, client_order_id)
            try:
                return super()._submit_order(order)
            finally:
                self.api = original_api

    def cancel_order(self, order):
        self._require_mutations("cancel_order")
        return super().cancel_order(order)

    def _modify_order(self, order, limit_price=None, stop_price=None):
        self._require_mutations("modify_order")
        return super()._modify_order(order, limit_price, stop_price)

    def cancel_orders(self, orders):
        self._require_mutations("cancel_orders")
        return super().cancel_orders(orders)

    def cancel_open_orders(self, strategy):
        self._require_mutations("cancel_open_orders")
        return super().cancel_open_orders(strategy)

    def sell_all(self, *args, **kwargs):
        self._require_mutations("sell_all")
        return super().sell_all(*args, **kwargs)


def get_broker(run_mode: str = RUN_MODE) -> Alpaca:
    """Create and return a Lumibot Alpaca broker configured for paper trading.

    Reads API credentials from config.py (sourced from the .env file).
    Raises a RuntimeError if credentials are missing.

    Returns:
        Alpaca: A Lumibot Alpaca broker instance ready for paper trading.

    Raises:
        RuntimeError: If ALPACA_API_KEY or ALPACA_SECRET_KEY are empty strings.
    """
    if run_mode not in RUN_MODES:
        raise ValueError(f"Unsupported run mode: {run_mode}")
    if not PAPER_ONLY:
        raise RuntimeError("Refusing to create a non-paper broker")
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError(
            "Alpaca API credentials are missing. "
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
        )

    logger.info(
        "Creating Alpaca paper trading broker (key: %s...)",
        ALPACA_API_KEY[:6] if len(ALPACA_API_KEY) >= 6 else "***",
    )
    return GuardedAlpaca(ALPACA_CONFIG, run_mode=run_mode)
