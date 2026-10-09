"""Explicitly opted-in, bounded Alpaca paper round-trip smoke test."""

from __future__ import annotations

import time
import uuid
from decimal import Decimal
from typing import Any

from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from config import MAX_ORDER_NOTIONAL, TRADING_UNIVERSE
from execution.preflight import verify_paper_client

TERMINAL = {"filled", "canceled", "cancelled", "expired", "rejected"}


def _status(order: Any) -> str:
    return str(getattr(order, "status", "")).split(".")[-1].lower()


def _symbol(item: Any) -> str:
    return str(getattr(item, "symbol", getattr(getattr(item, "asset", None), "symbol", "")))


def run_paper_round_trip(
    client: Any,
    *,
    symbol: str,
    notional: Decimal,
    timeout_seconds: float,
    max_orders: int = 2,
    confirmed: bool = False,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    """Buy a bounded fractional notional and sell only the resulting test position."""
    if not confirmed:
        raise RuntimeError("Paper smoke requires explicit --confirm-paper-smoke")
    verify_paper_client(client)
    if symbol not in TRADING_UNIVERSE:
        raise ValueError(f"Smoke symbol must be in the configured universe: {TRADING_UNIVERSE}")
    if notional <= 0 or notional > Decimal(str(MAX_ORDER_NOTIONAL)):
        raise ValueError(f"Smoke notional must be in (0, {MAX_ORDER_NOTIONAL}]")
    if max_orders != 2:
        raise ValueError("The bounded round trip requires exactly two order slots")
    asset = client.get_asset(symbol)
    if not getattr(asset, "tradable", False) or not getattr(asset, "fractionable", False):
        raise RuntimeError(f"{symbol} must be tradable and fractionable for bounded smoke")

    existing_positions = [
        position for position in client.get_all_positions() if _symbol(position) == symbol
    ]
    existing_orders = [
        order
        for order in client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        if _symbol(order) == symbol
    ]
    if existing_positions or existing_orders:
        raise RuntimeError(
            f"Smoke requires clean {symbol} state; found "
            f"{len(existing_positions)} position(s), {len(existing_orders)} open order(s)"
        )

    clock = getattr(client, "get_clock", None)
    if not callable(clock):
        raise RuntimeError("Paper smoke requires a broker clock")
    if not bool(getattr(clock(), "is_open", False)):
        raise RuntimeError("Paper smoke requires an open, tradable session")

    prefix = f"ds-smoke-{symbol.lower()}-{uuid.uuid4().hex[:12]}"
    own_orders: list[Any] = []

    def lookup(order: Any) -> Any:
        client_id = getattr(order, "client_order_id", None)
        if client_id and hasattr(client, "get_order_by_client_id"):
            return client.get_order_by_client_id(client_id)
        return client.get_order_by_id(order.id)

    def wait_terminal(order: Any) -> Any:
        current = order
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            current = lookup(current)
            if _status(current) in TERMINAL:
                return current
            sleep_fn(0.5)
        client.cancel_order_by_id(current.id)
        cancel_deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < cancel_deadline:
            current = lookup(current)
            if _status(current) in TERMINAL:
                return current
            sleep_fn(0.5)
        raise TimeoutError(f"Paper smoke order {current.id} did not become terminal after cancellation")

    def submit(request: Any, client_id: str) -> Any:
        try:
            order = client.submit_order(request)
        except Exception:
            if not hasattr(client, "get_order_by_client_id"):
                raise
            order = client.get_order_by_client_id(client_id)
        if not getattr(order, "client_order_id", None):
            order.client_order_id = client_id
        return order

    failure: Exception | None = None
    try:
        buy_id = f"{prefix}-buy"
        buy = submit(
            MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id=buy_id,
            ),
            buy_id,
        )
        own_orders.append(buy)
        buy = wait_terminal(buy)
        if _status(buy) != "filled":
            raise RuntimeError(f"Paper smoke buy ended as {_status(buy)}")
    except Exception as exc:
        failure = exc

    positions = [
        position
        for position in client.get_all_positions()
        if _symbol(position) == symbol and Decimal(str(position.qty)) != 0
    ]
    if positions:
        if len(positions) != 1 or Decimal(str(positions[0].qty)) <= 0:
            raise RuntimeError("Smoke cleanup found an unexpected test-symbol position") from failure
        test_qty = Decimal(str(positions[0].qty))
        if len(own_orders) >= max_orders:
            raise RuntimeError("Smoke exhausted its bounded order count before flattening") from failure
        sell_id = f"{prefix}-sell"
        sell = submit(
            MarketOrderRequest(
                symbol=symbol,
                qty=test_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=sell_id,
            ),
            sell_id,
        )
        own_orders.append(sell)
        sell = wait_terminal(sell)
        if _status(sell) != "filled":
            raise RuntimeError(f"Paper smoke sell ended as {_status(sell)}") from failure
    elif failure is not None:
        raise failure

    remaining = [
        position
        for position in client.get_all_positions()
        if _symbol(position) == symbol and Decimal(str(position.qty)) != 0
    ]
    terminal_orders = [lookup(order) for order in own_orders]
    if remaining or any(_status(order) not in TERMINAL for order in terminal_orders):
        raise RuntimeError("Broker did not confirm smoke position flat and orders terminal")
    return {
        "symbol": symbol,
        "notional": str(notional),
        "orders": [str(order.id) for order in terminal_orders],
        "flat_confirmed": True,
        "all_orders_terminal": True,
    }
