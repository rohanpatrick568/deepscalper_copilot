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


class UnresolvedSmokeExposure(RuntimeError):
    """The smoke position cannot be flattened without violating its limits."""

    def __init__(
        self,
        *,
        symbol: str,
        quantity: Decimal,
        estimated_notional: Decimal | None,
        max_order_notional: Decimal,
        reason: str,
        order_ids: tuple[str, ...] = (),
        client_order_ids: tuple[str, ...] = (),
    ) -> None:
        notional = (
            f"${estimated_notional:.2f}"
            if estimated_notional is not None
            else "unknown"
        )
        identifiers = (
            f" Orders: {', '.join(order_ids)}." if order_ids else ""
        )
        client_identifiers = (
            f" Client order IDs: {', '.join(client_order_ids)}."
            if client_order_ids
            else ""
        )
        super().__init__(
            f"UNRESOLVED PAPER-SMOKE EXPOSURE: {quantity} {symbol}; "
            f"estimated notional={notional}, order cap=${max_order_notional:.2f}. "
            f"{reason} No further smoke orders were submitted."
            f"{identifiers}{client_identifiers}"
            " Recover manually: inspect the order(s) above in the Alpaca paper "
            "dashboard, wait for or cancel any non-terminal order, then flatten "
            f"the remaining {symbol} quantity yourself."
        )
        self.symbol = symbol
        self.quantity = quantity
        self.estimated_notional = estimated_notional
        self.max_order_notional = max_order_notional
        self.order_ids = tuple(order_ids)
        self.client_order_ids = tuple(client_order_ids)


def _status(order: Any) -> str:
    return str(getattr(order, "status", "")).split(".")[-1].lower()


def _symbol(item: Any) -> str:
    return str(getattr(item, "symbol", getattr(getattr(item, "asset", None), "symbol", "")))


def _position_notional(position: Any, quantity: Decimal) -> Decimal | None:
    current_price = getattr(position, "current_price", None)
    if current_price is not None:
        price = Decimal(str(current_price))
        if price > 0:
            return quantity * price
    market_value = getattr(position, "market_value", None)
    if market_value is not None:
        value = abs(Decimal(str(market_value)))
        if value > 0:
            return value
    return None


def run_paper_round_trip(
    client: Any,
    *,
    symbol: str,
    notional: Decimal,
    timeout_seconds: float,
    max_orders: int = 2,
    confirmed: bool = False,
    cleanup_timeout_seconds: float | None = None,
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
    if timeout_seconds <= 0:
        raise ValueError("Paper smoke requires a positive timeout")
    # The cleanup sell gets its own bounded budget so a slow buy can never
    # consume the time needed to flatten the resulting position.
    cleanup_budget = (
        timeout_seconds if cleanup_timeout_seconds is None else cleanup_timeout_seconds
    )
    if cleanup_budget <= 0:
        raise ValueError("Paper smoke requires a positive cleanup timeout")
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

    def wait_terminal(order: Any, budget: float) -> Any:
        """Poll, then cancel, then re-poll until the broker confirms a terminal state.

        A cancel *request* is never treated as confirmation: the order is only
        considered resolved once the broker itself reports a terminal status.
        """
        current = order
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            current = lookup(current)
            if _status(current) in TERMINAL:
                return current
            sleep_fn(0.5)
        client.cancel_order_by_id(current.id)
        cancel_deadline = time.monotonic() + budget
        while time.monotonic() < cancel_deadline:
            current = lookup(current)
            if _status(current) in TERMINAL:
                return current
            sleep_fn(0.5)
        raise TimeoutError(
            f"Paper smoke order {current.id} is still {_status(current) or 'unknown'} "
            "after a cancellation request"
        )

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

    def live_position() -> Any | None:
        for position in client.get_all_positions():
            if _symbol(position) == symbol and Decimal(str(position.qty)) != 0:
                return position
        return None

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

    # The buy MUST be broker-confirmed terminal before any cleanup is even
    # calculated: submitting a sell against a still-working buy can oversell.
    try:
        buy = wait_terminal(buy, timeout_seconds)
    except TimeoutError as exc:
        stranded = live_position()
        quantity = Decimal(str(stranded.qty)) if stranded is not None else Decimal("0")
        raise UnresolvedSmokeExposure(
            symbol=symbol,
            quantity=quantity,
            estimated_notional=(
                _position_notional(stranded, quantity) if stranded is not None else None
            ),
            max_order_notional=Decimal(str(MAX_ORDER_NOTIONAL)),
            reason=(
                "The buy never reached a broker-confirmed terminal state, so its "
                "filled quantity is unknown and no cleanup sell was calculated."
            ),
            order_ids=(str(getattr(buy, "id", "unknown")),),
            client_order_ids=(buy_id,),
        ) from exc

    own_orders[0] = buy
    # Reconcile late fills: re-read the broker position now that the buy is terminal.
    failure: Exception | None = None
    if _status(buy) != "filled":
        failure = RuntimeError(f"Paper smoke buy ended as {_status(buy)}")

    positions = [
        position
        for position in client.get_all_positions()
        if _symbol(position) == symbol and Decimal(str(position.qty)) != 0
    ]
    if positions:
        if len(positions) != 1 or Decimal(str(positions[0].qty)) <= 0:
            raise RuntimeError("Smoke cleanup found an unexpected test-symbol position") from failure
        test_qty = Decimal(str(positions[0].qty))
        order_cap = Decimal(str(MAX_ORDER_NOTIONAL))
        cleanup_notional = _position_notional(positions[0], test_qty)
        if cleanup_notional is None:
            raise UnresolvedSmokeExposure(
                symbol=symbol,
                quantity=test_qty,
                estimated_notional=None,
                max_order_notional=order_cap,
                reason="The broker position has no usable current price or market value.",
                order_ids=(str(getattr(buy, "id", "unknown")),),
                client_order_ids=(buy_id,),
            ) from failure
        if cleanup_notional > order_cap:
            raise UnresolvedSmokeExposure(
                symbol=symbol,
                quantity=test_qty,
                estimated_notional=cleanup_notional,
                max_order_notional=order_cap,
                reason="Flattening the full position would exceed the configured per-order cap.",
                order_ids=(str(getattr(buy, "id", "unknown")),),
                client_order_ids=(buy_id,),
            ) from failure
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
        try:
            sell = wait_terminal(sell, cleanup_budget)
        except TimeoutError as exc:
            stranded = live_position()
            quantity = (
                Decimal(str(stranded.qty)) if stranded is not None else Decimal("0")
            )
            raise UnresolvedSmokeExposure(
                symbol=symbol,
                quantity=quantity,
                estimated_notional=(
                    _position_notional(stranded, quantity)
                    if stranded is not None
                    else None
                ),
                max_order_notional=order_cap,
                reason=(
                    "The cleanup sell never reached a broker-confirmed terminal "
                    "state within its own timeout budget."
                ),
                order_ids=(str(getattr(buy, "id", "unknown")), str(getattr(sell, "id", "unknown"))),
                client_order_ids=(buy_id, sell_id),
            ) from exc
        own_orders[-1] = sell
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
