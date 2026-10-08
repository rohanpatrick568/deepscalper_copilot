"""Read-only Alpaca paper account, entitlement, calendar, and data checks."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    DATA_ADJUSTMENT,
    DATA_DELAY_MAX_SECONDS,
    DATA_FEED,
    TRADING_UNIVERSE,
)
from execution.market_data import AlpacaBarSource
from execution.validation import checkpoint_path, validate_bars, validate_checkpoint


def _paper_endpoint(client: Any) -> str:
    return str(getattr(client, "_base_url", getattr(client, "_base_url_override", "")))


def verify_paper_client(client: Any) -> None:
    endpoint = _paper_endpoint(client)
    if endpoint and "paper-api.alpaca.markets" not in endpoint:
        raise RuntimeError(f"Refusing non-paper Alpaca endpoint: {endpoint}")


def run_read_only_preflight(
    *,
    trading_client: Any | None = None,
    bar_source: AlpacaBarSource | None = None,
) -> dict[str, Any]:
    """Perform network checks without invoking any broker mutation."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("Paper credentials are required for read-only preflight")
    if trading_client is None:
        from alpaca.trading.client import TradingClient

        trading_client = TradingClient(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True
        )
    verify_paper_client(trading_client)
    bar_source = bar_source or AlpacaBarSource(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        feed=DATA_FEED,
        adjustment=DATA_ADJUSTMENT,
    )

    account = trading_client.get_account()
    positions = list(trading_client.get_all_positions())
    orders = list(
        trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    )
    clock = trading_client.get_clock()
    sessions = list(
        trading_client.get_calendar(
            GetCalendarRequest(start=date.today(), end=date.today())
        )
    )

    assets: dict[str, Any] = {}
    bars: dict[str, Any] = {}
    model_readiness: dict[str, Any] = {}
    for symbol in TRADING_UNIVERSE:
        asset = trading_client.get_asset(symbol)
        eligible = bool(
            getattr(asset, "tradable", False)
            and getattr(asset, "status", "") in {"active", "AssetStatus.ACTIVE"}
        )
        assets[symbol] = {
            "tradable": bool(getattr(asset, "tradable", False)),
            "fractionable": bool(getattr(asset, "fractionable", False)),
            "shortable": bool(getattr(asset, "shortable", False)),
            "eligible": eligible,
        }
        frame = bar_source.completed_minute_bars(symbol, 65)
        valid, reason = validate_bars(
            frame,
            now=pd.Timestamp.now(tz="UTC"),
            max_age_seconds=DATA_DELAY_MAX_SECONDS,
        )
        last = frame.index[-1] if len(frame) else None
        bars[symbol] = {
            "valid": valid,
            "reason": reason,
            "last_timestamp": last.isoformat() if last is not None else None,
            "rows": len(frame),
            "feed": DATA_FEED,
            "adjustment": DATA_ADJUSTMENT,
        }
        try:
            validate_checkpoint(
                checkpoint_path(symbol), symbol=symbol, require_approved=True
            )
        except (FileNotFoundError, ValueError) as exc:
            model_readiness[symbol] = {"ready": False, "reason": str(exc)}
        else:
            model_readiness[symbol] = {"ready": True, "reason": "approved"}

    return {
        "paper_endpoint": _paper_endpoint(trading_client) or "paper=True",
        "account": {
            "status": str(account.status),
            "buying_power": str(account.buying_power),
            "trading_blocked": bool(getattr(account, "trading_blocked", False)),
            "account_blocked": bool(getattr(account, "account_blocked", False)),
        },
        "positions": len(positions),
        "open_orders": len(orders),
        "clock": {
            "is_open": bool(clock.is_open),
            "timestamp": str(clock.timestamp),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        },
        "calendar": [
            {"date": str(item.date), "open": str(item.open), "close": str(item.close)}
            for item in sessions
        ],
        "assets": assets,
        "bars": bars,
        "model_readiness": model_readiness,
    }

