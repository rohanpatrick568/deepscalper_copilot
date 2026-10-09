"""Explicit Alpaca market-data access without Lumibot's delayed-bar fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from alpaca.common.enums import Sort
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


_FEEDS = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
    "delayed_sip": DataFeed.DELAYED_SIP,
}
_ADJUSTMENTS = {
    "raw": Adjustment.RAW,
    "split": Adjustment.SPLIT,
    "dividend": Adjustment.DIVIDEND,
    "all": Adjustment.ALL,
}


class AlpacaBarSource:
    """Fetch completed minute bars with the configured feed and adjustment."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        feed: str,
        adjustment: str,
        client: Any | None = None,
    ) -> None:
        try:
            self.feed = _FEEDS[feed]
            self.adjustment = _ADJUSTMENTS[adjustment]
        except KeyError as exc:
            raise ValueError(f"Unsupported Alpaca data setting: {exc.args[0]}") from exc
        self.client = client or StockHistoricalDataClient(api_key, api_secret)

    def completed_minute_bars(
        self,
        symbol: str,
        limit: int,
        *,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        if limit <= 0:
            raise ValueError("Bar limit must be positive")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        completed_before = current.astimezone(timezone.utc).replace(second=0, microsecond=0)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=completed_before - timedelta(days=10),
            end=completed_before,
            limit=limit,
            sort=Sort.DESC,
            feed=self.feed,
            adjustment=self.adjustment,
        )
        response = self.client.get_stock_bars(request)
        frame = response.df.copy()
        if isinstance(frame.index, pd.MultiIndex):
            try:
                frame = frame.xs(symbol, level="symbol")
            except (KeyError, ValueError):
                frame = frame.droplevel(0)
        frame.index = pd.DatetimeIndex(frame.index)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        frame = frame.loc[frame.index < pd.Timestamp(completed_before)]
        return frame.sort_index().tail(limit)
