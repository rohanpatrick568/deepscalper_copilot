"""Shared historical-data and feature preparation CLI for Local and Colab."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import compute_macro_features, compute_micro_features


def prepare_features(input_path: Path, output_path: Path) -> Path:
    frame = pd.read_parquet(input_path)
    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input data is missing {sorted(required - set(frame.columns))}")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Input data contains duplicate timestamps")
    values = frame[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input data contains non-finite OHLCV values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_dates = pd.DatetimeIndex(frame.index).normalize()
    day_starts = np.r_[0, np.flatnonzero(normalized_dates[1:] != normalized_dates[:-1]) + 1]
    np.savez_compressed(
        output_path,
        lob=compute_micro_features(frame, use_proxy=True).astype(np.float32),
        macro=compute_macro_features(frame).astype(np.float32),
        close=frame["close"].to_numpy(dtype=np.float64),
        day_starts=day_starts.astype(np.int64),
    )
    return output_path


def fetch_alpaca_bars(
    output_path: Path,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    feed: str,
    adjustment: str,
) -> Path:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
    client = StockHistoricalDataClient(key, secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed(feed),
        adjustment=Adjustment(adjustment),
    )
    frame = client.get_stock_bars(request).df
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.xs(symbol, level="symbol")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--symbol", default="AAPL")
    fetch.add_argument("--start", type=datetime.fromisoformat, required=True)
    fetch.add_argument("--end", type=datetime.fromisoformat, required=True)
    fetch.add_argument("--feed", choices=("iex", "sip", "delayed_sip"), default="iex")
    fetch.add_argument(
        "--adjustment", choices=("raw", "split", "dividend", "all"), default="raw"
    )
    features = commands.add_parser("features")
    features.add_argument("--input", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fetch":
        path = fetch_alpaca_bars(
            args.output,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            feed=args.feed,
            adjustment=args.adjustment,
        )
    else:
        path = prepare_features(args.input, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
