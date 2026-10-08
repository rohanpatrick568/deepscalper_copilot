"""Startup and data validation shared by offline and paper workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from config import (
    DATA_DELAY_MAX_SECONDS,
    DATA_FEED,
    FC_HIDDEN,
    GRU_HIDDEN,
    LOB_DIM,
    LOOKBACK_BARS,
    MACRO_DIM,
    MACRO_EMBED_DIM,
    N_DIR,
    N_SIZE,
    PRIV_DIM,
    TRADING_UNIVERSE,
    WEIGHTS_DIR,
)


def validate_bars(
    bars: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    max_age_seconds: int = DATA_DELAY_MAX_SECONDS,
) -> tuple[bool, str]:
    """Require finite, strictly ordered, completed OHLCV bars."""
    if bars is None or len(bars) == 0:
        return False, "no bars"
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(bars.columns):
        return False, f"missing columns: {sorted(required - set(bars.columns))}"
    if not isinstance(bars.index, pd.DatetimeIndex) or not bars.index.is_monotonic_increasing:
        return False, "bar index is not ordered"
    if bars.index.has_duplicates:
        return False, "duplicate bar timestamps"
    if not bars[list(required)].apply(pd.to_numeric, errors="coerce").replace(
        [float("inf"), float("-inf")], pd.NA
    ).notna().all().all():
        return False, "non-finite bar values"
    if (bars["high"] < bars["low"]).any() or (bars["volume"] < 0).any():
        return False, "invalid OHLCV values"
    if now is not None:
        last = bars.index[-1]
        current = pd.Timestamp(now)
        if last.tzinfo is None and current.tzinfo is not None:
            current = current.tz_localize(None)
        age = (current - last).total_seconds()
        if age < 0 or age > max_age_seconds:
            return False, f"bar age {age:.1f}s exceeds {max_age_seconds}s"
    return True, "ok"


def checkpoint_filename(symbol: str) -> str:
    return f"{symbol.replace('/', '_')}.pth"


def checkpoint_path(symbol: str, weights_dir: Path = WEIGHTS_DIR) -> Path:
    return Path(weights_dir) / checkpoint_filename(symbol)


def validate_checkpoint(
    path: Path,
    *,
    symbol: str | None = None,
    require_approved: bool = False,
) -> dict:
    """Load and validate architecture metadata before a model can run."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unreadable checkpoint {path}: {exc}") from exc
    state = checkpoint.get("online_net", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint {path} has no model state")
    required = ("macro_enc", "micro_enc")
    if not any(key.startswith(required[0]) for key in state) or not any(
        key.startswith(required[1]) for key in state
    ):
        raise ValueError(f"Checkpoint {path} is incompatible with the DeepScalper architecture")
    manifest = checkpoint.get("manifest")
    if symbol and manifest and manifest.get("symbol") not in (None, symbol):
        raise ValueError(f"Checkpoint {path} manifest symbol does not match {symbol}")
    if require_approved:
        from colab.deepscalper.training import validate_artifact

        artifact_manifest = validate_artifact(
            path, path.with_name(f"{path.stem}.manifest.json")
        )
        if not artifact_manifest["accepted"]:
            raise ValueError(f"Checkpoint {path} did not pass acceptance gates")
        if symbol and artifact_manifest["symbol"] != symbol:
            raise ValueError(f"Checkpoint {path} manifest symbol does not match {symbol}")
        expected_model = {
            "macro_dim": MACRO_DIM,
            "lob_dim": LOB_DIM,
            "priv_dim": PRIV_DIM,
            "n_dir": N_DIR,
            "n_size": N_SIZE,
            "gru_hidden": GRU_HIDDEN,
            "macro_embed": MACRO_EMBED_DIM,
            "fc_hidden": FC_HIDDEN,
            "lookback_bars": LOOKBACK_BARS,
        }
        if artifact_manifest["model"] != expected_model:
            raise ValueError(f"Checkpoint {path} is not runtime-compatible")
    return checkpoint


def validate_startup_configuration(
    symbols: Iterable[str] = TRADING_UNIVERSE,
    weights_dir: Path = WEIGHTS_DIR,
) -> None:
    symbols = list(symbols)
    if not symbols:
        raise ValueError("Trading universe must contain at least one symbol")
    if DATA_FEED not in {"iex", "sip", "delayed_sip"}:
        raise ValueError(f"Unsupported Alpaca data feed: {DATA_FEED}")
    if N_DIR != 3 or N_SIZE < 1:
        raise ValueError(f"Unsupported model action shape: n_dir={N_DIR}, n_size={N_SIZE}")
    for symbol in symbols:
        validate_checkpoint(
            checkpoint_path(symbol, weights_dir),
            symbol=symbol,
            require_approved=True,
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
