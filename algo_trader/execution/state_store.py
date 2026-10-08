"""Durable execution state used to recover safely after a restart."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any


class ExecutionStateStore:
    """Small atomic JSON store for order, position, protection, and loss state."""

    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "sequence": 0,
            "orders": {},
            "symbols": {},
            "session": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Execution state is unreadable: {self.path}: {exc}") from exc
        if state.get("version") != self.VERSION:
            raise RuntimeError(
                f"Unsupported execution-state version {state.get('version')} in {self.path}"
            )
        for key in ("orders", "symbols", "session"):
            if not isinstance(state.get(key), dict):
                raise RuntimeError(f"Execution state field {key!r} is invalid in {self.path}")
        return state

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def next_client_order_id(self, symbol: str, intent: str) -> str:
        with self._lock:
            self._state["sequence"] += 1
            sequence = self._state["sequence"]
            client_id = f"ds-{symbol.lower().replace('.', '-')}-{intent}-{sequence:08d}"
            self._write()
            return client_id

    def set_order(self, client_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._state["orders"][client_id] = deepcopy(value)
            self._write()

    def patch_order(self, client_id: str, **values: Any) -> None:
        with self._lock:
            order = self._state["orders"].setdefault(client_id, {})
            order.update(deepcopy(values))
            self._write()

    def set_symbol(self, symbol: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._state["symbols"][symbol] = deepcopy(value)
            self._write()

    def clear_symbol(self, symbol: str) -> None:
        with self._lock:
            self._state["symbols"].pop(symbol, None)
            self._write()

    def set_session(self, **values: Any) -> None:
        with self._lock:
            self._state["session"].update(deepcopy(values))
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

