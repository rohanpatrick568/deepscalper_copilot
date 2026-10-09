"""Execution-policy rules shared by evaluation and paper runtime."""

from __future__ import annotations

ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2


def normalize_action(action: int, position: int, *, long_only: bool = True) -> int:
    """Prevent evaluation from opening shorts when paper mode is long-only."""
    if long_only and action == ACTION_SHORT:
        return ACTION_FLAT
    if position > 0 and action == ACTION_LONG:
        return ACTION_FLAT
    if position == 0 and action == ACTION_FLAT:
        return ACTION_FLAT
    return action
