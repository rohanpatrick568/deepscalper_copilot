"""Execution-policy rules shared by training, evaluation, and paper runtime.

Actions use *target-position* semantics, matching ``ScalperEnv``:

* ``LONG``  -> target position ``+1``
* ``FLAT``  -> target position ``0``
* ``SHORT`` -> target position ``-1`` (suppressed to ``0`` when long-only)

A repeated ``LONG`` therefore *maintains* an existing long without pyramiding;
it must never be rewritten to ``FLAT``, which would alternate exposure and
charge turnover on every bar.

``decide_target_position`` is the single definition of the runtime entry
filters, holding period, cooldown, and protective-exit precedence.  Evaluation
reuses it so approval reflects the rules paper execution actually applies.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

POLICY_VERSION = "long_only_execution_v2"

ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2

ACTION_TO_POSITION = {ACTION_SHORT: -1, ACTION_FLAT: 0, ACTION_LONG: 1}
POSITION_TO_ACTION = {-1: ACTION_SHORT, 0: ACTION_FLAT, 1: ACTION_LONG}


@dataclass(frozen=True)
class PolicyConfig:
    """Runtime execution rules mirrored by the approval simulator.

    The protective-exit percentages approximate the runtime ATR stops with a
    fixed fraction of price, and simulated exits trigger on bar closes only.
    Simulated fills therefore approximate broker behaviour; they are not fill
    parity.
    """

    long_only: bool = True
    allow_pyramiding: bool = False
    entry_q_edge_min: float = 0.02
    entry_confidence_min: float = 0.58
    min_hold_bars: int = 3
    entry_cooldown_bars: int = 2
    stop_loss_pct: float = 0.010
    take_profit_pct: float = 0.020
    trailing_stop_pct: float = 0.0075
    use_trailing_stop: bool = True
    version: str = POLICY_VERSION

    def schema(self) -> dict:
        return dict(asdict(self))


@dataclass(frozen=True)
class PolicyDecision:
    """A target position plus the rule that selected it."""

    target_position: int
    reason: str
    confidence: float = 0.0
    protective: bool = False

    def changes_position(self, position: int) -> bool:
        return self.target_position != position


def normalize_action(action: int, position: int, *, long_only: bool = True) -> int:
    """Apply the long-only restriction while preserving target-position semantics.

    ``LONG`` while already long maintains the position; no pyramiding occurs
    because the action selects a target rather than an increment.  Only
    ``SHORT`` is rewritten, and only when shorts are disallowed.
    """
    if long_only and action == ACTION_SHORT:
        return ACTION_FLAT
    return action


def softmax_confidence(q_values) -> float:
    """Uncalibrated softmax score over direction Q-values.

    This is a relative preference score, not a calibrated win probability.
    """
    values = [float(value) for value in q_values]
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    return max(exponentials) / total if total > 0 else 0.0


def entry_filters(q_values, config: PolicyConfig) -> tuple[float, bool, bool]:
    """Return ``(confidence, long_entry_allowed, short_entry_allowed)``."""
    confidence = softmax_confidence(q_values)
    q_short = float(q_values[ACTION_SHORT])
    q_flat = float(q_values[ACTION_FLAT])
    q_long = float(q_values[ACTION_LONG])
    confident = confidence >= config.entry_confidence_min
    long_allowed = (q_long - q_flat) > config.entry_q_edge_min and confident
    short_allowed = (q_short - q_flat) > config.entry_q_edge_min and confident
    return confidence, long_allowed, short_allowed


def protective_exit_reason(
    position: int,
    entry_price: float,
    current_price: float,
    extreme_price: float,
    config: PolicyConfig,
) -> str | None:
    """Simulated stop/target/trailing check used by approval evaluation."""
    if position == 0 or entry_price <= 0 or current_price <= 0:
        return None
    if position > 0:
        if current_price <= entry_price * (1.0 - config.stop_loss_pct):
            return "RISK_STOP"
        if config.use_trailing_stop and extreme_price > 0:
            trailing = extreme_price * (1.0 - config.trailing_stop_pct)
            if (
                trailing > entry_price * (1.0 - config.stop_loss_pct)
                and current_price <= trailing
            ):
                return "TRAILING_STOP"
        if current_price >= entry_price * (1.0 + config.take_profit_pct):
            return "TAKE_PROFIT"
        return None
    if current_price >= entry_price * (1.0 + config.stop_loss_pct):
        return "RISK_STOP"
    if config.use_trailing_stop and extreme_price > 0:
        trailing = extreme_price * (1.0 + config.trailing_stop_pct)
        if (
            trailing < entry_price * (1.0 + config.stop_loss_pct)
            and current_price >= trailing
        ):
            return "TRAILING_STOP"
    if current_price <= entry_price * (1.0 - config.take_profit_pct):
        return "TAKE_PROFIT"
    return None


def decide_target_position(
    q_values,
    position: int,
    *,
    bars_held: int,
    bars_since_exit: int | None,
    config: PolicyConfig,
    entries_enabled: bool = True,
    protective_exit: str | None = None,
) -> PolicyDecision:
    """Resolve the target position from Q-values and execution state.

    A protective exit always wins and suppresses any model-driven change in the
    same iteration.
    """
    confidence, long_allowed, short_allowed = entry_filters(q_values, config)
    if protective_exit is not None and position != 0:
        return PolicyDecision(0, protective_exit, confidence, protective=True)

    action = int(max(range(len(q_values)), key=lambda index: float(q_values[index])))
    action = normalize_action(action, position, long_only=config.long_only)
    can_exit = bars_held >= config.min_hold_bars
    in_cooldown = (
        bars_since_exit is not None and bars_since_exit < config.entry_cooldown_bars
    )

    if action == ACTION_LONG:
        if position > 0:
            return PolicyDecision(position, "HOLD_LONG", confidence)
        if position < 0:
            if can_exit:
                return PolicyDecision(0, "REVERSE_TO_LONG", confidence)
            return PolicyDecision(position, "MIN_HOLD_BLOCKS_EXIT", confidence)
        if not entries_enabled:
            return PolicyDecision(0, "ENTRIES_DISABLED", confidence)
        if in_cooldown:
            return PolicyDecision(0, "ENTRY_COOLDOWN", confidence)
        if not long_allowed:
            return PolicyDecision(0, "ENTRY_FILTERED", confidence)
        if config.allow_pyramiding:
            return PolicyDecision(0, "PYRAMIDING_DISABLED", confidence)
        return PolicyDecision(1, "ENTER_LONG", confidence)

    if action == ACTION_SHORT:
        if position < 0:
            return PolicyDecision(position, "HOLD_SHORT", confidence)
        if position > 0:
            if can_exit:
                return PolicyDecision(0, "REVERSE_TO_SHORT", confidence)
            return PolicyDecision(position, "MIN_HOLD_BLOCKS_EXIT", confidence)
        if not entries_enabled:
            return PolicyDecision(0, "ENTRIES_DISABLED", confidence)
        if config.long_only:
            return PolicyDecision(0, "LONG_ONLY", confidence)
        if in_cooldown:
            return PolicyDecision(0, "ENTRY_COOLDOWN", confidence)
        if not short_allowed:
            return PolicyDecision(0, "ENTRY_FILTERED", confidence)
        return PolicyDecision(-1, "ENTER_SHORT", confidence)

    if position != 0:
        if can_exit:
            return PolicyDecision(0, "MODEL_FLAT", confidence)
        return PolicyDecision(position, "MIN_HOLD_BLOCKS_EXIT", confidence)
    return PolicyDecision(0, "STAY_FLAT", confidence)
