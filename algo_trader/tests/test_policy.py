"""Regressions for the execution policy shared by training, evaluation, runtime."""

from __future__ import annotations

import pytest

from colab.deepscalper.policy import (
    ACTION_FLAT,
    ACTION_LONG,
    ACTION_SHORT,
    ACTION_TO_POSITION,
    POSITION_TO_ACTION,
    PolicyConfig,
    decide_target_position,
    normalize_action,
)

# Q-values that clear the default entry edge (0.02) and confidence (0.58) gates.
STRONG_LONG = [0.0, 0.0, 6.0]
STRONG_SHORT = [6.0, 0.0, 0.0]
STRONG_FLAT = [0.0, 6.0, 0.0]
WEAK_LONG = [0.0, 0.0, 0.01]

HELD = {"bars_held": 10, "bars_since_exit": None}


def decide(q, position, **overrides):
    kwargs = dict(HELD)
    kwargs.update(overrides)
    config = kwargs.pop("config", PolicyConfig())
    return decide_target_position(q, position, config=config, **kwargs)


def test_repeated_long_maintains_the_position_without_pyramiding():
    """Regression: LONG while long previously normalised to FLAT and churned."""
    position = 0
    reasons = []
    for _ in range(5):
        decision = decide(STRONG_LONG, position)
        reasons.append(decision.reason)
        position = decision.target_position

    assert position == 1
    assert reasons == ["ENTER_LONG"] + ["HOLD_LONG"] * 4
    # Exposure is maintained, never alternated.
    assert all(reason != "MODEL_FLAT" for reason in reasons[1:])


def test_repeated_long_never_increases_exposure():
    decision = decide(STRONG_LONG, 1)
    assert decision.target_position == 1
    assert not decision.changes_position(1)


def test_flat_action_exits_an_open_position():
    decision = decide(STRONG_FLAT, 1)
    assert decision.target_position == 0
    assert decision.reason == "MODEL_FLAT"
    assert decision.changes_position(1)


def test_minimum_hold_blocks_an_early_model_exit():
    decision = decide(STRONG_FLAT, 1, bars_held=0)
    assert decision.target_position == 1
    assert decision.reason == "MIN_HOLD_BLOCKS_EXIT"


def test_long_only_short_cannot_create_a_short():
    decision = decide(STRONG_SHORT, 0)
    assert decision.target_position == 0
    assert decision.reason in {"LONG_ONLY", "STAY_FLAT"}


def test_long_only_short_exits_an_existing_long():
    decision = decide(STRONG_SHORT, 1)
    assert decision.target_position == 0
    assert decision.changes_position(1)


def test_shorts_are_reachable_only_when_long_only_is_disabled():
    config = PolicyConfig(long_only=False)
    decision = decide(STRONG_SHORT, 0, config=config)
    assert decision.target_position == -1
    assert decision.reason == "ENTER_SHORT"


def test_normalize_action_preserves_long_while_long():
    assert normalize_action(ACTION_LONG, 1, long_only=True) == ACTION_LONG
    assert normalize_action(ACTION_LONG, 0, long_only=True) == ACTION_LONG
    assert normalize_action(ACTION_SHORT, 0, long_only=True) == ACTION_FLAT
    assert normalize_action(ACTION_SHORT, 0, long_only=False) == ACTION_SHORT
    assert normalize_action(ACTION_FLAT, 1, long_only=True) == ACTION_FLAT


def test_action_and_position_maps_are_inverse():
    for action, position in ACTION_TO_POSITION.items():
        assert POSITION_TO_ACTION[position] == action


def test_weak_edge_is_filtered_out_of_entries():
    decision = decide(WEAK_LONG, 0)
    assert decision.target_position == 0
    assert decision.reason == "ENTRY_FILTERED"


def test_cooldown_blocks_a_re_entry_but_not_an_exit():
    blocked = decide(STRONG_LONG, 0, bars_since_exit=0)
    assert blocked.target_position == 0
    assert blocked.reason == "ENTRY_COOLDOWN"

    exit_decision = decide(STRONG_FLAT, 1, bars_since_exit=0)
    assert exit_decision.target_position == 0
    assert exit_decision.reason == "MODEL_FLAT"


def test_disabled_entries_still_permit_exits():
    """A loss halt stops new entries while position protection continues."""
    blocked = decide(STRONG_LONG, 0, entries_enabled=False)
    assert blocked.target_position == 0
    assert blocked.reason == "ENTRIES_DISABLED"

    exit_decision = decide(STRONG_FLAT, 1, entries_enabled=False)
    assert exit_decision.target_position == 0
    assert exit_decision.changes_position(1)


def test_protective_exit_overrides_the_model_in_the_same_iteration():
    decision = decide(STRONG_LONG, 1, protective_exit="RISK_STOP")
    assert decision.target_position == 0
    assert decision.reason == "RISK_STOP"
    assert decision.protective is True


def test_protective_exit_is_ignored_when_already_flat():
    decision = decide(STRONG_LONG, 0, protective_exit="RISK_STOP")
    assert decision.reason == "ENTER_LONG"
    assert decision.protective is False


@pytest.mark.parametrize("position", [-1, 0, 1])
def test_decisions_are_deterministic_from_any_starting_position(position):
    first = decide(STRONG_LONG, position)
    second = decide(STRONG_LONG, position)
    assert (first.target_position, first.reason) == (
        second.target_position,
        second.reason,
    )


def test_policy_schema_is_versioned_and_round_trips():
    config = PolicyConfig()
    schema = config.schema()
    assert schema["version"] == config.version
    assert PolicyConfig(**schema) == config
