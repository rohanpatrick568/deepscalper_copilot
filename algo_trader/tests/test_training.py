"""Shared local/Colab training entry-point and artifact contract tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import torch

from colab.deepscalper.data import prepare_features
from colab.deepscalper.policy import POLICY_VERSION
from colab.deepscalper.training import (
    MarketData,
    ModelConfig,
    TrainConfig,
    evaluate,
    promote_model,
    validate_artifact,
)


@pytest.fixture
def artifact_dir():
    path = Path.cwd() / f".training-test-artifacts-{os.getpid()}"
    shutil.rmtree(path, ignore_errors=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _run(module: str, *arguments: str, pythonpath: Path | None = None):
    env = os.environ.copy()
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=Path.cwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_tiny_shared_entry_point_resume_and_colab_compatibility(artifact_dir):
    first = _run(
        "colab.deepscalper.training",
        "train",
        "--synthetic",
        "--tiny",
        "--epochs",
        "1",
        "--output-dir",
        str(artifact_dir),
        "--symbol",
        "AAPL",
        "--location",
        "local",
    )
    result = json.loads(first.stdout)
    best = Path(result["checkpoint"])
    manifest_path = Path(result["manifest"])
    latest = artifact_dir / "latest.pth"
    state = artifact_dir / "training_state.json"
    assert best.is_file() and latest.is_file() and state.is_file()
    assert best != latest

    manifest = validate_artifact(best, manifest_path)
    # A tiny untrained run is genuinely flat, so the activity gate must reject
    # it: a zero-trade policy can never be approved for paper execution.
    assert manifest["accepted"] is False
    assert manifest["data_source"] == "synthetic"
    assert any("position_changes" in reason for reason in manifest["rejections"])
    assert manifest["policy_schema"]["version"] == POLICY_VERSION
    assert manifest["splits"]["method"] == "chronological"
    assert manifest["splits"]["holdout_used_for_selection"] is False
    assert manifest["selection"]["best_reloaded_before_holdout"] is True
    assert manifest["metrics"]["holdout"]["exploration_rate"] == 0.0
    assert manifest["metrics"]["holdout"]["terminal_liquidations"] in (0, 1)
    assert {"backend", "seed", "dependencies", "code_revision", "data_sha256"} <= set(
        manifest["provenance"]
    )
    assert manifest["provenance"]["backend"]["location"] == "local"
    assert manifest["provenance"]["config_sha256"]
    assert manifest["action_schema"]["size"].startswith("fixed")

    colab_import_root = Path.cwd() / "colab"
    verified = _run(
        "deepscalper.training",
        "verify",
        "--checkpoint",
        str(best),
        "--manifest",
        str(manifest_path),
        pythonpath=colab_import_root,
    )
    assert json.loads(verified.stdout)["valid"] is True

    # Synthetic smoke output must never reach the paper-approved weights dir.
    imported = artifact_dir / "imported"
    with pytest.raises(ValueError, match="accepted"):
        promote_model(best, manifest_path, imported, symbol="AAPL", allow_nonproduction=True)
    assert not imported.exists()

    # The checkpoint still loads on CPU regardless of the training device.
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)

    _run(
        "colab.deepscalper.training",
        "train",
        "--synthetic",
        "--tiny",
        "--epochs",
        "2",
        "--resume",
        "--output-dir",
        str(artifact_dir),
        "--location",
        "local",
        "--symbol",
        "AAPL",
    )
    resumed_manifest = validate_artifact(best, manifest_path)
    assert resumed_manifest["provenance"]["resumed"] is True
    assert json.loads(state.read_text())["completed_epoch"] == 1


def test_checksum_tampering_is_rejected(artifact_dir):
    _run(
        "colab.deepscalper.training",
        "train",
        "--synthetic",
        "--tiny",
        "--epochs",
        "1",
        "--output-dir",
        str(artifact_dir),
        "--location",
        "local",
    )
    checkpoint = artifact_dir / "best.pth"
    manifest = artifact_dir / "best.manifest.json"
    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_artifact(checkpoint, manifest)


def test_colab_location_exports_local_compatible_artifact(artifact_dir):
    _run(
        "colab.deepscalper.training",
        "train",
        "--synthetic",
        "--tiny",
        "--epochs",
        "1",
        "--output-dir",
        str(artifact_dir),
        "--location",
        "colab",
    )
    manifest = validate_artifact(
        artifact_dir / "best.pth", artifact_dir / "best.manifest.json"
    )
    assert manifest["provenance"]["backend"]["location"] == "colab"
    assert manifest["feature_schema"]
    assert manifest["model"]["n_size"] == 1


def test_shared_data_launcher_builds_portable_feature_contract(artifact_dir):
    index = pd.date_range("2025-01-02 14:30", periods=80, freq="1min", tz="UTC")
    close = np.linspace(100.0, 102.0, len(index))
    raw = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(len(index), 100.0),
        },
        index=index,
    )
    raw_path = artifact_dir / "AAPL_raw.parquet"
    features_path = artifact_dir / "AAPL_features.npz"
    artifact_dir.mkdir(parents=True)
    raw.to_parquet(raw_path)
    prepare_features(raw_path, features_path)
    with np.load(features_path) as payload:
        assert payload["lob"].shape == (80, 5)
        assert payload["macro"].shape == (80, 11)
        assert payload["close"].shape == (80,)
        assert payload["day_starts"].tolist() == [0]


def test_evaluation_visits_each_day_once_without_exploration():
    class FlatAgent:
        def __init__(self):
            self.explore_values = []

        def select_action(self, observation, explore):
            self.explore_values.append(explore)
            return 1, 0

        def action_values(self, observation):
            self.explore_values.append(False)
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    bars_per_day = 15
    days = 3
    count = bars_per_day * days
    data = MarketData(
        lob=np.zeros((count, 5), dtype=np.float32),
        macro=np.zeros((count, 11), dtype=np.float32),
        close=np.linspace(100.0, 101.0, count),
        day_starts=np.arange(0, count, bars_per_day),
    )
    agent = FlatAgent()
    metrics = evaluate(
        agent,
        data,
        ModelConfig(lookback_bars=10),
        TrainConfig(),
        seed=7,
    )
    assert metrics["days_visited"] == [0, 1, 2]
    assert all(value is False for value in agent.explore_values)


def test_training_epoch_visits_every_training_day(monkeypatch):
    from colab.deepscalper import training as training_module

    class Agent:
        def __init__(self):
            self.days = []

        def select_action(self, observation, explore):
            self.days.append(int(observation["day"]))
            return 1, 0

        def action_values(self, observation):
            self.days.append(int(observation["day"]))
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

        def store(self, *args):
            return None

        def update_net(self):
            return None

    class Env:
        day_starts = [0, 10, 20]

        def reset(self, seed, options):
            self.day = options["day_idx"]
            self.step_count = 0
            return {"day": self.day}, {}

        def step(self, action):
            self.step_count += 1
            done = self.step_count == 1
            return (
                {"day": self.day},
                0.0,
                done,
                False,
                {"vol_target": 0.0},
            )

    agent = Agent()
    training_module._run_training_epoch(
        agent, Env(), seed=1, max_steps=None
    )
    assert agent.days == [0, 1, 2]
