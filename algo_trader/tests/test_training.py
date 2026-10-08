"""Shared local/Colab training entry-point and artifact contract tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from colab.deepscalper.training import validate_artifact


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
    )
    result = json.loads(first.stdout)
    best = Path(result["checkpoint"])
    manifest_path = Path(result["manifest"])
    latest = artifact_dir / "latest.pth"
    state = artifact_dir / "training_state.json"
    assert best.is_file() and latest.is_file() and state.is_file()
    assert best != latest

    manifest = validate_artifact(best, manifest_path)
    assert manifest["accepted"] is True
    assert manifest["splits"]["method"] == "chronological"
    assert manifest["splits"]["holdout_used_for_selection"] is False
    assert manifest["selection"]["best_reloaded_before_holdout"] is True
    assert manifest["metrics"]["holdout"]["exploration_rate"] == 0.0
    assert manifest["metrics"]["holdout"]["terminal_liquidations"] in (0, 1)
    assert {"backend", "seed", "dependencies", "code_revision", "data_sha256"} <= set(
        manifest["provenance"]
    )

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

    imported = artifact_dir / "imported"
    _run(
        "colab.deepscalper.training",
        "promote",
        "--checkpoint",
        str(best),
        "--manifest",
        str(manifest_path),
        "--weights-dir",
        str(imported),
        "--symbol",
        "AAPL",
        "--allow-nonproduction",
    )
    validate_artifact(imported / "AAPL.pth", imported / "AAPL.manifest.json")

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
    )
    checkpoint = artifact_dir / "best.pth"
    manifest = artifact_dir / "best.manifest.json"
    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_artifact(checkpoint, manifest)
