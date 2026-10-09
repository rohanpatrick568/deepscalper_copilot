"""Verify a training artifact loads on CPU and respects the promotion rules.

Used by CI (and usable locally) to confirm that an artifact produced on any
device can be loaded for CPU inference, and that a rejected or synthetic
artifact can never reach the paper-approved weights directory.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

import torch

from colab.deepscalper.training import promote_model, validate_artifact


def verify(checkpoint: Path, manifest_path: Path, *, expect_rejected: bool) -> dict:
    manifest = validate_artifact(checkpoint, manifest_path)

    # Loading with map_location="cpu" must work regardless of the training device.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"checkpoint payload must be a dict, got {type(payload)}")

    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu_load": True,
        "accepted": manifest["accepted"],
        "data_source": manifest["data_source"],
        "policy_version": manifest["policy_schema"]["version"],
        "rejections": manifest["rejections"],
    }

    if expect_rejected:
        if manifest["accepted"]:
            raise SystemExit("a flat synthetic smoke run must never be accepted")
        with tempfile.TemporaryDirectory() as weights_dir:
            try:
                promote_model(
                    checkpoint,
                    manifest_path,
                    Path(weights_dir),
                    symbol=manifest["symbol"],
                    allow_nonproduction=True,
                )
            except ValueError as exc:
                result["promotion_refused"] = str(exc)
            else:
                raise SystemExit("promotion must refuse a rejected artifact")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expect-rejected",
        action="store_true",
        help="also assert the artifact is rejected and cannot be promoted",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            verify(
                args.checkpoint,
                args.manifest,
                expect_rejected=args.expect_rejected,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
