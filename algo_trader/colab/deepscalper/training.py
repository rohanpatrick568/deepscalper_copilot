"""Shared local/Colab DeepScalper training, evaluation, and artifact tooling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .agent import DeepScalperAgent
from .environment import ScalperEnv
from .policy import normalize_action

ARTIFACT_FORMAT = "deepscalper-shared-v1"
FEATURE_SCHEMA = {
    "macro": [
        "z_open", "z_high", "z_low", "z_close", "z_adj_close",
        "z_d_5", "z_d_10", "z_d_15", "z_d_20", "z_d_25", "z_d_30",
    ],
    "micro": [
        "relative_spread", "depth_imbalance", "microprice_deviation",
        "trade_intensity", "short_return",
    ],
    "private": ["position_sign", "unrealized_pnl_fraction"],
}
ACTION_SCHEMA = {
    "direction": {"0": "SHORT", "1": "FLAT", "2": "LONG"},
    "size": "fixed risk allocation; model size branch disabled",
    "execution_timing": "action at close[t], fill at close[t], reward close[t] to close[t+1]",
}


class TrainingCancelled(RuntimeError):
    pass
REQUIRED_MANIFEST_KEYS = {
    "format",
    "symbol",
    "checkpoint_sha256",
    "accepted",
    "model",
    "splits",
    "metrics",
    "provenance",
    "feature_schema",
    "action_schema",
    "universe",
}


@dataclass(frozen=True)
class ModelConfig:
    macro_dim: int = 11
    lob_dim: int = 5
    priv_dim: int = 2
    n_dir: int = 3
    n_size: int = 1
    gru_hidden: int = 128
    macro_embed: int = 64
    fc_hidden: int = 128
    lookback_bars: int = 60


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    buffer_capacity: int = 1_000_000
    learning_rate: float = 1e-3
    gamma: float = 0.9
    soft_update_tau: float = 0.005
    explore_rate: float = 0.25
    transaction_cost_pct: float = 0.0018
    train_fraction: float = 0.70
    validation_fraction: float = 0.10
    seed: int = 42
    max_steps_per_epoch: int | None = None


@dataclass(frozen=True)
class MarketData:
    lob: np.ndarray
    macro: np.ndarray
    close: np.ndarray
    day_starts: np.ndarray | None = None

    def validate(self) -> None:
        lengths = {len(self.lob), len(self.macro), len(self.close)}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 3:
            raise ValueError("lob, macro, and close must have the same non-trivial length")
        if self.lob.ndim != 2 or self.macro.ndim != 2 or self.close.ndim != 1:
            raise ValueError("expected lob/macro matrices and a close vector")
        if not all(np.isfinite(values).all() for values in (self.lob, self.macro, self.close)):
            raise ValueError("training data contains non-finite values")
        if np.any(self.close <= 0):
            raise ValueError("close prices must be positive")
        if self.day_starts is not None:
            starts = np.asarray(self.day_starts, dtype=int)
            if (
                starts.ndim != 1
                or len(starts) == 0
                or starts[0] != 0
                or np.any(np.diff(starts) <= 0)
                or starts[-1] >= len(self.close)
            ):
                raise ValueError("day_starts must be ordered unique indices beginning at zero")


def synthetic_market_data(
    bars: int = 180,
    *,
    seed: int = 42,
    lob_dim: int = 5,
    macro_dim: int = 11,
) -> MarketData:
    """Create deterministic offline data for CI and installation smoke tests."""
    if bars < 45:
        raise ValueError("synthetic data needs at least 45 bars")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00005, 0.001, bars)
    close = 100.0 * np.exp(np.cumsum(returns))
    macro = rng.normal(0.0, 0.01, (bars, macro_dim)).astype(np.float32)
    macro[:, 3] = np.r_[0.0, np.diff(np.log(close))].astype(np.float32)
    lob = rng.normal(0.0, 0.01, (bars, lob_dim)).astype(np.float32)
    data = MarketData(lob=lob, macro=macro, close=close.astype(np.float64))
    data.validate()
    return data


def load_market_data(path: Path) -> MarketData:
    """Load the portable NPZ contract shared by local and Colab launchers."""
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = {"lob", "macro", "close"} - set(payload.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {sorted(missing)}")
        data = MarketData(
            lob=np.asarray(payload["lob"], dtype=np.float32),
            macro=np.asarray(payload["macro"], dtype=np.float32),
            close=np.asarray(payload["close"], dtype=np.float64),
            day_starts=(
                np.asarray(payload["day_starts"], dtype=np.int64)
                if "day_starts" in payload.files
                else None
            ),
        )
    data.validate()
    return data


def chronological_split(
    data: MarketData,
    train_fraction: float,
    validation_fraction: float,
    *,
    minimum_bars: int,
) -> tuple[dict[str, MarketData], dict[str, list[int]]]:
    """Split in time order; holdout is untouched until best-model selection."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave a holdout")
    n = len(data.close)
    if data.day_starts is not None and len(data.day_starts) >= 3:
        day_starts = np.asarray(data.day_starts, dtype=int)
        day_count = len(day_starts)
        train_days = max(1, int(day_count * train_fraction))
        validation_days = max(1, int(day_count * validation_fraction))
        if train_days + validation_days >= day_count:
            train_days = day_count - 2
            validation_days = 1
        train_end = int(day_starts[train_days])
        validation_end = int(day_starts[train_days + validation_days])
    else:
        train_end = int(n * train_fraction)
        validation_end = int(n * (train_fraction + validation_fraction))
    ranges = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "holdout": (validation_end, n),
    }
    if any(end - start < minimum_bars for start, end in ranges.values()):
        raise ValueError(
            f"each chronological split needs at least {minimum_bars} bars; got {ranges}"
        )

    def subset(bounds: tuple[int, int]) -> MarketData:
        start, end = bounds
        subset_starts = None
        if data.day_starts is not None:
            within = np.asarray(data.day_starts)
            within = within[(within >= start) & (within < end)] - start
            subset_starts = np.unique(np.r_[0, within]).astype(np.int64)
        return MarketData(
            data.lob[start:end],
            data.macro[start:end],
            data.close[start:end],
            subset_starts,
        )

    return (
        {name: subset(bounds) for name, bounds in ranges.items()},
        {name: [start, end] for name, (start, end) in ranges.items()},
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_sha256(data: MarketData) -> str:
    digest = hashlib.sha256()
    arrays = [data.lob, data.macro, data.close]
    if data.day_starts is not None:
        arrays.append(np.asarray(data.day_starts))
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _save_checkpoint(
    path: Path,
    agent: DeepScalperAgent,
    *,
    epoch: int,
    best_validation_return: float,
    model: ModelConfig,
    training: TrainConfig,
    run_signature: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "format": ARTIFACT_FORMAT,
            "online_net": agent.online_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "training_state": {
                "completed_epoch": epoch,
                "best_validation_return": best_validation_return,
                "steps": agent._steps,
                "run_signature": run_signature,
            },
            "model_config": asdict(model),
            "train_config": asdict(training),
        },
        partial,
    )
    partial.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_agent(model: ModelConfig, training: TrainConfig, device: str) -> DeepScalperAgent:
    return DeepScalperAgent(
        macro_dim=model.macro_dim,
        lob_dim=model.lob_dim,
        priv_dim=model.priv_dim,
        n_dir=model.n_dir,
        n_size=model.n_size,
        gru_hidden=model.gru_hidden,
        macro_embed=model.macro_embed,
        fc_hidden=model.fc_hidden,
        lr=training.learning_rate,
        gamma=training.gamma,
        soft_update_tau=training.soft_update_tau,
        batch_size=training.batch_size,
        buffer_capacity=training.buffer_capacity,
        explore_rate=training.explore_rate,
        device=device,
    )


def _make_env(
    data: MarketData,
    model: ModelConfig,
    training: TrainConfig,
    *,
    is_training: bool,
) -> ScalperEnv:
    return ScalperEnv(
        lob_features=data.lob,
        macro_features=data.macro,
        close_prices=data.close,
        day_starts=(
            np.asarray(data.day_starts, dtype=int).tolist()
            if data.day_starts is not None
            else [0]
        ),
        random_day_reset=False,
        lookback_bars=model.lookback_bars,
        transaction_cost_pct=training.transaction_cost_pct,
        hindsight_weight=0.2 if is_training else 0.0,
        training_mode=is_training,
    )


def _run_training_epoch(
    agent: DeepScalperAgent,
    env: ScalperEnv,
    *,
    seed: int,
    max_steps: int | None,
) -> dict[str, float]:
    rewards: list[float] = []
    losses: list[float] = []
    steps = 0
    for day_idx in range(len(env.day_starts)):
        obs, _ = env.reset(seed=seed + day_idx, options={"day_idx": day_idx})
        done = False
        while not done and (max_steps is None or steps < max_steps):
            direction, size = agent.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(direction)
            done = terminated or truncated
            agent.store(obs, direction, size, reward, next_obs, done, info["vol_target"])
            loss = agent.update_net()
            if loss is not None:
                losses.append(loss)
            rewards.append(float(reward))
            obs = next_obs
            steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {
        "return": float(np.sum(rewards)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "steps": float(steps),
    }


def evaluate(
    agent: DeepScalperAgent,
    data: MarketData,
    model: ModelConfig,
    training: TrainConfig,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Evaluate greedily with costs, no hindsight/exploration, and liquidation."""
    env = _make_env(data, model, training, is_training=False)
    net_log_returns: list[float] = []
    positions: list[int] = []
    costs = 0.0
    liquidations = 0
    days_visited: list[int] = []
    for day_idx in range(len(env.day_starts)):
        obs, _ = env.reset(seed=seed + day_idx, options={"day_idx": day_idx})
        days_visited.append(day_idx)
        done = False
        while not done:
            direction, _ = agent.select_action(obs, explore=False)
            direction = normalize_action(direction, int(env._position), long_only=True)
            obs, reward, terminated, truncated, info = env.step(direction)
            done = terminated or truncated
            net_log_returns.append(float(info["net_log_return"]))
            positions.append(int(info["position"]))
            costs += float(info["transaction_cost"])
            liquidations += int(info["terminal_liquidation"])
    values = np.asarray(net_log_returns, dtype=np.float64)
    equity = np.exp(np.cumsum(values))
    net_return = float(equity[-1] - 1.0) if len(equity) else 0.0
    peaks = np.maximum.accumulate(np.r_[1.0, equity])
    drawdowns = 1.0 - equity / peaks[1:]
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    periods_per_year = 252 * 390
    return {
        "return": net_return,
        "net_log_return": float(values.sum()),
        "sharpe": float(values.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0,
        "annualization_periods": periods_per_year,
        "max_drawdown": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "steps": len(net_log_returns),
        "position_changes": int(np.count_nonzero(np.diff(np.r_[0, positions]))),
        "transaction_cost": costs,
        "terminal_liquidations": liquidations,
        "exploration_rate": 0.0,
        "policy": "long_only_execution_v1",
        "days_visited": days_visited,
    }


def _dependencies() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "torch", "gymnasium"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _code_revision() -> str:
    repository = Path(__file__).resolve().parents[3]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _load_training_checkpoint(agent: DeepScalperAgent, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=agent.device, weights_only=True)
    if checkpoint.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"cannot resume unsupported checkpoint format: {path}")
    agent.online_net.load_state_dict(checkpoint["online_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    state = checkpoint.get("training_state", {})
    agent._steps = int(state.get("steps", 0))
    return state


def train_shared(
    data: MarketData,
    output_dir: Path,
    *,
    symbol: str,
    model: ModelConfig = ModelConfig(),
    training: TrainConfig = TrainConfig(),
    device: str = "cpu",
    location: str,
    cancel_path: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    resume: bool = False,
    min_validation_return: float = 0.0,
    min_holdout_return: float = 0.0,
    max_holdout_drawdown: float = 0.10,
) -> dict[str, Any]:
    """Run the canonical entry point used by both local and Colab launchers."""
    if location not in {"local", "colab"}:
        raise ValueError("location must be explicitly selected as local or colab")
    data.validate()
    if data.lob.shape[1] != model.lob_dim or data.macro.shape[1] != model.macro_dim:
        raise ValueError("dataset feature dimensions do not match model configuration")
    if not 0.0 < training.soft_update_tau <= 1.0:
        raise ValueError("soft_update_tau must be in (0, 1]")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pth"
    best_path = output_dir / "best.pth"
    state_path = output_dir / "training_state.json"
    manifest_path = output_dir / "best.manifest.json"

    split_data, split_ranges = chronological_split(
        data,
        training.train_fraction,
        training.validation_fraction,
        minimum_bars=model.lookback_bars + 2,
    )
    _seed_everything(training.seed)
    agent = _make_agent(model, training, device)
    signature_training = asdict(training)
    signature_training.pop("epochs", None)
    start_epoch = 0
    best_validation_return = float("-inf")
    resumed = False
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume requested but {latest_path} does not exist")
        state = _load_training_checkpoint(agent, latest_path)
        expected_signature = hashlib.sha256(
            json.dumps(
                {
                    "symbol": symbol,
                    "data": data_sha256(data),
                    "model": asdict(model),
                    "training": signature_training,
                    "features": FEATURE_SCHEMA,
                    "actions": ACTION_SCHEMA,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if state.get("run_signature") != expected_signature:
            raise ValueError(
                "resume artifact does not match symbol/data/config; start a new run "
                "or explicitly migrate the checkpoint"
            )
        start_epoch = int(state.get("completed_epoch", -1)) + 1
        best_validation_return = float(state.get("best_validation_return", float("-inf")))
        resumed = True

    history: list[dict[str, Any]] = []
    run_signature = hashlib.sha256(
        json.dumps(
            {
                "symbol": symbol,
                "data": data_sha256(data),
                "model": asdict(model),
                "training": signature_training,
                "features": FEATURE_SCHEMA,
                "actions": ACTION_SCHEMA,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    train_env = _make_env(split_data["train"], model, training, is_training=True)
    for epoch in range(start_epoch, training.epochs):
        if cancel_path is not None and Path(cancel_path).exists():
            raise TrainingCancelled(
                f"Cancellation requested before epoch {epoch}; resume from {latest_path}"
            )
        epoch_seed = training.seed + epoch
        _seed_everything(epoch_seed)
        train_metrics = _run_training_epoch(
            agent,
            train_env,
            seed=epoch_seed,
            max_steps=training.max_steps_per_epoch,
        )
        validation_metrics = evaluate(
            agent,
            split_data["validation"],
            model,
            training,
            seed=training.seed,
        )
        validation_return = float(validation_metrics["return"])
        if validation_return > best_validation_return or not best_path.exists():
            best_validation_return = validation_return
            _save_checkpoint(
                best_path,
                agent,
                epoch=epoch,
                best_validation_return=best_validation_return,
                model=model,
                training=training,
                run_signature=run_signature,
            )
        _save_checkpoint(
            latest_path,
            agent,
            epoch=epoch,
            best_validation_return=best_validation_return,
            model=model,
            training=training,
            run_signature=run_signature,
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        _write_json(
            state_path,
            {
                "format": ARTIFACT_FORMAT,
                "completed_epoch": epoch,
                "best_validation_return": best_validation_return,
                "latest_checkpoint": latest_path.name,
                "best_checkpoint": best_path.name,
                "run_signature": run_signature,
            },
        )
        if progress is not None:
            progress(
                {
                    "epoch": epoch + 1,
                    "epochs": training.epochs,
                    "validation_return": validation_return,
                    "best_validation_return": best_validation_return,
                }
            )

    if not best_path.is_file():
        raise ValueError(
            "training completed no epochs and no best checkpoint exists; "
            "increase --epochs or resume the original run"
        )
    selected_state = _load_training_checkpoint(agent, best_path)
    validation_metrics = evaluate(
        agent, split_data["validation"], model, training, seed=training.seed
    )
    holdout_metrics = evaluate(
        agent, split_data["holdout"], model, training, seed=training.seed
    )
    accepted = (
        np.isfinite(float(validation_metrics["return"]))
        and np.isfinite(float(holdout_metrics["return"]))
        and float(validation_metrics["return"]) >= min_validation_return
        and float(holdout_metrics["return"]) >= min_holdout_return
        and float(holdout_metrics["max_drawdown"]) <= max_holdout_drawdown
    )
    manifest: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "checkpoint": best_path.name,
        "checkpoint_sha256": file_sha256(best_path),
        "accepted": bool(accepted),
        "universe": [symbol],
        "feature_schema": FEATURE_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "acceptance": {
            "min_validation_return": min_validation_return,
            "min_holdout_return": min_holdout_return,
            "max_holdout_drawdown": max_holdout_drawdown,
            "minimum_position_changes": 0,
        },
        "model": asdict(model),
        "training": signature_training,
        "selection": {
            "criterion": "validation.return",
            "best_epoch": int(selected_state.get("completed_epoch", -1)),
            "latest_is_separate": True,
            "best_reloaded_before_holdout": True,
        },
        "splits": {
            "method": "chronological",
            "ranges": split_ranges,
            "holdout_used_for_selection": False,
        },
        "metrics": {
            "validation": validation_metrics,
            "holdout": holdout_metrics,
            "history": history,
        },
        "provenance": {
            "backend": {
                "location": location,
                "requested_device": device,
                "torch_device": str(next(agent.online_net.parameters()).device),
                "cuda_available": torch.cuda.is_available(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "seed": training.seed,
            "dependencies": _dependencies(),
            "code_revision": _code_revision(),
            "data_sha256": data_sha256(data),
            "config_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "model": asdict(model),
                        "training": asdict(training),
                        "features": FEATURE_SCHEMA,
                        "actions": ACTION_SCHEMA,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "resumed": resumed,
            "replay_buffer_restored": False,
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def validate_artifact(checkpoint_path: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else checkpoint_path.with_name(f"{checkpoint_path.stem}.manifest.json")
    )
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("checkpoint and manifest must both exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise ValueError(f"manifest is missing fields: {sorted(missing)}")
    if manifest["format"] != ARTIFACT_FORMAT:
        raise ValueError(f"unsupported artifact format: {manifest['format']}")
    actual_checksum = file_sha256(checkpoint_path)
    if manifest["checkpoint_sha256"] != actual_checksum:
        raise ValueError("checkpoint checksum does not match manifest")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != ARTIFACT_FORMAT or not checkpoint.get("online_net"):
        raise ValueError("checkpoint is not a shared DeepScalper artifact")
    if checkpoint.get("model_config") != manifest["model"]:
        raise ValueError("checkpoint and manifest model configurations differ")
    if manifest["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("artifact feature schema is incompatible")
    if manifest["action_schema"] != ACTION_SCHEMA:
        raise ValueError("artifact action schema is incompatible")
    return manifest


def promote_model(
    checkpoint_path: Path,
    manifest_path: Path,
    weights_dir: Path,
    *,
    symbol: str | None = None,
    allow_nonproduction: bool = False,
) -> tuple[Path, Path]:
    """Import an accepted artifact into runtime weights after strict validation."""
    manifest = validate_artifact(checkpoint_path, manifest_path)
    if not manifest["accepted"]:
        raise ValueError("only accepted models may be promoted")
    target_symbol = symbol or str(manifest["symbol"])
    if symbol is not None and symbol != manifest["symbol"]:
        raise ValueError("promotion symbol must match the manifest")
    production = asdict(ModelConfig())
    if not allow_nonproduction and manifest["model"] != production:
        raise ValueError("artifact architecture is not production-compatible")
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = weights_dir / f"{target_symbol.replace('/', '_')}.pth"
    target_manifest = weights_dir / f"{target_symbol.replace('/', '_')}.manifest.json"
    partial = target_checkpoint.with_suffix(".pth.partial")
    shutil.copyfile(checkpoint_path, partial)
    partial.replace(target_checkpoint)
    promoted = dict(manifest)
    promoted["checkpoint"] = target_checkpoint.name
    promoted["checkpoint_sha256"] = file_sha256(target_checkpoint)
    promoted["promoted_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(target_manifest, promoted)
    validate_artifact(target_checkpoint, target_manifest)
    return target_checkpoint, target_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train/evaluate with the shared entry point")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="NPZ with lob, macro, and close arrays")
    source.add_argument("--synthetic", action="store_true", help="use deterministic tiny data")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--symbol", default="AAPL")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    train.add_argument("--location", choices=("local", "colab"), required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--cancel-file", type=Path)
    train.add_argument("--tiny", action="store_true", help="small network and bounded steps")
    train.add_argument("--min-validation-return", type=float)
    train.add_argument("--min-holdout-return", type=float)
    train.add_argument("--max-holdout-drawdown", type=float)

    verify = subparsers.add_parser("verify", help="validate manifest and checksum")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    evaluation = subparsers.add_parser("evaluate", help="greedily evaluate an artifact")
    evaluation_source = evaluation.add_mutually_exclusive_group(required=True)
    evaluation_source.add_argument("--data", type=Path)
    evaluation_source.add_argument("--synthetic", action="store_true")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "holdout"), default="holdout")
    evaluation.add_argument("--device", default="cpu")

    promote = subparsers.add_parser("promote", help="import an accepted model")
    promote.add_argument("--checkpoint", type=Path, required=True)
    promote.add_argument("--manifest", type=Path, required=True)
    promote.add_argument("--weights-dir", type=Path, required=True)
    promote.add_argument("--symbol")
    promote.add_argument("--allow-nonproduction", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        manifest = validate_artifact(args.checkpoint, args.manifest)
        print(json.dumps({"valid": True, "accepted": manifest["accepted"]}, sort_keys=True))
        return 0
    if args.command == "evaluate":
        manifest = validate_artifact(args.checkpoint, args.manifest)
        model = ModelConfig(**manifest["model"])
        training_values = {
            key: value
            for key, value in manifest["training"].items()
            if key in TrainConfig.__dataclass_fields__
        }
        training = TrainConfig(**training_values)
        data = (
            synthetic_market_data(
                bars=180,
                seed=training.seed,
                lob_dim=model.lob_dim,
                macro_dim=model.macro_dim,
            )
            if args.synthetic
            else load_market_data(args.data)
        )
        splits, _ = chronological_split(
            data,
            training.train_fraction,
            training.validation_fraction,
            minimum_bars=model.lookback_bars + 2,
        )
        agent = _make_agent(model, training, args.device)
        _load_training_checkpoint(agent, args.checkpoint)
        metrics = evaluate(
            agent, splits[args.split], model, training, seed=training.seed
        )
        print(json.dumps(metrics, sort_keys=True))
        return 0
    if args.command == "promote":
        checkpoint, manifest = promote_model(
            args.checkpoint,
            args.manifest,
            args.weights_dir,
            symbol=args.symbol,
            allow_nonproduction=args.allow_nonproduction,
        )
        print(json.dumps({"checkpoint": str(checkpoint), "manifest": str(manifest)}))
        return 0

    model = ModelConfig()
    max_steps = None
    batch_size = 64
    buffer_capacity = 1_000_000
    if args.tiny:
        model = ModelConfig(
            gru_hidden=8,
            macro_embed=4,
            fc_hidden=8,
            lookback_bars=10,
        )
        max_steps = 12
        batch_size = 4
        buffer_capacity = 128
    data = (
        synthetic_market_data(
            bars=180 if args.tiny else 1200,
            seed=args.seed,
            lob_dim=model.lob_dim,
            macro_dim=model.macro_dim,
        )
        if args.synthetic
        else load_market_data(args.data)
    )
    training = TrainConfig(
        epochs=args.epochs,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        seed=args.seed,
        max_steps_per_epoch=max_steps,
    )
    thresholds = (
        (-1.0, -1.0, 1.0)
        if args.tiny
        else (
            0.0 if args.min_validation_return is None else args.min_validation_return,
            0.0 if args.min_holdout_return is None else args.min_holdout_return,
            0.10 if args.max_holdout_drawdown is None else args.max_holdout_drawdown,
        )
    )
    try:
        manifest = train_shared(
            data,
            args.output_dir,
            symbol=args.symbol,
            model=model,
            training=training,
            device=args.device,
            location=args.location,
            cancel_path=args.cancel_file,
            progress=lambda update: print(
                json.dumps({"progress": update}, sort_keys=True), file=sys.stderr
            ),
            resume=args.resume,
            min_validation_return=thresholds[0],
            min_holdout_return=thresholds[1],
            max_holdout_drawdown=thresholds[2],
        )
    except (TrainingCancelled, KeyboardInterrupt) as exc:
        print(f"training canceled: {exc}", file=sys.stderr)
        return 130
    print(
        json.dumps(
            {
                "accepted": manifest["accepted"],
                "checkpoint": str(Path(args.output_dir) / "best.pth"),
                "manifest": str(Path(args.output_dir) / "best.manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
