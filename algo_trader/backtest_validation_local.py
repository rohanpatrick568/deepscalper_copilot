"""
Local backtest validator for DeepScalper (SHORT/FLAT/LONG actions).

This script mirrors the logic in colab/05_backtest_validation.ipynb and adds
trade-frequency diagnostics to investigate under-trading behavior.

Action semantics:
- 0 = SHORT
- 1 = FLAT
- 2 = LONG

Example:
    python backtest_validation_local.py \
        --data-path data/raw/BTC_USD.parquet \
        --weights-path weights/BTC_USD.pth \
        --output-dir results/backtest_local \
        --save-plots
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from colab.deepscalper.architecture import DeepScalperNet
from colab.deepscalper.utils import (
    compute_day_starts,
    compute_macro_features,
    compute_micro_features,
)
from config import (
    FC_HIDDEN,
    GRU_HIDDEN,
    LOB_DIM,
    LOOKBACK_BARS,
    MACRO_DIM,
    MACRO_EMBED_DIM,
    N_DIR,
    N_SIZE,
    PRIV_DIM,
    TRANSACTION_COST_LAMBDA,
)

ANNUALISE_24_7 = np.sqrt(525_960.0)
ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2


@dataclass
class TradeRecord:
    trade_id: int
    side: str
    entry_bar: int
    exit_bar: int
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    hold_bars: int
    q_short_entry: float
    q_flat_entry: float
    q_long_entry: float
    q_edge_entry: float
    q_short_exit: float
    q_flat_exit: float
    q_long_exit: float
    q_edge_exit: float


@dataclass
class BarDiagnostic:
    bar_idx: int
    timestamp: str
    action: int
    action_name: str
    position: int
    q_short: float
    q_flat: float
    q_long: float
    q_edge: float
    close_price: float
    step_return: float
    tc_cost: float


@dataclass
class BacktestResult:
    summary: Dict[str, Any]
    trade_records: List[TradeRecord]
    bar_records: List[BarDiagnostic]
    equity_curve: List[float]


class BacktestError(RuntimeError):
    pass


def _iso_ts(ts: Any) -> str:
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return str(ts)


def _ensure_supported_action_space() -> None:
    if N_DIR != 3:
        raise BacktestError(
            f"Expected N_DIR=3 for this validator, found N_DIR={N_DIR}."
        )


def _load_bars(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    if data_path.suffix.lower() == ".parquet":
        bars = pd.read_parquet(str(data_path))
        bars.columns = [c.lower() for c in bars.columns]
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in bars.columns]
        if missing:
            raise BacktestError(f"Missing required columns in parquet: {missing}")
        bars = bars[required].astype(float)
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise BacktestError("Parquet index must be a DatetimeIndex for day splitting.")
        return bars

    raise BacktestError("Unsupported data format. Use .parquet input for local validation.")


def _split_test_window(n_bars: int, train_split: float, val_split: float) -> Tuple[int, int]:
    train_end = int(n_bars * train_split)
    val_end = int(n_bars * (train_split + val_split))
    return train_end, val_end


def _build_model(weights_path: Path, device: str) -> DeepScalperNet:
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    ckpt = torch.load(str(weights_path), map_location=device, weights_only=True)
    state_dict = ckpt.get("online_net", ckpt)

    model = DeepScalperNet(
        macro_dim=MACRO_DIM,
        lob_dim=LOB_DIM,
        priv_dim=PRIV_DIM,
        gru_hidden=GRU_HIDDEN,
        macro_embed=MACRO_EMBED_DIM,
        fc_hidden=FC_HIDDEN,
        n_dir=N_DIR,
        n_size=N_SIZE,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _entropy_from_counts(counts: Sequence[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def _pctiles(values: Sequence[float], q: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(x)}": 0.0 for x in q}
    arr = np.array(values, dtype=np.float64)
    return {f"p{int(x)}": float(np.percentile(arr, x)) for x in q}


def run_backtest_with_diagnostics(
    model: DeepScalperNet,
    macro_feats: np.ndarray,
    lob_feats: np.ndarray,
    close_arr: np.ndarray,
    timestamps: Sequence[Any],
    day_starts: Sequence[int],
    lookback: int,
    tc_pct: float,
    device: str,
) -> BacktestResult:
    portfolio = 1.0
    equity_curve = [1.0]
    step_returns: List[float] = []
    trade_pnls: List[float] = []
    trade_records: List[TradeRecord] = []
    bar_records: List[BarDiagnostic] = []
    entry_bars: List[int] = []

    n_bars = len(close_arr)
    trade_id = 0

    for day_i, day_start in enumerate(day_starts):
        day_end = day_starts[day_i + 1] - 1 if day_i + 1 < len(day_starts) else n_bars - 1
        if day_start + lookback >= day_end:
            continue

        position = 0
        entry_price = 0.0
        entry_bar_idx: Optional[int] = None
        entry_side = ""
        q_short_entry = 0.0
        q_flat_entry = 0.0
        q_long_entry = 0.0
        q_edge_entry = 0.0

        priv_history = deque([np.zeros(2, dtype=np.float32)] * lookback, maxlen=lookback)

        for t in range(day_start + lookback - 1, day_end):
            win_start = max(0, t - lookback + 1)
            lob_seq = lob_feats[win_start : t + 1]
            if len(lob_seq) < lookback:
                pad = np.zeros((lookback - len(lob_seq), lob_feats.shape[1]), dtype=np.float32)
                lob_seq = np.vstack([pad, lob_seq])

            priv_seq = np.array(list(priv_history), dtype=np.float32)
            macro = macro_feats[t]

            with torch.no_grad():
                lob_t = torch.tensor(lob_seq[None], dtype=torch.float32, device=device)
                prv_t = torch.tensor(priv_seq[None], dtype=torch.float32, device=device)
                mac_t = torch.tensor(macro[None], dtype=torch.float32, device=device)
                q_dir, _ = model(lob_t, prv_t, mac_t)

            q_np = q_dir.squeeze(0).detach().cpu().numpy()
            action = int(q_np.argmax())
            q_short = float(q_np[ACTION_SHORT])
            q_flat = float(q_np[ACTION_FLAT])
            q_long = float(q_np[ACTION_LONG])
            q_edge = max(q_long, q_short) - q_flat

            current_price = float(close_arr[t])
            next_price = float(close_arr[min(t + 1, day_end)])
            tc_cost = 0.0

            if action == ACTION_LONG and position <= 0:
                if position == -1 and entry_price > 0:
                    pnl = (entry_price - current_price) / (entry_price + 1e-10)
                    trade_pnls.append(float(pnl))
                    tc_cost += tc_pct
                    if entry_bar_idx is not None:
                        trade_id += 1
                        trade_records.append(
                            TradeRecord(
                                trade_id=trade_id,
                                side="SHORT",
                                entry_bar=entry_bar_idx,
                                exit_bar=t,
                                entry_ts=_iso_ts(timestamps[entry_bar_idx]),
                                exit_ts=_iso_ts(timestamps[t]),
                                entry_price=float(entry_price),
                                exit_price=float(current_price),
                                pnl_pct=float(pnl),
                                hold_bars=int(t - entry_bar_idx),
                                q_short_entry=float(q_short_entry),
                                q_flat_entry=float(q_flat_entry),
                                q_long_entry=float(q_long_entry),
                                q_edge_entry=float(q_edge_entry),
                                q_short_exit=float(q_short),
                                q_flat_exit=float(q_flat),
                                q_long_exit=float(q_long),
                                q_edge_exit=float(q_edge),
                            )
                        )
                position = 1
                entry_price = current_price
                entry_bar_idx = t
                entry_side = "LONG"
                q_short_entry = q_short
                q_flat_entry = q_flat
                q_long_entry = q_long
                q_edge_entry = q_edge
                entry_bars.append(t)
                tc_cost += tc_pct
            elif action == ACTION_SHORT and position >= 0:
                if position == 1 and entry_price > 0:
                    pnl = (current_price - entry_price) / (entry_price + 1e-10)
                    trade_pnls.append(float(pnl))
                    tc_cost += tc_pct
                    if entry_bar_idx is not None:
                        trade_id += 1
                        trade_records.append(
                            TradeRecord(
                                trade_id=trade_id,
                                side="LONG",
                                entry_bar=entry_bar_idx,
                                exit_bar=t,
                                entry_ts=_iso_ts(timestamps[entry_bar_idx]),
                                exit_ts=_iso_ts(timestamps[t]),
                                entry_price=float(entry_price),
                                exit_price=float(current_price),
                                pnl_pct=float(pnl),
                                hold_bars=int(t - entry_bar_idx),
                                q_short_entry=float(q_short_entry),
                                q_flat_entry=float(q_flat_entry),
                                q_long_entry=float(q_long_entry),
                                q_edge_entry=float(q_edge_entry),
                                q_short_exit=float(q_short),
                                q_flat_exit=float(q_flat),
                                q_long_exit=float(q_long),
                                q_edge_exit=float(q_edge),
                            )
                        )
                position = -1
                entry_price = current_price
                entry_bar_idx = t
                entry_side = "SHORT"
                q_short_entry = q_short
                q_flat_entry = q_flat
                q_long_entry = q_long
                q_edge_entry = q_edge
                entry_bars.append(t)
                tc_cost += tc_pct
            elif action == ACTION_FLAT and position != 0:
                if position == 1:
                    pnl = (current_price - entry_price) / (entry_price + 1e-10)
                else:
                    pnl = (entry_price - current_price) / (entry_price + 1e-10)
                trade_pnls.append(float(pnl))
                tc_cost += tc_pct
                if entry_bar_idx is not None:
                    trade_id += 1
                    trade_records.append(
                        TradeRecord(
                            trade_id=trade_id,
                            side=entry_side,
                            entry_bar=entry_bar_idx,
                            exit_bar=t,
                            entry_ts=_iso_ts(timestamps[entry_bar_idx]),
                            exit_ts=_iso_ts(timestamps[t]),
                            entry_price=float(entry_price),
                            exit_price=float(current_price),
                            pnl_pct=float(pnl),
                            hold_bars=int(t - entry_bar_idx),
                            q_short_entry=float(q_short_entry),
                            q_flat_entry=float(q_flat_entry),
                            q_long_entry=float(q_long_entry),
                            q_edge_entry=float(q_edge_entry),
                            q_short_exit=float(q_short),
                            q_flat_exit=float(q_flat),
                            q_long_exit=float(q_long),
                            q_edge_exit=float(q_edge),
                        )
                    )
                position = 0
                entry_price = 0.0
                entry_bar_idx = None
                entry_side = ""

            log_ret = np.log(next_price / (current_price + 1e-10)) if current_price > 0 else 0.0
            step_r = (log_ret * position) - tc_cost
            step_returns.append(float(step_r))
            portfolio *= float(np.exp(step_r))
            equity_curve.append(portfolio)

            bar_records.append(
                BarDiagnostic(
                    bar_idx=t,
                    timestamp=_iso_ts(timestamps[t]),
                    action=action,
                    action_name=({ACTION_SHORT: "SHORT", ACTION_FLAT: "FLAT", ACTION_LONG: "LONG"})[action],
                    position=position,
                    q_short=float(q_short),
                    q_flat=float(q_flat),
                    q_long=float(q_long),
                    q_edge=float(q_edge),
                    close_price=float(current_price),
                    step_return=float(step_r),
                    tc_cost=float(tc_cost),
                )
            )

            unreal_pnl = (
                (current_price - entry_price) / (entry_price + 1e-10)
                if position == 1 and entry_price > 0
                else 0.0
            )
            if position == -1 and entry_price > 0:
                unreal_pnl = (entry_price - current_price) / (entry_price + 1e-10)
            priv_history.append(
                np.array([float(position), float(np.clip(unreal_pnl, -0.5, 0.5))], dtype=np.float32)
            )

        if position != 0 and entry_price > 0:
            eod_price = float(close_arr[day_end])
            if position == 1:
                pnl = (eod_price - entry_price) / (entry_price + 1e-10)
            else:
                pnl = (entry_price - eod_price) / (entry_price + 1e-10)
            trade_pnls.append(float(pnl))
            if entry_bar_idx is not None:
                trade_id += 1
                # Exit Q values at EOD are unknown in this branch; reuse last seen bar-level Qs.
                last = bar_records[-1]
                trade_records.append(
                    TradeRecord(
                        trade_id=trade_id,
                        side=entry_side,
                        entry_bar=entry_bar_idx,
                        exit_bar=day_end,
                        entry_ts=_iso_ts(timestamps[entry_bar_idx]),
                        exit_ts=_iso_ts(timestamps[day_end]),
                        entry_price=float(entry_price),
                        exit_price=float(eod_price),
                        pnl_pct=float(pnl),
                        hold_bars=int(day_end - entry_bar_idx),
                        q_short_entry=float(q_short_entry),
                        q_flat_entry=float(q_flat_entry),
                        q_long_entry=float(q_long_entry),
                        q_edge_entry=float(q_edge_entry),
                        q_short_exit=float(last.q_short),
                        q_flat_exit=float(last.q_flat),
                        q_long_exit=float(last.q_long),
                        q_edge_exit=float(last.q_edge),
                    )
                )

    arr = np.array(step_returns, dtype=np.float64)
    sharpe = float((arr.mean() / (arr.std() + 1e-10)) * ANNUALISE_24_7) if len(arr) > 1 else 0.0

    eq = np.array(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / (peak + 1e-10)).min())

    action_counts = Counter([r.action_name for r in bar_records])
    hold_bars = [r.hold_bars for r in trade_records]
    q_edges = [r.q_edge for r in bar_records]
    entry_gaps = [
        int(entry_bars[i] - entry_bars[i - 1]) for i in range(1, len(entry_bars))
    ]

    bars_total = max(len(bar_records), 1)
    entries_per_1k = float((len(entry_bars) / bars_total) * 1000.0)
    exits_per_1k = float((len(trade_records) / bars_total) * 1000.0)

    summary = {
        "action_space": {"n_dir": 3, "mapping": {"0": "SHORT", "1": "FLAT", "2": "LONG"}},
        "backtest": {
            "n_bars_evaluated": int(len(bar_records)),
            "total_return": float(portfolio - 1.0),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(np.mean([p > 0 for p in trade_pnls])) if trade_pnls else 0.0,
            "n_trades": int(len(trade_records)),
        },
        "actions": {
            "short_count": int(action_counts.get("SHORT", 0)),
            "flat_count": int(action_counts.get("FLAT", 0)),
            "long_count": int(action_counts.get("LONG", 0)),
            "short_ratio": float(action_counts.get("SHORT", 0) / bars_total),
            "flat_ratio": float(action_counts.get("FLAT", 0) / bars_total),
            "long_ratio": float(action_counts.get("LONG", 0) / bars_total),
            "entropy_nats": float(
                _entropy_from_counts([
                    action_counts.get("SHORT", 0),
                    action_counts.get("FLAT", 0),
                    action_counts.get("LONG", 0),
                ])
            ),
        },
        "frequency": {
            "entries_per_1000_bars": entries_per_1k,
            "exits_per_1000_bars": exits_per_1k,
            "median_bars_between_entries": float(np.median(entry_gaps)) if entry_gaps else 0.0,
            "mean_bars_between_entries": float(np.mean(entry_gaps)) if entry_gaps else 0.0,
        },
        "hold_time": {
            "mean_hold_bars": float(np.mean(hold_bars)) if hold_bars else 0.0,
            "median_hold_bars": float(np.median(hold_bars)) if hold_bars else 0.0,
            "p90_hold_bars": float(np.percentile(hold_bars, 90)) if hold_bars else 0.0,
        },
        "q_edge": {
            "mean": float(np.mean(q_edges)) if q_edges else 0.0,
            "std": float(np.std(q_edges)) if q_edges else 0.0,
            **_pctiles(q_edges, [10, 25, 50, 75, 90]),
        },
    }

    return BacktestResult(
        summary=summary,
        trade_records=trade_records,
        bar_records=bar_records,
        equity_curve=equity_curve,
    )


def _save_outputs(
    result: BacktestResult,
    output_dir: Path,
    save_plots: bool,
    undertrade_threshold_entries_per_1k: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(result.summary)
    warnings: List[str] = []
    if summary["frequency"]["entries_per_1000_bars"] < undertrade_threshold_entries_per_1k:
        warnings.append(
            "Entry frequency below threshold: "
            f"{summary['frequency']['entries_per_1000_bars']:.3f} < {undertrade_threshold_entries_per_1k:.3f} per 1000 bars"
        )
    if summary["actions"]["flat_ratio"] > 0.97:
        warnings.append(
            f"Very high FLAT ratio detected: {summary['actions']['flat_ratio']:.3f}."
        )
    summary["warnings"] = warnings

    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    trades_df = pd.DataFrame([asdict(t) for t in result.trade_records])
    bars_df = pd.DataFrame([asdict(b) for b in result.bar_records])

    trades_df.to_csv(output_dir / "trades.csv", index=False)
    bars_df.to_csv(output_dir / "bar_diagnostics.csv", index=False)

    if save_plots:
        import matplotlib.pyplot as plt

        eq = np.array(result.equity_curve, dtype=np.float64)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(eq, linewidth=1.3)
        ax.axhline(1.0, linestyle="--", linewidth=0.8)
        ax.set_title("Equity Curve")
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio")
        plt.tight_layout()
        fig.savefig(output_dir / "equity_curve.png", dpi=150)
        plt.close(fig)

        if not bars_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars_df["action_name"].value_counts().reindex(["SHORT", "FLAT", "LONG"]).fillna(0).plot(
                kind="bar", ax=ax
            )
            ax.set_title("Action Count")
            ax.set_ylabel("Count")
            plt.tight_layout()
            fig.savefig(output_dir / "action_counts.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 4))
            bars_df["q_edge"].plot(kind="hist", bins=50, ax=ax)
            ax.set_title("Q Edge Distribution (Q_LONG - Q_FLAT)")
            ax.set_xlabel("Q Edge")
            plt.tight_layout()
            fig.savefig(output_dir / "q_edge_hist.png", dpi=150)
            plt.close(fig)

        if not trades_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            trades_df["hold_bars"].plot(kind="hist", bins=40, ax=ax)
            ax.set_title("Hold Bars Distribution")
            ax.set_xlabel("Bars")
            plt.tight_layout()
            fig.savefig(output_dir / "hold_bars_hist.png", dpi=150)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local backtest validator (SHORT/FLAT/LONG).")
    parser.add_argument("--data-path", required=True, type=Path, help="Path to BTC_USD.parquet")
    parser.add_argument("--weights-path", required=True, type=Path, help="Path to BTC_USD.pth")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for outputs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_BARS, help="Lookback bars")
    parser.add_argument(
        "--tc-pct",
        type=float,
        default=TRANSACTION_COST_LAMBDA,
        help="One-way transaction cost as decimal",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.70,
        help="Train split fraction used for time split",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.10,
        help="Validation split fraction used for time split",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save diagnostic plots")
    parser.add_argument(
        "--undertrade-threshold-entries-per-1k",
        type=float,
        default=1.0,
        help="Warn if entries per 1000 bars fall below this threshold",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_supported_action_space()
    args = parse_args()

    bars = _load_bars(args.data_path)
    macro_feats = compute_macro_features(bars)
    lob_feats = compute_micro_features(bars, use_proxy=True)
    close_arr = bars["close"].values.astype(np.float64)
    ts_arr = bars.index
    day_starts = compute_day_starts(bars.index)

    n_bars = len(bars)
    _train_end, val_end = _split_test_window(n_bars, args.train_split, args.val_split)

    if val_end >= n_bars:
        raise BacktestError("Invalid split produced empty test window.")

    test_macro = macro_feats[val_end:]
    test_lob = lob_feats[val_end:]
    test_close = close_arr[val_end:]
    test_ts = ts_arr[val_end:]
    test_day_starts = [d - val_end for d in day_starts if d >= val_end]

    if len(test_day_starts) < 1:
        raise BacktestError("Insufficient test days after split.")

    model = _build_model(args.weights_path, args.device)

    result = run_backtest_with_diagnostics(
        model=model,
        macro_feats=test_macro,
        lob_feats=test_lob,
        close_arr=test_close,
        timestamps=test_ts,
        day_starts=test_day_starts,
        lookback=args.lookback,
        tc_pct=args.tc_pct,
        device=args.device,
    )

    _save_outputs(
        result=result,
        output_dir=args.output_dir,
        save_plots=args.save_plots,
        undertrade_threshold_entries_per_1k=args.undertrade_threshold_entries_per_1k,
    )

    s = result.summary
    print(
        "Backtest complete | "
        f"Return={s['backtest']['total_return']*100:+.2f}% | "
        f"Sharpe={s['backtest']['sharpe']:+.3f} | "
        f"MaxDD={s['backtest']['max_drawdown']*100:.2f}% | "
        f"Trades={s['backtest']['n_trades']} | "
        f"Entries/1k={s['frequency']['entries_per_1000_bars']:.3f}"
    )
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
