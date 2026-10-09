"""Safe operational commands for offline, read-only, dry-run, and paper modes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from config import LOOKBACK_BARS, MARKET_TIMEZONE, TRADING_UNIVERSE

COLAB_NOTEBOOK = "algo_trader/colab/03_train_deepscalper.ipynb"
COLAB_NOTEBOOK_URL = (
    "https://colab.research.google.com/github/rohanpatrick568/deepscalper_copilot/"
    "blob/fix/supervised-paper-reliability/" + COLAB_NOTEBOOK
)


def training_launch_plan(
    location: str,
    *,
    data: Path | None = None,
    output_dir: Path = Path("runs/AAPL"),
    symbol: str = "AAPL",
    epochs: int = 20,
    device: str = "cpu",
    resume: bool = False,
) -> dict:
    """Describe the concrete launch route for the user's Local/Colab choice.

    Both locations run the same shared training CLI and produce the same
    checkpoint/manifest format; only the host differs.
    """
    if location not in ("local", "colab"):
        raise ValueError(f"location must be 'local' or 'colab', not {location!r}")
    argv = [
        "train",
        "--data",
        str(data) if data is not None else "DATA.npz",
        "--output-dir",
        str(output_dir),
        "--symbol",
        symbol,
        "--location",
        location,
        "--device",
        device,
        "--epochs",
        str(epochs),
    ]
    if resume:
        argv.append("--resume")
    plan = {
        "location": location,
        "module": "colab.deepscalper.training",
        "argv": argv,
        "command": "python -m colab.deepscalper.training " + " ".join(argv),
        "credentials_required": False,
        "runs_here": location == "local",
    }
    if location == "colab":
        # Colab compute is started by the user in their own authorized session;
        # this process never launches a remote job.
        plan["notebook"] = COLAB_NOTEBOOK
        plan["notebook_url"] = COLAB_NOTEBOOK_URL
        plan["next"] = (
            "Open the notebook, run the bootstrap cell to mount Drive and check "
            "out this revision, choose the device, then run the training cell. "
            "Download best.pth and best.manifest.json and import them locally "
            "with the promote command."
        )
    else:
        plan["next"] = (
            "Runs on this machine with the command above. Use --resume to "
            "continue an interrupted run and --cancel-file to stop it cleanly."
        )
    return plan


def offline_validation() -> dict:
    """Validate shared feature/training imports without credentials or network."""
    from colab.deepscalper.training import synthetic_market_data

    data = synthetic_market_data(bars=180)
    return {
        "network_used": False,
        "credentials_required": False,
        "symbols": TRADING_UNIVERSE,
        "synthetic_bars": len(data.close),
        "finite": bool(
            np.isfinite(data.close).all()
            and np.isfinite(data.lob).all()
            and np.isfinite(data.macro).all()
        ),
    }


def synthetic_dry_run() -> dict:
    """Execute one real strategy decision on fresh synthetic bars with no broker."""
    import pytz

    import execution.strategy as strategy_module
    from dashboard.data_bridge import DataBridge
    from execution.state_store import ExecutionStateStore
    from execution.strategy import EquityDeepScalper

    original_mode = strategy_module.RUN_MODE
    original_volatility_sizing = strategy_module.USE_VOLATILITY_SIZING
    strategy_module.RUN_MODE = "dry-run"
    strategy_module.USE_VOLATILITY_SIZING = False
    try:
        strategy = EquityDeepScalper.__new__(EquityDeepScalper)
        strategy._data_bridge = DataBridge()
        strategy._circuit_breaker = None
        strategy._trade_history = defaultdict(lambda: deque(maxlen=50))
        strategy._entry_prices = {}
        strategy._entry_side = {}
        strategy._stop_prices = {}
        strategy._tp_prices = {}
        strategy._peak_prices = {}
        strategy._trough_prices = {}
        strategy._filled_entry_qty = {}
        strategy._private_history = defaultdict(lambda: deque(maxlen=LOOKBACK_BARS))
        strategy._iteration_index = 1
        strategy._entry_iteration = {}
        strategy._last_exit_iteration = {}
        strategy._exit_iteration = {}
        strategy._order_state = {}
        strategy._last_bar_timestamp = {}
        strategy._entries_enabled = True
        strategy._shutdown_requested = False
        strategy._calendar_cache = {}
        strategy._bar_source = None
        strategy._tz = pytz.timezone(MARKET_TIMEZONE)
        strategy._state_store = ExecutionStateStore(
            Path(".dry-run-state.json")
        )

        class LongModel:
            def __call__(self, lob, private, macro):
                return torch.tensor([[0.0, 0.0, 2.0]]), torch.zeros((1, 1))

        strategy._models = {TRADING_UNIVERSE[0]: LongModel()}
        end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
        index = pd.date_range(end=end, periods=LOOKBACK_BARS + 5, freq="1min")
        close = np.linspace(19.5, 20.0, len(index))
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.full(len(index), 100.0),
            },
            index=index,
        )
        strategy.get_historical_prices = lambda *args, **kwargs: SimpleNamespace(df=bars)
        strategy.get_positions = lambda: []
        strategy.get_position = lambda asset: None
        strategy.get_orders = lambda: []
        strategy.get_cash = lambda: 1_000.0
        strategy.get_last_price = lambda asset: 20.0
        created = []

        def create_order(asset, quantity, side, **kwargs):
            created.append((asset.symbol, float(quantity), side))
            return SimpleNamespace(
                asset=asset,
                quantity=quantity,
                side=side,
                custom_params={},
            )

        strategy.create_order = create_order
        strategy.submit_order = lambda order: (_ for _ in ()).throw(
            AssertionError("dry-run crossed the broker boundary")
        )
        strategy.broker = SimpleNamespace(api=None)
        strategy._process_symbol(
            TRADING_UNIVERSE[0], 1_000.0, {}, entry_allowed=True
        )
        return {
            "decision_loop_executed": True,
            "orders_created": len(created),
            "broker_submissions": 0,
            "signals": len(strategy._data_bridge.get_all_signals()),
        }
    finally:
        strategy_module.RUN_MODE = original_mode
        strategy_module.USE_VOLATILITY_SIZING = original_volatility_sizing
        Path(".dry-run-state.json").unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("offline")
    subparsers.add_parser("preflight")
    subparsers.add_parser("dry-run")
    setup = subparsers.add_parser(
        "setup", help="prepare a local or Colab training workspace without credentials"
    )
    setup.add_argument("--location", choices=("local", "colab"), required=True)
    setup.add_argument("--output-dir", type=Path, default=Path("runs/AAPL"))
    setup.add_argument("--symbol", default=TRADING_UNIVERSE[0])
    setup.add_argument("--data", type=Path)
    setup.add_argument("--device", default="cpu")

    train = subparsers.add_parser(
        "train", help="run the shared training CLI for the selected location"
    )
    train.add_argument("--location", choices=("local", "colab"), required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, default=Path("runs/AAPL"))
    train.add_argument("--symbol", default=TRADING_UNIVERSE[0])
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--resume", action="store_true")
    smoke = subparsers.add_parser("paper-smoke")
    smoke.add_argument("--confirm-paper-smoke", action="store_true")
    smoke.add_argument("--symbol", default=TRADING_UNIVERSE[0])
    smoke.add_argument("--notional", type=Decimal, default=Decimal("20"))
    smoke.add_argument("--timeout-seconds", type=float, default=120.0)
    smoke.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        help="separate budget for flattening the test position (defaults to --timeout-seconds)",
    )
    smoke.add_argument("--max-orders", type=int, default=2)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "setup":
        from colab.deepscalper.training import resolve_device

        args.output_dir.mkdir(parents=True, exist_ok=True)
        device, device_note = resolve_device(args.device)
        result = training_launch_plan(
            args.location,
            data=args.data,
            output_dir=args.output_dir.resolve(),
            symbol=args.symbol,
            device=device,
        )
        result["output_dir"] = str(args.output_dir.resolve())
        result["device"] = device
        result["device_note"] = device_note
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        plan = training_launch_plan(
            args.location,
            data=args.data,
            output_dir=args.output_dir,
            symbol=args.symbol,
            epochs=args.epochs,
            device=args.device,
            resume=args.resume,
        )
        if args.location == "colab":
            # Never pretend to start remote compute the user has not authorized.
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        from colab.deepscalper.training import main as training_main

        return training_main(plan["argv"])
    if args.command == "offline":
        result = offline_validation()
    elif args.command == "dry-run":
        result = synthetic_dry_run()
    elif args.command == "preflight":
        from execution.preflight import run_read_only_preflight

        result = run_read_only_preflight()
        mandatory_ready = (
            result["account"]["status"].lower().endswith("active")
            and not result["account"]["trading_blocked"]
            and not result["account"]["account_blocked"]
            and all(asset["eligible"] for asset in result["assets"].values())
            and all(bar["valid"] for bar in result["bars"].values())
        )
        result["connectivity_ready"] = mandatory_ready
    else:
        if not args.confirm_paper_smoke:
            raise RuntimeError("Paper smoke requires --confirm-paper-smoke")
        from alpaca.trading.client import TradingClient
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
        from execution.paper_smoke import run_paper_round_trip

        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("Paper credentials are required for paper smoke")
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        result = run_paper_round_trip(
            client,
            symbol=args.symbol,
            notional=args.notional,
            timeout_seconds=args.timeout_seconds,
            cleanup_timeout_seconds=args.cleanup_timeout_seconds,
            max_orders=args.max_orders,
            confirmed=True,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.command == "preflight" and not result["connectivity_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
