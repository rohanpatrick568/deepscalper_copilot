"""Safe operational entry points for the paper-only workflow."""

from __future__ import annotations

import argparse
import os

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, DATA_FEED, TRADING_UNIVERSE, WEIGHTS_DIR
from execution.validation import validate_startup_configuration


def read_only_preflight() -> None:
    """Validate local artifacts and inspect Alpaca without submitting orders."""
    validate_startup_configuration()
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("Paper credentials are required for read-only preflight")
    from alpaca.trading.client import TradingClient

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    account = client.get_account()
    positions = client.get_all_positions()
    orders = client.get_orders()
    clock = client.get_clock()
    print(
        f"paper={account.account_number} status={account.status} "
        f"buying_power={account.buying_power} positions={len(positions)} "
        f"open_orders={len(orders)} market_open={clock.is_open} feed={DATA_FEED}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepScalper safe workflow commands")
    parser.add_argument("command", choices=("offline", "preflight", "dry-run"))
    args = parser.parse_args()
    if args.command == "preflight":
        read_only_preflight()
    else:
        validate_startup_configuration()
        os.environ["ALGO_TRADER_RUN_MODE"] = args.command
        print(f"{args.command}: validated {', '.join(TRADING_UNIVERSE)} using {WEIGHTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
