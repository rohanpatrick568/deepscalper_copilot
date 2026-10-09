from __future__ import annotations

from datetime import datetime, time, timezone
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

import execution.preflight as preflight_module
from execution.paper_smoke import run_paper_round_trip
from execution.preflight import run_read_only_preflight
from workflow import offline_validation, synthetic_dry_run
from dashboard.data_bridge import DataBridge
from main import EngineController


class FakePaperClient:
    _base_url = "https://paper-api.alpaca.markets"

    def __init__(self):
        self.positions = [
            SimpleNamespace(symbol="MSFT", qty="3"),
        ]
        self.orders = {}
        self.submitted = []
        self.cancelled = []

    def get_asset(self, symbol):
        return SimpleNamespace(
            symbol=symbol,
            tradable=True,
            fractionable=True,
            shortable=True,
            status="active",
        )

    def get_clock(self):
        return SimpleNamespace(is_open=True)

    def get_all_positions(self):
        return list(self.positions)

    def get_orders(self, request=None):
        return [order for order in self.orders.values() if order.status == "new"]

    def submit_order(self, request):
        fields = request.to_request_fields()
        order = SimpleNamespace(
            id=f"order-{len(self.submitted) + 1}",
            symbol=fields["symbol"],
            side=fields["side"],
            status="filled",
            client_order_id=fields.get("client_order_id"),
        )
        self.submitted.append(fields)
        self.orders[order.id] = order
        if str(fields["side"]).lower().endswith("buy"):
            self.positions.append(
                SimpleNamespace(symbol=fields["symbol"], qty="0.1")
            )
        else:
            self.positions = [
                position
                for position in self.positions
                if position.symbol != fields["symbol"]
            ]
        return order

    def get_order_by_id(self, order_id):
        return self.orders[order_id]

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(order_id)
        self.orders[order_id].status = "canceled"


def test_offline_and_dry_run_need_no_credentials_or_network():
    assert offline_validation()["network_used"] is False
    result = synthetic_dry_run()
    assert result == {
        "decision_loop_executed": True,
        "orders_created": 1,
        "broker_submissions": 0,
        "signals": 1,
    }


def test_setup_is_reachable_without_credentials(tmp_path, capsys):
    from workflow import main

    assert main(["setup", "--location", "local", "--output-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert '"credentials_required": false' in output


def test_bounded_paper_smoke_reconciles_only_its_symbol():
    client = FakePaperClient()
    result = run_paper_round_trip(
        client,
        symbol="AAPL",
        notional=Decimal("20"),
        timeout_seconds=1,
        confirmed=True,
        sleep_fn=lambda _: None,
    )
    assert result["flat_confirmed"]
    assert len(client.submitted) == 2
    assert [(position.symbol, position.qty) for position in client.positions] == [
        ("MSFT", "3")
    ]
    assert client.cancelled == []


def test_paper_smoke_requires_explicit_confirmation_and_clean_symbol():
    client = FakePaperClient()
    with pytest.raises(RuntimeError, match="explicit"):
        run_paper_round_trip(
            client,
            symbol="AAPL",
            notional=Decimal("20"),
            timeout_seconds=1,
        )
    client.positions.append(SimpleNamespace(symbol="AAPL", qty="1"))
    with pytest.raises(RuntimeError, match="clean AAPL"):
        run_paper_round_trip(
            client,
            symbol="AAPL",
            notional=Decimal("20"),
            timeout_seconds=1,
            confirmed=True,
        )


def test_read_only_preflight_checks_account_calendar_asset_and_fresh_bars(
    monkeypatch,
):
    monkeypatch.setattr(preflight_module, "ALPACA_API_KEY", "key")
    monkeypatch.setattr(preflight_module, "ALPACA_SECRET_KEY", "secret")

    class Client(FakePaperClient):
        def get_account(self):
            return SimpleNamespace(
                status="active",
                buying_power="1000",
                trading_blocked=False,
                account_blocked=False,
            )

        def get_clock(self):
            now = datetime.now(timezone.utc)
            return SimpleNamespace(
                is_open=True,
                timestamp=now,
                next_open=now,
                next_close=now,
            )

        def get_calendar(self, request):
            return [
                SimpleNamespace(
                    date=datetime.now().date(), open=time(9, 30), close=time(16, 0)
                )
            ]

    now = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    index = pd.date_range(end=now, periods=65, freq="1min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
        },
        index=index,
    )
    source = SimpleNamespace(completed_minute_bars=lambda symbol, limit: bars)
    result = run_read_only_preflight(
        trading_client=Client(), bar_source=source
    )
    assert result["account"]["status"] == "active"
    assert result["calendar"]
    assert result["assets"]["AAPL"]["eligible"]
    assert result["bars"]["AAPL"]["valid"]
    assert result["model_readiness"]["AAPL"]["ready"] is False


def test_engine_controller_surfaces_failure_and_requires_confirmed_shutdown():
    bridge = DataBridge()
    controller = EngineController(bridge)
    controller.fail(RuntimeError("engine stopped"))
    assert "engine stopped" in bridge.engine_error
    assert controller.shutdown(timeout_seconds=0) is False

    strategy = SimpleNamespace(
        request_close_position=lambda symbol: symbol == "AAPL",
        graceful_shutdown=lambda timeout: True,
        stop_live_engine=lambda: None,
    )
    controller.attach(strategy)
    assert controller.close_position("AAPL")
    assert controller.shutdown(timeout_seconds=1)
    assert controller.shutdown_confirmed is True
