from __future__ import annotations

from datetime import datetime, time
from types import SimpleNamespace

import pandas as pd
import pytest

import execution.strategy as strategy_module
from dashboard.data_bridge import DataBridge
from execution.broker import BrokerMutationBlocked, GuardedAlpaca
from execution.market_data import AlpacaBarSource
from execution.state_store import ExecutionStateStore
from execution.strategy import EquityDeepScalper


class FakeOrder:
    def __init__(self, asset, quantity, side):
        self.asset = asset
        self.quantity = quantity
        self.side = side
        self.status = "new"
        self.identifier = f"broker-{side}-{quantity}"
        self.custom_params = {}
        self.error_message = None


class RecordingRuntime:
    def __init__(self, positions=()):
        self.positions = list(positions)
        self.orders = []
        self.submissions = []
        self.created = []
        self.cash = 10_000.0

    def create_order(self, asset, quantity, side, **kwargs):
        order = FakeOrder(asset, quantity, side)
        order.creation_kwargs = kwargs
        self.created.append(order)
        return order

    def submit_order(self, order):
        self.submissions.append(order)
        self.orders.append(order)
        return order

    def get_position(self, asset):
        return next(
            (position for position in self.positions if position.asset.symbol == asset.symbol),
            None,
        )


def _position(strategy, symbol="AAPL", quantity=5, avg_fill_price=100.0):
    return SimpleNamespace(
        asset=strategy._get_equity_asset(symbol),
        quantity=quantity,
        avg_fill_price=avg_fill_price,
    )


def _strategy(tmp_path, monkeypatch, mode="paper", quantity=5):
    monkeypatch.setattr(strategy_module, "RUN_MODE", mode)
    instance = EquityDeepScalper.__new__(EquityDeepScalper)
    runtime = RecordingRuntime()
    instance._data_bridge = DataBridge()
    instance._models = {}
    instance._circuit_breaker = None
    instance._trade_history = strategy_module.defaultdict(
        lambda: strategy_module.deque(maxlen=50)
    )
    instance._entry_prices = {}
    instance._entry_side = {}
    instance._stop_prices = {}
    instance._tp_prices = {}
    instance._peak_prices = {}
    instance._trough_prices = {}
    instance._filled_entry_qty = {}
    instance._private_history = strategy_module.defaultdict(
        lambda: strategy_module.deque(maxlen=strategy_module.LOOKBACK_BARS)
    )
    instance._iteration_index = 10
    instance._entry_iteration = {}
    instance._last_exit_iteration = {}
    instance._exit_iteration = {}
    instance._order_state = {}
    instance._last_bar_timestamp = {}
    instance._entries_enabled = True
    instance._shutdown_requested = False
    instance._calendar_cache = {}
    instance._bar_source = None
    instance._tz = strategy_module.pytz.timezone(strategy_module.MARKET_TIMEZONE)
    instance._state_store = ExecutionStateStore(tmp_path / "execution-state.json")
    position = _position(instance, quantity=quantity)
    runtime.positions = [position] if quantity else []
    instance.create_order = runtime.create_order
    instance.submit_order = runtime.submit_order
    instance.get_position = runtime.get_position
    instance.get_positions = lambda: list(runtime.positions)
    instance.get_orders = lambda: list(runtime.orders)
    instance.get_cash = lambda: runtime.cash
    instance.get_last_price = lambda asset: 100.0
    instance.get_portfolio_value = lambda: 1_000.0
    instance.broker = SimpleNamespace(api=None)
    return instance, runtime, position


def test_five_share_position_gets_one_five_share_exit(tmp_path, monkeypatch):
    strategy, runtime, position = _strategy(tmp_path, monkeypatch)
    strategy._entry_side["AAPL"] = "buy"
    strategy._stop_prices["AAPL"] = 95.0
    strategy._tp_prices["AAPL"] = 110.0

    strategy._submit_exit_flat("AAPL", position.asset, 99.0, "RISK_STOP")
    strategy._submit_exit_flat("AAPL", position.asset, 99.0, "MODEL_FLAT")

    assert [(order.side, order.quantity) for order in runtime.submissions] == [
        ("sell", 5)
    ]
    assert strategy._stop_prices["AAPL"] == 95.0
    assert strategy._tp_prices["AAPL"] == 110.0


@pytest.mark.parametrize("mode", ["offline", "preflight", "dry-run"])
def test_no_order_modes_block_eod_risk_manual_and_shutdown(
    tmp_path, monkeypatch, mode
):
    strategy, runtime, position = _strategy(tmp_path, monkeypatch, mode=mode)
    strategy._entry_side["AAPL"] = "buy"
    strategy._stop_prices["AAPL"] = 101.0

    strategy._risk_exit_check(
        "AAPL",
        position.asset,
        pd.DataFrame(
            {"high": [101.0, 101.0], "low": [99.0, 99.0], "close": [100.0, 100.0]}
        ),
        100.0,
        "buy",
    )
    strategy._close_all_positions("EOD_CLOSE")
    assert strategy.request_close_position("AAPL") is False
    assert strategy.graceful_shutdown(timeout_seconds=0) is False
    assert runtime.submissions == []


@pytest.mark.parametrize("mode", ["offline", "preflight", "dry-run"])
def test_broker_boundary_rejects_every_no_order_mode(mode):
    broker = GuardedAlpaca.__new__(GuardedAlpaca)
    broker.run_mode = mode
    broker.is_paper = True
    with pytest.raises(BrokerMutationBlocked):
        broker._require_mutations("submit_order")
    with pytest.raises(BrokerMutationBlocked):
        broker._require_mutations("cancel_order")


def test_actual_partial_fills_replace_estimate_with_weighted_price(
    tmp_path, monkeypatch
):
    strategy, _, position = _strategy(tmp_path, monkeypatch, quantity=5)
    strategy._entry_side["AAPL"] = "buy"
    strategy._order_state["client-1"] = {
        "symbol": "AAPL",
        "intent": "entry",
        "side": "buy",
        "qty": 5,
        "filled_qty": 0,
        "remaining_qty": 5,
        "estimated_price": 100.0,
        "status": "new",
    }
    order = FakeOrder(position.asset, 5, "buy")
    order.custom_params = {"client_order_id": "client-1"}

    strategy.on_partially_filled_order(position, order, 100.0, 2, 1)
    strategy.on_filled_order(position, order, 104.0, 3, 1)

    assert strategy._entry_prices["AAPL"] == pytest.approx(102.4)
    assert strategy._entry_prices["AAPL"] != 100.0


def test_102_fill_replaces_100_submission_estimate(tmp_path, monkeypatch):
    strategy, _, position = _strategy(tmp_path, monkeypatch, quantity=1)
    strategy._entry_side["AAPL"] = "buy"
    strategy._order_state["client-1"] = {
        "symbol": "AAPL",
        "intent": "entry",
        "side": "buy",
        "qty": 1,
        "filled_qty": 0,
        "remaining_qty": 1,
        "estimated_price": 100.0,
        "status": "new",
    }
    order = FakeOrder(position.asset, 1, "buy")
    order.custom_params = {"client_order_id": "client-1"}
    strategy.on_filled_order(position, order, 102.0, 1, 1)
    assert strategy._entry_prices["AAPL"] == 102.0


def test_200_dollar_share_is_skipped_under_30_dollar_order_cap(
    tmp_path, monkeypatch
):
    strategy, runtime, position = _strategy(tmp_path, monkeypatch, quantity=0)
    bars = pd.DataFrame(
        {
            "open": [200.0] * 20,
            "high": [201.0] * 20,
            "low": [199.0] * 20,
            "close": [200.0] * 20,
            "volume": [100.0] * 20,
        }
    )
    strategy._submit_entry(
        "AAPL", position.asset, bars, 200.0, 1_000.0, side="buy"
    )
    assert runtime.created == []
    assert runtime.submissions == []


def test_entry_uses_bracket_protection_and_persisted_client_id(
    tmp_path, monkeypatch
):
    strategy, runtime, position = _strategy(tmp_path, monkeypatch, quantity=0)
    monkeypatch.setattr(strategy_module, "USE_VOLATILITY_SIZING", False)
    bars = pd.DataFrame(
        {
            "open": [20.0] * 20,
            "high": [20.2] * 20,
            "low": [19.8] * 20,
            "close": [20.0] * 20,
            "volume": [100.0] * 20,
        }
    )
    strategy._submit_entry(
        "AAPL", position.asset, bars, 20.0, 1_000.0, side="buy"
    )
    assert len(runtime.submissions) == 1
    order = runtime.submissions[0]
    assert order.creation_kwargs["order_class"] == "bracket"
    assert order.creation_kwargs["secondary_stop_price"] < 20.0
    assert order.creation_kwargs["secondary_limit_price"] > 20.0
    client_id = order.custom_params["client_order_id"]
    assert client_id.startswith("ds-aapl-entry-")
    reloaded = ExecutionStateStore(tmp_path / "execution-state.json")
    assert client_id in reloaded.snapshot()["orders"]


def test_restart_reconstructs_position_and_protection(tmp_path, monkeypatch):
    store = ExecutionStateStore(tmp_path / "execution-state.json")
    store.set_symbol(
        "AAPL",
        {
            "quantity": 5,
            "entry_side": "buy",
            "entry_price": 100.0,
            "filled_entry_qty": 5,
            "stop_price": 95.0,
            "target_price": 110.0,
            "peak_price": 103.0,
            "trough_price": None,
        },
    )
    strategy, _, position = _strategy(tmp_path, monkeypatch)
    position.avg_fill_price = 102.0
    strategy._restore_persisted_state()
    strategy._reconcile_broker_state()

    assert strategy._entry_prices["AAPL"] == 102.0
    assert strategy._stop_prices["AAPL"] == 95.0
    assert strategy._tp_prices["AAPL"] == 110.0
    assert strategy._entries_enabled


def test_rejected_and_canceled_exits_keep_protection_for_late_fills(
    tmp_path, monkeypatch
):
    strategy, runtime, position = _strategy(tmp_path, monkeypatch)
    strategy._entry_side["AAPL"] = "buy"
    strategy._entry_prices["AAPL"] = 100.0
    strategy._filled_entry_qty["AAPL"] = 5
    strategy._stop_prices["AAPL"] = 95.0
    strategy._tp_prices["AAPL"] = 110.0

    def reject(order):
        order.status = "rejected"
        order.error_message = "paper rejection"
        runtime.submissions.append(order)
        return order

    strategy.submit_order = reject
    strategy._submit_exit_flat("AAPL", position.asset, 99.0, "RISK_STOP")
    assert strategy._stop_prices["AAPL"] == 95.0
    assert strategy._tp_prices["AAPL"] == 110.0
    assert not strategy._has_pending_exit("AAPL")

    strategy._iteration_index += 1
    strategy.submit_order = runtime.submit_order
    strategy._submit_exit_flat("AAPL", position.asset, 99.0, "RISK_STOP")
    order = runtime.submissions[-1]
    strategy.on_canceled_order(order)
    assert strategy._stop_prices["AAPL"] == 95.0
    order.status = "partially_filled"
    position.quantity = 3
    strategy.on_partially_filled_order(position, order, 99.0, 2, 1)
    assert strategy._stop_prices["AAPL"] == 95.0
    position.quantity = 0
    strategy.on_filled_order(position, order, 98.0, 3, 1)
    assert "AAPL" not in strategy._stop_prices
    assert "AAPL" not in strategy._tp_prices


def test_uncertain_restart_submission_disables_entries(tmp_path, monkeypatch):
    store = ExecutionStateStore(tmp_path / "execution-state.json")
    store.set_order(
        "ds-aapl-entry-00000001",
        {
            "symbol": "AAPL",
            "intent": "entry",
            "side": "buy",
            "qty": 1,
            "remaining_qty": 1,
            "estimated_price": 20.0,
            "status": "submitting",
        },
    )
    strategy, _, _ = _strategy(tmp_path, monkeypatch, quantity=0)
    strategy._restore_persisted_state()
    strategy._reconcile_broker_state()
    assert strategy._entries_enabled is False


def test_session_loss_includes_unrealized_equity_and_restores_latch(
    tmp_path, monkeypatch
):
    strategy, _, _ = _strategy(tmp_path, monkeypatch, quantity=0)
    from execution.circuit_breakers import CircuitBreaker

    strategy._circuit_breaker = CircuitBreaker(0.03, 1_000.0)
    strategy._state_store.set_session(
        date=datetime.now(strategy._tz).date().isoformat(),
        baseline=1_000.0,
        halted=False,
    )
    strategy._update_session_loss(960.0)
    assert strategy._circuit_breaker.daily_pnl == -40.0
    assert strategy._circuit_breaker.is_halted_by_loss

    recovered, _, _ = _strategy(tmp_path, monkeypatch, quantity=0)
    recovered._circuit_breaker = CircuitBreaker(0.03, 1_000.0)
    recovered._restore_session_loss(980.0)
    assert recovered._circuit_breaker.is_halted_by_loss


def test_exposure_uses_positions_and_pending_orders(tmp_path, monkeypatch):
    strategy, _, _ = _strategy(tmp_path, monkeypatch)
    strategy._order_state["pending"] = {
        "symbol": "AAPL",
        "intent": "entry",
        "status": "new",
        "remaining_qty": 2,
        "estimated_price": 101.0,
    }
    symbol, aggregate, pending = strategy._exposure_snapshot("AAPL", 102.0)
    assert symbol == 712.0
    assert aggregate == 510.0
    assert pending == 202.0


def test_early_close_comes_from_broker_calendar(tmp_path, monkeypatch):
    strategy, _, _ = _strategy(tmp_path, monkeypatch, quantity=0)
    strategy.broker.api = SimpleNamespace(
        get_calendar=lambda request: [SimpleNamespace(close=time(13, 0))]
    )
    now = strategy._tz.localize(datetime(2026, 11, 27, 12, 56))
    assert strategy._is_eod_close_window(now)


def test_loss_halt_still_allows_model_exit(tmp_path, monkeypatch):
    strategy, runtime, position = _strategy(tmp_path, monkeypatch)
    strategy._entry_side["AAPL"] = "buy"
    strategy._entry_prices["AAPL"] = 100.0
    strategy._filled_entry_qty["AAPL"] = 5

    class FlatModel:
        def __call__(self, lob, private, macro):
            return (
                strategy_module.torch.tensor([[0.0, 2.0, 0.0]]),
                strategy_module.torch.zeros((1, 1)),
            )

    strategy._models = {"AAPL": FlatModel()}
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    index = pd.date_range(end=end, periods=65, freq="1min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 100.0,
        },
        index=index,
    )
    strategy.get_historical_prices = lambda *args, **kwargs: SimpleNamespace(df=bars)
    strategy._process_symbol(
        "AAPL", 1_000.0, {"AAPL": position}, entry_allowed=False
    )
    assert [(order.side, order.quantity) for order in runtime.submissions] == [
        ("sell", 5)
    ]


def test_repeated_bar_runs_model_once(tmp_path, monkeypatch):
    strategy, _, _ = _strategy(tmp_path, monkeypatch, quantity=0)

    class CountingFlatModel:
        calls = 0

        def __call__(self, lob, private, macro):
            self.calls += 1
            return (
                strategy_module.torch.tensor([[0.0, 2.0, 0.0]]),
                strategy_module.torch.zeros((1, 1)),
            )

    model = CountingFlatModel()
    strategy._models = {"AAPL": model}
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    index = pd.date_range(end=end, periods=65, freq="1min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 100.0,
        },
        index=index,
    )
    strategy.get_historical_prices = lambda *args, **kwargs: SimpleNamespace(df=bars)
    strategy._process_symbol("AAPL", 1_000.0, {})
    strategy._process_symbol("AAPL", 1_000.0, {})
    assert model.calls == 1


def test_repeated_long_signal_maintains_the_live_position(tmp_path, monkeypatch):
    """Regression: a repeated LONG must not churn an existing long flat."""
    strategy, runtime, position = _strategy(tmp_path, monkeypatch)
    strategy._entry_side["AAPL"] = "buy"
    strategy._entry_prices["AAPL"] = 100.0
    strategy._filled_entry_qty["AAPL"] = 5

    class LongModel:
        def __call__(self, lob, private, macro):
            return (
                strategy_module.torch.tensor([[0.0, 0.0, 6.0]]),
                strategy_module.torch.zeros((1, 1)),
            )

    strategy._models = {"AAPL": LongModel()}
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    for offset in range(3):
        index = pd.date_range(
            end=end - pd.Timedelta(minutes=offset), periods=65, freq="1min"
        )
        bars = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.5,
                "low": 99.8,
                "close": 100.0,
                "volume": 100.0,
            },
            index=index,
        )
        strategy.get_historical_prices = lambda *a, _bars=bars, **k: SimpleNamespace(
            df=_bars
        )
        strategy._iteration_index += 1
        strategy._process_symbol("AAPL", 1_000.0, {"AAPL": position})

    # No exit and no pyramiding entry: the existing long is simply maintained.
    assert runtime.submissions == []


def test_flat_model_exits_the_live_position_once(tmp_path, monkeypatch):
    strategy, runtime, position = _strategy(tmp_path, monkeypatch)
    strategy._entry_side["AAPL"] = "buy"
    strategy._entry_prices["AAPL"] = 100.0
    strategy._filled_entry_qty["AAPL"] = 5

    class FlatModel:
        def __call__(self, lob, private, macro):
            return (
                strategy_module.torch.tensor([[0.0, 6.0, 0.0]]),
                strategy_module.torch.zeros((1, 1)),
            )

    strategy._models = {"AAPL": FlatModel()}
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    index = pd.date_range(end=end, periods=65, freq="1min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.5,
            "low": 99.8,
            "close": 100.0,
            "volume": 100.0,
        },
        index=index,
    )
    strategy.get_historical_prices = lambda *a, **k: SimpleNamespace(df=bars)
    strategy._process_symbol("AAPL", 1_000.0, {"AAPL": position})
    assert [(order.side, order.quantity) for order in runtime.submissions] == [
        ("sell", 5)
    ]


def test_long_only_short_signal_exits_rather_than_shorting(tmp_path, monkeypatch):
    strategy, runtime, position = _strategy(tmp_path, monkeypatch)
    strategy._entry_side["AAPL"] = "buy"
    strategy._entry_prices["AAPL"] = 100.0
    strategy._filled_entry_qty["AAPL"] = 5

    class ShortModel:
        def __call__(self, lob, private, macro):
            return (
                strategy_module.torch.tensor([[6.0, 0.0, 0.0]]),
                strategy_module.torch.zeros((1, 1)),
            )

    strategy._models = {"AAPL": ShortModel()}
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    index = pd.date_range(end=end, periods=65, freq="1min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.5,
            "low": 99.8,
            "close": 100.0,
            "volume": 100.0,
        },
        index=index,
    )
    strategy.get_historical_prices = lambda *a, **k: SimpleNamespace(df=bars)
    strategy._process_symbol("AAPL", 1_000.0, {"AAPL": position})

    sides = [order.side for order in runtime.submissions]
    assert sides == ["sell"]
    # The long is closed; no short is opened in long-only mode.
    assert all(order.quantity == 5 for order in runtime.submissions)


def test_long_only_short_signal_opens_nothing_when_flat(tmp_path, monkeypatch):
    strategy, runtime, _ = _strategy(tmp_path, monkeypatch, quantity=0)

    class ShortModel:
        def __call__(self, lob, private, macro):
            return (
                strategy_module.torch.tensor([[6.0, 0.0, 0.0]]),
                strategy_module.torch.zeros((1, 1)),
            )

    strategy._models = {"AAPL": ShortModel()}
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    index = pd.date_range(end=end, periods=65, freq="1min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.5,
            "low": 99.8,
            "close": 100.0,
            "volume": 100.0,
        },
        index=index,
    )
    strategy.get_historical_prices = lambda *a, **k: SimpleNamespace(df=bars)
    strategy._process_symbol("AAPL", 1_000.0, {})
    assert runtime.submissions == []


def test_shutdown_only_succeeds_after_fake_broker_confirms_flat(
    tmp_path, monkeypatch
):
    strategy, runtime, _ = _strategy(tmp_path, monkeypatch)

    def fill_and_flatten(order):
        order.status = "filled"
        runtime.orders.append(order)
        runtime.submissions.append(order)
        runtime.positions.clear()
        return order

    strategy.submit_order = fill_and_flatten
    strategy.sleep = lambda _: None
    assert strategy.graceful_shutdown(timeout_seconds=1)
    assert len(runtime.submissions) == 1
    assert strategy._shutdown_requested
    assert not strategy._entries_enabled


def test_market_data_request_carries_feed_adjustment_and_excludes_open_bar():
    captured = {}

    class DataClient:
        def get_stock_bars(self, request):
            captured.update(request.to_request_fields())
            index = pd.MultiIndex.from_product(
                [
                    ["AAPL"],
                    pd.to_datetime(
                        ["2026-01-05T14:29:00Z", "2026-01-05T14:30:00Z"]
                    ),
                ],
                names=["symbol", "timestamp"],
            )
            frame = pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.0, 101.0],
                    "volume": [10, 11],
                },
                index=index,
            )
            return SimpleNamespace(df=frame)

    source = AlpacaBarSource(
        "key", "secret", feed="iex", adjustment="raw", client=DataClient()
    )
    bars = source.completed_minute_bars(
        "AAPL", 60, now=datetime.fromisoformat("2026-01-05T14:30:30+00:00")
    )
    assert captured["feed"].value == "iex"
    assert captured["adjustment"].value == "raw"
    assert captured["sort"].value == "desc"
    assert list(bars.index) == [pd.Timestamp("2026-01-05T14:29:00Z")]
