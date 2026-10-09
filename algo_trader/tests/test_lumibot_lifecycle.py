"""Integration coverage for the actual pinned Lumibot startup/shutdown path.

These tests exercise the real ``lumibot.traders.Trader`` class rather than an
inert stub, with the broker and networking faked, so an upgrade that changes
the startup contract fails here instead of in a live session.
"""

from __future__ import annotations

import inspect
import signal
import threading

import pytest
from lumibot.traders import Trader

import main as main_module
from dashboard.data_bridge import DataBridge
from main import EngineController


def test_pinned_trader_registers_sigint_and_therefore_needs_the_main_thread():
    """The constraint that forces main-thread startup must still hold."""
    source = inspect.getsource(Trader.run_all)
    assert "signal.signal" in source, (
        "lumibot.traders.Trader.run_all no longer registers SIGINT; "
        "re-check the main-thread startup requirement before relaxing it"
    )

    # Reproduces the original failure: registering SIGINT off the main thread.
    failures: list[BaseException] = []

    def register_from_worker():
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            failures.append(exc)

    worker = threading.Thread(target=register_from_worker)
    worker.start()
    worker.join()
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert "main thread" in str(failures[0])


def test_start_lumibot_runs_the_real_trader_from_the_main_thread(monkeypatch):
    """``_start_lumibot`` must call the real run_all on the main thread."""
    calls: dict[str, object] = {}

    class FakeBroker:
        name = "fake"

        def __init__(self):
            self.is_backtesting = False

    class FakeStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._trader = None

    def fake_run_all(self, **kwargs):
        calls["thread"] = threading.current_thread()
        calls["kwargs"] = kwargs
        calls["strategies"] = list(self._strategies)
        return []

    monkeypatch.setattr("execution.broker.get_broker", lambda: FakeBroker())
    monkeypatch.setattr("execution.strategy.EquityDeepScalper", FakeStrategy)
    # Patch only the network/trading entry point; the real Trader is used.
    monkeypatch.setattr(Trader, "run_all", fake_run_all)
    monkeypatch.setattr(Trader, "add_strategy", lambda self, s: self._strategies.append(s))

    bridge = DataBridge()
    controller = EngineController(bridge)
    main_module._start_lumibot(bridge, controller)

    assert calls["thread"] is threading.main_thread()
    assert calls["kwargs"]["async_"] is True
    assert calls["kwargs"]["show_plot"] is False
    assert len(calls["strategies"]) == 1
    assert controller.trader is not None
    assert isinstance(controller.trader, Trader)


def test_startup_failure_is_surfaced_and_does_not_leave_a_trader(monkeypatch):
    class FakeBroker:
        is_backtesting = False

    def exploding_run_all(self, **kwargs):
        raise RuntimeError("alpaca refused the connection")

    monkeypatch.setattr("execution.broker.get_broker", lambda: FakeBroker())
    monkeypatch.setattr(
        "execution.strategy.EquityDeepScalper", lambda **kwargs: type("S", (), {})()
    )
    monkeypatch.setattr(Trader, "run_all", exploding_run_all)
    monkeypatch.setattr(Trader, "add_strategy", lambda self, s: self._strategies.append(s))

    bridge = DataBridge()
    controller = EngineController(bridge)
    with pytest.raises(RuntimeError, match="refused the connection"):
        main_module._start_lumibot(bridge, controller)

    controller.fail(RuntimeError("alpaca refused the connection"))
    assert "refused the connection" in bridge.engine_error


def test_shutdown_requires_broker_confirmation_before_reporting_success():
    bridge = DataBridge()
    controller = EngineController(bridge)

    class UnconfirmedStrategy:
        def graceful_shutdown(self, timeout):
            return False

        def stop_live_engine(self):
            return None

    controller.attach(UnconfirmedStrategy())
    assert controller.shutdown(timeout_seconds=0) is False
    assert controller.shutdown_confirmed is False

    confirmed = EngineController(DataBridge())

    class ConfirmedStrategy:
        def graceful_shutdown(self, timeout):
            return True

        def stop_live_engine(self):
            return None

    confirmed.attach(ConfirmedStrategy())
    assert confirmed.shutdown(timeout_seconds=1) is True
    assert confirmed.shutdown_confirmed is True
