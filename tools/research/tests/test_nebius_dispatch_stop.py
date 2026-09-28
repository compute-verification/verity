"""Node 1's dispatcher under its unit: `systemctl stop` (KillMode=mixed) sends SIGTERM to the loop alone, which lets the tick in
flight finish and then ends, or ends at once between ticks, so an install's restart never cuts a tick short."""
from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path

import pytest
from research.pods import registry

NEBIUS = Path(registry.__file__).resolve().parent / "nebius"


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path / "dispatch"))
    monkeypatch.setenv("VY_READY", str(tmp_path / "ready"))
    spec = importlib.util.spec_from_file_location("dispatch_stop", NEBIUS / "dispatch.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "dispatch_stop", mod)
    spec.loader.exec_module(mod)
    return mod


def term() -> None:
    """SIGTERM's delivery, at this point exactly: the handler the loop installed."""
    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)


def test_sigterm_during_a_tick_lets_it_finish_and_ends_the_loop_without_another(dispatch, capsys):
    before = signal.getsignal(signal.SIGTERM)
    rcs, ticks, sleeps = [1, dispatch.TICK_FAILED, 0], [], []

    def tick() -> int:
        n = len(ticks)
        ticks.append(n)
        if n == 2:
            term()
            ticks.append("finished")
        return rcs[n]

    assert dispatch.run_loop(tick, 60, sleeps.append) == 0
    assert ticks == [0, 1, 2, "finished"] and sleeps == [60, 60]
    assert capsys.readouterr().err.count("tick failed: exit") == 1             # TICK_FAILED's tick said so itself
    assert signal.getsignal(signal.SIGTERM) is before


def test_sigterm_between_ticks_ends_the_loop_at_once(dispatch):
    before = signal.getsignal(signal.SIGTERM)
    ticks, slept = [], []

    def sleep(s: float) -> None:
        term()
        slept.append(s)

    assert dispatch.run_loop(lambda: ticks.append(1) or 0, 60, sleep) == 0
    assert ticks == [1] and slept == []
    assert signal.getsignal(signal.SIGTERM) is before
