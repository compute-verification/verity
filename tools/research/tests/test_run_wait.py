"""`research run --on ... --wait`: `run --on` returns once the machine has started the run, so a loop of launches never serialized
them (a compute-accounting worker held 5 GPUs against a 2-GPU budget, INF-36). `--wait` fetches the run's records until it ends and
exits with its rc; an unreachable machine is waited out, for at most WAIT_CAP_S (no tool call may block more than 20
minutes)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research import remote, waitcap
from research.cli import cmd_run, wait_remote


def _status(rd: Path, state: str, rc: int | None) -> None:
    (rd / "status.json").write_text(json.dumps({"state": state, "rc": rc, "transitions": [{"state": "running"}, {"state": state}]}))


def _waiter(rd: Path, steps: list):
    """A fetch that applies one step to the run dir per call (None: nothing new; "unreachable": the machine didn't answer)."""
    calls = iter(steps)

    def fetch(d: Path) -> dict:
        step = next(calls)
        if step == "unreachable":
            return {"last_fetch_error": {"rc": 255}}
        if step is not None:
            step(d)
        return {}
    return fetch


class Clock:
    """Simulated time: it moves only by the sleeps the wait takes and the time each fetch is given to take."""

    def __init__(self, fetch_s: float = 0.0):
        self.now, self.fetch_s, self.slept = 0.0, fetch_s, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s

    def fetching(self, fetch):
        def timed(d: Path) -> dict:
            self.now += self.fetch_s
            return fetch(d)
        return timed


@pytest.mark.parametrize("end, rc", [
    (lambda d: _status(d, "done", 0), 0),
    (lambda d: _status(d, "failed", 2), 2),
    (lambda d: _status(d, "failed", 0), 1),
    (lambda d: (d / "refused.json").write_text("{}"), 3),
], ids=["done", "failed", "failed-without-rc", "refused"])
def test_wait_returns_the_runs_own_exit_once_it_ends(tmp_path, capsys, end, rc):
    rd = tmp_path / "r1"
    rd.mkdir()
    clock = Clock()
    first = None if rc == 3 else (lambda d: _status(d, "running", None))   # a refused run never writes status.json
    steps = [first, "unreachable", None, end]
    assert wait_remote("r1", rd, every_s=7, fetch=_waiter(rd, steps), sleep=clock.sleep, clock=clock) == rc
    assert clock.slept == [7, 7, 7], "one fetch per interval until the run ends, and none after"
    err = capsys.readouterr().err
    assert "machine unreachable (rc 255); still waiting" in err and "research: run r1 " in err


@pytest.mark.parametrize("max_s, fetch_s", [(waitcap.WAIT_CAP_S, 0.0), (waitcap.WAIT_CAP_S, 45.0), (300.0, 0.0)],
                         ids=["cap", "slow-fetches", "wait-max"])
def test_wait_stops_at_its_cap_with_the_run_still_going_and_says_how_to_check_back(tmp_path, capsys, max_s, fetch_s):
    """Agents that blocked in one long `--wait` looked silent and missed their wake-ups. The wait ends within its
    bound however long each fetch takes, cancels nothing, and exits STILL_GOING, which no other outcome of the wait gives."""
    rd = tmp_path / "r1"
    rd.mkdir()
    _status(rd, "running", None)
    clock = Clock(fetch_s)
    fetch = clock.fetching(lambda d: {})
    assert wait_remote("r1", rd, every_s=30, max_s=max_s, fetch=fetch, sleep=clock.sleep, clock=clock) == waitcap.STILL_GOING
    assert max_s <= clock.now <= max_s + fetch_s, "it stops at the first fetch at or past its bound, never a sleep past it"
    assert waitcap.STILL_GOING not in (0, 1, 3, 75), "75 is slot.py's and gpu-lease's never-started, which a run's rc passes on"
    err = capsys.readouterr().err
    assert "research: run r1 (running) is still going" in err and "Nothing was cancelled" in err
    assert "Check back with `research status r1`" in err and "End your turn now; the run's end will ring your lane." in err
    assert json.loads((rd / "status.json").read_text())["state"] == "running"


def test_the_default_fetch_gets_the_time_left_under_the_cap(tmp_path, monkeypatch):
    """`remote.fetch` may take its ssh's own 900 s; under `--wait` it gets what is left of the cap instead."""
    rd = tmp_path / "r1"
    rd.mkdir()
    _status(rd, "running", None)
    clock = Clock()
    given: list[float] = []
    monkeypatch.setattr(remote, "fetch", lambda d, timeout=None: given.append(timeout) or {})
    assert wait_remote("r1", rd, every_s=500, max_s=1200, sleep=clock.sleep, clock=clock) == waitcap.STILL_GOING
    assert given == [1200, 700, 200, 0] and clock.slept == [500, 500, 200]


@pytest.mark.parametrize("flags, says", [
    (["--wait", "--wait-max", "1201"], "--wait-max 1201: a wait may last 0 to 1200 s"),
    (["--wait", "--wait-max", "0"], "--wait-max 0: a wait may last 0 to 1200 s"),
    (["--wait-max", "600"], "--wait-max is for --wait"),
])
def test_wait_max_may_lower_the_cap_and_never_raise_it(capsys, flags, says):
    """Refused before anything launches."""
    assert cmd_run(["--on", "no-such-machine", "--project", "p", *flags, "--", "true"]) == 2
    assert says in capsys.readouterr().err
    assert waitcap.max_s(None) == waitcap.WAIT_CAP_S == 1200 and waitcap.max_s(600) == 600
