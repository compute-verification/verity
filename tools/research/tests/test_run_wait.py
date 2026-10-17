"""`research run --on ... --wait`: `run --on` returns once the machine has started the run, so a loop of launches never serialized
them (a compute-accounting worker held 5 GPUs against a 2-GPU budget, INF-36). `--wait` fetches the run's records until it ends and
exits with its rc; an unreachable machine is waited out."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.cli import wait_remote


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


@pytest.mark.parametrize("end, rc", [
    (lambda d: _status(d, "done", 0), 0),
    (lambda d: _status(d, "failed", 2), 2),
    (lambda d: _status(d, "failed", 0), 1),
    (lambda d: (d / "refused.json").write_text("{}"), 3),
], ids=["done", "failed", "failed-without-rc", "refused"])
def test_wait_returns_the_runs_own_exit_once_it_ends(tmp_path, capsys, end, rc):
    rd = tmp_path / "r1"
    rd.mkdir()
    slept: list[float] = []
    first = None if rc == 3 else (lambda d: _status(d, "running", None))   # a refused run never writes status.json
    steps = [first, "unreachable", None, end]
    assert wait_remote("r1", rd, every_s=7, fetch=_waiter(rd, steps), sleep=slept.append) == rc
    assert slept == [7, 7, 7], "one fetch per interval until the run ends, and none after"
    err = capsys.readouterr().err
    assert "machine unreachable (rc 255); still waiting" in err and "research: run r1 " in err
