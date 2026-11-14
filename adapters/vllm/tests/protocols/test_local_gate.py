"""The local gate's own two steps that need no tree of infra's (`local_gate`): the start of a session's unit, and the
clock its hooks read. The unit itself, and every verb, run infra's code from a pinned tree and are the rehearsal's."""
from __future__ import annotations

import io
import subprocess
import sys
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.protocols import local_gate as LG


class Refused(Exception):
    """The gate's refusal, as `auditor.Refused` is to the hook."""


def test_the_start_of_a_unit_returns_once_it_says_running_and_is_the_gates_refusal_when_it_does_not(tmp_path, monkeypatch):
    """`local_gate._start`: the unit is this script's ``run`` of the session in a session of its own, its stderr the
    session's log under ``units/``. The start returns when the unit's first line is ``running`` and raises the gate's
    refusal, naming the session and its log, when the unit ends without it."""
    first = [b"running\n", b""]
    started: list[dict[str, Any]] = []

    def popen(argv: list[str], **given: Any) -> Any:
        started.append({"argv": argv, "log": given["stderr"].name, "mode": given["stderr"].mode,
                        **{k: v for k, v in given.items() if k != "stderr"}})
        return SimpleNamespace(stdout=io.BytesIO(first.pop(0)))

    monkeypatch.setattr(LG.subprocess, "Popen", popen)
    start = LG._start(tmp_path, SimpleNamespace(Refused=Refused))
    assert start("s1") is None
    assert started == [{"argv": [sys.executable, "-I", str(Path(LG.__file__).resolve()), str(tmp_path), "run", "s1"],
                        "log": str(tmp_path / "units" / "s1.log"), "mode": "ab", "stdout": subprocess.PIPE,
                        "stdin": subprocess.DEVNULL, "start_new_session": True}]
    with pytest.raises(Refused, match=r"^session s2 was admitted but its unit didn't start \(see units/s2\.log\)"):
        start("s2")
    assert [s["argv"][-1] for s in started] == ["s1", "s2"] and (tmp_path / "units" / "s2.log").exists()


def test_the_gates_clock_is_asked_for_utc(monkeypatch):
    """`local_gate._now`: what the gate's hooks stamp a pin, a reservation and an admission with is the clock's time
    in UTC. The clock here is a stand-in that says which zone it was asked for."""
    asked: list[Any] = []

    class Clock:
        @staticmethod
        def now(zone: Any) -> str:
            asked.append(zone)
            return "then"

    monkeypatch.setattr(LG, "datetime", Clock)
    assert LG._now() == "then" and asked == [UTC]
