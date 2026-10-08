"""Every memory challenger in this suite runs on an injected clock (AGENTS.md "Tests"). `tests/test_no_wall_clock.py`
scans for clock reads in test source, and cannot see an `audit.UnsignedRounds`, `ContinuousMemoryChallenger` or
`memory_challenger.MemoryChallenger` left on its default `time.perf_counter_ns`; such a test passes or fails with the load on
`check`'s slot. Here the default clock fails the test that reads it, every time."""

from __future__ import annotations

import inspect

import pytest

from verity.core.protocols.pous import audit, memory_challenger
from verity.core.protocols.pous.continuous import ContinuousMemoryChallenger


def _wall_clock() -> int:
    raise AssertionError("a memory challenger read its default wall clock: pass clock=, e.g. itertools.count(0, 1000).__next__")


@pytest.fixture(autouse=True)
def _no_default_clock(monkeypatch):
    for cls in (audit.UnsignedRounds, ContinuousMemoryChallenger, memory_challenger.MemoryChallenger):
        init = cls.__init__
        if inspect.signature(init).parameters["clock"].kind is inspect.Parameter.KEYWORD_ONLY:
            monkeypatch.setattr(init, "__kwdefaults__", {**init.__kwdefaults__, "clock": _wall_clock})
            continue
        defaulted = [p.name for p in inspect.signature(init).parameters.values()
                     if p.kind is p.POSITIONAL_OR_KEYWORD and p.default is not p.empty]
        defaults = list(init.__defaults__)
        defaults[defaulted.index("clock")] = _wall_clock
        monkeypatch.setattr(init, "__defaults__", tuple(defaults))
