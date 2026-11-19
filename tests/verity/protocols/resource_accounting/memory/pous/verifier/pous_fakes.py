"""The suite's one injected clock (AGENTS.md "Tests"): what a memory attester, a wipe's parties and a timed audit read in
place of their default clock (`time.perf_counter_ns`) and of the raw clock a deployed one reads, which `conftest.py`
fails a test for leaving them on."""

from __future__ import annotations


class Clock:
    """Nanoseconds from `now`, each reading `step` after the last; at `step = 0` time moves only when a test sets `now`, as
    a responder spends it."""

    def __init__(self, now: int = 0, step: int = 0) -> None:
        self.now, self.step = now, step

    def __call__(self) -> int:
        self.now += self.step
        return self.now
