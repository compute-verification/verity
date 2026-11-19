"""The suite's fakes: its one injected clock (AGENTS.md "Tests"): what a memory attester, a wipe's parties and a timed audit read in
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


class Server:
    """A listening socket for a side's `main`: it accepts one connection, itself, whose file is itself."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getsockname(self):
        return ("127.0.0.1", 1)

    def accept(self):
        return self, None

    def makefile(self, *_):
        return self

    def setblocking(self, *_):
        pass

    def setsockopt(self, *_):
        pass
