"""Waiting without a clock (tests/test_no_wall_clock.py at the repository root): on a process's exit, on a gate the test
holds, or on a fake clock that code under test sleeps on."""
from __future__ import annotations

import json
import os
import select
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable


def wait_exit(pid: int) -> None:
    """Return once `pid` has exited. A child of this process (a runner `remote.launch_request` started) is reaped; any other
    (a runner a `research launch-request` subprocess left behind) is waited on through a pidfd, which is readable from its exit
    on, reaped or not. A pid that is already gone has exited."""
    try:
        os.waitpid(pid, 0)
        return
    except ChildProcessError:           # not a child, or a child subprocess's cleanup of a dropped Popen already reaped
        pass
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        select.select([fd], [], [])
    finally:
        os.close(fd)


def wait_runner(run_dir: Path) -> None:
    """Return once the runner that `launch-request` started for `run_dir` (launch.json's runner_pid) has exited; it writes the
    run's terminal status and releases its locks and reservations before it does."""
    wait_exit(json.loads((run_dir / "launch.json").read_text())["runner_pid"])


class Gate:
    """A FIFO the test holds open for reading and writing from the start, so no open of it ever waits and a read waits for
    data, never sees EOF. Either a workload blocks on it, running `open(path, 'rb').read(1)` (``Gate.WAIT``), until
    `release()`, whether it got there first or not; or the code under test writes a line to it at each step
    (`echo tick >> path`) and `readline()` waits for the next one. `close()` once the other side is done."""

    WAIT = "import sys; open(sys.argv[1], 'rb').read(1)"

    def __init__(self, path: Path):
        os.mkfifo(path)
        self.path = path
        self._fd: int | None = os.open(path, os.O_RDWR)

    def release(self) -> None:
        os.write(self._fd, b"x")

    def readline(self) -> bytes:
        line = b""
        while not line.endswith(b"\n"):
            line += os.read(self._fd, 1)
        return line

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class FakeClock:
    """Stands in for a module's `time` (``monkeypatch.setattr(mod, "time", clock.module)``): `sleep` advances `now` and runs
    whatever `at(t, fn)` scheduled up to then; `strftime` and `gmtime` stay real."""

    def __init__(self, now: float = 0.0):
        self.now = now
        self._due: list[tuple[float, Callable[[], object]]] = []
        self.module = SimpleNamespace(monotonic=lambda: self.now, time=lambda: self.now, sleep=self.sleep,
                                      strftime=time.strftime, gmtime=time.gmtime)

    def at(self, t: float, fn: Callable[[], object]) -> None:
        self._due.append((t, fn))

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        due = [f for t, f in self._due if t <= self.now]
        self._due = [(t, f) for t, f in self._due if t > self.now]
        for f in due:
            f()
