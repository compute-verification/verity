"""Where serve-openai seals its closed windows off the engine loop (`Sealer`): in two ordered stages, a thread each, so that one
window's seal (`serve_seal.seal_window`, or its execution blocks in `seal_process.ProcessSealer`'s process) runs beside the next
window's finalize and release, while the windows queued or sealing hold under a bound of committed bytes."""
from __future__ import annotations

import functools
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any


class Sealer:
    """Seals closed windows off the engine loop in two stages, each a thread of its own taking windows in order: the first runs
    each window's job (its finalize and records when the close detached it, then its release), and the second what the job
    returns when that is callable (its seal and its line in seals.jsonl), so that one window's seal runs beside the next one's
    finalize. `submit` returns at once while the windows queued or sealing hold under ``hold_bytes`` of committed values (the
    pinned host memory the double-buffered windows take), else it waits for room and says how long; `drain` waits for every job
    and its seal. A job's own failure, or its seal's, is its window's: the next one still runs."""
    process: Any = None                                  # the `seal_process.SealProcess` its seals run in (`seal_process.ProcessSealer`)

    def __init__(self, hold_bytes: int, clock: Callable[[], float] = time.perf_counter) -> None:
        self.hold_bytes, self.clock = int(hold_bytes), clock
        self.held = 0                                    # committed bytes of the windows queued or sealing
        self.errors: list[str] = []
        self._cv = threading.Condition()
        self._jobs: deque = deque()
        self._seals: deque = deque()                     # (seal or None, the bytes its window holds, whether it is the last)
        self.thread = threading.Thread(target=self._run, name="serve-seal", daemon=True)
        self.seal_thread = threading.Thread(target=self._run_seals, name="serve-seal-2", daemon=True)
        self.thread.start()
        self.seal_thread.start()

    def submit(self, job: Callable[[float], Any], nbytes: int = 0) -> float:
        """Queue ``job``, which holds ``nbytes`` of a window's values until it and the seal it returns have run, and is called with the
        seconds the caller waited for room; returns them too."""
        t0 = self.clock()
        with self._cv:
            while nbytes and self.held and self.held + nbytes > self.hold_bytes:
                self._cv.wait()
            waited = self.clock() - t0
            self.held += int(nbytes)
            self._jobs.append((functools.partial(job, waited), int(nbytes)))
            self._cv.notify_all()
        return waited

    def _call(self, f: Callable[[], Any]) -> Any:
        try:
            return f()
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"{type(e).__name__}: {e}"[:300])
            return None

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait()
                job, nbytes = self._jobs[0]
            seal = self._call(job) if job is not None else None
            with self._cv:
                self._jobs.popleft()
                self._seals.append((seal if callable(seal) else None, nbytes, job is None))
                self._cv.notify_all()
            if job is None:
                return

    def _run_seals(self) -> None:
        while True:
            with self._cv:
                while not self._seals:
                    self._cv.wait()
                seal, nbytes, last = self._seals[0]
            if seal is not None:
                self._call(seal)
            with self._cv:
                self._seals.popleft()
                self.held -= nbytes
                self._cv.notify_all()
            if last:
                return

    def drain(self) -> None:
        with self._cv:
            while self._jobs or self._seals:
                self._cv.wait()

    def close(self) -> None:
        """Every job and seal run, then both threads ended."""
        with self._cv:
            self._jobs.append((None, 0))
            self._cv.notify_all()
        self.thread.join()
        self.seal_thread.join()

    def summary(self) -> dict[str, Any]:
        return {"hold_bytes": self.hold_bytes, "held": self.held, "errors": self.errors[:10], "errors_n": len(self.errors)}
