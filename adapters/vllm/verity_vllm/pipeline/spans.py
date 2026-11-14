"""Span accounting for the commitment components (host wall clock, `time.perf_counter`).

Component names are fixed (`COMPONENTS`) so every committer reports the same breakdown and the table in
`out/gen/r9/cmt-integ/delta_table.md` has one column per component:

  acquisition        getting hold of the value bytes on the device (hook dispatch, view bookkeeping, any .contiguous())
  movement           moving bytes to where they are hashed (host: D2H issue time; device: bytes are recorded, the GPU-side copy time
                     is measured by CUDA events where the committer uses a side stream, else calculated from an isolated D2H microbench)
  hashing            leaf hashing + tree folding (SHA-256 here)
  root_finalisation  context binding of the per-step root, run-root fold
  opening_generation generating openings for sampled positions (after the run; never on the serving path)
  wait               time the SERVING thread spent blocked on the committer (backpressure) — must be ~0 in steady state

`snapshot()["throughput"]` is the per-scheme view of the same accumulators: the commitment scheme the committer hashes with and, per
stage that ran, its wall time, bytes, leaves and the two rates.  A device-side stage's wall time is the host's wait for it.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

__all__ = ["Spans"]

COMPONENTS = ("acquisition", "movement", "hashing", "root_finalisation", "opening_generation", "wait")


class Spans:
    """Thread-safe accumulators: per component total seconds, call counts, bytes and leaves, plus per-step buckets."""

    def __init__(self, scheme: str | None = None) -> None:
        self._lock = threading.Lock()
        self.scheme = scheme
        self.total: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.per_step: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bytes: dict[str, int] = defaultdict(int)
        self.leaves: dict[str, int] = defaultdict(int)

    def add(self, component: str, seconds: float, *, step: int | None = None, nbytes: int = 0, leaves: int = 0) -> None:
        with self._lock:
            self.total[component] += seconds
            self.calls[component] += 1
            if nbytes:
                self.bytes[component] += nbytes
            if leaves:
                self.leaves[component] += leaves
            if step is not None:
                self.per_step[step][component] += seconds

    @contextmanager
    def span(self, component: str, *, step: int | None = None, nbytes: int = 0) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(component, time.perf_counter() - t0, step=step, nbytes=nbytes)

    def throughput(self) -> dict:
        """Per stage that ran: ``wall_s``, ``bytes``, ``leaves``, ``bytes_per_s`` and ``leaves_per_s`` (``None`` without wall time)."""
        with self._lock:
            stages = {}
            for c in [*COMPONENTS, *sorted(set(self.calls) - set(COMPONENTS))]:
                if not self.calls.get(c):
                    continue
                wall, nb, nl = self.total.get(c, 0.0), self.bytes.get(c, 0), self.leaves.get(c, 0)
                stages[c] = {"wall_s": wall, "bytes": nb, "leaves": nl, "bytes_per_s": nb / wall if wall > 0 else None,
                             "leaves_per_s": nl / wall if wall > 0 else None}
            return {"scheme": self.scheme, "stages": stages}

    def snapshot(self) -> dict:
        throughput = self.throughput()
        with self._lock:
            return {
                "total_s": {c: self.total.get(c, 0.0) for c in COMPONENTS},
                "calls": {c: self.calls.get(c, 0) for c in COMPONENTS},
                "bytes": dict(self.bytes),
                "leaves": dict(self.leaves),
                "per_step": {int(s): {c: v for c, v in d.items()} for s, d in sorted(self.per_step.items())},
                "throughput": throughput,
            }


COMMIT_S = {"serving": ("acquisition", "movement", "wait"), "worker": ("hashing", "worker_idle", "root_finalisation", "device_pool_wait")}


def commit_s_of(rc: Any) -> dict[str, dict[str, float]] | None:
    """A window's Commit time by thread, summed over its steps (the committer's per-step spans). The serving thread's
    ``movement`` is the copies' native issue, ``acquisition`` the step ends and the sources' takes (their waits included), and
    ``wait`` its waits for a free slot of the staging ring, the worker behind. The worker's ``hashing`` holds ``worker_idle``, its
    waits for the serving thread's windows. The finalize's wait for the worker has no step and isn't counted."""
    spans = getattr(rc, "spans", None)
    per_step = spans.get("per_step") if spans else None             # None: the committer recorded no per-step spans
    if not per_step:
        return None
    per = per_step.values()
    return {side: {c: round(sum(float(d.get(c, 0.0)) for d in per), 4) for c in cs} for side, cs in COMMIT_S.items()}


def gc_of(rc: Any) -> dict[str, Any] | None:
    """A window's Python collections (the committer's `GcPauses` record, while it was installed): how many, their total and
    longest, and each of 50 ms or more as (generation, start on `time.perf_counter`, seconds, thread)."""
    g = (getattr(rc, "extra", None) or {}).get("gc")
    if not g:
        return None
    p = list(g.get("pauses") or [])
    return {"n": len(p), "total_s": round(sum(x[2] for x in p), 4), "max_s": round(max((x[2] for x in p), default=0.0), 4),
            "long": [[x[0], round(x[1], 3), round(x[2], 3), x[3]] for x in p if x[2] >= 0.05], "freeze_s": g.get("freeze_s")}


def phases_of(laps: Sequence[tuple[str, float]]) -> dict[str, float]:
    """A close's phases, each one's seconds by its name: the time between two laps (`end_lap` marks them)."""
    return {name: round(t - s, 4) for (_, s), (name, t) in zip(laps, laps[1:])}


def end_lap(laps: list[tuple[str, float]], clock: Callable[[], float], name: str) -> None:
    """The end of a close's phase ``name``, marked in ``laps`` (`phases_of` reads them)."""
    laps.append((name, clock()))
