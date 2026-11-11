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
from contextlib import contextmanager
from typing import Iterator

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
