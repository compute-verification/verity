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
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

COMPONENTS = ("acquisition", "movement", "hashing", "root_finalisation", "opening_generation", "wait")


class Spans:
    """Thread-safe accumulators: per component total seconds and call counts, plus per-step buckets."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.per_step: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bytes: dict[str, int] = defaultdict(int)

    def add(self, component: str, seconds: float, *, step: int | None = None, nbytes: int = 0) -> None:
        with self._lock:
            self.total[component] += seconds
            self.calls[component] += 1
            if nbytes:
                self.bytes[component] += nbytes
            if step is not None:
                self.per_step[step][component] += seconds

    @contextmanager
    def span(self, component: str, *, step: int | None = None, nbytes: int = 0) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(component, time.perf_counter() - t0, step=step, nbytes=nbytes)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total_s": {c: self.total.get(c, 0.0) for c in COMPONENTS},
                "calls": {c: self.calls.get(c, 0) for c in COMPONENTS},
                "bytes": dict(self.bytes),
                "per_step": {int(s): {c: v for c, v in d.items()} for s, d in sorted(self.per_step.items())},
            }
