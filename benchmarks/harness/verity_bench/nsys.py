"""Busy SM-time from an Nsight Systems report's SQLite export: the time each SM has a resident warp, summed over the card's
SMs, read from the "SMs Active" GPU metric (the share of the SMs with a warp in flight, sampled at 10 kHz or more), for each
NVTX range the profiled program marked (`bench.py --traced`: one range an item of the traced pass).

A range's work is the kernels the CUDA calls made inside it launched (CUPTI's runtime and driver records on the range's
thread, joined to its kernel records by correlation id, a graph's nodes included with `--cuda-graph-trace=node`); its busy
SM-time is the metric integrated over those kernels' span on the GPU, times the card's SMs. Its cross-check is each kernel's
duration times the most SMs its blocks can occupy, min(SMs, blocks): a block is resident on one SM, and the block
scheduler spreads a small grid over as many SMs as it has blocks, so ⌈blocks ÷ blocks an SM holds⌉ is the fewest SMs a
grid occupies, not the most.

    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \\
        --gpu-metrics-frequency=10000 --capture-range=cudaProfilerApi --capture-range-end=repeat -o PROF CMD...
    nsys export --type sqlite -o PROF.sqlite PROF.nsys-rep      (each report: one a capture range)

`bench.py --traced` primes the profiler with empty capture ranges first, whose reports hold no range; a report that holds
kernels its ranges didn't launch, beyond one flush a range, collected past the traced pass (`report`'s `stray`).

Standard library only.
"""

from __future__ import annotations

import bisect
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

#: the metric's names across nsys versions and metric sets, first match wins (lower case, a prefix)
SM_ACTIVE = ("sms active", "sm active")
#: the CUDA calls a range's kernels are launched by: the runtime's, and the driver's (a JIT library's cuLaunchKernel)
CALLS = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")


def _has(db: sqlite3.Connection, table: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _cols(db: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def _strings(db: sqlite3.Connection) -> dict[int, str]:
    return dict(db.execute("SELECT id, value FROM StringIds")) if _has(db, "StringIds") else {}


def sm_active(db: sqlite3.Connection) -> tuple[list[int], list[float]]:
    """The SMs Active samples (ns, share 0-1), by time."""

    if not _has(db, "GPU_METRICS"):
        raise ValueError("the report has no GPU_METRICS: profile with --gpu-metrics-devices")
    names = {str(n): int(i) for i, n in db.execute("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS")}
    metric = next((i for prefix in SM_ACTIVE for n, i in names.items() if n.lower().startswith(prefix)), None)
    if metric is None:
        raise ValueError(f"no SMs Active metric among {sorted(names)}")
    ts, vs = [], []
    for t, v in db.execute("SELECT timestamp, value FROM GPU_METRICS WHERE metricId = ? ORDER BY timestamp", (metric,)):
        ts.append(int(t))
        vs.append(float(v) / 100)
    if len(ts) < 2:
        raise ValueError("fewer than two SMs Active samples")
    return ts, vs


def integral(ts: list[int], vs: list[float], lo: int, hi: int) -> float:
    """∫ share dt over [lo, hi] in ns, each sample's value held over the period since the sample before it."""

    total = 0.0
    i = max(1, bisect.bisect_right(ts, lo))
    while i < len(ts) and ts[i - 1] < hi:
        a, b = max(ts[i - 1], lo), min(ts[i], hi)
        if b > a:
            total += vs[i] * (b - a)
        i += 1
    return total


@dataclass(frozen=True)
class Kernel:
    start: int
    end: int
    blocks: int


@dataclass(frozen=True)
class Range:
    """One NVTX range: its text, its kernels and their span on the GPU."""

    text: str
    kernels: tuple[Kernel, ...]

    @property
    def span(self) -> tuple[int, int] | None:
        if not self.kernels:
            return None
        return min(k.start for k in self.kernels), max(k.end for k in self.kernels)


def ranges(db: sqlite3.Connection, prefix: str = "") -> list[Range]:
    """Every NVTX push/pop range whose text starts with `prefix`, with the kernels the CUDA calls inside it launched."""

    if not _has(db, "NVTX_EVENTS"):
        return []
    strings = _strings(db)
    cols = _cols(db, "NVTX_EVENTS")
    text = "COALESCE(text, textId)" if "textId" in cols else "text"
    marks = [(int(a), int(b), int(tid), t if isinstance(t, str) else strings.get(t, str(t)))
             for a, b, tid, t in db.execute(f"SELECT start, end, globalTid, {text} FROM NVTX_EVENTS "
                                            "WHERE end IS NOT NULL ORDER BY start")]
    marks = [m for m in marks if str(m[3]).startswith(prefix)]
    if not marks:
        return []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if not _has(db, table):
            raise ValueError(f"the report has no {table}: profile with --trace=cuda,nvtx")
    calls: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for table in CALLS:
        if _has(db, table):
            for start, tid, corr in db.execute(f"SELECT start, globalTid, correlationId FROM {table}"):
                calls[int(tid)].append((int(start), int(corr)))
    for on in calls.values():
        on.sort()
    by_corr: dict[int, list[Kernel]] = defaultdict(list)
    for row in db.execute("SELECT correlationId, start, end, gridX * gridY * gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        corr, *rest = (int(x) for x in row)
        by_corr[corr].append(Kernel(*rest))
    out = []
    for a, b, tid, t in marks:
        on = calls.get(tid, [])
        lo, hi = bisect.bisect_left(on, (a, -1)), bisect.bisect_right(on, (b, math.inf))
        kernels = tuple(k for _, corr in on[lo:hi] for k in by_corr.get(corr, ()))
        out.append(Range(str(t), kernels))
    return out


def occupied_ns(k: Kernel, card) -> float:
    """The cross-check's SM-time of one kernel: its duration times the most SMs its blocks can occupy, one a block."""

    return (k.end - k.start) * min(card.sms, k.blocks)


def report(db_path: Path, card, prefix: str = "") -> dict:
    """One report's ranges whose text starts with `prefix`: `busy`, each one's busy SM-time (SM·ns) from SMs Active, its
    cross-check and their ratio, by its text; `ranges`, how many; and `stray`, the report's kernels none of them launched
    (a capture range around the ranges alone holds one flush a range besides). A report with no such range (a priming
    range's) has an empty `busy` and needs no metric."""

    db = sqlite3.connect(db_path)
    try:
        rs = ranges(db, prefix)
        kernels = db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] if _has(
            db, "CUPTI_ACTIVITY_KIND_KERNEL") else 0
        out = {}
        ts, vs = sm_active(db) if rs else ([], [])
        for r in rs:
            span = r.span
            if span is None:
                out[r.text] = {"sm_ns": None, "occupied_sm_ns": None, "ratio": None, "kernels": 0}
                continue
            sm_ns = card.sms * integral(ts, vs, *span)
            occupied = sum(occupied_ns(k, card) for k in r.kernels)
            out[r.text] = {"sm_ns": sm_ns, "occupied_sm_ns": occupied, "ratio": sm_ns / occupied if occupied else None,
                           "kernels": len(r.kernels), "span_ns": span[1] - span[0]}
        return {"busy": out, "ranges": len(rs), "stray": kernels - sum(len(r.kernels) for r in rs)}
    finally:
        db.close()
