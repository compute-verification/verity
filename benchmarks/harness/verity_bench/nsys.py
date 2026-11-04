"""Busy SM-time from an Nsight Systems report's SQLite export: the time each SM has a resident warp, summed over the card's
SMs, read from the "SMs Active" GPU metric (the share of the SMs with a warp in flight, sampled at 10 kHz or more), for each
NVTX range the profiled program marked (`bench.py --traced`: one range an item of the traced pass).

A range's work is the kernels the CUDA calls made inside it launched (CUPTI's runtime and driver records on the range's
thread, joined to its kernel records by correlation id, a graph's nodes included with `--cuda-graph-trace=node`); its busy
SM-time is the metric integrated over those kernels' span on the GPU, times the card's SMs. Its cross-check is each kernel's
duration times the SMs its blocks occupy, min(SMs, ⌈blocks ÷ blocks an SM holds⌉), the blocks an SM holds from the
kernel's threads, registers and shared memory under the card's limits (`overhead.Card`).

    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --gpu-metrics-devices=cuda-visible \\
        --gpu-metrics-frequency=10000 --capture-range=cudaProfilerApi --capture-range-end=repeat -o PROF CMD...
    nsys export --type sqlite -o PROF.sqlite PROF.nsys-rep      (each report: one a capture range)

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
#: the shared memory the driver reserves per block on sm_80 and later
RESERVED_SMEM = 1024
REG_UNIT = 256


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
    threads: int
    registers: int
    smem: int


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

    for table in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if not _has(db, table):
            raise ValueError(f"the report has no {table}: profile with --trace=cuda,nvtx")
    strings = _strings(db)
    cols = _cols(db, "NVTX_EVENTS")
    text = "COALESCE(text, textId)" if "textId" in cols else "text"
    marks = [(int(a), int(b), int(tid), t if isinstance(t, str) else strings.get(t, str(t)))
             for a, b, tid, t in db.execute(f"SELECT start, end, globalTid, {text} FROM NVTX_EVENTS "
                                            "WHERE end IS NOT NULL ORDER BY start")]
    marks = [m for m in marks if str(m[3]).startswith(prefix)]
    calls: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for table in CALLS:
        if _has(db, table):
            for start, tid, corr in db.execute(f"SELECT start, globalTid, correlationId FROM {table}"):
                calls[int(tid)].append((int(start), int(corr)))
    for on in calls.values():
        on.sort()
    kcols = _cols(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    smem = " + ".join(c for c in ("staticSharedMemory", "dynamicSharedMemory") if c in kcols) or "0"
    by_corr: dict[int, list[Kernel]] = defaultdict(list)
    for row in db.execute("SELECT correlationId, start, end, gridX * gridY * gridZ, blockX * blockY * blockZ, "
                          f"registersPerThread, {smem} FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        corr, *rest = (int(x) for x in row)
        by_corr[corr].append(Kernel(*rest))
    out = []
    for a, b, tid, t in marks:
        on = calls.get(tid, [])
        lo, hi = bisect.bisect_left(on, (a, -1)), bisect.bisect_right(on, (b, math.inf))
        kernels = tuple(k for _, corr in on[lo:hi] for k in by_corr.get(corr, ()))
        out.append(Range(str(t), kernels))
    return out


def blocks_per_sm(k: Kernel, card) -> int:
    """The blocks of `k` one SM of `card` holds at once: the tightest of its warps, registers and shared memory."""

    warps = -(-k.threads // 32)
    limits = [card.max_blocks_per_sm, card.max_warps_per_sm // max(warps, 1)]
    if k.registers:
        per_warp = -(-k.registers * 32 // REG_UNIT) * REG_UNIT
        limits.append(card.regs_per_sm // (per_warp * warps))
    limits.append(card.smem_per_sm // (k.smem + RESERVED_SMEM))
    return max(min(limits), 1)


def occupied_ns(k: Kernel, card) -> float:
    """The cross-check's SM-time of one kernel: its duration times the SMs its blocks occupy."""

    return (k.end - k.start) * min(card.sms, -(-k.blocks // blocks_per_sm(k, card)))


def busy(db_path: Path, card, prefix: str = "") -> dict[str, dict]:
    """Each range's busy SM-time (SM·ns) from SMs Active, its cross-check and their ratio, by the range's text."""

    db = sqlite3.connect(db_path)
    try:
        ts, vs = sm_active(db)
        out = {}
        for r in ranges(db, prefix):
            span = r.span
            if span is None:
                out[r.text] = {"sm_ns": None, "occupied_sm_ns": None, "ratio": None, "kernels": 0}
                continue
            sm_ns = card.sms * integral(ts, vs, *span)
            occupied = sum(occupied_ns(k, card) for k in r.kernels)
            out[r.text] = {"sm_ns": sm_ns, "occupied_sm_ns": occupied, "ratio": sm_ns / occupied if occupied else None,
                           "kernels": len(r.kernels), "span_ns": span[1] - span[0]}
        return out
    finally:
        db.close()
