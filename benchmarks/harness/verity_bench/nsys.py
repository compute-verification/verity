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

import argparse
import bisect
import json
import math
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
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


# ---- GPU kernel time by kernel -------------------------------------------------------------------------------------------

KERNELS = "CUPTI_ACTIVITY_KIND_KERNEL"
#: a CUDA graph traced whole (nsys's default, `--cuda-graph-trace=graph`): one record a launch, in place of its kernels'
GRAPHS = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
WHOLE_GRAPH = "CUDA graph traced whole: its kernels need --cuda-graph-trace=node"
NOT_BUSY = ("Kernel time is not busy time: kernels on overlapping streams each count their whole duration "
            "({kernel_ms:.3f} ms summed over {streams} stream(s); the union of their intervals, device by device, is "
            "{union_ms:.3f} ms).")
NOT_COST = "Kernel time is not a cost: the timed repetitions give the cost, and a profiled repetition is never one of them."


def _col(cols: set[str], *names: str) -> str:
    """The first of `names` a table has, else NULL."""

    return next((n for n in names if n in cols), "NULL")


def _marks(db: sqlite3.Connection, prefix: str) -> list[tuple[int, int]]:
    """The (start, end) of every NVTX range whose text starts with `prefix`, on any thread."""

    if not _has(db, "NVTX_EVENTS"):
        return []
    strings = _strings(db)
    text = "COALESCE(text, textId)" if "textId" in _cols(db, "NVTX_EVENTS") else "text"
    return [(int(a), int(b)) for a, b, t in db.execute(f"SELECT start, end, {text} FROM NVTX_EVENTS WHERE end IS NOT NULL")
            if str(t if isinstance(t, str) else strings.get(t, t)).startswith(prefix)]


def _merged(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """The intervals' union, as disjoint intervals by start."""

    out: list[tuple[int, int]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def kernel_table(db_path: Path, prefix: str | None = None, steps: str | int | None = None) -> dict:
    """GPU kernel time by kernel over a capture range of one export: the NVTX ranges whose text starts with `prefix`, on any
    thread, or with no `prefix` the whole capture. `kernels` holds, by time, each kernel's demangled `name`, its `launches`,
    its `ms` (its launches' summed durations), its `ms_step` and its `share` of the range's kernel time (`kernel_ms`). A
    step is an NVTX range whose text starts with `steps` and that opens inside the capture range, or `steps` is their
    count (the engine's own counter, for a server that marks no range); `ms_step` is the kernel's time over the range's
    steps (ms a decode step, of a range whose steps all decode), the work launched between them (a side stream's)
    included, and None without steps.

    A kernel is in the range when the CUDA call that launched it started while one of the range's NVTX ranges was open: a
    runtime call or a driver call (a JIT library's cuLaunchKernel) on any thread of the capture, joined by correlation id
    within its process. So a worker thread's launches count (`ranges` keeps to the range's own thread), and a kernel still
    running when the range closes counts whole. In an export made with `--cuda-graph-trace=node` every node of a CUDA
    graph is a kernel of its cudaGraphLaunch. Without it nsys records a graph launch as one span and none of its kernels:
    their names, launches and times are not in the export, and the table holds one `WHOLE_GRAPH` row, the launches'
    spans, in their place. Copies and memsets are not kernels and are not counted.

    `kernel_ms` is not busy time: kernels on overlapping streams each count whole, so `union_ms`, the union of the
    kernels' intervals on each device, is beside it. Nor is it a cost. `notes` says both, in every rendering.

        nsys launch --session-new=S --trace=cuda,nvtx --cuda-graph-trace=node SERVER...
        nsys start --session=S -o PROF; LOAD...; nsys stop --session=S       (a bounded capture on a live server)
        nsys export --type sqlite -o PROF.sqlite PROF.nsys-rep
    """

    db = sqlite3.connect(db_path)
    try:
        if not _has(db, KERNELS):
            raise ValueError(f"the export has no {KERNELS}: profile with --trace=cuda,nvtx")
        marks = [] if prefix is None else _marks(db, prefix)
        if prefix is not None and not marks:
            raise ValueError(f"no NVTX range of the export starts with {prefix!r}: name a range the profiled program "
                             "marked and closed inside the capture, or no prefix for the whole capture")
        span = _merged(marks)
        opens = [a for a, _ in span]

        def inside(t: int | None) -> bool:
            if prefix is None:
                return True
            i = bisect.bisect_right(opens, t) - 1 if t is not None else -1
            return i >= 0 and t <= span[i][1]

        launched: dict[tuple[int | None, int], int] = {}
        for table in CALLS if prefix is not None else ():
            if _has(db, table):
                for start, tid, corr in db.execute(f"SELECT start, globalTid, correlationId FROM {table}"):
                    for key in ((int(tid) >> 24, int(corr)), (None, int(corr))):
                        launched[key] = min(launched.get(key, int(start)), int(start))
        strings = _strings(db)
        by_name: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        busy: dict[tuple[int | None, int | None], list[tuple[int, int]]] = defaultdict(list)
        streams: set[tuple[int | None, int | None, int | None]] = set()
        for table in (KERNELS, GRAPHS):
            if not _has(db, table):
                continue
            cols = _cols(db, table)
            for start, end, corr, name, pid, device, stream in db.execute(
                    f"SELECT start, end, correlationId, {_col(cols, 'demangledName', 'shortName')}, "
                    f"{_col(cols, 'globalPid')} >> 24, {_col(cols, 'deviceId')}, {_col(cols, 'streamId')} FROM {table}"):
                if not inside(launched.get((pid, corr))):
                    continue
                row = by_name[WHOLE_GRAPH if table == GRAPHS else strings.get(name, str(name))]
                row[0] += 1
                row[1] += end - start
                busy[pid, device].append((start, end))
                streams.add((pid, device, stream))
        n = sum(inside(a) for a, _ in _marks(db, steps)) if isinstance(steps, str) else steps
        total = sum(ns for _, ns in by_name.values())
        union = sum(b - a for on in busy.values() for a, b in _merged(on))
        kernels = [{"name": name, "launches": count, "ms": ns / 1e6, "ms_step": ns / 1e6 / n if n else None,
                    "share": ns / total if total else None}
                   for name, (count, ns) in sorted(by_name.items(), key=lambda kv: (-kv[1][1], kv[0]))]
        return {"range": prefix, "ranges": len(marks), "steps": n or None, "launches": sum(k["launches"] for k in kernels),
                "streams": len(streams), "kernel_ms": total / 1e6, "union_ms": union / 1e6, "kernels": kernels,
                "notes": [NOT_BUSY.format(kernel_ms=total / 1e6, streams=len(streams), union_ms=union / 1e6), NOT_COST]}
    finally:
        db.close()


def extra_over_stock(arm: dict, stock: dict) -> dict:
    """Where an arm's extra GPU kernel time goes: each kernel's ms a step in `arm` less its ms a step in `stock` (two
    `kernel_table`s with steps), kernels matched by name. A kernel only the arm runs is all extra, a kernel both run is the
    difference (more launches a step, or longer ones: a kernel beside it on another stream lengthens it), and a kernel only
    stock runs is negative, so the rows sum to `extra_ms_step`, the arm's kernel time a step less stock's. `share` is a
    row's part of that extra, None unless the arm's kernel time is the larger. The notes are the arm's."""

    for label, table in (("arm", arm), ("stock", stock)):
        if not table["steps"]:
            raise ValueError(f"the {label}'s table has no steps: give `kernel_table` its `steps`, the step ranges' prefix "
                             "or the engine's count of steps")
    a, s = ({k["name"]: k["ms_step"] for k in table["kernels"]} for table in (arm, stock))
    arm_ms, stock_ms = arm["kernel_ms"] / arm["steps"], stock["kernel_ms"] / stock["steps"]
    extra = arm_ms - stock_ms
    rows = [{"name": name, "arm_ms_step": a.get(name, 0.0), "stock_ms_step": s.get(name, 0.0),
             "extra_ms_step": a.get(name, 0.0) - s.get(name, 0.0)} for name in a.keys() | s.keys()]
    for r in rows:
        r["share"] = r["extra_ms_step"] / extra if extra > 0 else None
    rows.sort(key=lambda r: (-r["extra_ms_step"], r["name"]))
    return {"arm_ms_step": arm_ms, "stock_ms_step": stock_ms, "extra_ms_step": extra, "kernels": rows,
            "notes": arm["notes"]}


def kernel_lines(table: dict, top: int = 25) -> str:
    """`kernel_table`'s or `extra_over_stock`'s table as Markdown: its first `top` kernels, the rest summed in one row, all
    of them in the last, and its notes."""

    if "extra_ms_step" in table:
        title = (f"Extra GPU kernel time a step over stock, by kernel: {table['extra_ms_step']:+.3f} ms (the arm's "
                 f"{table['arm_ms_step']:.3f} ms less stock's {table['stock_ms_step']:.3f} ms)")
        columns = (("arm, ms a step", "arm_ms_step", ".3f"), ("stock, ms a step", "stock_ms_step", ".3f"),
                   ("extra, ms a step", "extra_ms_step", "+.3f"), ("share of the extra", "share", ".1%"))
    else:
        where = "the whole capture" if table["range"] is None else f"{table['ranges']} NVTX range(s) `{table['range']}`"
        title = f"GPU kernel time by kernel, {where}" + (f", {table['steps']} step(s)" if table["steps"] else "")
        columns = (("launches", "launches", ","), ("ms", "ms", ".3f"), ("ms a step", "ms_step", ".3f"),
                   ("share", "share", ".1%"))
    rows = table["kernels"]

    def line(label: str, part: list[dict]) -> str:
        cells = []
        for _, key, form in columns:
            values = [r[key] for r in part if r[key] is not None]
            cells.append(format(sum(values), form) if values else "")
        return f"| {label} | " + " | ".join(cells) + " |"

    def cut(name: str) -> str:
        return (name if len(name) <= 100 else name[:99] + "…").replace("|", "\\|")

    lines = [title, "", "| kernel | " + " | ".join(head for head, _, _ in columns) + " |", "|---|" + "---:|" * len(columns)]
    lines += [line(f"`{cut(r['name'])}`", [r]) for r in rows[:top]]
    if len(rows) > top:
        lines.append(line(f"{len(rows) - top} more", rows[top:]))
    lines.append(line("all", rows))
    return "\n".join([*lines, "", *table["notes"]]) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m verity_bench.nsys ARM.sqlite [--range PREFIX] [--steps PREFIX|COUNT] [--stock STOCK.sqlite] [--json]`."""

    ap = argparse.ArgumentParser(prog="python -m verity_bench.nsys", description="GPU kernel time by kernel over a capture "
                                 "range of an `nsys export --type sqlite`, and an arm's extra a step over stock's export.")
    ap.add_argument("export", type=Path)
    ap.add_argument("--range", dest="prefix", help="the NVTX ranges whose text starts with this (default: the whole "
                    "capture)")
    ap.add_argument("--steps", help="the step ranges' prefix, or the count of steps")
    ap.add_argument("--stock", type=Path, help="stock's export, read over the same --range")
    ap.add_argument("--stock-steps", help="stock's --steps (default: the arm's)")
    ap.add_argument("--top", type=int, default=25, help="the rows a table prints before it sums the rest")
    ap.add_argument("--json", action="store_true", help="every table as one JSON object in place of Markdown")
    a = ap.parse_args(argv)

    def steps(value: str | None) -> str | int | None:
        return int(value) if value is not None and value.isdigit() else value

    out = {"arm": kernel_table(a.export, a.prefix, steps(a.steps))}
    if a.stock:
        out["stock"] = kernel_table(a.stock, a.prefix, steps(a.stock_steps or a.steps))
        out["extra"] = extra_over_stock(out["arm"], out["stock"])
    print(json.dumps(out, indent=1) if a.json
          else "\n".join(kernel_lines(table, a.top) for key, table in out.items() if key != "stock"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
