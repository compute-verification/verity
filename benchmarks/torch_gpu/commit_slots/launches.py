"""The host's enqueue calls inside each NVTX range of a name, from an nsys report exported to SQLite: how long each call held
the thread, and from which call on they block (does the launch queue's depth pace ``StepCommitter.add``?).

    nsys profile -t cuda,nvtx --export sqlite -o OUT python step.py time ...      # step.py marks each step's enqueue
    python benchmarks/torch_gpu/commit_slots/launches.py OUT.sqlite [--range enqueue]

A JSON line a range: its length; the CUDA runtime calls on its thread (count and seconds by name); of those that enqueue
work on a stream (launches, async copies and memsets, event records and waits, graph launches), how many took over
``--slow-us``, their seconds, the first one's index, and the median call before and from it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

ENQUEUE = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaMemcpyAsync",
           "cudaMemsetAsync", "cudaEventRecord", "cudaEventRecordWithFlags", "cudaStreamWaitEvent", "cudaGraphLaunch")


def base(name: str) -> str:
    """A runtime API name without its version suffix (``cudaLaunchKernel_v7000``)."""
    head, sep, tail = name.rpartition("_v")
    return head if sep and tail.isdigit() else name


def ranges(db: sqlite3.Connection, name: str) -> list[tuple[int, int, int]]:
    cols = {r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text = "COALESCE(e.text, s.value)" if "textId" in cols else "e.text"
    join = "LEFT JOIN StringIds s ON s.id = e.textId" if "textId" in cols else ""
    q = f"SELECT e.start, e.end, e.globalTid FROM NVTX_EVENTS e {join} WHERE {text} = ? AND e.end IS NOT NULL ORDER BY e.start"
    return list(db.execute(q, (name,)))


def summary(db: sqlite3.Connection, start: int, end: int, tid: int, slow_ns: int) -> dict:
    calls = db.execute("SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
                       "WHERE r.globalTid = ? AND r.start >= ? AND r.start < ? ORDER BY r.start", (tid, start, end)).fetchall()
    by: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    enq = []
    for s, e, n in calls:
        b = base(n)
        by[b][0] += 1
        by[b][1] += (e - s) / 1e9
        if b in ENQUEUE:
            enq.append(e - s)
    slow = [i for i, d in enumerate(enq) if d > slow_ns]
    first = slow[0] if slow else None
    out = {"range_s": round((end - start) / 1e9, 4), "calls": len(calls),
           "by_name": {k: {"calls": v[0], "s": round(v[1], 4)} for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])[:12]},
           "enqueue_calls": len(enq), "enqueue_s": round(sum(enq) / 1e9, 4), "slow_calls": len(slow),
           "slow_s": round(sum(enq[i] for i in slow) / 1e9, 4), "first_slow": first}
    if first is not None:
        out["median_us_before_first_slow"] = round(statistics.median(enq[:first]) / 1e3, 2) if first else None
        out["median_us_from_first_slow"] = round(statistics.median(enq[first:]) / 1e3, 2)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--range", default="enqueue", help="the NVTX range's name")
    ap.add_argument("--slow-us", type=float, default=50.0)
    a = ap.parse_args(argv)
    db = sqlite3.connect(a.sqlite)
    rs = ranges(db, a.range)
    if not rs:
        raise SystemExit(f"no NVTX range named {a.range!r} in {a.sqlite}")
    for i, (s, e, tid) in enumerate(rs):
        print(json.dumps({"range": a.range, "index": i, **summary(db, s, e, tid, int(a.slow_us * 1e3))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
