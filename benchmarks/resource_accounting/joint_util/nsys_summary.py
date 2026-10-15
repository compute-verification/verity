"""One nsys report's GPU metrics and kernels over its capture range (`loop.py`'s timed window), from its SQLite export:
each GPU metric's mean, median, p5 and p95 over the samples, and each protocol's kernels (by name, `PROTOCOL_OF`): their
count, summed time and the union of their intervals as a fraction of the window. Stdlib only.

    python3 nsys_summary.py REPORT.sqlite > SUMMARY.json
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys

#: the bandwidth audit's kernels are in `benchmarks/remote_session/session.cu`'s namespace; every other kernel is Pearl-C's
AUDIT_NAMESPACE = "rb::"


def protocol_of(name: str) -> str:
    return "audit" if name.startswith(AUDIT_NAMESPACE) else "pouw"


def union(intervals: list[tuple[int, int]]) -> int:
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def q(xs: list[float], p: float) -> float:
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main(path: str) -> dict:
    db = sqlite3.connect(path)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    out: dict = {"tables": sorted(t for t in tables if "GPU_METRIC" in t or "KERNEL" in t)}
    t0 = t1 = None
    if "GPU_METRICS" in tables and "TARGET_INFO_GPU_METRICS" in tables:
        names = dict(db.execute("select distinct metricId, metricName from TARGET_INFO_GPU_METRICS"))
        by: dict[int, list[float]] = {}
        for mid, v in db.execute("select metricId, value from GPU_METRICS"):
            by.setdefault(mid, []).append(v)
        t0, t1 = db.execute("select min(timestamp), max(timestamp) from GPU_METRICS").fetchone()
        out["window_ns"] = t1 - t0
        out["metrics"] = {}
        for mid, vs in sorted(by.items()):
            vs.sort()
            out["metrics"][names.get(mid, str(mid))] = {"mean": statistics.fmean(vs), "median": q(vs, 0.5), "p05": q(vs, 0.05),
                                                        "p95": q(vs, 0.95), "n": len(vs)}
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        strings = dict(db.execute("select id, value from StringIds"))
        rows = db.execute("select start, end, demangledName, streamId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
        lo = min(r[0] for r in rows) if rows else 0
        hi = max(r[1] for r in rows) if rows else 0
        w0, w1 = (t0, t1) if t0 is not None else (lo, hi)
        span = max(w1 - w0, 1)
        prot: dict[str, dict] = {}
        names: dict[str, dict] = {}
        for s, e, nid, stream in rows:
            s, e = max(s, w0), min(e, w1)
            if e <= s:
                continue
            nm = strings.get(nid, str(nid))
            p = prot.setdefault(protocol_of(nm), {"kernels": 0, "sum_ns": 0, "iv": [], "streams": set()})
            p["kernels"] += 1
            p["sum_ns"] += e - s
            p["iv"].append((s, e))
            p["streams"].add(stream)
            k = names.setdefault(nm.split("(")[0][:80], {"protocol": protocol_of(nm), "count": 0, "sum_ns": 0})
            k["count"] += 1
            k["sum_ns"] += e - s
        allv = [iv for p in prot.values() for iv in p["iv"]]
        out["kernels"] = {"window_ns": span, "any_busy_frac": union(allv) / span,
                          "by_protocol": {nm: {"kernels": p["kernels"], "sum_ns": p["sum_ns"], "busy_frac": union(p["iv"]) / span,
                                               "streams": sorted(p["streams"])} for nm, p in prot.items()},
                          "top": dict(sorted(names.items(), key=lambda kv: -kv[1]["sum_ns"])[:40])}
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))
