"""Average nsys's sampled GPU metrics over a profile's active span.

`python3 nsys_metrics.py PROFILE.sqlite LABEL` prints one JSON line: for each GPU the metrics sampled, each averaged over
the middle 80% of the span between the first CUDA memcpy or kernel's start and the last one's end (CUPTI's activity
tables), so the ramp at either end is left out.  The gb20x set reports DRAM read and write bandwidth and PCIe RX and TX
throughput as percentages of their peaks.
"""

import json
import sqlite3
import sys


def tables(db):
    return {r[0] for r in db.execute("select name from sqlite_master where type='table'")}


def span(db, have):
    lo, hi = None, None
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if t in have:
            a, b = db.execute(f"select min(start), max(end) from {t}").fetchone()
            if a is not None:
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
    return lo, hi


def main(path, label):
    db = sqlite3.connect(path)
    have = tables(db)
    out = {"label": label, "tables": sorted(t for t in have if "GPU_METRIC" in t)}
    if "GPU_METRICS" not in have:
        print(json.dumps(out))
        return
    lo, hi = span(db, have)
    if lo is None:
        out["error"] = "no CUDA activity"
        print(json.dumps(out))
        return
    a, b = lo + (hi - lo) // 10, hi - (hi - lo) // 10
    names = {}
    for row in db.execute("select typeId, metricId, metricName from TARGET_INFO_GPU_METRICS"):
        names[(row[0], row[1])] = row[2]
    gpus = {}
    q = "select typeId, metricId, avg(value), count(*) from GPU_METRICS where timestamp between ? and ? group by typeId, metricId"
    for type_id, metric_id, mean, n in db.execute(q, (a, b)):
        name = names.get((type_id, metric_id), str(metric_id))
        gpus.setdefault(str(type_id), {})[name] = round(mean, 3)
        gpus[str(type_id)]["_samples"] = n
    if "TARGET_INFO_GPU" in have:
        cols = [c[1] for c in db.execute("pragma table_info(TARGET_INFO_GPU)")]
        out["gpu_info"] = [dict(zip(cols, r)) for r in db.execute("select * from TARGET_INFO_GPU")]
        for g in out["gpu_info"]:
            for k in list(g):
                if isinstance(g[k], bytes):
                    del g[k]
    out["span_ns"] = hi - lo
    out["gpus"] = gpus
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
