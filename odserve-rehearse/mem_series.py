"""A finished run's memory over time from its resources.jsonl, per minute: the run's total RSS and its largest process.

    python3 mem_series.py RUN_DIR OUT/mem-series.tsv

Each row is the minute's peak sample: UTC minute, the run's summed RSS (GiB), the largest process's comm and its RSS
and high-water mark (GiB). Only process names and sizes, so it holds no witness.
"""

import json
import sys

run, to = sys.argv[1], sys.argv[2]
peak: dict[str, tuple[float, str, float, float]] = {}
with open(f"{run}/resources.jsonl") as f:
    for line in f:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ps = r.get("processes") or []
        if not ps:
            continue
        minute = r["t_utc"][:16]
        total = sum(p.get("rss_bytes") or 0 for p in ps)
        top = max(ps, key=lambda p: p.get("rss_bytes") or 0)
        if minute not in peak or total > peak[minute][0]:
            peak[minute] = (total, top.get("comm") or "?", top.get("rss_bytes") or 0, top.get("rss_hwm_bytes") or 0)
g = 2**30
with open(to, "w") as out:
    out.write("minute_utc\trun_rss_gib\ttop_comm\ttop_rss_gib\ttop_hwm_gib\n")
    for m in sorted(peak):
        total, comm, rss, hwm = peak[m]
        out.write(f"{m}\t{total / g:.1f}\t{comm}\t{rss / g:.1f}\t{hwm / g:.1f}\n")
print(f"MEM-SERIES {len(peak)} minutes; peak run RSS {max(v[0] for v in peak.values()) / g:.1f} GiB")
