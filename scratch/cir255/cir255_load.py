"""The node's other load while a pytest shard's circuit_check suite built its catalog: from resources.jsonl, every 60 s between
two UTC times, the run's own PSS sum (the pss records), the RSS of the processes outside the run that the sampler lists
(`outsiders`, the 32 largest of `outsiders_total`), their top commands, and the run's cgroup CPU rate.

python cir255_load.py RESOURCES.jsonl HH:MM:SS HH:MM:SS
"""
import collections
import json
import sys
from datetime import datetime

path, lo, hi = sys.argv[1:4]
last_cpu = None
pss = {}
rows = []
for line in open(path):
    r = json.loads(line)
    t = r["t_utc"][11:19]
    if not (lo <= t <= hi):
        continue
    k = r.get("kind")
    if k == "pss":
        pss[t[:5]] = r["pss_sum_bytes"]
    if k == "cgroup":
        u = r["cgroup"]["cpu_stat"]["usage_usec"] / 1e6
        tt = datetime.fromisoformat(r["t_utc"].replace("Z", "+00:00")).timestamp()
        if last_cpu is None or tt - last_cpu[0] >= 60:
            rate = (u - last_cpu[1]) / (tt - last_cpu[0]) if last_cpu else None
            last_cpu = (tt, u)
            rows.append(("cpu", t, rate, r["cgroup"]["memory_current_bytes"]))
    if k == "process" and t.endswith(":00") and t[3:5] in {f"{m:02d}" for m in range(0, 60, 2)}:
        outs = r.get("outsiders") or []
        by = collections.Counter()
        for o in outs:
            by[o["comm"]] += o.get("rss_bytes") or 0
        rows.append(("out", t, sum(o.get("rss_bytes") or 0 for o in outs), by.most_common(4), r.get("outsiders_total")))
for row in rows:
    if row[0] == "cpu":
        print(f"{row[1]} run cgroup: {row[2] and round(row[2], 1)} CPUs, memory.current {row[3] / 2**30:.0f} GiB"
              f"{'' if row[1][:5] not in pss else f', run pss {pss[row[1][:5]] / 2**30:.0f} GiB'}")
    else:
        print(f"{row[1]} outside the run: {row[4]} processes, the listed 32 hold {row[2] / 2**30:.0f} GiB RSS: "
              + ", ".join(f"{c} {b / 2**30:.0f}" for c, b in row[3]))
