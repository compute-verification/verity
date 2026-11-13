"""Each fetched run's memory: python3 peak.py RUN...  (each under ~/.research/runs, `research fetch RUN --all`).

The sampler sees the pod's cgroup, which other jobs share, and the run's own process tree.  Two upper bounds on what the run
holds, each sample: the cgroup's anon (the run's and every other job's) and the sum of its own processes' anon RSS (pages a
fork shares counted once per process).  ``bound`` is the smaller of the two at each second, its peak over the run; ``pss`` the
peak of the tree's PSS sum, which leaves out processes the sampler skipped or was denied (a floor, not a bound)."""
import json
import os
import re
import sys

ROOT = os.path.expanduser("~/.research/runs")
G = 2**30


def declared(d):
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            m = re.search(r'CHECK_RUN_MEM_GB"\s*:\s*"(\d+)"', open(os.path.join(d, f)).read())
            if m:
                return m.group(1)
    return "?"


for run in sys.argv[1:]:
    d = os.path.join(ROOT, run)
    cg, own = {}, {}
    pss = 0
    with open(os.path.join(d, "resources.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = (r.get("t_utc") or "")[:19]
            if r.get("kind") == "cgroup":
                a = ((r.get("cgroup") or {}).get("memory_stat") or {}).get("anon")
                if a is not None:
                    cg[t] = a
            elif r.get("kind") == "process" and r.get("processes") is not None:
                own[t] = sum(p.get("rss_anon_bytes") or 0 for p in r["processes"])
            elif r.get("kind") == "pss":
                pss = max(pss, r.get("pss_sum_bytes") or 0)
    both = sorted(set(cg) & set(own))
    bound = max(((min(cg[t], own[t]), t) for t in both), default=(0, None))
    last = both[-1] if both else None
    print(f"{run}: declared {declared(d)} GB; bound peak {bound[0] / G:.1f} GiB at {bound[1]}Z (pod cgroup anon "
          f"{cg.get(bound[1], 0) / G:.1f}, own anon rss {own.get(bound[1], 0) / G:.1f}); pod cgroup anon peak "
          f"{max(cg.values(), default=0) / G:.1f}; pss peak {pss / G:.1f}; last sample {last}Z bound "
          f"{(min(cg[last], own[last]) / G if last else 0):.1f} GiB")
