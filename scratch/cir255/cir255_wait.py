"""How long the circuit_check suite's session waited on its catalog build: from resources.jsonl, the controller (the parent of the
xdist worker that forked a 20+ GiB child), its workers, and for each worker but the builder the first sample after its first
10 s at which its CPU rose by 3 s or more within 10 s (it left the lock `_built` holds and ran tests).  Also the builder's
largest child (the served merge) and the node's whole-machine CPU and memory over the suite from the cgroup records.

python cir255_wait.py RUN_DIR...
"""
import collections
import json
import sys
from datetime import datetime
from pathlib import Path


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def scan(run: Path) -> None:
    series = collections.defaultdict(list)
    info: dict[int, dict] = {}
    for line in open(run / "resources.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "process":
            continue
        t = ts(r["t_utc"])
        for p in r["processes"]:
            info.setdefault(p["pid"], {"ppid": p["ppid"], "comm": p["comm"]})
            series[p["pid"]].append((t, p.get("cpu_s") or 0, p.get("rss_bytes") or 0))
    kids = collections.defaultdict(list)
    for pid, s in info.items():
        kids[s["ppid"]].append(pid)
    builder = None
    for parent, cs in kids.items():
        if len(cs) >= 30 and any(max(x[2] for x in series[c]) > 20 * 2**30 for c in cs):
            if len(kids.get(info.get(parent, {}).get("ppid"), [])) < 30:
                builder = parent
                break
    if builder is None:
        print(run.name, "no builder")
        return
    ctl = info[builder]["ppid"]
    workers = [c for c in kids[ctl] if info[c]["comm"].startswith("python")]
    t0 = series[ctl][0][0]
    t1 = series[ctl][-1][0]
    resumed = []
    for w in workers:
        if w == builder:
            continue
        s = series[w]
        for i, (t, c, _) in enumerate(s):
            if t - s[0][0] < 90:
                continue
            j = next((k for k in range(i, len(s)) if s[k][0] - t >= 10), None)
            if j is not None and s[j][1] - c >= 3:
                resumed.append(t)
                break
    big = max(kids[builder], key=lambda c: max(x[2] for x in series[c]))
    bs = series[big]
    print(f"{run.name}: controller {datetime.utcfromtimestamp(t0):%H:%M:%S}..{datetime.utcfromtimestamp(t1):%H:%M:%S} "
          f"({t1 - t0:.0f} s), {len(workers)} workers; the waiting workers resumed after "
          f"{sorted(x - t0 for x in resumed)[:1] and round(sorted(x - t0 for x in resumed)[0])} s (first) / "
          f"{resumed and round(sorted(x - t0 for x in resumed)[len(resumed) // 2])} s (median of {len(resumed)}); "
          f"served-merge child {bs[0][0] - t0:.0f}..{bs[-1][0] - t0:.0f} s, cpu {bs[-1][1]:.0f} s, peak "
          f"{max(x[2] for x in bs) / 2**30:.1f} GiB")


for a in sys.argv[1:]:
    scan(Path(a))
