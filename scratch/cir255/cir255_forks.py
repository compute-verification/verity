"""From a run's resources.jsonl: python processes that forked many children (circuit-check's catalog built apart forks one
per binding), with the parent's own parent, its siblings, and the children's first and last sighting.

python cir255_forks.py RUN_DIR [MIN_CHILDREN]
"""
import collections
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
low = int(sys.argv[2]) if len(sys.argv) > 2 else 40
seen: dict[int, dict] = {}
trunc = 0
for line in open(run / "resources.jsonl"):
    r = json.loads(line)
    if r.get("kind") != "process":
        continue
    trunc += bool(r.get("truncated"))
    t = r["t_utc"]
    for p in r["processes"]:
        s = seen.setdefault(p["pid"], {"ppid": p["ppid"], "comm": p["comm"], "first": t, "last": t, "hwm": 0, "cpu": 0})
        s["last"] = t
        s["hwm"] = max(s["hwm"], p.get("rss_bytes") or 0)
        s["cpu"] = max(s["cpu"], p.get("cpu_s") or 0)
kids = collections.defaultdict(list)
for pid, s in seen.items():
    kids[s["ppid"]].append(pid)
print(f"{len(seen)} pids, {trunc} truncated samples")
for parent, cs in sorted(kids.items(), key=lambda kv: -len(kv[1])):
    if len(cs) < low:
        break
    ps = seen.get(parent, {})
    pys = [c for c in cs if seen[c]["comm"].startswith("python")]
    gp = ps.get("ppid")
    sib = [c for c in kids.get(gp, []) if seen[c]["comm"].startswith("python")]
    first = min(seen[c]["first"] for c in cs)
    last = max(seen[c]["last"] for c in cs)
    print(f"parent {parent} {ps.get('comm')} seen {ps.get('first')}..{ps.get('last')} cpu {ps.get('cpu')}: {len(cs)} children "
          f"({len(pys)} python) seen {first}..{last}; its parent {gp} {seen.get(gp, {}).get('comm')} has {len(sib)} python "
          f"children; children's peak rss {max(seen[c]['hwm'] for c in cs) / 2**30:.1f} GiB, cpu sum {sum(seen[c]['cpu'] for c in cs):.0f} s")
