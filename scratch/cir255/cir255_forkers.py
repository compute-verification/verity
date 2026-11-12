"""Every process in resources.jsonl that forked 50+ children: its span, the children's count, CPU sum, largest RSS, and the
children over 8 GiB.  python cir255_forkers.py RUN_DIR"""
import collections, json, sys
from pathlib import Path
seen = {}
for line in open(Path(sys.argv[1]) / "resources.jsonl"):
    r = json.loads(line)
    if r.get("kind") != "process":
        continue
    t = r["t_utc"][11:19]
    for p in r["processes"]:
        s = seen.setdefault(p["pid"], {"ppid": p["ppid"], "comm": p["comm"], "first": t, "last": t, "hwm": 0, "cpu": 0})
        s["last"] = t
        s["hwm"] = max(s["hwm"], p.get("rss_bytes") or 0)
        s["cpu"] = max(s["cpu"], p.get("cpu_s") or 0)
kids = collections.defaultdict(list)
for pid, s in seen.items():
    kids[s["ppid"]].append(pid)
G = 2**30
for pid, ks in kids.items():
    if len(ks) < 50:
        continue
    s = seen.get(pid, {})
    cs = [seen[k] for k in ks]
    big = sorted(((seen[k]["hwm"] / G, seen[k]["cpu"], seen[k]["first"], seen[k]["last"]) for k in ks if seen[k]["hwm"] > 8 * G), reverse=True)
    print(f"pid {pid} ({s.get('comm')}, ppid {s.get('ppid')}) {s.get('first')}-{s.get('last')}: {len(ks)} children "
          f"{min(c['first'] for c in cs)}-{max(c['last'] for c in cs)}, cpu {sum(c['cpu'] for c in cs):.0f} s, "
          f"max rss {max(c['hwm'] for c in cs)/G:.2f} GiB, own cpu {s.get('cpu', 0):.0f} s")
    for b in big[:5]:
        print(f"    big child {b[0]:.2f} GiB cpu {b[1]:.0f} s {b[2]}-{b[3]}")
