"""Did a pytest shard's circuit_check suite build the catalog apart?  From resources.jsonl: the xdist worker that forked one child
per catalog binding (`units.built_apart`), found as the python process whose children are many and one of whom grew past
20 GiB (the served merge's binding, `heavy.json` 30.96 GiB); the children's span, count, CPU sum, and the big ones.

python cir255_apart.py RUN_DIR...
"""
import collections
import json
import sys
from pathlib import Path


def scan(run: Path) -> None:
    seen: dict[int, dict] = {}
    first_t = last_t = None
    for line in open(run / "resources.jsonl"):
        r = json.loads(line)
        if r.get("kind") != "process":
            continue
        t = r["t_utc"]
        first_t = first_t or t
        last_t = t
        for p in r["processes"]:
            s = seen.setdefault(p["pid"], {"ppid": p["ppid"], "comm": p["comm"], "first": t, "last": t, "hwm": 0, "cpu": 0})
            s["last"] = t
            s["hwm"] = max(s["hwm"], p.get("rss_bytes") or 0)
            s["cpu"] = max(s["cpu"], p.get("cpu_s") or 0)
    kids = collections.defaultdict(list)
    for pid, s in seen.items():
        kids[s["ppid"]].append(pid)
    found = False
    for parent, cs in kids.items():
        big = [c for c in cs if seen[c]["hwm"] > 20 * 2**30]
        if len(cs) < 30 or not big:
            continue
        found = True
        ctl = seen.get(parent, {}).get("ppid")
        workers = [c for c in kids.get(ctl, []) if seen[c]["comm"].startswith("python")]
        print(f"{run.name}: worker {parent} of controller {ctl} ({len(workers)} python children; controller seen "
              f"{seen.get(ctl, {}).get('first')}..{seen.get(ctl, {}).get('last')}): {len(cs)} children sampled "
              f"{min(seen[c]['first'] for c in cs)}..{max(seen[c]['last'] for c in cs)}, cpu sum {sum(seen[c]['cpu'] for c in cs):.0f} s")
        for c in sorted(cs, key=lambda c: -seen[c]["hwm"])[:4]:
            s = seen[c]
            print(f"   child {c}: {s['first']}..{s['last']}, peak {s['hwm'] / 2**30:.1f} GiB, cpu {s['cpu']:.0f} s")
    if not found:
        print(f"{run.name}: no python process with 30+ children one of whom grew past 20 GiB (samples {first_t}..{last_t})")


for a in sys.argv[1:]:
    scan(Path(a))
