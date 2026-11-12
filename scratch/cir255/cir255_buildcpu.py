"""The catalog build's children of builder PID that started by T (HH:MM:SS, when the builder resumed): count, CPU sum, the
top five by CPU.  python cir255_buildcpu.py RUN_DIR PID T"""
import json, sys
from pathlib import Path
b, cut = int(sys.argv[2]), sys.argv[3]
k = {}
for line in open(Path(sys.argv[1]) / "resources.jsonl"):
    r = json.loads(line)
    if r.get("kind") != "process":
        continue
    t = r["t_utc"][11:19]
    for p in r["processes"]:
        if p["ppid"] == b:
            s = k.setdefault(p["pid"], {"first": t, "cpu": 0, "hwm": 0})
            s["cpu"] = max(s["cpu"], p.get("cpu_s") or 0)
            s["hwm"] = max(s["hwm"], p.get("rss_bytes") or 0)
ks = [s for s in k.values() if s["first"] <= cut]
print(f"{len(ks)} children by {cut}: cpu {sum(s['cpu'] for s in ks):.0f} s; top: " +
      ", ".join(f"{s['cpu']:.0f} s/{s['hwm']/2**30:.1f} GiB" for s in sorted(ks, key=lambda s: -s["cpu"])[:6]))
