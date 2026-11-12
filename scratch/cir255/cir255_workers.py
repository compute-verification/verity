"""The xdist workers (children of the controller PPID) in resources.jsonl: each one's peak RSS, when it peaked, and its CPU
before and after T (HH:MM:SS).  python cir255_workers.py RUN_DIR CONTROLLER_PID T"""
import json, sys
from pathlib import Path
ctl, cut = int(sys.argv[2]), sys.argv[3]
w = {}
for line in open(Path(sys.argv[1]) / "resources.jsonl"):
    r = json.loads(line)
    if r.get("kind") != "process":
        continue
    t = r["t_utc"][11:19]
    for p in r["processes"]:
        if p["ppid"] != ctl:
            continue
        s = w.setdefault(p["pid"], {"hwm": 0, "at": "", "cpu_cut": 0, "cpu": 0})
        rss = p.get("rss_bytes") or 0
        if rss > s["hwm"]:
            s["hwm"], s["at"] = rss, t
        s["cpu"] = max(s["cpu"], p.get("cpu_s") or 0)
        if t <= cut:
            s["cpu_cut"] = s["cpu"]
for pid, s in sorted(w.items()):
    print(f"worker {pid}: peak {s['hwm']/2**30:.2f} GiB at {s['at']}, cpu {s['cpu_cut']:.0f} s by {cut}, {s['cpu']-s['cpu_cut']:.0f} s after")
