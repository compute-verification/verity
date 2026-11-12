"""When the xdist workers of controller PPID resumed after the catalog build: for each worker, the first sample at least 90 s
after the session's first at which its CPU rose by 3 s or more within 10 s.  python cir255_resume.py RUN_DIR PPID"""
import collections, json, sys
from datetime import datetime
from pathlib import Path
ctl = int(sys.argv[2])
ser = collections.defaultdict(list)
for line in open(Path(sys.argv[1]) / "resources.jsonl"):
    r = json.loads(line)
    if r.get("kind") != "process":
        continue
    t = datetime.fromisoformat(r["t_utc"].replace("Z", "+00:00")).timestamp()
    for p in r["processes"]:
        if p["ppid"] == ctl:
            ser[p["pid"]].append((t, p.get("cpu_s") or 0))
t0 = min(s[0][0] for s in ser.values())
fmt = lambda t: datetime.utcfromtimestamp(t).strftime("%H:%M:%S")
out = []
for pid, s in ser.items():
    hit = None
    for i, (t, c) in enumerate(s):
        if t - t0 < 90:
            continue
        j = i
        while j < len(s) and s[j][0] - t <= 10:
            j += 1
        if j - 1 > i and s[j - 1][1] - c >= 3:
            hit = t
            break
    out.append((hit or 0, pid))
print("session's first sample", fmt(t0))
for hit, pid in sorted(out):
    print(f"worker {pid}: resumed {fmt(hit) if hit else '-'} ({hit - t0:.0f} s in)" if hit else f"worker {pid}: -")
