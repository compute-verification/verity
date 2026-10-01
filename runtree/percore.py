"""Per-core busy % from /proc/stat deltas, one JSON line per second, between START and END (UTC ISO times)."""
import datetime as dt
import json
import sys
import time

start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
ts = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
t0, t1 = ts(start), ts(end)


def read():
    rows = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                name, *v = line.split()
                v = list(map(int, v[:8]))
                rows[int(name[3:])] = (sum(v), v[3] + v[4])
    return rows


if time.time() < t0:
    time.sleep(t0 - time.time())
with open(out, "a", buffering=1) as f:
    prev = read()
    f.write(json.dumps({"schema": "percore-busy/v0", "cores": len(prev), "start": start, "end": end}) + "\n")
    while time.time() < t1:
        time.sleep(1.0)
        cur = read()
        busy = []
        for c in sorted(cur):
            dt_all = cur[c][0] - prev[c][0]
            dt_idle = cur[c][1] - prev[c][1]
            busy.append(round(100 * (dt_all - dt_idle) / dt_all) if dt_all else 0)
        prev = cur
        f.write(json.dumps({"t": dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ"), "busy": busy}) + "\n")
