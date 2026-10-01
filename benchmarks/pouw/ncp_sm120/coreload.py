#!/usr/bin/env python3
"""Per-core CPU load beside a timed window: /proc/stat deltas every INTERVAL s into OUT (JSONL), until SIGTERM or its parent
exits.

The first line is {"part": "header", "cores": N, "interval_s": ..., "pinning": PINNING}; then one line per interval,
{"t": UTC ISO, "busy": [busy fraction of each core over the interval, cpu0 first]}, busy = 1 - (idle + iowait) / total over
user..steal (guest is counted inside user).

  python3 coreload.py OUT [INTERVAL]   (env: PINNING, the window's own host-thread cores, recorded in the header)
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import sys
import time


def read() -> list[tuple[int, int]]:
    rows = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                name, *v = line.split()
                v = [int(x) for x in v]
                rows[int(name[3:])] = (sum(v[:8]), v[3] + v[4])
    return [rows[i] for i in sorted(rows)]


def main(argv: list[str]) -> int:
    interval = float(argv[2]) if len(argv) > 2 else 0.5
    parent = os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    prev = read()
    with open(argv[1], "w") as f:
        f.write(json.dumps({"part": "header", "cores": len(prev), "interval_s": interval,
                            "pinning": os.environ.get("PINNING")}) + "\n")
        while os.getppid() == parent:
            time.sleep(interval)
            cur = read()
            busy = [round(1 - (ci - pi) / (ct - pt), 3) if ct > pt else 0.0 for (ct, ci), (pt, pi) in zip(cur, prev)]
            f.write(json.dumps({"t": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                                "busy": busy}) + "\n")
            f.flush()
            prev = cur
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
