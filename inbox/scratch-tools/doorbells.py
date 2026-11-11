#!/usr/bin/env python3
"""top's unread doorbells, minus monitor noise, for the quarterback tick.

Usage: doorbells.py [--mark]
Prints `DOORBELL HH:MMZ | text` for each unread doorbell that a person or lane sent, and a count line.
With --mark, marks every doorbell read afterwards (`research msg read --as top`), so the next tick shows only new ones.
Exits 1 when any actionable doorbell is unread.
"""
import datetime
import json
import os
import re
import subprocess
import sys

CWD = "/tmp/rchk" if os.path.isdir("/tmp/rchk/.agents") else None

NOISE = re.compile(
    r"Agent watch|^from runs|from dashboard|from sweep|main has gone|registered no run|"
    r"from vy-monitors",
    re.I,
)


def main() -> int:
    out = subprocess.run(
        ["research", "msg", "read", "--as", "top", "--peek", "--json"],
        capture_output=True, text=True, timeout=120, cwd=CWD,
    )
    try:
        bells = json.loads(out.stdout)
    except json.JSONDecodeError:
        print("  DOORBELL check failed: " + (out.stderr or out.stdout)[-200:])
        return 1
    keep = [b for b in bells if not NOISE.search(b.get("text") or "")]
    for b in keep:
        at = datetime.datetime.fromtimestamp(float(b.get("ts") or 0), datetime.UTC).strftime("%d %H:%MZ")
        print(f"  DOORBELL {at} | {(b.get('text') or '')[:240].replace(chr(10), ' / ')}")
    print(f"  doorbells: {len(keep)} actionable of {len(bells)} unread")
    if "--mark" in sys.argv:
        subprocess.run(["research", "msg", "read", "--as", "top", "--json"], capture_output=True, timeout=120, cwd=CWD)
    return 1 if keep else 0


if __name__ == "__main__":
    sys.exit(main())
