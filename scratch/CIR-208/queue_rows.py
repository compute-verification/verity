#!/usr/bin/env python3
"""queue_rows.py STATUS.txt [OLD.json]: `research queue status`'s rows as JSON, and what moved since an older read.

A row is `pos #PR sha8 state lane why`.  A pull request counts as marked when the queue admits it (`ready`), refuses it
(`refused`: only a marked one is judged), or holds it `waiting` for a reason other than "not marked ready".
"""
import json
import re
import sys
from pathlib import Path

ROW = re.compile(r"^\s*(\S+)\s+#(\d+)\s+([0-9a-f]{8})\s+(ready|waiting|refused)\s+(\S+)\s+(.*)$")


def rows(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            pos, pr, sha8, state, lane, why = m.groups()
            out.append({"pos": pos, "pr": pr, "sha8": sha8, "state": state, "lane": lane, "why": why,
                        "marked": state != "waiting" or "not marked ready" not in why})
    return out


if __name__ == "__main__":
    now = rows(Path(sys.argv[1]).read_text())
    if len(sys.argv) > 2:
        old = {r["pr"]: r for r in json.loads(Path(sys.argv[2]).read_text())}
        cur = {r["pr"]: r for r in now}
        for pr in sorted(set(cur) - set(old), key=int):
            print("new", pr, cur[pr]["sha8"], cur[pr]["state"], cur[pr]["lane"], "marked" if cur[pr]["marked"] else "unmarked")
        for pr in sorted(set(old) - set(cur), key=int):
            print("gone", pr, old[pr]["sha8"], old[pr]["state"], old[pr]["lane"], "marked" if old[pr]["marked"] else "unmarked")
        for pr in sorted(set(old) & set(cur), key=int):
            o, c = old[pr], cur[pr]
            if (o["sha8"], o["state"], o["marked"]) != (c["sha8"], c["state"], c["marked"]):
                print("moved", pr, f"{o['sha8']} {o['state']} {'marked' if o['marked'] else 'unmarked'} ->",
                      f"{c['sha8']} {c['state']} {'marked' if c['marked'] else 'unmarked'}", c["lane"])
    else:
        print(json.dumps(now, indent=1))
