#!/usr/bin/env python3
"""VLLM-289 scratch: a round's branches in N groups, from `meta.py`'s JSON -> DIR/args-K.txt (the driver's PR=HEAD
arguments) and DIR/parents.txt (every head, for the carrier commit a run ships).

    groups.py META.json N DIR [--only PR,PR...]

Left out: the pull requests the base already holds, and #8027 and #8656 on the lead's word. Each mend goes in as `mPR`.
Order: marked with a land verdict, marked, the five today's workers build on, the rest.
"""
import json
import os
import sys

meta, N, DIR = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3]
only = {x for x in sys.argv[5].split(",")} if len(sys.argv) > 5 and sys.argv[4] == "--only" else None
LEFT_OUT = {8027, 8656}
PRIORITY = [8448, 8529, 8567, 8566, 8585]

items = []
for r in meta["rows"]:
    n = r["pr"]
    if n in LEFT_OUT or r["in_base"]:
        continue
    k = 0 if r["marked"] and r["verdict"] == "yes" else 1 if r["marked"] else 2 if n in PRIORITY else 3
    items.append((k, n, str(n), r["head"]))
for label, m in meta["extra"].items():
    n = int(label.lstrip("m"))
    items.append((1 if n == 8384 else 2 if n in PRIORITY else 3, n, label, m["head"]))
if only:
    items = [i for i in items if i[2] in only]
items.sort()
groups = [[] for _ in range(N)]
for i, (_k, _n, label, h) in enumerate(items):
    groups[i % N].append(f"{label}={h}")
os.makedirs(DIR, exist_ok=True)
for k, g in enumerate(groups, 1):
    with open(os.path.join(DIR, f"args-{k}.txt"), "w") as f:
        f.write(" ".join(g) + "\n")
    print(k, len(g), " ".join(x.split("=")[0] for x in g))
with open(os.path.join(DIR, "parents.txt"), "w") as f:
    f.write(" ".join(sorted({h for *_x, h in items})) + "\n")
