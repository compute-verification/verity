"""Time-weighted critical path from Lake's build log (`Built M (t)`) and/or times.jsonl, over the import graph of ROOT.

    python3 chain_time.py REPO BUILD_LOG [TIMES_JSONL] [--root M] [--default S]

A module with no measured time weighs --default seconds (or its lines times the measured seconds per line)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("log")
ap.add_argument("times", nargs="?")
ap.add_argument("--root", default="Proofs.Flock")
ap.add_argument("--out")
a = ap.parse_args()
sys.argv = ["x", a.repo]
import graph  # noqa: E402

built = {}
for m in re.finditer(r"\] Built (\S+) \(([\d.]+)(m?s)\)", Path(a.log).read_text()):
    built[m.group(1)] = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
prof = {}
if a.times:
    for line in Path(a.times).read_text().splitlines():
        r = json.loads(line)
        if r["rc"] == 0:
            prof[r["module"]] = r["wall_s"]
cl = graph.closure(a.root)
meas = {m: built.get(m, prof.get(m)) for m in cl}
known = [m for m in cl if meas[m] is not None]
spl = sum(meas[m] for m in known) / max(1, sum(graph.lines(m) for m in known))
base = min((meas[m] for m in known), default=3.0)
est = lambda m: meas[m] if meas[m] is not None else base + spl * graph.lines(m)
tot, path = graph.longest(a.root, est)
print(f"closure {len(cl)}; measured {len(known)} (build log {sum(m in built for m in cl)}, profiler {sum(m in prof and m not in built for m in cl)}); "
      f"{spl * 1000:.1f} ms/line, floor {base:.1f} s")
print(f"critical path by time: {tot / 60:.1f} min over {len(path)} modules; measured on it {sum(meas[m] is not None for m in path)}")
print(f"sum of all closure times (CPU): {sum(est(m) for m in cl) / 60:.0f} min")
for m in path:
    print(f"{est(m):7.1f} {'*' if meas[m] is None else ' '} {graph.lines(m):5d} {m}")
if a.out:
    json.dump({m: est(m) for m in cl}, open(a.out, "w"))
