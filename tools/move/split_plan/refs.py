"""Append to Extract.lean's deps.tsv the references each module's `.ilean` records: `R	module	declaration	constant`, one
per in-tree constant a declaration names. The constant graph reads proof terms, which leave out what elaboration consumed
without a trace (a `simp` lemma proved by `rfl`); the `.ilean` holds every identifier as written.

    python3 refs.py REPO mods.txt deps.tsv        (after `lake build`, which writes each module's .ilean)"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("mods")
ap.add_argument("deps")
a = ap.parse_args()
sys.argv = ["x", a.repo]
sys.path.insert(0, str(Path(__file__).parent))
import graph  # noqa: E402

mods = [m for m in Path(a.mods).read_text().split() if m in graph.MODS]
tracked = set(graph.MODS)
seen, missing = set(), []
for m in mods:
    p = graph.MODS[m][2] / ".lake/build/lib/lean" / (m.replace(".", "/") + ".ilean")
    if not p.exists():
        missing.append(m)
        continue
    for key, info in json.loads(p.read_text()).get("references", {}).items():
        c = json.loads(key).get("c")
        if not c or c.get("m") not in tracked:
            continue
        for u in info.get("usages", []):
            if len(u) > 4 and u[4]:
                seen.add((m, u[4], c["n"]))
with open(a.deps, "a") as f:
    f.writelines(f"R\t{m}\t{d}\t{n}\n" for m, d, n in sorted(seen))
print(f"{len(seen)} references from {len(mods) - len(missing)} modules' .ilean; {len(missing)} without one"
      + (f" ({', '.join(missing[:5])}{', …' if len(missing) > 5 else ''})" if missing else ""))
sys.exit(1 if len(missing) * 10 > len(mods) else 0)
