"""Scratch, never lands: the library's rows in three units from library3.sh's output: OUT/UNIT/ENTRY/library.json,
UNIT 0 gates as main prints them, 1 AND gates of the Definitions, 2 rows m of each Definition lowered alone."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
U = ("0", "1", "2")
rows = {}
for u in U:
    for f in sorted((root / u).glob("*/library.json")):
        for r in json.loads(f.read_text())["rows"]:
            rows.setdefault(r["entry"], {})[u] = r
out = []
for e, by in sorted(rows.items()):
    if set(by) != set(U):
        out.append({"entry": e, "missing": sorted(set(U) - set(by))})
        continue
    a = by["0"]
    row = {"entry": e, "start": a["start"], "route": a["route"], "steps": a["budget"]["steps"], "k": a["policy"]["k_bits"],
           "binds": {u: by[u].get("binds") for u in U}, "priced_in": (a.get("step") or {}).get("priced_in")}
    for u in U:
        r = by[u]
        C, rho = (r.get("step") or {}).get("gates"), r.get("slowdown")
        row |= {f"C{u}": C, f"c{u}": r["policy"]["c_gates"], f"rho{u}": rho,
                f"blocks{u}": (r.get("bound") or {}).get("blocks"),
                f"run{u}": a["budget"]["steps"] * C * rho if C and rho else None,
                f"printed_run_and{u}": (r.get("periods") or {}).get("run_and_gates")}
    out.append(row)
json.dump(out, sys.stdout, indent=1)
