"""Every count in CIR-271's cover comment, from cover.json's rows alone: per column, templates and instances in each state."""
import json, sys
from collections import Counter
rows = [r for r in json.load(open(sys.argv[1] if len(sys.argv) > 1 else __file__.replace("cover_check.py", "cover.json")))["rows"] if r["in_list"]]
state = {"pin": lambda r: r["pin"], "cut": lambda r: "cut" if r["cut"] else "none", "accepted": lambda r: r["accepted"],
         "staged": lambda r: "none" if r["staged"] is None else ("census" if r["staged"].startswith("census:") else "table")}
counts = {c: {s: [sum(1 for r in rows if f(r) == s), sum(r["instances"] for r in rows if f(r) == s)] for s in sorted(Counter(map(f, rows)))} for c, f in state.items()}
uncovered = sorted(([r["template"], r["instances"]] for r in rows if r["accepted"] not in ("yes", "boolean-version")), key=lambda x: (-x[1], x[0]))
print(json.dumps({"templates": len(rows), "instances": sum(r["instances"] for r in rows), "counts": counts,
                  "uncovered": [len(uncovered), sum(n for _, n in uncovered)]}, indent=1))
