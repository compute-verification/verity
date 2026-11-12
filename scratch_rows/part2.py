"""Scratch, never lands: the library's rows in both units, from two runs' library.json (main's unit, then rows)."""
import json
import sys
import tomllib
from pathlib import Path

main, rows = (json.loads(Path(p).read_text())["rows"] for p in sys.argv[1:3])
held = {k for k in tomllib.loads(Path("benchmarks/training_slowdown/training_slowdown/of_record.toml").read_text())
        .get("held", {})}
by = {r["entry"]: r for r in rows}
out = []
for a in main:
    b = by[a["entry"]]
    g0, g1 = (a.get("step") or {}).get("gates"), (b.get("step") or {}).get("gates")
    r0, r1 = a.get("slowdown"), b.get("slowdown")
    p0, p1 = a.get("periods") or {}, b.get("periods") or {}
    steps = a["budget"].get("steps")
    f_C = g0 / g1 if g0 and g1 else None
    row = {"entry": a["entry"], "held": a["entry"] in held, "start": a["start"], "route": a["route"],
           "priced_in": (a.get("step") or {}).get("priced_in"), "binds": a.get("binds"), "binds_rows": b.get("binds"),
           "C": g0, "C_rows": g1, "f_C": f_C,
           "c": a["policy"]["c_gates"], "c_rows": b["policy"]["c_gates"], "f_c": a["policy"]["c_gates"] / b["policy"]["c_gates"],
           "k": a["policy"]["k_bits"], "rho": r0, "rho_rows": r1, "rho_factor": r1 / r0 if r0 and r1 else None,
           "run_and": p0.get("run_and_gates"), "run_rows": p1.get("run_and_gates"),
           "run_factor": p1["run_and_gates"] / p0["run_and_gates"] if p0.get("run_and_gates") and p1.get("run_and_gates") else None,
           "true_run_and": steps * g0 * r0 / f_C if steps and g0 and r0 and f_C else None}
    if row["true_run_and"] and row["run_and"]:
        row["per_period_undercount"] = 1 - row["run_and"] / row["true_run_and"]
    out.append(row)
json.dump(out, sys.stdout, indent=1)
