"""[E6 CURVE 2026-09-13, D54] Does L5's population RULE (census_v2.small_scaling basis at the registry's FA2 block) equal the BUILT P1/cu=whole plan's kind
table at a shape?  Builds the plan in-process (laptop: shapes with P = LP + STEPS <= ~140 fit the 2.5 GB guardian with E2's columnar runs), derives
the population both ways through population_v3 -- `expected_population` (rule) and `expected_population_serve_plan` (plan) -- and prints the per-kind
differences.  Exit 0 iff equal.  MEASURED evidence for the rule at a block size the rule was not fitted on (BN 128 after E7's `dfcea003`).

    PYTHONPATH=. python scripts/rule_vs_plan.py B0 100 32 [--json out.json] [--attention '{"kernel":"fa2-splitkv","kBlockM":64,"kBlockN":128,"num_splits":1}']
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(ROOT), str(POC), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main(argv: list[str]) -> int:
    case, lp, steps = argv[0], int(argv[1]), int(argv[2])
    out = Path(argv[argv.index("--json") + 1]) if "--json" in argv else None
    attention = json.loads(argv[argv.index("--attention") + 1]) if "--attention" in argv else None  # [D57] declared geometry; None = legacy
    from census_v2 import small_scaling
    from e2e.population_v3 import PLAN_NAME_P1, expected_population, expected_population_serve_plan, pinned_instance

    pn = PLAN_NAME_P1 + "/token-select=whole-scan/literals=compiled"
    pin = pinned_instance({"case": case, "lp": lp, "steps": steps, "plan_name": pn, "instances": 1, "attention": attention, "family": "Serve_v4" if attention else "Serve_v1"})
    t0 = time.perf_counter()
    rule = expected_population(pin) if case == "B0" else None
    t1 = time.perf_counter()
    plan = expected_population_serve_plan(pin, plan_name=pn, token_select="whole-scan", literals="compiled")
    t2 = time.perf_counter()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)
    pk = {k["kind"]: int(k["count"]) for k in plan["kinds"]}
    doc = {"case": case, "lp": lp, "steps": steps, "P": lp + steps, "attention": attention, "fa2_block_n": small_scaling.fa2_block_n(attention), "plan": {"N_V": int(plan["vu_count_per_ru"]), "kinds": len(pk), "program_digest": plan.get("program_digest") or (plan.get("source") or {}),
           "seconds": round(t2 - t1, 1)}, "peak_rss_gb": round(rss, 2), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if rule is not None:
        rk = {k["kind"]: int(k["count"]) for k in rule["kinds"]}
        diff = {k: {"rule": rk.get(k), "plan": pk.get(k)} for k in sorted(set(rk) | set(pk)) if rk.get(k) != pk.get(k)}
        doc.update(rule={"N_V": int(rule["vu_count_per_ru"]), "kinds": len(rk), "seconds": round(t1 - t0, 2)}, differences=diff, equal=not diff)
        print(f"[rule_vs_plan] {case} LP={lp} STEPS={steps} (P={lp + steps}, BN={doc['fa2_block_n']}): rule N_V {rk and sum(rk.values()):,} vs plan N_V {sum(pk.values()):,}; "
              f"{'EQUAL over ' + str(len(pk)) + ' kinds' if not diff else 'DIFFER in ' + str(len(diff)) + ' kinds: ' + json.dumps(diff)}; plan {t2 - t1:.1f} s, peak RSS {rss:.2f} GB")
    else:
        print(f"[rule_vs_plan] {case} LP={lp} STEPS={steps}: no rule for this case; plan N_V {sum(pk.values()):,} over {len(pk)} kinds; {t2 - t1:.1f} s, {rss:.2f} GB")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0 if doc.get("equal", True) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
