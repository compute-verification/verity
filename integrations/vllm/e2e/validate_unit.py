"""Validate one plan unit (``run_composed.unit_plans``) in its own process and write the report entry.

    python -m e2e.validate_unit --case B1 --unit gemm:1536x17920 --out plan/unit-gemm-1536x17920.json \
        [--bound-steps 200000 --q 1/64 --s 1/8]

Used by ``run_composed``'s plan stage so that a unit's circuit (RA's validator holds one interval per gate;
``Gemm{1536,17920}`` of B1 peaks at ~20 GB) is released to the OS before the GPU stages start -- the runner
process must stay small beside the vLLM prover/verifier under the pod's 31 GB cgroup.  The entry has the same
fields ``stage_plan`` wrote in-process (INT): per partition status, codes, VUs/CUs, edges, visits, validate
seconds, the bound, plus ``maxrss_mb``.  Exit 0 = every partition valid, 1 = a partition invalid, 2 = error,
3 = MemoryError.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _maxrss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bound-steps", type=int, default=200_000)
    ap.add_argument("--q", default="1/64")
    ap.add_argument("--s", default="1/8")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {"unit": a.unit, "case": a.case, "status": "starting", "partitions": {}}

    def dump() -> None:
        doc["maxrss_mb"] = _maxrss_mb()
        out.write_text(json.dumps(doc, indent=1, sort_keys=True))

    dump()
    try:
        from e2e.run_composed import PLAN_CHECKS, plans_digest, static_configuration, unit_plans, unit_program
        from veritor.core.policy import VerificationPolicy
        from veritor.plans.bound import BoundLimits, bound_of, hierarchical_table
        from veritor.plans.validate import validate_plan

        config = static_configuration(a.case)
        t0 = time.perf_counter()
        Q, c, partitions = unit_plans(config, a.unit)
        root = Q.root
        doc.update({"program": Q.name, "gates": unit_program(config, a.unit).gates, "build_seconds": round(time.perf_counter() - t0, 2),
                    "root_nodes": [{"index": k, "form": n.form, "function": n.fn.name, "n": getattr(n, "n", None)} for k, n in enumerate(root.nodes)],
                    "plan_digest": plans_digest(Q, partitions), "status": "built"})
        dump()
        policy = VerificationPolicy(q=Fraction(a.q), s=Fraction(a.s))
        all_valid = True
        for tag, plan in partitions:
            t = time.perf_counter()
            v = validate_plan(c, plan)
            vt = time.perf_counter() - t
            entry: dict = {"name": plan.name, "vus": len(plan.verification_units), "cus": len(plan.commitment_units), "status": v.status, "codes": [str(x) for x in v.codes],
                           "reasons": list(v.reasons)[:5], "vu_edges": v.vu_edges, "cu_edges": v.cu_edges, "visits": v.visits, "validate_seconds": round(vt, 2),
                           "checks": PLAN_CHECKS}
            all_valid &= v.status == "valid"
            if v.status == "valid" and a.bound_steps > 0:
                t = time.perf_counter()
                try:
                    tab = hierarchical_table(c, plan, v, BoundLimits(max_reach_steps=a.bound_steps))
                    r = bound_of(tab.table, policy, Fraction(1, 64))
                    entry["bound"] = {"bits": float(r.bits) if r.bits is not None else None, "capped": bool(r.capped), "out_bits": int(r.out_bits),
                                      "vu_kinds": tab.vu_kinds, "cu_kinds": tab.cu_kinds, "fallback_units": tab.fallback_units,
                                      "policy": {"q": str(policy.q), "s": str(policy.s)}, "seconds": round(time.perf_counter() - t, 2)}
                except Exception as e:  # noqa: BLE001
                    entry["bound"] = {"error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - t, 2)}
            elif a.bound_steps == 0:
                entry["bound"] = {"skipped": "bound not computed (--bound-steps 0)"}
            doc["partitions"][tag] = entry
            doc["status"] = f"validated {tag}"
            dump()
            print(f"[validate_unit] {a.unit} {tag}: {v.status} vus={len(plan.verification_units)} edges={v.vu_edges}/{v.cu_edges} {vt:.1f}s", flush=True)
        doc["status"] = "valid" if all_valid else "invalid"
        doc["valid"] = all_valid
        dump()
        return 0 if all_valid else 1
    except MemoryError:
        doc["status"] = "MemoryError"
        dump()
        return 3
    except Exception as e:  # noqa: BLE001
        doc["status"] = f"error: {type(e).__name__}: {e}"[:400]
        dump()
        return 2


if __name__ == "__main__":
    sys.exit(main())
