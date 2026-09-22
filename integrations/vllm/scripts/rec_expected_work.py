"""REC: expected selected counts and proving work BY FAMILY, to print into the record directory BEFORE the challenges.

E[|Q|] = q * s * N_V over the ACTUAL VU population of the partition the record certifies:

* P1 (RU = request, VU = scalar coordinate relation): N_V(family) = number of output coordinates of the family's operator
  calls = ``len(circuit.call(...).returns)`` summed over the P3 operator units (each P3 unit is one operator call; its
  returns are exactly its coordinate VUs).  Counted on the laptop from the plan-IR (~4 s, < 300 MB for whole B0).
* P3 (VU = operator call): N_V(family) = number of operator units of the family.

Costs (GPU-s per obligation, one RTX 4090, MEASURED elsewhere and cited): GEMM coordinate 0.476 core + 0.595 compressed
(WARM-B0 prove-core/prove-compressed: 22 obligations, 39.75 M cycles); other families from docs/vllm-poc/proof-families.md
(core s/unit, unit = row/head, so they are UPPER bounds per coordinate).  Families without a measured host are UNPRICED.

    python scripts/rec_expected_work.py --case B0 --lp 32 --steps 15 --q 1/2 --s 1/16384 --out <record>/expected_work.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e import population as popmod  # noqa: E402

# GPU-s per obligation (core + compressed where measured), with provenance
COST = {
    "gemm": (0.476 + 0.595, "WARM-B0 prove-core 10.47 s + prove-compressed 13.08 s over 22 GEMM-coordinate obligations (MEASURED)"),
    "rmsnorm": (1.69 + 2.46, "proof-families.md rms-triton/n576 core 1.69 + compressed 2.46 s/unit (unit = row; upper bound per coordinate)"),
    "rope": (0.256 + 0.395, "proof-families.md rope/d64 core 0.256 + compressed 0.395 s/unit (unit = head)"),
    "residual_add": (1.998 + 2.893, "proof-families.md residual_add/h576 core + compressed s/unit (unit = row)"),
    "silu_mul": (5.503, "proof-families.md silu_mul/i1536 core s/unit (unit = row; no compressed row)"),
}


def family_of(fn_name: str) -> str:
    n = fn_name.lower()
    for key, fam in (("gemm", "gemm"), ("matmul", "gemm"), ("linear", "gemm"), ("rmsnorm", "rmsnorm"), ("rope", "rope"), ("add", "residual_add"),
                     ("silu", "silu_mul"), ("attn", "attention"), ("softmax", "attention"), ("embed", "embedding"), ("gather", "embedding"),
                     ("argmax", "argmax"), ("input", "input")):
        if key in n:
            return fam
    return "other"


def count(case: str, lp: int, steps: int) -> tuple[dict, dict, dict]:
    if case == "TINY":
        config = popmod.TINY
    else:
        from verity_vllm.registry import b1

        config = dict(getattr(b1, f"{case}_CONFIG"))
    program, adapted, circuit, pdigest = popmod.build(config, lp, steps, digest=True)
    plan, _ = popmod.make_plan("P3/cu=whole", adapted, circuit)  # 14,197 operator units for whole B0; dense is fine at P3
    p3: dict[str, int] = defaultdict(int)
    p1: dict[str, int] = defaultdict(int)
    fns: dict[str, set] = defaultdict(set)
    for desc in plan.verification_units:
        parts = desc.parts
        if len(parts) != 1:
            p3["(explicit gate set)"] += 1
            continue
        part = parts[0]
        if part.b - part.a != 1:  # a batch node counted as one P3 unit: members x returns per member
            scope = circuit.scope_at(part.path)
            node = scope.fn.nodes[part.node_idx]
            fam = family_of(node.fn.name)
            fns[fam].add(node.fn.name)
            p3[fam] += 1
            fc = circuit.call((*part.path, (part.node_idx, part.a)))
            p1[fam] += (part.b - part.a) * sum(1 for _ in fc.iter_returns())
            continue
        fc = circuit.call((*part.path, (part.node_idx, part.a)))
        fam = family_of(fc.name)
        fns[fam].add(fc.name)
        p3[fam] += 1
        p1[fam] += sum(1 for _ in fc.iter_returns())
    meta = {"program": program.fn.id if hasattr(program, "fn") else str(program), "program_digest": pdigest, "gates": circuit.size,
            "p3_units": len(plan.verification_units)}
    return dict(p3), dict(p1), {"meta": meta, "functions": {k: sorted(v) for k, v in fns.items()}}


def table(p3: dict, p1: dict, q: Fraction, s: Fraction, instances: int) -> list[dict]:
    rows = []
    for fam in sorted(set(p3) | set(p1)):
        cost, prov = COST.get(fam, (None, "UNPRICED: no measured family host"))
        for part_name, n in (("P1", p1.get(fam, 0)), ("P3", p3.get(fam, 0))):
            eq = float(q * s) * n * instances
            rows.append({"family": fam, "partition": part_name, "N_V_per_instance": n, "instances": instances, "E_Q": round(eq, 3),
                         "gpu_s_per_obligation": cost, "expected_gpu_h": (round(eq * cost / 3600, 4) if cost else None), "cost_source": prov})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="B0")
    ap.add_argument("--lp", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--instances", type=int, default=1, help="requests in the record (program instances)")
    ap.add_argument("--q", default="1/2")
    ap.add_argument("--s", default="1/16384")
    ap.add_argument("--p1-units-validated", type=int, default=None,
                    help="N_V of the VALIDATED P1 plan (compact validation artefact); the per-family P1 counts here are operator OUTPUT "
                         "coordinates only and undercount interior coordinate VUs (TINY: 27,949 vs 53,113 validated) -- the ratio is reported")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = time.perf_counter()
    p3, p1, extra = count(a.case, a.lp, a.steps)
    q, s = Fraction(a.q), Fraction(a.s)
    rows = table(p3, p1, q, s, a.instances)
    tot = {p: {"N_V": sum(r["N_V_per_instance"] for r in rows if r["partition"] == p) * a.instances,
               "E_Q": round(sum(r["E_Q"] for r in rows if r["partition"] == p), 3),
               "priced_gpu_h": round(sum(r["expected_gpu_h"] or 0 for r in rows if r["partition"] == p), 4),
               "unpriced_E_Q": round(sum(r["E_Q"] for r in rows if r["partition"] == p and r["expected_gpu_h"] is None), 3)} for p in ("P1", "P3")}
    doc = {"schema": "veritor/rec-expected-work/v1", "label": "DERIVED (N_V counted on the plan-IR; costs MEASURED as cited)", "q": str(q), "s": str(s),
           **extra, "rows": rows, "totals": tot, "seconds": round(time.perf_counter() - t, 1),
           "p1_units_validated": a.p1_units_validated,
           "p1_undercount_ratio": (round(a.p1_units_validated / max(1, tot["P1"]["N_V"]), 3) if a.p1_units_validated else None),
           "caveat": "P1 rows count operator OUTPUT coordinates (call returns); the validated P1 plan also has interior coordinate VUs, so "
                     "scale E|Q| by p1_undercount_ratio (TINY: 1.9x) or use the validation artefact's vus directly",
           "note": "E_Q = q*s*N_V*instances; a Bernoulli draw realises |Q| ~ Binomial; budget on E_Q + 3 sqrt(E_Q)"}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"[expected-work] {extra['meta']['program']} gates={extra['meta']['gates']:,} P3 units={extra['meta']['p3_units']:,} q={q} s={s} x{a.instances}")
    print(f"{'family':14} {'part':4} {'N_V':>12} {'E|Q|':>10} {'GPU-s/obl':>10} {'GPU-h':>8}")
    for r in rows:
        print(f"{r['family']:14} {r['partition']:4} {r['N_V_per_instance']:>12,} {r['E_Q']:>10.2f} {str(r['gpu_s_per_obligation'] or '-'):>10} {str(r['expected_gpu_h'] if r['expected_gpu_h'] is not None else 'UNPRICED'):>8}")
    for p, tv in tot.items():
        print(f"TOTAL {p}: N_V={tv['N_V']:,} E|Q|={tv['E_Q']} priced GPU-h={tv['priced_gpu_h']} (unpriced E|Q| {tv['unpriced_E_Q']}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
