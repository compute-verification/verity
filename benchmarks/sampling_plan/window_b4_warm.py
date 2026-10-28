"""`window_b4.py`'s work and joint plans, and `pouw_tile.py`'s tile, under a warm prover: GD-PROVER's refit of C-Flock's
RTX PRO 6000 sessions (r20261018-060433-b55e, note:compute-research/20261028T1955Z-report-gd-prover), a session of k drawn
blocks of A ANDs costing a + b·A + c·A·P GPU-s with the prover process kept, and the Lean verifier 12.1 + 6.6e-6·A CPU-s.

Both cost models, fitted (`window_b4.costs`, a cold process's statement) and warm, report what the planner leaves out:
C-Flock pads a session's k drawn blocks to P = max(8, 2^ceil(log2 k)), so `padding_gpu_s` is c·A·(P − k) summed, and
`eps_padded` the free compute certified when the padded blocks are drawn too (they cost nothing more).

    uv run python benchmarks/sampling_plan/window_b4_warm.py CENSUS.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from math import ceil, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pouw_tile as T
import window_b4 as W

from verity.core.protocols.sampled_proofs.optimize import Candidate, CostModel, Objective, Plan, certify, for_target

#: GD-PROVER's warm-session refit: a (s), b (s a block AND), c (s a padded drawn AND); the Lean verifier's CPU-s
WARM = {"a": 2.05, "b": 1.14e-6, "c": 1.20e-8, "verify_fixed": 12.1, "verify_per_and": 6.6e-6}


def warm(cpu: float) -> tuple[CostModel, dict[str, CostModel]]:
    prove = CostModel(per_and=WARM["c"], statement_fixed=WARM["a"], statement_per_and=WARM["b"], shared=True)
    verify = CostModel(statement_fixed=WARM["verify_fixed"], statement_per_and=WARM["verify_per_and"], shared=True)
    total = CostModel(per_and=WARM["c"], statement_fixed=WARM["a"] + cpu * WARM["verify_fixed"],
                      statement_per_and=WARM["b"] + cpu * WARM["verify_per_and"], shared=True,
                      source="GD-PROVER warm refit of r20261018-060433-b55e")
    return total, {"prove_gpu_s": prove, "verify_cpu_s": verify}


def padded(p: Plan, cands: list[Candidate]) -> dict[str, int]:
    """Each template's drawn blocks as C-Flock proves them: its k split evenly over ceil(k / cap) sessions, each padded
    to P = max(8, 2^ceil(log2 k_s)) blocks, at most the template's units."""
    out = {}
    for c in cands:
        k = p.sizes.get(c.template, 0)
        if k:
            s = ceil(k / c.cap)
            ks = ceil(k / s)
            out[c.template] = min(c.units, s * max(8, 1 << ceil(log2(ks))))
    return out


def plan(cands: list[Candidate], cm: CostModel, split: dict[str, CostModel], *, grid: int, joint: bool) -> dict:
    F = W.window_work(cands)
    work = for_target(cands, cm, Objective(W.DELTA, work=1.0), W.EPS * F, by="free_compute", exact=False, grid=grid)
    chosen = [c for c in cands if c.template in work.sizes]
    out = {"work": W.report(work, chosen, cm, F, split)}
    if joint:
        out["joint ℓ≤0.01"] = W.joint(work, cands, cm, split, F, 0.01 * out["work"]["window_out_bits"], grid=grid)
    pads = padded(work, chosen)
    out["work"]["padding_gpu_s"] = sum(cm.per_and * cm.ands(c) * (pads[c.template] - work.sizes[c.template])
                                       for c in chosen)
    out["work"]["padded_draws"] = sum(pads.values())
    out["work"]["eps_padded"] = certify(pads, chosen, Objective(W.DELTA, work=1.0)).free_compute / F
    for x in out.values():
        if x:
            x["grains"] = {op: g["cut"] for op, g in x["grains"].items()}
            x.pop("sizes", None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("census")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=61)
    args = ap.parse_args()
    census = json.loads(Path(args.census).read_text())
    f = W.fitted()
    cm_warm, split_warm = warm(f.cpu)
    models = {"fitted": (W.costs(), W.parts(f)), "warm": (cm_warm, split_warm)}
    b4 = W.candidates(census)
    tiles, table = T.tiles(census, 1 << 27)
    doc = {"warm": WARM, "cpu": f.cpu, "runs": {}}
    for name, (cm, split) in models.items():
        doc["runs"][f"{name} N grains"] = plan(b4, cm, split, grid=args.grid, joint=True)
        doc["runs"][f"{name} N grains, attention buckets"] = plan(W.bucketed(b4), cm, split, grid=args.grid, joint=True)
        doc["runs"][f"{name} served grain, attention buckets"] = plan(W.bucketed(W.candidates(census, widths=())), cm,
                                                                      split, grid=args.grid, joint=True)
        tc = T.lanes_priced(cm, table)
        r = plan(tiles, tc, {k: T.lanes_priced(m, table) for k, m in split.items()}, grid=args.grid, joint=False)
        doc["runs"][f"{name} pouw tile"] = r
    for name, r in doc["runs"].items():
        for tgt, x in r.items():
            if x:
                s = ", ".join(f"{op.split(',DOT')[0]} {cut}" for op, cut in x["grains"].items())
                print(f"{name:40s} {tgt:12s} s*: {s} | draws {x['draws']} | proving ANDs {x['proving_ands']:.4g} | "
                      f"{x['total_gpu_s']:.1f} GPU-s (statements {x['statements_gpu_s']:.1f}, prove {x['prove_gpu_s']:.1f}, "
                      f"verify {x['verify_cpu_s']:.0f} CPU-s) | eps {x['eps']:.4g} | leak {x['leak_bits']:.4g}"
                      + (f" | padding {x['padding_gpu_s']:.1f} GPU-s, {x['padded_draws']} draws, eps {x['eps_padded']:.4g}"
                         if "padding_gpu_s" in x else ""))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "window_b4_warm.json").write_text(json.dumps(doc, indent=1, default=str))


if __name__ == "__main__":
    main()
