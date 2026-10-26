"""Demo A's frontier on one H200 (PRF-166): plain INT8 weights, no P2 decode, a block of 16,384 requests at a 7.84 s step
(compute-accounting, CMP-92 00:50Z), on `h200.templates`' partition, planned by `optimize.frontier` under Daniel's
objective (25 Oct 5:45 PM PDT): half skipped credited work, half leaked output bits.

**Harm.** A unit's credited work is PoUW's: the static INT8 coordinates' K MACs, and 0 for every unit PoUW credits nothing
(attention's BF16 GEMMs, every row op); no unit carries influence, since no unit here is one PoUW's own check reads (noise,
form). Every unit carries its output bits. At each budget B the exchange rate is top's: each half is divided by the
least bound B certifies for that half alone, `Objective(delta, work=0.5/H_work(B), bits=0.5/H_bits(B))`, which is
scale-free, and the plan reports the certificate's two bounds from its one draw, `free_compute` and `leak`, at delta = 1%
and 1e-6.

**Certificate.** `certify`'s: one stratified harm bound over every template (core's `Stratified.harm_bound`), with no
delta split by template, its leak's location term counting the wrong units by the unit-weighted harm bound in place of the
exact `count_bound` (`certificate`). `stratified_isHarmBound_exact` (Security/Proofs/Flock/Soundness/Audit/HarmBound.lean) proves the
knapsack's value an `IsHarmBound` of the stratified law; the locked `Partition.audit_influence` and
`Partition.audit_exfiltration` turn such a bound into the accepted audit's bound under the backend's open obligations
(KnowledgeSound, LinkSound), for the abstract one-stage audit, not a Lean function; and no difftest holds core's Python
`harm_bound` to the Lean.

**Cost.** `h200.Rates`' fit with statements shared (`CostModel.shared`: one session a template a block, `ceil(k / cap)`
statements of drawn units), every unit's outputs committed at the GPU row committer's 225 GB/s (CMP-92 00:45Z), and a
drawn request replayed at its share of the block's GPU time (a stand-in).

    uv run python benchmarks/sampling_plan/demo_a.py --out DIR [--pad]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from math import floor, inf, log, log2, nextafter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h200 as H  # noqa: E402
import transformer as X  # noqa: E402

from verity.core.protocols.profile import MARGIN, Stratified, Stratum  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import (  # noqa: E402
    Candidate, Certificate, CostModel, Objective, Plan, ReplayDomain, _plan, density_bound, for_budget, frontier)

#: Demo A's block (CMP-92 00:50Z): 16,384 requests, a 7.84 s step (GEMM 5.08 s, attention and elementwise ~0.25 s,
#: commit 2.51 s), 32 steps a request
SHAPE = H.Shape(batch=16_384, step_s=7.84, steps=32)
#: the GPU row committer (HASH-17, r20261025-234226-299c), bytes a second
COMMIT_BPS = 225e9
DELTAS = H.DELTAS
#: top's proposed budget, still with Daniel: proving at this share of the block's GPU time
BUDGET_SHARE = 0.01
#: the budgets swept, in multiples of the block's GPU time, past the cheapest plan's
GRID = (1e-2, 0.1, 1, 12.5, 15, 20, 30, 50, 100, 300, 1000, 3000, 1e4)
S8 = "ScaledMmS8Coordinate_v1"


def candidates(m: H.Shape = SHAPE, *, pad: bool = False) -> list[Candidate]:
    """One candidate a template, its unit's credited work PoUW's (an s8 coordinate's K MACs, else 0), its output bits
    committed, and its statement cap `_cap` of its ANDs with its rows' openings."""
    out = []
    for t in H.templates(m, pad=pad):
        work = int(t.id.split("K=")[1].split(",")[0].rstrip("}")) if t.id.startswith(S8) else 0
        out.append(Candidate(t.id, t.id, t.units, work, t.out_bits, ands=t.ands, rows=t.rows, commit_bits=t.out_bits,
                             cap=X._cap(t.proved), replay="request", per_replay=t.units // m.batch))
    return out


def costs(m: H.Shape = SHAPE) -> CostModel:
    r = H.Rates.fitted()
    return CostModel(per_and=r.per_and, statement_fixed=r.statement_fixed, statement_per_and=r.statement_per_and,
                     commit_per_bit=1 / (8 * COMMIT_BPS), replay={"request": ReplayDomain(m.batch, m.block_s / m.batch)},
                     shared=True, source="h200.Rates.fitted")


@dataclass(frozen=True)
class Point:
    budget: float
    h_work: float
    h_bits: float
    plan: Plan
    at: dict[str, dict[str, float]]


def location(n: int, count: int) -> float:
    """`consumers.location_bits` at a count bound `count` of n units: log2 of sum_{j <= count} C(n, j), rounded up."""
    if count <= 0 or n == 0:
        return 0.0
    if 2 * count >= n:
        return float(n)
    x = count / n
    return min(float(n), nextafter(n * (-x * log2(x) - (1 - x) * log2(1 - x)) * (1 + MARGIN), inf))


def certificate(sizes: dict[str, int], cands: list[Candidate], objective: Objective) -> Certificate:
    """`certify`'s certificate, its location term's count the floor of core's `harm_bound` with every unit weighing 1
    instead of the exact `count_bound`, whose exact binomials take hours at 1e12 units. Any set with more units escapes
    with less than delta, so that floor is `audit_exfiltration`'s k (`L.miss (k + 1) < delta`), and as it is at least
    `count_bound` the leak is at least `certify`'s."""
    chosen = [c for c in cands if c.template in sizes]
    d = objective.delta
    law = Stratified(tuple(Stratum(c.template, c.units, sizes[c.template]) for c in chosen))
    harm = law.harm_bound({c.template: objective.harm(c) for c in chosen}, d)
    work = law.harm_bound({c.template: float(c.work + c.influence) for c in chosen}, d)
    count = floor(law.harm_bound({c.template: 1.0 for c in chosen}, d))
    leak = law.harm_bound({c.template: float(c.out_bits) for c in chosen}, d) + location(law.population, count)
    dens = density_bound(((c.units, sizes[c.template], objective.harm(c)) for c in chosen), d)
    return Certificate(d, harm, dens, work, leak)


def _best(cands, cm, objective, budget, grid) -> Plan | None:
    """The least-harm plan within `budget`, the sweep screened by `density_bound` and only the plan taken certified
    (`certificate`)."""
    plans = frontier(cands, cm, objective, grid=grid, exact=False)
    p = for_budget(plans, budget, candidates=cands, costs=cm, objective=objective, exact=False)
    return None if p is None else replace(p, certificate=certificate(p.sizes, cands, objective))


def sweep(cands: list[Candidate], cm: CostModel, m: H.Shape = SHAPE, *, grid: int = 41,
          budgets: tuple[float, ...] = GRID) -> dict:
    """At each budget each half's least bound alone, then the plan under top's rate there, certified at both deltas from
    its one draw."""
    d = DELTAS[0]
    commit = sum(cm.commit(c) for c in cands)
    work_only, bits_only = Objective(d, work=1.0), Objective(d, work=0.0, bits=1.0)
    floor = _plan(cands, cm, bits_only, 0.0, frozenset(c.template for c in cands), frozenset())
    cheapest = floor.cost["total"] - commit
    points, skipped, complete = [], [], None
    for g in budgets:
        proving = g * m.block_s
        if proving < cheapest:
            skipped.append(proving)
            continue
        hw = _best(cands, cm, work_only, proving + commit, grid).certificate.free_compute
        hb = _best(cands, cm, bits_only, proving + commit, grid).certificate.leak
        if not (hw >= 1 and hb >= 1):
            complete = proving
            break
        mixed = Objective(d, work=0.5 / hw, bits=0.5 / hb)
        p = _best(cands, cm, mixed, proving + commit, grid)
        at = {}
        for delta in DELTAS:
            c = p.certificate if delta == d else certificate(p.sizes, cands, replace(mixed, delta=delta))
            at[str(delta)] = {"free_compute": c.free_compute, "leak": c.leak}
        points.append(Point(proving, hw, hb, p, at))
    return {"commit": commit, "cheapest": cheapest, "skipped": skipped, "points": points, "complete": complete}


def knee(points: list[Point]) -> int:
    """The point farthest below the chord between the ends of log(cost) against log(mixed harm), each half at its
    budget's rate (`kneedle` on the log-log frontier)."""
    xs = [log(p.budget) for p in points]
    ys = [log(0.5 * p.at["0.01"]["free_compute"] / p.h_work + 0.5 * p.at["0.01"]["leak"] / p.h_bits) for p in points]
    if len(points) < 3:
        return 0
    (x0, y0), (x1, y1) = (xs[0], ys[0]), (xs[-1], ys[-1])
    return max(range(len(points)), key=lambda i: (y0 + (y1 - y0) * (xs[i] - x0) / (x1 - x0)) - ys[i])


def report(m: H.Shape = SHAPE, *, pad: bool = False) -> dict:
    cands, cm = candidates(m, pad=pad), costs(m)
    s = sweep(cands, cm, m)
    pts = s["points"]
    k = knee(pts) if pts else None
    work = sum(c.units * c.work for c in cands)
    bits = sum(c.units * c.out_bits for c in cands)

    def row(p: Point) -> dict:
        return {"proving": p.budget, "of_block": p.budget / m.block_s, "cost": dict(p.plan.cost),
                "weights": {"work": 0.5 / p.h_work, "bits": 0.5 / p.h_bits}, "H_work": p.h_work, "H_bits": p.h_bits,
                "at": p.at, "drawn": sum(p.plan.sizes.values()), "sizes": dict(p.plan.sizes),
                "statements": dict(p.plan.statement_size)}

    return {"shape": vars(m) | {"block_s": m.block_s}, "pad": pad, "templates": len(cands),
            "units": sum(c.units for c in cands), "work": work, "bits": bits, "commit": s["commit"],
            "cheapest": s["cheapest"], "unaffordable": s["skipped"],
            "budget": BUDGET_SHARE * m.block_s, "budget_affordable": BUDGET_SHARE * m.block_s >= s["cheapest"],
            "frontier": [row(p) for p in pts], "knee": k}


def render(o: dict) -> str:
    b = o["shape"]["block_s"]
    lines = [f"# Demo A's frontier: {o['templates']} templates, {o['units']:.3g} units, {o['work']:.3g} credited MACs "
             f"and {o['bits']:.3g} output bits a block of {b:g} GPU-s", "",
             f"Committing every unit's outputs: {o['commit']:.4g} GPU-s a block, beside every plan. The cheapest plan "
             f"proves for {o['cheapest']:.4g} GPU-s ({o['cheapest'] / b:.3g} of the block); top's budget, "
             f"{o['budget']:.3g} GPU-s, " + ("affords it." if o["budget_affordable"] else "affords no plan."), "",
             "| proving GPU-s | of the block | H_work(B) | H_bits(B) | free_compute at 1% | leak at 1% (bits) | "
             "free_compute at 1e-6 | leak at 1e-6 | drawn | |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, f in enumerate(o["frontier"]):
        a, z = f["at"]["0.01"], f["at"]["1e-06"]
        lines.append(f"| {f['proving']:.4g} | {f['of_block']:g} | {f['H_work']:.3g} | {f['H_bits']:.3g} | "
                     f"{a['free_compute']:.3g} | {a['leak']:.3g} | {z['free_compute']:.3g} | {z['leak']:.3g} | "
                     f"{f['drawn']:,} | {'knee' if i == o['knee'] else ''} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", action="store_true")
    a = ap.parse_args(argv)
    o = report(pad=a.pad)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frontier.json").write_text(json.dumps(o, indent=1, default=str))
    print(render(o))


if __name__ == "__main__":
    main()
