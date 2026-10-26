"""Demo A's frontier on one H200 (PRF-166): plain INT8 weights, no P2 decode, a block of 16,384 requests at a 7.84 s step
(compute-accounting, CMP-92 00:50Z), on `h200.templates`' partition fitted to C-Flock's size cap (`fit`), planned by
`optimize.frontier` under Daniel's objective (25 Oct 5:45 PM PDT, its work half circuit size from 5:55 PM): half skipped
work, half leaked output bits.

**Harm.** A unit's work is its subcircuit's own ANDs (`Candidate.ands`, not its rows' openings), every subcircuit alike,
with no PoUW work table and no influence. Every unit carries its output bits. At each budget B the exchange rate is
top's: each half is divided by the least bound B certifies for that half alone,
`Objective(delta, work=0.5/H_work(B), bits=0.5/H_bits(B))`, which is scale-free, and the plan reports the certificate's
two bounds from its one draw, `free_compute` and `leak`, at delta = 1% and 1e-6.

**Units.** C-Flock refuses a statement past k_log 27 (a soundness bound, r20261026-004208-a28d), k_log being
`ceil(log2(ANDs with the rows' openings)) + 1` as `_cap` counts it. RMSNorm is `RMSNormFusedCuda_v2` at its measured
21.5M ANDs for N = 4096 (k_log 26), scaled to the row's width; every subcircuit still past the cap is cut into the
fewest power-of-two equal parts that fit, each part's ANDs, openings and outputs the whole's share and a 32-bit partial
carried to the next (a K-chunk's running sum, a row chunk's reduction). The parts are stubs: no Definition on main.

**Budget.** Daniel (6:05 PM PDT): proving within 1% of the block's compute. One serving block's 1% is less than one
statement's fixed cost, so the plan of record is also given for sampling blocks (the proof units one draw covers,
committed and registered before it) of the least length whose 1% affords a plan, a week, a month and a year: one draw
over all their units, its statements shared per subcircuit.

**Certificate.** `certify`'s: one stratified harm bound over every subcircuit (core's `Stratified.harm_bound`), with no
delta split by subcircuit, its leak's location term counting the wrong units by the unit-weighted harm bound in place of
the exact `count_bound` (`certificate`). `stratified_isHarmBound_exact` (Security/Proofs/Flock/Soundness/Audit/HarmBound.lean) proves the
knapsack's value an `IsHarmBound` of the stratified law; the locked `Partition.audit_influence` and
`Partition.audit_exfiltration` turn such a bound into the accepted audit's bound under the backend's open obligations
(KnowledgeSound, LinkSound), for the abstract one-stage audit, not a Lean function; and no difftest holds core's Python
`harm_bound` to the Lean.

**Cost.** `h200.Rates`' fit with statements shared (`CostModel.shared`: one session a subcircuit a sampling block,
`ceil(k / cap)` statements of drawn units), every unit's outputs committed at the GPU row committer's 225 GB/s (CMP-92
00:45Z), and a drawn request replayed at its share of the block's GPU time (a stand-in).

    uv run python benchmarks/sampling_plan/demo_a.py --out DIR [--pad]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from math import ceil, floor, inf, log, log2, nextafter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h200 as H  # noqa: E402
import transformer as X  # noqa: E402

from verity.core.protocols.profile import MARGIN, Stratified, Stratum  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import (  # noqa: E402
    Candidate, Certificate, CostModel, Objective, Plan, ReplayDomain, _plan, density_bound, for_budget, frontier,
    row_ands)

#: Demo A's block (CMP-92 00:50Z): 16,384 requests, a 7.84 s step (GEMM 5.08 s, attention and elementwise ~0.25 s,
#: commit 2.51 s), 32 steps a request
SHAPE = H.Shape(batch=16_384, step_s=7.84, steps=32)
#: the step's GPU time by what runs (CMP-92 00:50Z), seconds: the INT8 linears, and attention with the row ops
STEP_SPLIT = {"ScaledMmS8Coordinate_v1": 5.08, "attention and row ops": 0.25}
#: the GPU row committer (HASH-17, r20261025-234226-299c), bytes a second
COMMIT_BPS = 225e9
DELTAS = H.DELTAS
PERIODS = H.PERIODS
#: Daniel's budget (25 Oct 6:05 PM PDT): proving at this share of the sampling block's GPU time
BUDGET_SHARE = 0.01
#: the budgets swept, in multiples of one serving block's GPU time, past the cheapest plan's
GRID = (0.1, 1, 12.5, 15, 20, 30, 50, 100, 300, 1000, 3000, 1e4)
S8 = "ScaledMmS8Coordinate_v1"
#: C-Flock's largest statement (r20261026-004208-a28d refused k_log 28)
K_LOG_MAX = 27
#: `RMSNormFusedCuda_v2{N=4096}`'s ANDs (DRAWPROVE, PRF-170), a row element's share taken for any width
RMSNORM_FUSED = ("RMSNormFusedCuda_v2", 21_500_000, 4096)
PARTIAL_BITS = 32


def k_log(t: H.Template) -> int:
    return ceil(log2(max(t.proved, 2))) + 1


def fit(ts: list[H.Template]) -> list[H.Template]:
    """`ts` within `K_LOG_MAX`: RMSNorm as `RMSNormFusedCuda_v2`, then each subcircuit past the cap cut into the fewest
    power-of-two parts that fit, `split:P:` before its id."""
    out = []
    for t in ts:
        if t.id.startswith("stub:RmsNormRow_v1"):
            name, ands, n = RMSNORM_FUSED
            width = int(t.id.split("N=")[1].rstrip("}"))
            t = replace(t, id=f"{name}{{N={width}}}", ands=ands * width // n, real=width == n)
        p = 1
        while k_log(part := _part(t, p)) > K_LOG_MAX:
            p *= 2
        out.append(part)
    return out


def _part(t: H.Template, p: int) -> H.Template:
    if p == 1:
        return t
    rows = tuple(ceil(r / p) if r > 4 else r for r in t.rows) + (PARTIAL_BITS // 8,)
    return H.Template(f"split:{p}:{t.id}", t.units * p, ceil(t.ands / p), rows, ceil(t.out_bits / p) + PARTIAL_BITS)


def candidates(m: H.Shape = SHAPE, *, pad: bool = False, blocks: int = 1) -> list[Candidate]:
    """One candidate a subcircuit over a sampling block of `blocks` serving blocks, its unit's work its own ANDs, its
    output bits committed, and its statement cap `_cap` of its ANDs with its rows' openings."""
    return [Candidate(t.id, t.id, t.units * blocks, t.ands, t.out_bits, ands=t.ands, rows=t.rows,
                      commit_bits=t.out_bits, cap=X._cap(t.proved), replay="request", per_replay=t.units // m.batch)
            for t in fit(H.templates(m, pad=pad))]


def costs(m: H.Shape = SHAPE, blocks: int = 1) -> CostModel:
    r = H.Rates.fitted()
    return CostModel(per_and=r.per_and, statement_fixed=r.statement_fixed, statement_per_and=r.statement_per_and,
                     commit_per_bit=1 / (8 * COMMIT_BPS),
                     replay={"request": ReplayDomain(m.batch * blocks, m.block_s / m.batch)},
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


def commit(cands: list[Candidate], cm: CostModel) -> float:
    return sum(cm.commit(c) for c in cands)


def cheapest(cands: list[Candidate], cm: CostModel) -> float:
    """The proving cost of the least plan, every subcircuit at its floor of one drawn unit."""
    p = _plan(cands, cm, Objective(DELTAS[0], work=0.0, bits=1.0), 0.0, frozenset(c.template for c in cands),
              frozenset())
    return p.cost["total"] - commit(cands, cm)


def point(cands: list[Candidate], cm: CostModel, proving: float, *, grid: int = 41) -> Point | None:
    """At proving budget `proving` each half's least bound alone, then the plan under top's rate there, certified at
    both deltas from its one draw; None once a half is proved whole (its least bound below 1)."""
    d, total = DELTAS[0], proving + commit(cands, cm)
    hw = _best(cands, cm, Objective(d, work=1.0), total, grid).certificate.free_compute
    hb = _best(cands, cm, Objective(d, work=0.0, bits=1.0), total, grid).certificate.leak
    if not (hw >= 1 and hb >= 1):
        return None
    mixed = Objective(d, work=0.5 / hw, bits=0.5 / hb)
    p = _best(cands, cm, mixed, total, grid)
    at = {}
    for delta in DELTAS:
        c = p.certificate if delta == d else certificate(p.sizes, cands, replace(mixed, delta=delta))
        at[str(delta)] = {"free_compute": c.free_compute, "leak": c.leak}
    return Point(proving, hw, hb, p, at)


def sweep(cands: list[Candidate], cm: CostModel, m: H.Shape = SHAPE, *, grid: int = 41,
          budgets: tuple[float, ...] = GRID) -> dict:
    """The frontier over one serving block's proving budgets (`budgets`, in multiples of its GPU time)."""
    low = cheapest(cands, cm)
    points, skipped, complete = [], [], None
    for g in budgets:
        proving = g * m.block_s
        if proving < low:
            skipped.append(proving)
            continue
        p = point(cands, cm, proving, grid=grid)
        if p is None:
            complete = proving
            break
        points.append(p)
    return {"commit": commit(cands, cm), "cheapest": low, "skipped": skipped, "points": points, "complete": complete}


def least_blocks(m: H.Shape = SHAPE, *, pad: bool = False, share: float = BUDGET_SHARE) -> int:
    """The fewest serving blocks a sampling block spans for `share` of its GPU time to afford the cheapest plan. The
    cheapest plan's cost only grows with the span (its drawn units meet more requests to replay), so no span shorter
    than one block's cheapest over `share` of a block affords it, and past the first span that does every span does."""
    def affords(n: int) -> bool:
        return cheapest(candidates(m, pad=pad, blocks=n), costs(m, n)) <= share * n * m.block_s

    lo = max(1, ceil(cheapest(candidates(m, pad=pad), costs(m)) / (share * m.block_s)))
    if affords(lo):
        return lo
    hi = lo
    while not affords(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if affords(mid) else (mid, hi)
    return hi


def spans(m: H.Shape = SHAPE, *, pad: bool = False, share: float = BUDGET_SHARE,
          periods: dict[str, int] = PERIODS) -> dict[str, int]:
    """The sampling blocks the plan of record is given for, in serving blocks."""
    return ({"one block": 1, "least": least_blocks(m, pad=pad, share=share)}
            | {name: floor(seconds / m.block_s) for name, seconds in periods.items()})


def plan_at(m: H.Shape, name: str, n: int, *, pad: bool = False, share: float = BUDGET_SHARE, grid: int = 41) -> dict:
    """The plan at `share` of a sampling block of n serving blocks: its draw a subcircuit, its cost a serving block, and
    its two bounds at each delta, each also as a share of the sampling block's ANDs and output bits."""
    cands, cm = candidates(m, pad=pad, blocks=n), costs(m, n)
    ands, bits = sum(c.units * c.ands for c in cands), sum(c.units * c.out_bits for c in cands)
    budget, low = share * n * m.block_s, cheapest(cands, cm)
    row = {"span": name, "blocks": n, "hours": n * m.block_s / 3600, "budget": budget, "cheapest": low,
           "commit": commit(cands, cm), "ands": ands, "bits": bits, "affordable": budget >= low}
    p = point(cands, cm, budget, grid=grid) if budget >= low else None
    if p is not None:
        proving = p.plan.cost["total"] - row["commit"]
        row |= {"cost": dict(p.plan.cost), "proving": proving, "per_block": proving / n,
                "weights": {"work": 0.5 / p.h_work, "bits": 0.5 / p.h_bits}, "H_work": p.h_work, "H_bits": p.h_bits,
                "sizes": dict(p.plan.sizes), "drawn": sum(p.plan.sizes.values()),
                "statements": dict(p.plan.statement_size),
                "draw": [{"subcircuit": c.template, "units": c.units, "drawn": p.plan.sizes.get(c.template, 0),
                          "ands": c.ands, "proved_ands": c.ands + sum(row_ands(r) for r in c.rows),
                          "k_log": ceil(log2(max(c.ands + sum(row_ands(r) for r in c.rows), 2))) + 1,
                          "cap": c.cap, "out_bits": c.out_bits} for c in cands],
                "at": {d: v | {"work_share": v["free_compute"] / ands, "bits_share": v["leak"] / bits}
                       for d, v in p.at.items()}}
    return row


def ands_against_gpu(m: H.Shape = SHAPE, *, pad: bool = False) -> dict:
    """Each Definition's share of the block's own ANDs, of its proved ANDs (with the rows' openings, what proving costs)
    and of its units, beside the step's GPU split (`STEP_SPLIT`): where circuit size misstates GPU time."""
    by: dict[str, dict[str, float]] = {}
    for t in fit(H.templates(m, pad=pad)):
        g = by.setdefault(t.id.split("{")[0], {"units": 0, "ands": 0, "proved": 0})
        g["units"] += t.units
        g["ands"] += t.units * t.ands
        g["proved"] += t.units * t.proved
    ands, proved = sum(g["ands"] for g in by.values()), sum(g["proved"] for g in by.values())
    gpu = sum(STEP_SPLIT.values())
    s8 = sum(g["ands"] for k, g in by.items() if S8 in k)
    return {"definitions": {k: g | {"ands_share": g["ands"] / ands, "proved_share": g["proved"] / proved}
                            for k, g in sorted(by.items(), key=lambda kv: -kv[1]["ands"])},
            "step_split": STEP_SPLIT,
            "s8": {"ands_share": s8 / ands, "gpu_share": STEP_SPLIT[S8] / gpu},
            "rest": {"ands_share": 1 - s8 / ands, "gpu_share": 1 - STEP_SPLIT[S8] / gpu}}


def knee(points: list[Point]) -> int:
    """The point farthest below the chord between the ends of log(cost) against log(mixed harm), each half at its
    budget's rate (`kneedle` on the log-log frontier)."""
    xs = [log(p.budget) for p in points]
    ys = [log(0.5 * p.at["0.01"]["free_compute"] / p.h_work + 0.5 * p.at["0.01"]["leak"] / p.h_bits) for p in points]
    if len(points) < 3:
        return 0
    (x0, y0), (x1, y1) = (xs[0], ys[0]), (xs[-1], ys[-1])
    return max(range(len(points)), key=lambda i: (y0 + (y1 - y0) * (xs[i] - x0) / (x1 - x0)) - ys[i])


def frontier_report(m: H.Shape = SHAPE, *, pad: bool = False) -> dict:
    cands, cm = candidates(m, pad=pad), costs(m)
    s = sweep(cands, cm, m)
    pts = s["points"]

    def row(p: Point) -> dict:
        return {"proving": p.budget, "of_block": p.budget / m.block_s, "cost": dict(p.plan.cost),
                "weights": {"work": 0.5 / p.h_work, "bits": 0.5 / p.h_bits}, "H_work": p.h_work, "H_bits": p.h_bits,
                "at": p.at, "drawn": sum(p.plan.sizes.values()), "sizes": dict(p.plan.sizes),
                "statements": dict(p.plan.statement_size)}

    return {"shape": vars(m) | {"block_s": m.block_s}, "pad": pad, "templates": len(cands),
            "units": sum(c.units for c in cands), "ands": sum(c.units * c.ands for c in cands),
            "bits": sum(c.units * c.out_bits for c in cands), "commit": s["commit"], "cheapest": s["cheapest"],
            "unaffordable": s["skipped"], "complete": s["complete"], "share": BUDGET_SHARE,
            "frontier": [row(p) for p in pts], "knee": knee(pts) if pts else None,
            "ands_against_gpu": ands_against_gpu(m, pad=pad)}


def render(o: dict) -> str:
    b = o["shape"]["block_s"]
    lines = [f"# Demo A's frontier: {o['templates']} subcircuits, {o['units']:.3g} units, {o['ands']:.3g} ANDs and "
             f"{o['bits']:.3g} output bits a block of {b:g} GPU-s", "",
             f"Committing every unit's outputs: {o['commit']:.4g} GPU-s a block, beside every plan. The cheapest plan "
             f"proves for {o['cheapest']:.4g} GPU-s ({o['cheapest'] / b:.3g} of the block).", "",
             "| proving GPU-s | of the block | H_work(B) | H_bits(B) | free_compute at 1% | leak at 1% (bits) | "
             "free_compute at 1e-6 | leak at 1e-6 | drawn | |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, f in enumerate(o["frontier"]):
        a, z = f["at"]["0.01"], f["at"]["1e-06"]
        lines.append(f"| {f['proving']:.4g} | {f['of_block']:g} | {f['H_work']:.3g} | {f['H_bits']:.3g} | "
                     f"{a['free_compute']:.3g} | {a['leak']:.3g} | {z['free_compute']:.3g} | {z['leak']:.3g} | "
                     f"{f['drawn']:,} | {'knee' if i == o['knee'] else ''} |")
    g = o["ands_against_gpu"]
    lines += ["", f"ANDs against GPU time: the INT8 linears hold {g['s8']['ands_share']:.4g} of the ANDs and "
              f"{g['s8']['gpu_share']:.4g} of the step's GPU time outside the commit; the rest "
              f"{g['rest']['ands_share']:.3g} and {g['rest']['gpu_share']:.3g}.", "",
              "| Definition | units | own ANDs | share | proved ANDs share |", "|---|---|---|---|---|"]
    for k, d in g["definitions"].items():
        lines.append(f"| {k} | {d['units']:.3g} | {d['ands']:.3g} | {d['ands_share']:.2g} | {d['proved_share']:.2g} |")
    return "\n".join(lines)


def render_record(rows: list[dict], share: float = BUDGET_SHARE) -> str:
    lines = [f"Plan of record: proving within {share:g} of the sampling block's GPU time.", "",
             "| sampling block | blocks | budget GPU-s | cheapest | proving a block | drawn | free_compute at 1% | "
             "leak at 1% | free_compute at 1e-6 | leak at 1e-6 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "at" not in r:
            lines.append(f"| {r['span']} | {r['blocks']:,} | {r['budget']:.4g} | {r['cheapest']:.4g} | no plan | | | | "
                         "| |")
            continue
        a, z = r["at"]["0.01"], r["at"]["1e-06"]
        lines.append(f"| {r['span']} | {r['blocks']:,} | {r['budget']:.4g} | {r['cheapest']:.4g} | "
                     f"{r['per_block']:.4g} | {r['drawn']:,} | {a['free_compute']:.3g} ({a['work_share']:.2g}) | "
                     f"{a['leak']:.3g} ({a['bits_share']:.2g}) | {z['free_compute']:.3g} ({z['work_share']:.2g}) | "
                     f"{z['leak']:.3g} ({z['bits_share']:.2g}) |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, n in spans(SHAPE, pad=a.pad).items():
        rows.append(plan_at(SHAPE, name, n, pad=a.pad))
        (a.out / "plan_of_record.json").write_text(json.dumps(rows, indent=1, default=str))
        print(json.dumps({k: v for k, v in rows[-1].items() if k not in ("draw", "sizes", "statements")},
                         default=str), flush=True)
    print(render_record(rows), flush=True)
    o = frontier_report(pad=a.pad) | {"plan_of_record": rows}
    (a.out / "frontier.json").write_text(json.dumps(o, indent=1, default=str))
    print(render(o), flush=True)


if __name__ == "__main__":
    main()
