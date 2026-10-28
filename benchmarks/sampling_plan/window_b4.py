"""The sampling planner (`sampled_proofs.optimize`) on the served window's program: the remote demo's B4 window, class
4,32,4, from `window_b4_census.py`'s census (a unit a template instance of the class circuit's population).

Each template is a candidate at its H200 ANDs, credited them as work, opening its rows in circuit (`row_ands`), its
statement cap from its ANDs (`transformer._cap`, m = k_log + nbl <= 35). The three linear families are ops with one
alternative a grain (`cut` "N=w": w output channels a unit) of at most `MAX_PROVED_ANDS` a proof block; every other
template is its own op. Cost is C-Flock's fit
(`h200.Rates.fitted`, GPU-s), statements shared as the served path stages them (`CostModel.shared`).

Two modes:
  1. proving only, as the planner and `GranularityOptimum` price it (no serve-time term);
  2. plus the serve-time commitment n·b(s)·h at weight w: a unit's b(s) its rows' SHA-512 compressions at serve time
     (`class_rows`: an output row b bytes costs blocks(128 + b) + 6, an input row 4) at 128 bytes each, h a byte's
     GPU-s at the whole card's SHA-512 rate (398 GB/s at 4 KiB rows, art:2499e3f4…); `CostModel.commit_per_row`
     carries it, so the planner's per-op cut choice trades it against proving.

Two targets, each in both modes, at delta = 2^-40:
  work: the cheapest plan certifying free compute (`Certificate.free_compute`) at most eps = 1% of the window's work;
  exfiltration: the least-leak plan (`Objective(work=0, bits=1)`) at the work plan's cost, the leak ℓ certified
    (`consumers.exfiltration_bound`).
Every final plan is certified exactly by core's bounds, with both its free compute and its leak (and the leak's location
bits apart), and its cost is reported by fit too: proving GPU-s, and the verifier's and staging's CPU-s on one CPU.

    uv run python benchmarks/sampling_plan/window_b4.py CENSUS.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from math import ceil, fsum, log, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h200 as H
import transformer as X
from inputs import KEY, I, observations

from verity.core.protocols.profile import IntegrityProfile, Stratified, Stratum
from verity.core.protocols.sampled_proofs.consumers import location_bits
from verity.core.protocols.sampled_proofs.costfit import Fitted, fit
from verity.core.protocols.sampled_proofs.optimize import (
    Candidate, CostModel, Objective, Plan, certify, for_budget, for_target, frontier, row_ands, template_cost)

DELTA = 2.0 ** -40
EPS = 0.01
#: SHA-512 over the whole card at 4 KiB rows, 16 KiB rows, and what's free beside the GEMM (bytes/s; art:2499e3f4…)
SHA512_BPS = {"card 4 KiB": 398e9, "card 16 KiB": 521e9, "beside GEMM": 66.5e9}
COMPRESSION = 128
#: C-Flock's compose refuses a proof block of more ANDs, its rows' openings included (TOP-138, 19:14Z)
MAX_PROVED_ANDS = 1 << 27


def proved_ands(e: dict) -> int:
    return e["ands"] + sum(row_ands(r) for r in (*e["in"], *e["out"]))


def _candidate(e: dict, op: str, template: str, cut: str = "") -> Candidate:
    return Candidate(op, template, e["units"], e["ands"], e["out_bits"], cut=cut, ands=e["ands"],
                     rows=tuple(e["in"]) + tuple(e["out"]), commit_bits=0,
                     commit_rows=float(e["commit_compressions"]), cap=X._cap(proved_ands(e)))


def candidates(census: dict, *, widths: tuple[int, ...] | None = None) -> list[Candidate]:
    """One candidate a template, its op the served template; a linear family's op has one alternative a grain (the
    served grain's keeping the served cut's rows). `widths` restricts the grains to those (and the served); None offers
    every width the census does."""
    out = []
    for e in census["templates"]:
        op = e["label"]
        if "alternatives" not in e:
            out.append(_candidate(e, op, op))
            continue
        for a in e["alternatives"]:
            if "refused" in a or (widths is not None and a["N"] not in widths and a["N"] != e["N"]):
                continue
            served = a["N"] == e["N"]
            if not served and proved_ands(a) > MAX_PROVED_ANDS:
                continue
            out.append(_candidate(e if served else a, op, op if served else f"{op}@N={a['N']}", f"N={a['N']}"))
    return out


def costs(*, h: float = 0.0, w: float = 1.0) -> CostModel:
    """C-Flock's fit, statements shared; mode 2 when `h` (GPU-s a byte hashed) is given, at weight `w`."""
    r = H.Rates.fitted()
    return CostModel(per_and=r.per_and, statement_fixed=r.statement_fixed, statement_per_and=r.statement_per_and,
                     commit_per_row=w * h * COMPRESSION, shared=True, source="h200.Rates.fitted")


def fitted() -> Fitted:
    """C-Flock's three fits as `transformer.costs` takes them: proving (GPU-s), verifying and staging (CPU-s, one CPU),
    which `costs` sums at the price ratio `cpu`."""
    obs = observations()
    full = ["one", "ands", "ands*staged", "ands*drawn"]
    return Fitted(fit(obs, "prove", full), fit(obs, "verify", full), fit(obs, "stage", full[:3]), I["cpu"].value, KEY)


def parts(f: Fitted) -> dict[str, CostModel]:
    """One cost model a fit, its staging folded into the drawn units as `h200.Rates.fitted` folds it (statements shared)."""
    def model(coef: dict) -> CostModel:
        return CostModel(per_and=coef.get("ands*drawn", 0.0) + coef.get("ands*staged", 0.0),
                         statement_fixed=coef["one"], statement_per_and=coef["ands"], shared=True)
    return {"prove_gpu_s": model(f.prove.coef), "verify_cpu_s": model(f.verify.coef), "stage_cpu_s": model(f.stage.coef)}


def grain_table(cands: list[Candidate], cm: CostModel, h: float) -> list[dict]:
    """Per linear family and grain, c(s)'s pieces: a proof block's proved ANDs, its statement's cost (a0) and cap, a drawn
    block's cost, and the serve-time n·b(s)·h at `h`."""
    out = []
    for c in cands:
        if not c.cut:
            continue
        a0, a1 = cm.statement(c)
        out.append({"op": c.op, "N": int(c.cut[2:]), "units": c.units, "unit_ands": c.ands, "proved_ands": cm.ands(c),
                    "k_log": ceil(log2(cm.ands(c))) + 1, "cap": c.cap, "statement_gpu_s": a0,
                    "drawn_gpu_s": a1 + cm.unit(c), "commit_compressions": c.commit_rows,
                    "serve_commit_gpu_s": c.units * c.commit_rows * COMPRESSION * h})
    return out


def window_work(cands: list[Candidate]) -> float:
    """The window's credited ANDs: each op at one of its alternatives (the grains conserve the coordinates)."""
    first: dict[str, str] = {}
    for c in cands:
        first.setdefault(c.op, c.cut)
    return fsum(c.units * c.work for c in cands if first[c.op] == c.cut)


def report(p: Plan, cands: list[Candidate], cm: CostModel, F: float, split: dict[str, CostModel]) -> dict:
    chosen = [c for c in cands if c.template in p.sizes]
    cert = certify(p.sizes, chosen, Objective(DELTA, work=1.0, bits=1.0))
    law = Stratified(tuple(Stratum(c.template, c.units, p.sizes[c.template]) for c in chosen))
    location = location_bits(IntegrityProfile(("unit",), (law.population,), {"unit": law}, DELTA))
    proving = fsum(template_cost(c, cm, p.sizes[c.template])[0] for c in chosen)
    draws = sum(p.sizes.values())
    ands = sum(p.sizes[c.template] * cm.ands(c) for c in chosen)
    grains = {c.op: {"N": int(c.cut[2:]), "units": c.units, "unit_ands": c.ands, "proved_ands": cm.ands(c),
                     "commit_compressions": c.commit_rows, "drawn": p.sizes[c.template], "cap": c.cap}
              for c in chosen if c.cut}
    return {"grains": grains, "draws": draws, "proving_ands": ands, "proving_gpu_s": proving,
            "statements_gpu_s": p.cost["statements"], "units_gpu_s": p.cost["units"],
            "commit_gpu_s": p.cost["commit"], "total_gpu_s": p.cost["total"],
            **{k: fsum(template_cost(c, m, p.sizes[c.template])[0] for c in chosen) for k, m in split.items()},
            "free_compute": cert.free_compute, "eps": cert.free_compute / F, "leak_bits": cert.leak,
            "leak_location_bits": location, "window_out_bits": fsum(c.units * c.out_bits for c in chosen),
            "window_work": F, "units": sum(c.units for c in chosen), "templates": len(chosen),
            "sizes": dict(sorted(p.sizes.items()))}


def run(cands: list[Candidate], cm: CostModel, split: dict[str, CostModel], *, grid: int) -> dict:
    F = window_work(cands)
    work_obj = Objective(DELTA, work=1.0)
    work = for_target(cands, cm, work_obj, EPS * F, by="free_compute", exact=False, grid=grid)
    out = {"work": report(work, cands, cm, F, split)}
    bits_obj = Objective(DELTA, work=0.0, bits=1.0)
    front = frontier(cands, cm, bits_obj, exact=False, grid=grid)
    leak = for_budget(front, work.cost["total"], candidates=cands, costs=cm, objective=bits_obj, exact=False)
    out["exfiltration"] = None if leak is None else {**report(leak, cands, cm, F, split),
                                                     "budget_gpu_s": work.cost["total"]}
    out["leak_frontier"] = [{"total_gpu_s": p.cost["total"], "harm_bits": p.certificate.harm, "draws": sum(p.sizes.values()),
                             "grains": {c.op: c.cut for c in cands if c.template in p.sizes and c.cut}}
                            for p in front[:: max(1, len(front) // 12)]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("census")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--weights", default="1,10,100,1000,10000", help="mode 2's w, beside w = 1, for where s* moves")
    args = ap.parse_args()
    census = json.loads(Path(args.census).read_text())
    cands = candidates(census)
    served = candidates(census, widths=())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h = 1 / SHA512_BPS["card 4 KiB"]
    f = fitted()
    split = parts(f)
    runs = {"mode1": run(cands, costs(), split, grid=args.grid),
            "mode2": run(cands, costs(h=h, w=1.0), split, grid=args.grid),
            "served_grain_mode1": run(served, costs(), split, grid=args.grid),
            "served_grain_mode2": run(served, costs(h=h, w=1.0), split, grid=args.grid)}
    for name, bps in SHA512_BPS.items():
        if name != "card 4 KiB":
            runs[f"mode2 h@{name}"] = run(cands, costs(h=1 / bps, w=1.0), split, grid=args.grid)
    for w in (float(x) for x in args.weights.split(",") if float(x) != 1.0):
        runs[f"mode2 w={w:g}"] = {"work": run(cands, costs(h=h, w=w), split, grid=args.grid)["work"]}
    doc = {"delta": DELTA, "eps": EPS, "lambda": log(1 / DELTA), "census": args.census,
           "rates": asdict(H.Rates.fitted()), "fits": {t: dict(getattr(f, t).coef) for t in ("prove", "verify", "stage")},
           "cpu": f.cpu, "fit_key": dict(f.key), "sha512_bps": SHA512_BPS,
           "grain_table": grain_table(cands, costs(), h), "runs": runs}
    (out / "window_b4.json").write_text(json.dumps(doc, indent=1, default=str))
    for name, r in runs.items():
        for tgt in ("work", "exfiltration"):
            x = r.get(tgt)
            if not x:
                continue
            g = ", ".join(f"{op.split(',DOT')[0]} N={v['N']}" for op, v in x["grains"].items())
            print(f"{name:22s} {tgt:12s} s*: {g} | draws {x['draws']} | proving ANDs {x['proving_ands']:.4g} | "
                  f"{x['total_gpu_s']:.1f} GPU-s (prove {x['prove_gpu_s']:.1f}, verify {x['verify_cpu_s']:.0f} CPU-s), "
                  f"commit {x['commit_gpu_s']:.4g} | eps {x['eps']:.4g} | leak {x['leak_bits']:.4g} bits "
                  f"({x['leak_location_bits']:.4g} location)")


if __name__ == "__main__":
    main()
