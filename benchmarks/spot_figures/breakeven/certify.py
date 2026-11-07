"""Lean's certificate of the break-even surface: for each row of a `breakeven_surface.json` (`sweep.py`) and each
weights size it reaches, the bits Lean certifies for the planner's design at the row's β are at most the weights
(`total_cost.certify`, its count `TwoStageCheckPatternsSound`'s), β stepping up by 1 + `REL` where they are a hair
above.  Each case is rebuilt from the surface's own `meta.cases` (`sweep.build`).

    uv run python benchmarks/spot_figures/breakeven/certify.py SURFACE.json --out DIR [--workers N] [--scopes S ...]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levers import WEIGHTS, bound
from sweep import Case, build

from common import tc

#: β's step outward where Lean's bits are above the weights at the surface's β: `sweep.solve` stops within
#: 1 + rel/4 of the weights on either side, and Lean's bits are a hair above the planner's
REL = 0.001


def case_of(meta: dict) -> Case:
    kw = {k: v for k, v in meta.items() if k != "tax"}
    return Case(**kw | {"labels": tuple(kw.get("labels", ()))})


def check(job: tuple) -> dict:
    row, meta, w = job
    case, beta = case_of(meta), row["beta"][w]
    out = {"case": row["case"], "scope": row["scope"], "weights": w, "tax": row["tax"], "beta": beta}
    if beta is None:
        return out | {"lean": None}
    L = build(case)
    cert = tc.certify(lambda b: bound(L, b, row["scope"], case.scale, full=True, lean=True)[1], beta, WEIGHTS[w],
                      False, REL)
    return out | {"lean": cert}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("surface", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scopes", nargs="*", help="default: every scope the surface has")
    ap.add_argument("--only", nargs="*", help="cases; default all")
    args = ap.parse_args(argv)
    surface = json.loads(args.surface.read_text())
    cases = surface["meta"]["cases"]
    jobs = [(r, cases[r["case"]], w) for r in surface["rows"] for w in WEIGHTS
            if (not args.scopes or r["scope"] in args.scopes) and (not args.only or r["case"] in args.only)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(check, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)
    meta = {"surface": str(args.surface), "weights_bits": WEIGHTS, "rel": REL, "steps": tc.STEPS,
            "exe": tc.plancheck.EXE, "slack": str(tc.plancheck.SLACK), "search": tc.plancheck.SEARCH,
            "precision": tc.plancheck.TWO_P}
    (args.out / "breakeven_certified.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    lines = ["case | scope | weights | tax | beta | certified beta | steps | binds | Lean bits / weights | total"]
    for r in rows:
        c = r["lean"]
        if c is None:
            lines.append(" | ".join([r["case"], r["scope"], r["weights"], f"{r['tax']:.2%}", "none"] + ["-"] * 5))
            continue
        at = "uncertified" if c["at"] is None else f"{c['at']:.4g}"
        total = "-" if c["at"] is None else f"{r['tax'] + c['at']:.2%}"
        lines.append(" | ".join([r["case"], r["scope"], r["weights"], f"{r['tax']:.2%}", f"{r['beta']:.4g}", at,
                                 str(c["steps"]), c["binds"], f"{c['y'] / WEIGHTS[r['weights']]:.6f}", total]))
    (args.out / "breakeven_certified.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
