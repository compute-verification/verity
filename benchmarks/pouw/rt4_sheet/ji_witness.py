"""RT4-SHEET on DC-90-ri: does a joint-issue (JI90) line borrowed from SHFL hold up the conclusion, or only ε_cert?

A per-word route is a dict of class -> instructions a word, like a design's `[fold_routes]`. Each is priced on each of a
tree's four readings (DS-90, DC-90-ri, DS-100, DC-100-ri) as `red_price.py` prices a SASS loop: at π* against φ*
(FOLD-OPT's test), and as a full-work run, ♭'s word production and filler (`data.nw`, `data.nx`, a unit) plus N_w words
of the route on every row of each class, against (1 − the reading's bound) · B on its most-loaded row (the conclusion's
test).

The routes move LI's XOR work (its ALU) into a class that would lose its borrowed JI90 lines, keeping LI's multiply
(IMAD, which no memory, atomic or reduce class computes) and the generic CP_RED8 checkpoint. `free` is the class the
XORs go to; `k` is how many of its instructions replace one LOP3 (1 is the cheater's best case, below any real
program, so a hold at k = 1 holds for every such program).

    uv run python benchmarks/pouw/rt4_sheet/ji_witness.py --tree TREE [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc_price import _load, use_tree  # noqa: E402

#: the design's `[fold_routes]` counts thread instructions a word; a warp instruction serves 32 lanes
LANES = 32
LI_ALU, LI_IMAD, CP_ATOME = (Fraction(n, LANES) for n in (Fraction(1, 2), Fraction(1, 2), Fraction(1, 16)))


def routes(free: str) -> dict[str, dict[str, Fraction]]:
    out = {"cp_red8_generic": {"ALU": LI_ALU, "IMAD": LI_IMAD, "ATOME32": CP_ATOME}}
    for k in (1, 2, 3):
        route = {"IMAD": LI_IMAD, "ATOME32": CP_ATOME}
        route[free] = route.get(free, Fraction(0)) + LI_ALU * k
        out[f"xor_by_{free}_k{k}"] = route
    return out


def price(r, route: dict[str, Fraction]) -> dict:
    lam, c = r.data.lam, r.data.c
    phi, B, bound = (Fraction(r.exact[k]) for k in ("phi", "B", "bound"))
    per = sum((n * sum((w * c[k].get(row, Fraction(0)) for row, w in lam.items()), Fraction(0))
               for k, n in route.items()), Fraction(0))
    n: dict[str, Fraction] = {}
    for k, v in route.items():
        n[k] = n.get(k, Fraction(0)) + r.Nw * v
    for part in (r.data.nw, r.data.nx):
        for k, v in part.items():
            n[k] = n.get(k, Fraction(0)) + v
    load = {row: sum((c[k].get(row, Fraction(0)) * v for k, v in n.items()), Fraction(0)) for row in r.data.rows}
    top = max(load, key=load.get)
    return {"over_phi": float(per / phi), "below_phi": per < phi, "binding_row": top,
            "time_over_credit_max_row": float(load[top] / B), "one_minus_bound": float(1 - bound),
            "breaks_conclusion": load[top] / B < 1 - bound,
            "rows": {row: float(v / B) for row, v in sorted(load.items(), key=lambda kv: -kv[1])[:5]}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, default=Path.cwd())
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pouw = a.tree.resolve() / "benchmarks" / "pouw"
    use_tree(pouw)
    LHD = _load(pouw / "lean_headline_dc.py", "rt4_lean_headline_dc")
    res: dict = {}
    broke = False
    for fam in (LHD.SM90, LHD.SM100):
        for vector in (fam.sheet, fam.measured):
            r = LHD.reading(fam, vector)
            out = res[vector] = {"phi_star": float(Fraction(r.exact["phi"])), "B": float(Fraction(r.exact["B"])),
                                 "bound": float(Fraction(r.exact["bound"])), "routes": {}}
            print(f"{vector}: φ* {out['phi_star']:.6f} a word, ♭ {out['B']:.4f} a unit, bound {out['bound']:.6%}")
            for free in ("ATOMS32", "REDUX", "STS32", "ATOME32"):
                if free not in r.data.c:
                    continue
                for name, route in routes(free).items():
                    if name in out["routes"]:
                        continue
                    p = price(r, route)
                    out["routes"][name] = p
                    broke |= p["breaks_conclusion"]
                    print(f"  {name}: {p['over_phi']:.4f} φ*{' (under)' if p['below_phi'] else ''}; most-loaded "
                          f"row {p['binding_row']} {p['time_over_credit_max_row']:.5f} vs {p['one_minus_bound']:.5f}"
                          f"{' BREAKS' if p['breaks_conclusion'] else ''}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, sort_keys=True))
    return 1 if broke else 0


if __name__ == "__main__":
    raise SystemExit(main())
