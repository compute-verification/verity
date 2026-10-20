"""RT4-SHEET on the H200 and B200: red_generic.cu's timed loops (li_loop<CP, GEN>, red_generic.sh's SASS for sm_90a and
sm_100a) priced on each family's data sheet and measured vector at π*, the certificate `lean_headline_dc` proves each
reading at, against its fold floor φ* a word (the menu's cheapest checkpoint route there).

Each SASS opcode of the loop (its longest backward branch, 4 units of 128 words a lane an iteration, so 4 · 4096 words
a warp) is classed by the tree's `facet_break/loads.class_of` and priced π*·c[class]; NOP is dropped and loop control
is kept (the menu's routes leave it out, so this is the cheater's cost from above). An opcode whose class is None or
not among the vector's classes is priced 0, as neither family's machine counts it (no catch-all), and listed; the route
is priced again with each at π*'s issue weight alone, as an A2 catch-all would. A full-work run that folds by the
route and skips only ♭'s skippable work reads time/credit (B (1 − ε_skip) − N_w φ* + N_w route) / B at π*. Its model
time is its most-loaded row, which π*'s mix only bounds from below, so the route breaks the conclusion when ♭'s word
production and filler plus N_w words of the route load every row under (1 − the reading's bound) · B.

    uv run python benchmarks/pouw/rt4_sheet/red_price.py --tree TREE SASS_SM90 SASS_SM100 [--json OUT]

Exit 0, or 1 when a route prices under φ* on a reading.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc_price import INS, _load, targets  # noqa: E402

#: li_loop<CP, GEN> (red_generic.cu) or li_loop<CP, GEN, W> (imadw.cu, W = 1 the multiply-add as IMAD.WIDE)
NAME = re.compile(r"li_loopILi(\d)ELi(\d)E(?:Li(\d)E)?")
CP = ("red", "red2", "red8")
ISSUE_ROWS = ("DS_issue", "issue")


def loops(sass: str):
    parts = re.split(r"\n\s*Function : (\S+)\n", "\n" + sass)
    for name, body in zip(parts[1::2], parts[2::2]):
        m = NAME.search(name)
        if not m:
            continue
        ins = [(int(a, 16), op) for a, p, op, rest in INS.findall(body) for _ in [targets(rest)]]
        back = [(t, a) for a, p, op, rest in INS.findall(body) if op.startswith("BRA")
                for t in targets(rest) for a in [int(a, 16)] if t < a]
        head, tail = max(back, key=lambda lp: lp[1] - lp[0])
        route = f"cp_{CP[int(m[1])]}_{'generic' if m[2] == '1' else 'global'}{'_wide' if m[3] == '1' else ''}"
        yield route, [op for a, op in ins if head <= a <= tail and op != "NOP"]


def price_reading(LHD, L, fam, vector: str, sass: str) -> dict:
    r = LHD.reading(fam, vector)
    lam, c = r.data.lam, r.data.c
    phi, B, bound = (Fraction(r.exact[k]) for k in ("phi", "B", "bound"))
    issue = sum((lam.get(k, Fraction(0)) for k in ISSUE_ROWS), Fraction(0))
    skip = Fraction(r.eps_skip)
    out = {"family": fam.ns, "gpu": fam.gpu, "vector": vector, "pistar": r.lam, "phi_star": float(phi), "Nw": r.Nw,
           "B": float(B), "bound": float(bound), "eps_skip": r.eps_skip, "routes": {}}
    words = 4 * 4096
    for name, ops in loops(sass):
        free: collections.Counter = collections.Counter()

        def cost(catch_all: bool) -> Fraction:
            tot = Fraction(0)
            for op in ops:
                k = L.class_of(op)
                if k is None or k not in c:
                    if not catch_all:
                        free[f"{op} ({k})" if k else op] += 1
                    tot += issue if catch_all else 0
                    continue
                tot += sum((w * c[k].get(row, Fraction(0)) for row, w in lam.items()), Fraction(0))
            return tot

        per, per_ca = cost(False) / words, cost(True) / words
        tc = (B * (1 - skip) - r.Nw * phi + r.Nw * per) / B
        # the run's model time is its most-loaded row, which π*'s mix only bounds from below: ♭'s word production and
        # filler (`data.nw`, `data.nx`, a unit) plus N_w words of this route, each opcode on every row of its class
        n = collections.Counter()
        for op in ops:
            k = L.class_of(op)
            if k is not None and k in c:
                n[k] += Fraction(r.Nw, words)
        for part in (r.data.nw, r.data.nx):
            for k, v in part.items():
                n[k] += v
        rowload = {row: sum((c[k].get(row, Fraction(0)) * v for k, v in n.items()), Fraction(0))
                   for row in r.data.rows}
        top = max(rowload, key=rowload.get)
        tmax = rowload[top] / B
        out["routes"][name] = {"per_word": float(per), "over_phi": float(per / phi),
                               "over_phi_with_catch_all": float(per_ca / phi), "below_phi": per < phi,
                               "time_over_credit": float(tc), "one_minus_bound": float(1 - bound),
                               "time_over_credit_max_row": float(tmax), "binding_row": top,
                               "breaks_conclusion": tmax < 1 - bound, "unpriced": dict(free.most_common()),
                               "loop": dict(collections.Counter(ops).most_common())}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, default=Path.cwd())
    ap.add_argument("sass90", type=Path)
    ap.add_argument("sass100", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pouw = a.tree.resolve() / "benchmarks" / "pouw"
    LHD = _load(pouw / "lean_headline_dc.py", "rt4_lean_headline_dc")
    L = _load(pouw / "facet_break" / "loads.py", "rt4_loads")
    res = []
    for fam, sass in ((LHD.SM90, a.sass90.read_text()), (LHD.SM100, a.sass100.read_text())):
        for vector in (fam.sheet, fam.measured):
            res.append(price_reading(LHD, L, fam, vector, sass))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, sort_keys=True))
    under = []
    for f in res:
        print(f"{f['family']} ({f['gpu']}) on {f['vector']}: π* {f['pistar']}, φ* {f['phi_star']:.6f} a word, "
              f"♭ {f['B']:.4f} a unit, bound {f['bound']:.6%}, ε_skip {f['eps_skip']:.4%}")
        for n, v in f["routes"].items():
            print(f"  {n}: {v['per_word']:.6f} a word = {v['over_phi']:.4f} φ* (with a catch-all "
                  f"{v['over_phi_with_catch_all']:.4f} φ*); time/credit at π* {v['time_over_credit']:.5f}, on the "
                  f"most-loaded row ({v['binding_row']}) {v['time_over_credit_max_row']:.5f}, vs "
                  f"{v['one_minus_bound']:.5f}{'  BREAKS' if v['breaks_conclusion'] else ''}")
            if v["unpriced"]:
                print(f"    unpriced on {f['vector']}: {v['unpriced']}")
            if v["below_phi"]:
                under.append(f"{f['family']} {f['vector']} {n}")
    print(f"routes under φ*: {under or 'none'}")
    return 1 if under else 0


if __name__ == "__main__":
    raise SystemExit(main())
