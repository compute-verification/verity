"""FOLD-XOR's price in the Lean form: the B200 sheet's bound with li's XOR on an n-instruction IMAD- or FP-pipe route.

    python benchmarks/pouw/theory/fold_xor_lean.py [N ...] [--json PATH]

`lean_headline_dc.reading(SM100, "DS-100")` (#2699) prices li's fold on its own routes. Here its fold menu also has li's
fold with each of its XOR LOP3s (`fold_xor.XOR_SHARE` a word) replaced by n IMAD instructions, and by n FFMA. When no
program of n − 1 or fewer instructions on those pipes computes x ⊕ y ⊕ K (`fold_xor_search.py`), n is the least a
cheater pays a XOR there, and a program mixing the pipes, or a cheater routing only some XORs, is priced at a convex
combination of these routes and the honest fold, at least the LP's least fold price φ*. So the bound read with the
routes open is the Lean form's headline resting on the search instead of on FOLD-OPT.

The routes enter `reading` by one line of its source (`EXTRA`), which this script checks is there, so a change to the
exporter fails here rather than reading something else. Two controls hold the reading to INT-HB's
(note:20261019T0507Z-draft-int-hb, "FOLD-MIN on the B200"): no route is 0.790%, and the count-only route
`{FFMA: fold_cost/32}` is 1.408%. Needs `benchmarks/pouw/lean_headline_dc.py`, on #2699 until it lands; from
70adb5e954 on, #2699's ♭ folds by its `[fold_routes]` while its exporter's ♭ is still honest's fold, so `reading` stops
at its own floor check, and f7ae9f6d59 is the last commit this reads (copy this file into that tree and put its
`benchmarks/pouw/machine_model` first on PYTHONPATH).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
POUW = HERE.parent
LHDC = POUW / "lean_headline_dc.py"
XOR_SHARE = Fraction(1, 2)
LINE = "    folds = [{k: exact(v) / E.LANES for k, v in d.fold.items()}]\n"
EXTRA = "    folds += EXTRA_FOLDS(d, exact, E.LANES)\n"


def exporter():
    """`lean_headline_dc` with `EXTRA` after its fold menu's line: `EXTRA_FOLDS(d, exact, lanes)` adds routes."""
    src = LHDC.read_text(encoding="utf-8")
    if src.count(LINE) != 1:
        raise SystemExit(f"{LHDC}: the fold menu's line isn't there once; the routes would enter nothing")
    spec = importlib.util.spec_from_loader("pouw_lean_headline_dc_xor", loader=None, origin=str(LHDC))
    m = importlib.util.module_from_spec(spec)
    m.__file__ = str(LHDC)
    m.EXTRA_FOLDS = lambda d, exact, lanes: []
    sys.modules[spec.name] = m
    exec(compile(src.replace(LINE, LINE + EXTRA), str(LHDC), "exec"), m.__dict__)
    return m


def xor_routes(n: int):
    def routes(d, exact, lanes):
        f = {k: exact(v) for k, v in d.fold.items()}
        if f.get("ALU", 0) < XOR_SHARE:
            raise SystemExit(f"{d.name}: its fold has {f.get('ALU')} ALU a word, fewer than li's XORs")
        out = []
        for pipe in ("IMAD", "FFMA"):
            g = dict(f)
            g["ALU"] -= XOR_SHARE
            g[pipe] = g.get(pipe, Fraction(0)) + XOR_SHARE * n
            out.append({k: v / lanes for k, v in g.items() if v})
        return out
    return routes


def count_only(d, exact, lanes):
    return [{"FFMA": exact(d.fold_cost()) / lanes}]


def read(m, routes) -> dict:
    m.EXTRA_FOLDS = routes
    r = m.reading(m.SM100, m.SM100.sheet)
    return {"bound": r.bound, "folded": r.folded, "unfolded": r.unfolded, "one_minus_Y": r.one_minus_Y,
            "phi": r.phi, "w": r.w, "certificate": r.certificate, "exact_bound": r.exact["bound"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("n", nargs="*", type=int, default=[1, 2, 3, 4, 5, 63, 78])
    ap.add_argument("--json")
    a = ap.parse_args()
    if not LHDC.exists():
        print(f"{LHDC} is absent (it is on #2699 until that lands)")
        return 2
    m = exporter()
    out = {"none": read(m, lambda d, exact, lanes: []), "count_only": read(m, count_only)}
    ok = abs(out["none"]["bound"] - 0.0079) < 5e-5 and abs(out["count_only"]["bound"] - 0.01408) < 5e-6
    print(f"no route: {out['none']['bound']:.4%}; count-only (INT-HB's option b): {out['count_only']['bound']:.4%}"
          f"  -- controls {'hold' if ok else 'FAIL'}")
    for n in a.n:
        out[n] = read(m, xor_routes(n))
        print(f"n = {n}: bound {out[n]['bound']:.4%} (folded {out[n]['folded']:.4%}, unfolded "
              f"{out[n]['unfolded']:.4%}), φ* {out[n]['phi']:.6f}, π* {out[n]['certificate']}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({str(k): v for k, v in out.items()}, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
