"""RT4-SHEET: the cheapest accepted run on DS-120, every set's CP-W checkpoint emitted, on today's sheet and with A5b.

An accepted unit is word production (♭'s IMMA.16832 and LDSM operand floor), a fold route and a checkpoint route; the
cheater skips ♭'s skippable work. A run may split its units among routes, so the cheapest accepted run is the LP
  min t  s.t.  Σ_j x_j L_j(r) ≤ t on every DS-120 row r,  Σ_j x_j = 1,  x ≥ 0,
over the pool's routes j, `L_j(r)` the route's load on row `r` a unit (`lean_headline.load`). Its value is the
run's time a unit on the sheet's machine (the most-loaded row), against the 0.656% line: the conclusion fails below
(1 − 0.656%)·♭, and the proof's margin is gone below (1 − 0.656%)·♭/(1 − 41/S).

`--tree` is the Verity tree priced on: #2791 (today's sheet, a bulk instruction one uniform slot) or
cursor/ca-sheet-bulk-ea24 (A5b, `loads.bulk_counts`). `--fixture` is fold_bulk_loops.json (fbw's SASS counts, from
the A5b branch). Folds: the menu's lane fold (♭'s), fbw4096_K4_BT256 (built, bit-exact), and FOLD-PROOF's bulk floors
F1–F3 (`BULK_FLOORS`, counted, not built). Checkpoints a unit (a set-warp is 8 units; 16 state words a lane a set):
  fcp      built: fold_cp.cu's register XOR and REDG merge, 33 LOP3 + 2 REDG + IMAD.WIDE + LDC a set over 32 units
  pair2    built (RT2-120, r20261019-121345-6849): 6 LOP3 + 2 REDG.64 a set-warp, the issue-merged bound
  egr-off  counted: 2 STG.256 a set-warp, the merge off the GPU (no row prices it; infeasible in real time)
  egr-stg  counted: 2 STG.256 a set-warp, a bulk copy into shared memory and an in-shared XOR tree (2·256 B a unit of
           bulk bytes), and the sub-16-byte last step in lanes (6 instructions a group, a group a set-warp of y or z)
  egr-sts  counted: 4 STS.128 a set-warp into shared memory, the tree (256 B a unit), the last step
  lopW-sts counted: each lane XORs its 8 chain words to W by LOP3, one STS (A5's 128 B; DS-120 has no narrower
           STS class) each of y and z, the smaller tree (`tree`), the last step
  z-only   F3's variant with y's copy surviving its fold (F3's reductions write shared memory, u leaves by a bulk
           copy): z's stores alone, y's tree and last step still paid on the GPU (`z_only`), or nothing for y off it
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

S_CHECKS = 48000
EPS = Fraction(656, 100000)
BULK = 101376


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def tree_modules(tree: Path):
    sys.path.insert(0, str(tree / "benchmarks/pouw/machine_model"))
    sys.path.insert(0, str(tree / "benchmarks/pouw"))
    for k in [m for m in sys.modules if m.startswith("verity_machine_model")]:
        del sys.modules[k]
    H = _load("rt4_lean_headline", tree / "benchmarks/pouw/lean_headline.py")
    L = _load("rt4_loads", tree / "benchmarks/pouw/facet_break/loads.py")
    return H, L


def add(*ns: dict) -> dict:
    out: dict[str, Fraction] = {}
    for n in ns:
        for k, v in n.items():
            out[k] = out.get(k, Fraction(0)) + Fraction(v)
    return out


def bulk(L, nbytes: Fraction) -> dict:
    """`nbytes` of bulk bytes a unit at the largest instruction: A5b's UTMABIG count, or on today's sheet the same
    count of instructions, each a UTMABIG (one issue slot, no other row)."""
    return {"UTMABIG": Fraction(nbytes) / BULK}


def fbw_counts(H, L, loop: dict) -> dict:
    """fold_bulk.cu's timed loop a unit, as test_lean_headline.bulk_fold: every warp over 8K units, thread 0's bodies
    over the block's, MMA and NOP dropped; a bulk instruction at its bytes (A5b) or its class (today: UNI)."""
    W, units = loop["BT"] // 32, 8 * loop["K"]
    n: dict[str, Fraction] = {}
    for part, share in (("every_warp", Fraction(1, units)), ("body", Fraction(1, units * W))):
        for key, c in loop[part].items():
            op, _, nbytes = key.partition(" ")
            k = L.class_of(op)
            if op == "NOP" or k in L.MMA_FORMS:
                continue
            if nbytes and hasattr(L, "bulk_counts"):
                counts = L.bulk_counts(op, int(nbytes))
            else:
                counts = {k if k is not None else H.CATCH_ALL["sm_120"]: Fraction(1)}
            for kk, v in counts.items():
                n[kk] = n.get(kk, Fraction(0)) + c * share * v
    return n


def lop3s(w: int) -> int:
    """LOP3s to XOR a lane's 8 chain words of y (or z) down to `w` words: 8 − w two-input XORs, two a LOP3."""
    return (8 - w + 1) // 2


def tree(w: int) -> Fraction:
    """A warp's y (or z) at `w` words a lane merged in shared memory by two-input bulk reductions, a unit: 32·4w B
    down to the 16 B minimum charges at least 32·4w − 16 source bytes on A5b's one side, over 8 units a set-warp."""
    return Fraction(128 * w - 16, 8)


def checkpoints(H, L, scale: Fraction = Fraction(1)) -> dict[str, dict]:
    """The checkpoint routes a unit; `scale` multiplies the counted merges' trees and last steps."""
    norow = H.CATCH_ALL["sm_120"]
    last = {norow: Fraction(6, 32) * Fraction(2, 8) * scale}  # one group a set-warp of y and one of z, a lane each
    cps = {
        "fcp": {"ALU": Fraction(33, 32), "REDG32": Fraction(2, 32), "IMADW": Fraction(1, 32), norow: Fraction(1, 32)},
        "pair2": {"ALU": Fraction(6, 8), "REDG64": Fraction(2, 8)},
        "egr-off": {norow: Fraction(2, 8)},
        "egr-stg": add({norow: Fraction(2, 8)}, bulk(L, 256 + 2 * tree(8) * scale), last),
        "egr-sts": add({"STS128": Fraction(4, 8)}, bulk(L, 2 * tree(8) * scale), last),
    }
    for w in (4, 2, 1):
        cps[f"lop{w}-sts"] = add({"ALU": Fraction(2 * lop3s(w), 8), "STS128": Fraction(2, 8)},
                                 bulk(L, 2 * tree(w) * scale), last)
    return cps


def z_only(H, L, scale: Fraction = Fraction(1)) -> dict[str, dict]:
    """F3's y reaches the merge without a register store, but bulk sources are shared::cta only, so an on-GPU merge
    still pays y's tree (128 B a unit, tapped from y's shared copy before the next set's reductions overwrite it) and
    y's last step; only z is stored and, by STG.256, copied back in."""
    norow = H.CATCH_ALL["sm_120"]
    last = {norow: Fraction(6, 32) * Fraction(2, 8) * scale}
    zs = {"z-only-off": {norow: Fraction(1, 8)},
          "z-only-stg": add({norow: Fraction(1, 8)}, bulk(L, 128 + 2 * tree(8) * scale), last),
          "z-only-sts": add({"STS128": Fraction(2, 8)}, bulk(L, 2 * tree(8) * scale), last)}
    for w in (4, 2, 1):
        zs[f"z-lop{w}-sts"] = add({"ALU": Fraction(lop3s(w), 8), "STS128": Fraction(1, 8)},
                                  bulk(L, (tree(w) + tree(8)) * scale), last)
    return zs


def rows(H) -> list[str]:
    return ["DS_issue", "DS_tensor", "DS_smem", *H.SM120.pipes]


def loads_of(H, i, n: dict) -> dict[str, Fraction]:
    return {r: H.load(i, r, n) for r in rows(H)}


def price(H, i, cert: dict, n: dict) -> Fraction:
    return sum((Fraction(w) * H.load(i, r, n) for r, w in cert.items()), Fraction(0))


def cheapest(H, i, pool: dict[str, dict]) -> tuple[Fraction, dict[str, Fraction], dict[str, float]]:
    """The LP's value and mix, solved in floats and recomputed exactly on its support, and its dual: the row weights π
    (Σπ = 1) at which every route of the pool costs at least the value, the certificate a statement would be at."""
    names = list(pool)
    R = rows(H)
    A = np.array([[float(H.load(i, r, pool[j])) for j in names] for r in R])
    c = np.zeros(len(names) + 1)
    c[-1] = 1
    A_ub = np.hstack([A, -np.ones((len(R), 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(R)), A_eq=[[1.0] * len(names) + [0.0]], b_eq=[1.0],
                  bounds=[(0, None)] * len(names) + [(None, None)], method="highs")
    x = {names[j]: res.x[j] for j in range(len(names)) if res.x[j] > 1e-9}
    pi = {R[r]: -float(res.ineqlin.marginals[r]) for r in range(len(R)) if -res.ineqlin.marginals[r] > 1e-9}
    sup = list(x)
    if len(sup) == 1:
        return max(loads_of(H, i, pool[sup[0]]).values()), {sup[0]: Fraction(1)}, pi
    # exact: the binding rows of the support's mix, solved as equalities in Fractions
    mix = {k: Fraction(v).limit_denominator(10**9) for k, v in x.items()}
    tot = sum(mix.values())
    mix = {k: v / tot for k, v in mix.items()}
    return max_row(H, i, pool, mix), mix, pi


def max_row(H, i, pool: dict[str, dict], mix: dict[str, Fraction]) -> Fraction:
    n = add(*({kk: vv * w for kk, vv in pool[k].items()} for k, w in mix.items()))
    return max(loads_of(H, i, n).values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--floors-from", type=Path, help="lean_headline.py with BULK_FLOORS (the A5b branch)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    H, L = tree_modules(a.tree)
    i = H.instance()
    a5b = hasattr(L, "bulk_counts")
    floors = H.BULK_FLOORS if hasattr(H, "BULK_FLOORS") else _load("rt4_floors", a.floors_from).BULK_FLOORS
    B = i.B
    concl = (1 - EPS) * B
    proof = concl / (1 - Fraction(41, S_CHECKS))
    words = dict(i.nw)
    loop = json.loads(a.fixture.read_text())["loops"]["fbw4096_K4_BT256"]
    folds = {"lane": dict(i.nf), "fbw": fbw_counts(H, L, loop)}
    names = {"d′ by STS.128": "F1", "d′ by STG.256 and a bulk copy in": "F2",
             "into shared memory, u by a bulk copy out": "F3"}
    for long, short in names.items():
        folds[short] = dict(floors[long])
    def make_routes(scale: Fraction = Fraction(1)) -> dict[str, dict]:
        cps = checkpoints(H, L, scale)
        rs = {f"{f}+{c}": add(words, folds[f], cps[c]) for f in folds for c in cps}
        for c, n in z_only(H, L, scale).items():
            rs[f"F3+{c}"] = add(words, folds["F3"], n)
        return rs

    def on_gpu(rs: dict[str, dict]) -> dict[str, dict]:
        return {k: v for k, v in rs.items() if "off" not in k}

    routes = make_routes()
    fold_only = {f"{f} (no checkpoint)": add(words, folds[f]) for f in folds}

    print(f"tree {a.tree} ({'A5b' if a5b else 'today: a bulk instruction is one uniform slot'}); "
          f"π* {', '.join(f'{w} {r}' for r, w in (getattr(i, 'cert', None) or {i.pistar: 1}).items())}")
    print(f"♭ {float(B):.6f}; 0.656% lines: conclusion (1−ε)♭ {float(concl):.5f}, proof (1−ε)♭/(1−41/S) "
          f"{float(proof):.5f}")
    out = {"tree": str(a.tree), "a5b": a5b, "B": float(B), "conclusion_line": float(concl),
           "proof_line": float(proof), "routes": {}, "mixes": {}}
    print("\nroutes a unit (issue, smem, max row):")
    for name, n in {**fold_only, **routes}.items():
        ld = loads_of(H, i, n)
        top = max(ld, key=ld.get)
        out["routes"][name] = {r: float(v) for r, v in ld.items()}
        print(f"  {name:28s} issue {float(ld['DS_issue']):8.4f}  smem {float(ld['DS_smem']):8.4f}  "
              f"max {float(ld[top]):8.4f} ({top})")
    built = {k: v for k, v in routes.items() if k.split("+")[0] in ("lane", "fbw")}
    pools = {
        "fold only (FOLD-PROOF's run; rejected)": {k: fold_only[k] for k in ("lane (no checkpoint)", "fbw (no checkpoint)")},
        "built accepted (lane+pair2, fbw+fcp, lane+fcp)": {k: routes[k] for k in ("lane+pair2", "fbw+fcp", "lane+fcp")},
        "fbw and lanes, every checkpoint": on_gpu(built),
        "fbw and lanes, merge off the GPU allowed": built,
        "F1–F3 floors and lanes, on-GPU merges, y never kept": {k: v for k, v in routes.items()
                                                                if "off" not in k and "+z-" not in k},
        "F1–F3 floors and lanes, on-GPU merges": on_gpu(routes),
        "F1–F3 floors and lanes, merge off the GPU allowed": routes,
    }
    def report(label: str, t: Fraction, mix: dict, pi: dict | None) -> None:
        verdict = "BEATS the conclusion line" if t < concl else (
            "under the proof's line" if t < proof else "holds")
        eps = 1 - t * (1 - Fraction(41, S_CHECKS)) / B
        out["mixes"][label] = {"t": float(t), "mix": {k: float(v) for k, v in mix.items()}, "pi": pi,
                               "minus_conclusion": float(t - concl), "minus_proof": float(t - proof),
                               "eps_proof": float(eps)}
        print(f"  {label}: {float(t):.5f} = {float(t / B):.5f} ♭, {float(t - concl):+.5f} / {float(t - proof):+.5f} "
              f"against the lines, ε {float(eps) * 100:.4f}%: {verdict}")
        print(f"      mix {', '.join(f'{k} {float(v):.4f}' for k, v in mix.items())}")
        if pi:
            print(f"      π {', '.join(f'{r} {w:.5f}' for r, w in pi.items())}")

    print("\ncheapest run a unit (the LP over the pool; max-row time; ε the proof's at that time):")
    beta = None
    for label, pool in pools.items():
        t, mix, pi = cheapest(H, i, pool)
        if beta is None:
            beta = mix.get("fbw (no checkpoint)", Fraction(0))
        report(label, t, mix, pi)
    print(f"\nFOLD-PROOF's β-mixed run, fbw on β = {float(beta):.4f} of the units (the fold-only optimum), fixed:")
    for label, (lane, fbw) in {
        "β-mix, CP-W on fbw's share only (the lane share fails CP-W)": ("lane (no checkpoint)", "fbw+fcp"),
        "β-mix, accepted: lane+pair2, fbw+fcp": ("lane+pair2", "fbw+fcp"),
        "β-mix, accepted: lane+pair2, fbw+pair2": ("lane+pair2", "fbw+pair2"),
        "β-mix, counted egress: lane+egr-stg, fbw+egr-stg": ("lane+egr-stg", "fbw+egr-stg"),
    }.items():
        pool = {lane: {**fold_only, **routes}[lane], fbw: routes[fbw]}
        mix = {lane: 1 - beta, fbw: beta}
        report(label, max_row(H, i, pool, mix), mix, None)
    t1 = cheapest(H, i, on_gpu(routes))[0]
    if t1 >= proof and cheapest(H, i, on_gpu(make_routes(Fraction(0))))[0] < proof:
        lo, hi = Fraction(0), Fraction(1)
        for _ in range(20):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cheapest(H, i, on_gpu(make_routes(mid)))[0] < proof else (lo, mid)
        out["merge_break_scale"] = float(hi)
        print(f"\nthe on-GPU floors reach the proof's line when the merges' trees and last steps cost {float(hi):.4f} "
              f"of the count (the counted trees are the bulk floor already: S − 16 source bytes)")
    cert = getattr(i, "cert", None) or {i.pistar: Fraction(1)}
    stmt = {k: v for k, v in words.items() if not k.startswith("LDSM")}
    print("\nthe statement's floor N_w(w* + φ*) (no operand floor), the cheapest route at the exporter's π* and by "
          "the LP:")
    for label, pool in (("on-GPU merges", on_gpu(routes)), ("merge off the GPU allowed", routes)):
        bare = {k: add(stmt, {kk: -vv for kk, vv in words.items()}, v) for k, v in pool.items()}
        arg = min(bare, key=lambda k: price(H, i, cert, bare[k]))
        at = price(H, i, cert, bare[arg])
        lp, _, pi = cheapest(H, i, bare)

        def eps(t):
            return 1 - t * (1 - Fraction(41, S_CHECKS)) / B
        out["mixes"][f"statement, {label}"] = {"at_pistar": float(at), "route_at_pistar": arg, "lp": float(lp),
                                               "pi_lp": pi, "minus_proof_at_pistar": float(at - proof),
                                               "minus_proof_lp": float(lp - proof), "eps_at_pistar": float(eps(at)),
                                               "eps_lp": float(eps(lp))}
        print(f"  {label}: at π* {float(at):.5f} ({arg}; {float(at - proof):+.5f} vs the proof's line, ε "
              f"{float(eps(at)) * 100:.4f}%), by the LP {float(lp):.5f} ({float(lp - proof):+.5f}, ε "
              f"{float(eps(lp)) * 100:.4f}%, π {', '.join(f'{r} {w:.5f}' for r, w in pi.items())})")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
