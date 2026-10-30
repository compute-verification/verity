"""Exact Strassen and Winograd in s8 on design D's alphabets (E over n_E centered integers, R over n_R), priced by
HT-PREMISE's executor: the A = 0, x = 0 attack on D0, which iter_price's plane routes miss.

The executor (`execute.py`, `schedules.py`) and HT-CLAIM's `pure_scheme.py` are not in this tree: pass their directory,
e.g. `premise_85d2bf18bc/` of art:3885be78ccac935230cabe540f31bc935c0b646c990b22b93c7c2ffacfb65ba9.

    python iter_strassen.py --premise DIR [--json out.json]

Each tree is run once on random draws with both alphabets' extremes forced and checked against E·Rᵀ, then counted in
W1 at Llama-3.2-1B's linear shapes in the executor's three readings (fast, plain, spilled), MAC-weighted.
"""

import argparse
import json
import sys

import numpy as np

LLAMA = {"qkv": (4096, 2048, 3072), "o": (4096, 2048, 2048), "gate_up": (4096, 2048, 16384), "down": (4096, 8192, 2048)}
PAIRS = [(2, 16), (3, 10), (3, 16), (3, 19), (3, 32), (4, 16), (5, 16), (10, 10), (10, 16), (5, 32), (19, 3), (3, 64),
         (2, 129)]
#: K = 8,192 with wide outputs: a larger model's linears under HT-PREMISE's K_max
WIDE = {"k8_sq": (8192, 8192, 8192), "k8_wide": (8192, 8192, 16384)}
SEARCH = [(4, 16), (3, 19), (10, 16), (3, 10)]


def interval(n):
    return (-(n // 2), n - 1 - n // 2)


def price(ex, tree, m, scheme, depth, ne, nr):
    w = tree(m, scheme, depth, ne, nr)
    ex.heights(w)
    ia, ib = interval(ne), interval(nr)
    side = 2 ** depth * 4
    rng = np.random.default_rng(20261030)
    e = rng.integers(ia[0], ia[1] + 1, (side, side), dtype=np.int64)
    r = rng.integers(ib[0], ib[1] + 1, (side, side), dtype=np.int64)
    e[0, 0], e[-1, -1] = ia
    r[0, 0], r[-1, -1] = ib
    try:
        y = ex.run(w, e, r.T, ia, ib, (side, side, side), ex.Ops())
    except AssertionError:
        return {"fits": False}
    rec = {"fits": True, "exact": bool(np.array_equal(y, e @ r.T))}
    tot, macs = {k: 0.0 for k in ("fast", "plain", "spilled")}, 0
    for name, (mm, k, n) in LLAMA.items():
        o = ex.Ops()
        ex.run(w, None, None, ia, ib, (mm, k, n), o)
        o.add_out(mm * n)
        rec[name] = {r_: o.w1(r_) / (mm * k * n) for r_ in tot}
        for r_ in tot:
            tot[r_] += o.w1(r_)
        macs += mm * k * n
    rec["mac_weighted"] = {r_: v / macs for r_, v in tot.items()}
    return rec


def search(ex, witness, m, ne, nr, depths):
    """HT-PREMISE's argmin schedule (Strassen, Winograd, digit splits, every atom of the instance; tensor cost) at each
    depth, run exactly and counted with its additions at Llama-3.2-1B's shapes and at WIDE."""
    ia, ib = interval(ne), interval(nr)
    per = {}
    for d in depths:
        w = witness(m, d, ne, nr)
        ex.heights(w)
        side = 2 ** d * 4
        rng = np.random.default_rng(20261030)
        e = rng.integers(ia[0], ia[1] + 1, (side, side), dtype=np.int64)
        r = rng.integers(ib[0], ib[1] + 1, (side, side), dtype=np.int64)
        e[0, 0], e[-1, -1] = ia
        r[0, 0], r[-1, -1] = ib
        try:
            y = ex.run(w, e, r.T, ia, ib, (side, side, side), ex.Ops())
        except AssertionError:
            per[d] = {"fits": False}
            continue
        rec = {"fits": True, "exact": bool(np.array_equal(y, e @ r.T)), "tensor": w["cost"] / 8}
        for name, (mm, k, n) in {**LLAMA, **WIDE}.items():
            o = ex.Ops()
            ex.run(w, None, None, ia, ib, (mm, k, n), o)
            o.add_out(mm * n)
            rec[name] = {r_: o.w1(r_) / (mm * k * n) for r_ in ("fast", "plain", "spilled")}
        per[d] = rec
    fit = [v for v in per.values() if v.get("fits")]
    macs = sum(mm * k * n for mm, k, n in LLAMA.values())
    best = {r_: {"llama_1b": sum(min([1.0] + [v[name][r_] for v in fit]) * mm * k * n
                                 for name, (mm, k, n) in LLAMA.items()) / macs,
                 **{name: min([1.0] + [v[name][r_] for v in fit]) for name in WIDE}}
            for r_ in ("fast", "plain", "spilled")}
    return {"depths": per, "best": best}


def per_linear_min(out, ne, nr):
    """The attacker picks a tree (or stock s8, 1.0) per linear: MAC-weighted W1 by reading."""
    recs = [v for k, v in out.items() if k.startswith(f"e{ne}r{nr}_") and isinstance(v, dict) and v.get("fits")
            and "down" in v]
    macs = sum(m * k * n for m, k, n in LLAMA.values())
    return {r: sum(min([1.0] + [v[name][r] for v in recs]) * m * k * n for name, (m, k, n) in LLAMA.items()) / macs
            for r in ("fast", "plain", "spilled")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premise", required=True, help="directory with execute.py, schedules.py and pure_scheme.py")
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--search-depths", default="2,3,4,5", help="witness.py's argmin schedules on SEARCH; '' skips")
    ap.add_argument("--json")
    a = ap.parse_args()
    sys.path.insert(0, a.premise)
    import execute as ex
    from pure_scheme import tree
    from schedules import INSTANCES, STRASSEN, WINOGRAD, Model
    from witness import witness

    m = Model(INSTANCES["h200Measured"], [STRASSEN, WINOGRAD])
    out, ok = {}, True
    for ne, nr in PAIRS:
        best = None
        for scheme in ("strassen", "winograd"):
            for d in map(int, a.depths.split(",")):
                rec = price(ex, tree, m, scheme, d, ne, nr)
                out[f"e{ne}r{nr}_{scheme}_d{d}"] = rec
                if not rec["fits"]:
                    continue
                ok &= rec["exact"]
                mw = rec["mac_weighted"]
                if best is None or mw["fast"] < best[2]["fast"]:
                    best = (scheme, d, mw)
        out[f"e{ne}r{nr}_best"] = None if best is None else {"scheme": best[0], "depth": best[1], **best[2]}
        out[f"e{ne}r{nr}_attack"] = attack = per_linear_min(out, ne, nr)
        msg = "no level fits s8" if best is None else (
            f"{best[0]} d={best[1]}: fast {best[2]['fast']:.3f} plain {best[2]['plain']:.3f} "
            f"spilled {best[2]['spilled']:.3f}; per-linear best fast {attack['fast']:.3f} plain {attack['plain']:.3f}")
        print(f"(n_E, n_R) = ({ne}, {nr}): {msg}", flush=True)
    for ne, nr in SEARCH if a.search_depths else []:
        s = search(ex, witness, m, ne, nr, [int(d) for d in a.search_depths.split(",")])
        ok &= all(v["exact"] for v in s["depths"].values() if v.get("fits"))
        out[f"e{ne}r{nr}_search"] = s
        att = out[f"e{ne}r{nr}_attack"]
        out[f"e{ne}r{nr}_attack"] = {r_: min(att[r_], b["llama_1b"]) for r_, b in s["best"].items()}
        out[f"e{ne}r{nr}_attack_k8"] = {r_: min(b[w] for w in WIDE) for r_, b in s["best"].items()}
        print(f"search ({ne}, {nr}): " + "; ".join(
            f"{r_} Llama-1B {b['llama_1b']:.3f}, K 8,192 wide {min(b[w] for w in WIDE):.3f}"
            for r_, b in s["best"].items()), flush=True)
    out["ok"] = bool(ok)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
