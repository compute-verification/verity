#!/usr/bin/env python3
"""ε₈'s cost test (the assessor's 16:15Z line in `red-team/ratings.md`): `fp8-merge-rate/sm120` stays on the joint condition
if no chain-exact row pair of Results 13's one-sided fragments profits from a delta route.  CPU only.

The fragments are eps8_fragments.py's: the census pairing's A-only and B-only searches replayed exactly as node 2 ran them
(the same --expect gate, step for step), with its per-pair chain leg (a pair is chain-inexact when no column has 32 of the
fragment's matched k pairs in atoms exact for both words).  The route (14:08Z item 3) forms the derived words' window sums
from the base words': C21 = D·C11 + (A21 − D·A11)·B on the window, D a diagonal of powers of two ±2^e, one per pair, each
the ratio at which that pair's values meet most often (the zero set of Δ is the same in either direction).  At W1's prices
(FFMA 8.46, FADD 8.376 FP8 units per MAC) its cost per MAC, against the direct window's 1, is

    min(8.46·d, 0.5 + 8.46·x, s) + 8.376/w

with w the window's k positions, d Δ's nonzero density, x the density of nonzeros beyond 2 per aligned 4-group (the 2:4
sparse FP8 path at 0.5, the rest patched on CUDA cores), and s the share of positions where any row of the group sharing
the base has Δ ≠ 0 (a compacted dense GEMM, gather ignored).  Two readings of the group:

- `pair`: one derived row per base, so s = d.  The window is the prover's: any run of whole atoms on which both words are
  chain-exact for some column of the other side (from +0; v1 restarts every 4 atoms, and a window may cross a group
  boundary at one merge, the prover's side).  This prices a one-row compacted GEMM at 1 per useful MAC.
- `block`: the fragment's derived rows as one tile (16 A rows, or 8 B columns: the m16n8k32 fragment's M or N, since W1
  prices `mma.sync` by its whole fragment), so s is the union of their Δ ≠ 0 positions, and d and x are the tile's means.
  The window is any run of whole atoms, chain-exact or not (the prover's side).

A pair (or block) passes when its cheapest window costs ≥ 1; the sufficient form d ≥ 1/8, x ≥ 1/16, s ≥ 1 − 8.4/w is
reported beside it.  Under a reading, a family's fragment keeps the joint statement at a chain when each pair is
chain-inexact there or passes; otherwise it reverts to per side (16:15Z).  Diagnostics per pair: the columns on which its
cheapest window is chain-exact, the share of that window's Δ zeros where both codes are zero (operand zeros, which a direct
window could skip without the route), and d over the whole row and on the fragment's own 64 k indices.

    python benchmarks/pouw/sm120_exact/eps8_cost.py --expect DIR [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import eps8_fragments as EF  # noqa: E402
import pairing as PR  # noqa: E402
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from run import ATOM  # noqa: E402

FFMA, FADD, SPARSE = 8.46, 8.376, 0.5


def ratios(vu: np.ndarray, vv: np.ndarray) -> list[float]:
    """Every c = ±2^e with vv = c·vu at some index where both are nonzero, most frequent first (1.0 when there is none)."""
    m = (vu != 0) & (vv != 0)
    r = vv[m] / vu[m]
    mant, _ = np.frexp(np.abs(r))
    cs, n = np.unique(r[mant == 0.5], return_counts=True)
    return [float(c) for c in cs[np.argsort(-n, kind="stable")]] or [1.0]


def reach(f: np.ndarray) -> np.ndarray:
    """(S, T) exactness -> (S, T): for each column and atom t, the end b of the run t..b-1 of exact atoms (t if t isn't)."""
    S, T = f.shape
    r = np.full(S, T)
    out = np.empty((S, T), np.int64)
    for t in range(T - 1, -1, -1):
        r = np.where(f[:, t], r, t)
        out[:, t] = r
    return out


def window_costs(n: np.ndarray, e: np.ndarray, u: np.ndarray, rows: int) -> tuple:
    """Per-atom Δ ≠ 0 counts n, excess e (summed over the group's rows) and union counts u -> every window's (cost, d, x, s)."""
    T = len(n)
    a_, b_ = np.arange(T)[:, None], np.arange(1, T + 1)[None, :]
    P = [np.concatenate([[0], np.cumsum(v)]) for v in (n, e, u)]
    w = 32.0 * np.maximum(b_ - a_, 1)
    d, x, s = ((p[b_] - p[a_]) / (w * r) for p, r in zip(P, (rows, rows, 1)))
    cost = np.minimum(np.minimum(FFMA * d, SPARSE + FFMA * x), s) + FADD / w
    return np.where(b_ > a_, cost, np.inf), d, x, s


def atom_counts(nz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = nz.shape[-1] // 32
    return nz.reshape(-1, T, 32).sum(2).sum(0), np.maximum(nz.reshape(-1, T, 8, 4).sum(3) - 2, 0).sum(2).sum(0)


def verdict(cost: np.ndarray, d, x, s, i) -> dict:
    a, b = int(i[0]), int(i[1]) + 1
    w = 32 * (b - a)
    return {"cost": round(float(cost[i]), 6), "window_atoms": [a, b], "w": w, "d": round(float(d[i]), 6),
            "x": round(float(x[i]), 6), "s": round(float(s[i]), 6), "passes": bool(cost[i] >= 1.0),
            "sufficient": bool(d[i] >= 1 / 8 and x[i] >= 1 / 16 and s[i] >= 1 - 8.4 / w)}


def pair_route(vu: np.ndarray, vv: np.ndarray, Rc: np.ndarray) -> dict | None:
    """The pair reading: the cheapest (c, window) over windows [a, b) chain-exact for both words in some column of Rc."""
    T = Rc.shape[1]
    R = Rc.max(0)
    ok = (np.arange(1, T + 1)[None, :] > np.arange(T)[:, None]) & (np.arange(1, T + 1)[None, :] <= R[:, None])
    if not ok.any():
        return None
    best = None
    for c in ratios(vu, vv):
        nz = (vv - c * vu) != 0
        n, e = atom_counts(nz[None])
        cost, d, x, s = window_costs(n, e, n, 1)
        cost = np.where(ok, cost, np.inf)
        i = np.unravel_index(int(np.argmin(cost)), cost.shape)
        if best is None or cost[i] < best[0][best[4]]:
            best = (cost, d, x, s, i, c, nz)
    cost, d, x, s, i, c, nz = best
    out = verdict(cost, d, x, s, i)
    a, b = out["window_atoms"]
    idx = np.arange(32 * a, 32 * b)
    zero = ~nz[idx]
    both0 = (vu[idx] == 0) & (vv[idx] == 0)
    out.update({"c": c, "columns": int((Rc[:, a] >= b).sum()),
                "zeros_operand_share": round(float(both0.sum() / zero.sum()), 6) if zero.any() else None})
    return out


def block_route(X: np.ndarray, pairs: list, cs: list) -> dict:
    """The block reading: the fragment's derived rows as one tile, over every window of whole atoms."""
    nz = np.stack([(census.val(X[v]) - c * census.val(X[u])) != 0 for (u, v), c in zip(pairs, cs)])
    n, e = atom_counts(nz)
    un = nz.any(0).reshape(-1, 32).sum(1)
    cost, d, x, s = window_costs(n, e, un, len(pairs))
    out = verdict(cost, d, x, s, np.unravel_index(int(np.argmin(cost)), cost.shape))
    out["rows"] = len(pairs)
    out["union_density_row"] = round(float(nz.any(0).mean()), 6)
    return out


def test_fragment(side: str, X: np.ndarray, Y: np.ndarray, frag: dict) -> dict:
    """frag: eps8_fragments.analyse's record for this side (row pairs, chain legs), plus its matched k pairs."""
    rows = sorted({r for p in frag["pairs"] for r in p["rows"]})
    assert len(rows) == 2 * len(frag["pairs"]), "a fragment row in two pairs"
    loc = {r: q for q, r in enumerate(rows)}
    fl = {ch: EF.flags(X[rows], Y, G) for ch, G in EF.CHAINS.items()}
    K = np.array(frag["k_pairs"][:PR.KPAIRS]).reshape(-1)
    out = {"side": side, "pairs": []}
    cs = []
    for p in frag["pairs"]:
        u_, v_ = p["rows"]
        vu, vv = census.val(X[u_]), census.val(X[v_])
        c0 = ratios(vu, vv)[0]
        cs.append(c0)
        rec = {"rows": [u_, v_], "c_row": c0, "row_d": round(float(((vv - c0 * vu) != 0).mean()), 6),
               "fragment_k_d": round(float(((vv[K] - c0 * vu[K]) != 0).mean()), 6)}
        for ch in EF.CHAINS:
            Rc = reach(fl[ch][loc[u_]] & fl[ch][loc[v_]])
            rec[ch] = {"chain_exact": not p[ch]["chain_inexact"],
                       "exact_atoms_any_column": int((Rc.max(0) > np.arange(Rc.shape[1])).sum()),
                       "pair": pair_route(vu, vv, Rc)}
        out["pairs"].append(rec)
    out["block"] = block_route(X, [p["rows"] for p in frag["pairs"]], cs)
    for ch in EF.CHAINS:
        ex = [q for q, r in enumerate(out["pairs"]) if r[ch]["chain_exact"]]
        routes = [(q, out["pairs"][q][ch]["pair"]) for q in ex if out["pairs"][q][ch]["pair"]]
        fail = [q for q, r in routes if not r["passes"]]
        out[ch] = {"pairs": len(out["pairs"]), "chain_exact": len(ex),
                   "pair": {"fail": fail, "min_cost": min((r["cost"] for _, r in routes), default=None),
                            "max_cost": max((r["cost"] for _, r in routes), default=None),
                            "sufficient_all": all(r["sufficient"] for _, r in routes),
                            "verdict": "joint holds" if not fail else "reverts to per side"},
                   "block": {"cost": out["block"]["cost"], "sufficient": out["block"]["sufficient"],
                             "verdict": "joint holds" if not ex or out["block"]["passes"] else "reverts to per side"}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", required=True, help="node 2's pairing_census JSONs (unit-<family>.json)")
    ap.add_argument("--families", default=",".join(EF.FAMILIES))
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--k-search", type=int, default=8192)
    ap.add_argument("--cands", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    census.set_atom(ATOM)
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    items, ok = [], True
    for fam in args.families.split(","):
        t = time.time()
        path = Path(args.expect) / f"unit-{fam}.json"
        expect = json.loads(path.read_text())
        sides, res, rec = EF.replay(fam, expect, args)
        rec["expect_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        ok &= all(g["steps_equal"] and g["fragment"] == g["expected_fragment"] for g in rec["gate"].values())
        rec["fragments"] = {}
        for label, s, o in (("A_only", "A", "B"), ("B_only", "B", "A")):
            if res[label]["fragment"]:
                frag = EF.analyse(s, sides[s][0], sides[o][0], res[label]["fragment_pairs"][s], res[label]["fragment_k_pairs"])
                frag["k_pairs"] = res[label]["fragment_k_pairs"]
                rec["fragments"][s] = test_fragment(s, sides[s][0], sides[o][0], frag)
        rec["seconds"] = round(time.time() - t, 1)
        (out_dir / f"{fam}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({"family": fam, "gate": rec["gate"], "seconds": rec["seconds"],
                          "fragments": {s: {ch: f[ch] for ch in EF.CHAINS} for s, f in rec["fragments"].items()}}), flush=True)
        items.append(rec)
    frags = [(r["item"], s, f) for r in items for s, f in r["fragments"].items()]
    summary = {"tool": "pouw_sm120_eps8_cost", "gate_ok": ok, "prices": {"ffma": FFMA, "fadd": FADD, "sparse": SPARSE},
               "fragments": [{"item": i, "side": s, "block": f["block"], **{ch: f[ch] for ch in EF.CHAINS}}
                             for i, s, f in frags]}
    tot = {}
    for ch in EF.CHAINS:
        pc = [f[ch]["pair"]["min_cost"] for _, _, f in frags if f[ch]["pair"]["min_cost"] is not None]
        tot[ch] = {"chain_exact": sum(f[ch]["chain_exact"] for _, _, f in frags),
                   "pair_fail": sum(len(f[ch]["pair"]["fail"]) for _, _, f in frags),
                   "pair_reverting": [f"{i}/{s}" for i, s, f in frags if f[ch]["pair"]["fail"]],
                   "pair_min_cost": min(pc, default=None),
                   "block_reverting": [f"{i}/{s}" for i, s, f in frags if f[ch]["block"]["verdict"] != "joint holds"],
                   "block_min_cost": min((f["block"]["cost"] for _, _, f in frags if f[ch]["chain_exact"]), default=None)}
    summary["totals"] = tot
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(tot), flush=True)
    run_result.write(path, ok, "sm120-eps8-cost",
                     f"{len(frags)} one-sided fragments replayed (gate {'ok' if ok else 'FAILED'}); chain-exact pairs failing "
                     f"the 16:15Z cost test per pair: v2 {tot['v2']['pair_fail']} of {tot['v2']['chain_exact']}, "
                     f"v1 {tot['v1']['pair_fail']} of {tot['v1']['chain_exact']}; fragments failing it per block: "
                     f"v2 {len(tot['v2']['block_reverting'])}, v1 {len(tot['v1']['block_reverting'])}",
                     [("fragments", len(frags), "count"),
                      *((f"{m}_{ch}", tot[ch][m], "count") for ch in EF.CHAINS for m in ("chain_exact", "pair_fail")),
                      *((f"block_fail_{ch}", len(tot[ch]["block_reverting"]), "count") for ch in EF.CHAINS),
                      *((f"{m}_{ch}", tot[ch][m], "per MAC") for ch in EF.CHAINS for m in ("pair_min_cost", "block_min_cost")
                        if tot[ch][m] is not None)])
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
