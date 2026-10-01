#!/usr/bin/env python3
"""ε₈'s leak where it reverts to per side (the assessor's 16:33Z line: outliers-first A, at v1): an upper bound on the share
of the side's MACs a prover saves through the delta route's 2:4 leg, over every window and every disjoint grouping, with
the saving counted only on words chain-exact on the window.  CPU only.

A tile of `need` derived rows saves, on a window of w positions, w·(1 − cost) per word with
cost = 0.5 + 8.46·x̄ + 8.376/w, and that is the mean over its rows of 0.5·w − 8.46·excess_r − 8.376.  So each pair's own
best set of disjoint windows, with negative terms dropped, bounds what any tile holding it saves, and the disjoint pairs
are a matching.  The excess is bounded below per atom by mantissa matches, then exactly with c free per atom where that
leaves a pair a positive saving.  The d leg saves on a window only if some atom of it has 32 − 8.46·nnz > 0, that is
29 or more zeros, and mantissa matches bound the zeros, so it is checked per atom over every pair and reported.
eps8_group.py found no window where the s leg profits (no group of common zeros reaches 9), so the 2:4 leg is the route.  With `--chain`, a word saves on a window only when both of its pair's words are
exact there (from +0, restarting every 4 atoms at v1) in that column of the other side; each column takes its own windows,
which is an upper bound again.  The share is the saved MACs over all MACs of the side (rows × k per column).

    python benchmarks/pouw/sm120_exact/eps8_leak.py [--family outliers-first --side A --chain v1] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import eps8_cost as EC  # noqa: E402
import eps8_fragments as EF  # noqa: E402
import eps8_group as EG  # noqa: E402
import pairing as PR  # noqa: E402
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from run import ATOM  # noqa: E402


def best_windows(exc: np.ndarray) -> np.ndarray:
    """(P, T) excess counts per atom -> (P,) the largest total of max(0, 16·(b − a) − 8.46·excess[a:b] − 8.376) over
    sets of disjoint windows [a, b) of whole atoms."""
    P, T = exc.shape
    C = np.zeros((P, T + 1))
    np.cumsum(exc, 1, out=C[:, 1:])
    best = np.zeros((P, T + 1))
    for b in range(1, T + 1):
        cur = best[:, b - 1].copy()
        for a in range(b):
            g = 16.0 * (b - a) - EC.FFMA * (C[:, b] - C[:, a]) - EC.FADD
            np.maximum(cur, best[:, a] + np.maximum(g, 0.0), out=cur)
        best[:, b] = cur
    return best[:, T]


def best_windows_masked(exc: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """One pair's per-atom excess (T,) and per-column exactness (S, T) -> (S,) each column's best windows, counting a
    window only where the column is exact on all of its atoms."""
    S, T = ok.shape
    C = np.concatenate([[0.0], np.cumsum(exc)])
    run = np.zeros((S, T + 1), np.int64)
    for t in range(T):
        run[:, t + 1] = np.where(ok[:, t], run[:, t] + 1, 0)
    best = np.zeros((S, T + 1))
    for b in range(1, T + 1):
        cur = best[:, b - 1].copy()
        for a in range(b):
            g = 16.0 * (b - a) - EC.FFMA * (C[b] - C[a]) - EC.FADD
            if g <= 0:
                continue
            np.maximum(cur, np.where(run[:, b] >= b - a, best[:, a] + g, cur), out=cur)
        best[:, b] = cur
    return best[:, T]


def max_matching(n: int, edges: list[tuple[int, int, float]]) -> tuple[float, list]:
    import networkx as nx
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    M = nx.max_weight_matching(G)
    w = {(min(u, v), max(u, v)): x for u, v, x in edges}
    pairs = [(min(u, v), max(u, v)) for u, v in M]
    return float(sum(w[p] for p in pairs)), pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="outliers-first")
    ap.add_argument("--side", default="A")
    ap.add_argument("--chain", default="v1", choices=tuple(EF.CHAINS))
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    census.set_atom(ATOM)
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sides = PR.item_sides(f"unit:{args.family}", args.k, args.rows, None, args.seed)[0]
    X, okr = sides[args.side]
    Y = sides["B" if args.side == "A" else "A"][0]
    adm = np.flatnonzero(okr)
    V = census.val(X)[adm]
    R, k = V.shape
    T = k // 32
    M = EG.mantissa_class(V)
    idx = np.array([(u, v) for u in range(R - 1) for v in range(u + 1, R)], np.int32)
    E = np.empty((len(idx), T), np.float64)
    q, dmax = 0, 0
    for u in range(R - 1):
        n = R - 1 - u
        eq = (M[u + 1:] == M[u]).reshape(n, T, 8, 4)
        E[q:q + n] = np.maximum(2 - eq.sum(3), 0).sum(2)
        dmax = max(dmax, int(eq.sum((2, 3)).max()))
        q += n
    ub_mant = best_windows(E)
    pos = np.flatnonzero(ub_mant > 0)
    exact = np.empty((len(pos), T))
    for i, p in enumerate(pos):
        exact[i] = EG.excess_by_c(V[idx[p, 0]], V[idx[p, 1]])[1].min(0)
    ub_exact = best_windows(exact) if len(pos) else np.zeros(0)
    live = ub_exact > 0
    edges = [(int(idx[p, 0]), int(idx[p, 1]), float(s)) for p, s, l in zip(pos, ub_exact, live) if l]
    total = R * k
    unchained, matched = max_matching(R, edges) if edges else (0.0, [])
    rec = {"family": args.family, "side": args.side, "rows_admitted": int(R), "k": k, "pairs": int(len(idx)),
           "pairs_positive_mantissa": int(len(pos)), "pairs_positive_exact": int(live.sum()),
           "saved_macs_per_column_unchained": round(unchained, 3),
           "share_unchained": unchained / total, "share_unchained_pp": round(100 * unchained / total, 6),
           "matched_pairs": len(matched), "d_leg_max_atom_zeros": dmax,
           "d_leg_atom_gain_bound": round(32 - EC.FFMA * (32 - dmax), 3), "d_leg_profits": bool(32 - EC.FFMA * (32 - dmax) > 0)}
    if edges:
        rows = sorted({r for e in edges for r in e[:2]})
        loc = {r: q_ for q_, r in enumerate(rows)}
        fl = EF.flags(X[adm[rows]], Y, EF.CHAINS[args.chain])
        ex_of = {(int(idx[p, 0]), int(idx[p, 1])): exact[i] for i, p in enumerate(pos) if live[i]}
        per_pair = {}
        for u, v, _ in edges:
            ok = fl[loc[u]] & fl[loc[v]]
            per_pair[(u, v)] = float(best_windows_masked(ex_of[(u, v)], ok).mean())
        chained, cm = max_matching(R, [(u, v, w) for (u, v), w in per_pair.items() if w > 0]) \
            if any(w > 0 for w in per_pair.values()) else (0.0, [])
        share_exact = {}
        for t in range(min(T, 16)):
            share_exact[t] = round(float(np.mean([(fl[loc[u]] & fl[loc[v]])[:, t].mean() for u, v, _ in edges])), 6)
        rec.update({"chain": args.chain, "saved_macs_per_column_chained": round(chained, 3),
                    "share_chained": chained / total, "share_chained_pp": round(100 * chained / total, 6),
                    "matched_pairs_chained": len(cm), "chain_exact_share_by_atom_first16": share_exact,
                    "positive_atoms": sorted({int(t) for i, p in enumerate(pos) if live[i]
                                              for t in np.flatnonzero(16.0 - EC.FFMA * exact[i] - EC.FADD > 0)})})
    rec["seconds"] = round(time.time() - t0, 1)
    path = out_dir / f"leak-{args.family}-{args.side}-{args.chain}.json"
    path.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)
    pp = rec.get("share_chained_pp", rec["share_unchained_pp"])
    run_result.write(path, True, "sm120-eps8-leak",
                     f"{args.family} {args.side} at {args.chain}: the 2:4 leg saves at most {pp} pp of the side's MACs",
                     [("share_unchained_pp", rec["share_unchained_pp"], "pp"),
                      *((("share_chained_pp", rec["share_chained_pp"], "pp"),) if "share_chained_pp" in rec else ())])


if __name__ == "__main__":
    main()
