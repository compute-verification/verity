#!/usr/bin/env python3
"""The re-search's threshold (coordinator, server.md 4:35 PM PDT): at window length L and M rows, the larger of the padded
free family's least width N_min(L, M) and the staircase W*(L, M), in `hot_blocks.py --staircase`'s format.

The padded free family (bc-d9842080's `free_family_padded.py`, over `pearlc_region_family.search`'s states): a
composition of Πa, Πb, Πc on a (lm, ln, lk) leaf at `price` per MAC occupies R₀ = lm·Πa rows and C₀ = ln·Πb columns (or
mirrored) and costs T = price·Πρ·max(1, lk·Πc/(32·L)); on M rows it pays at N_min(L, M) = min C₀·T·max(1, R₀/M) over the
states with T·max(1, R₀/M) < 1. Families: FP16/BF16 (16, 8, 16) at 2.0 and TF32 (16, 8, 8) at 4.0, `all` ranks, 8,192³.
Both are evaluated at every M = 8, 16, …, 8,192, the rows `hot_blocks.py` reads a block of M rows at (the next multiple
of 8 up; N_min doesn't increase with M, so reading it there can only lower the threshold). The staircase is read at its
step at or below M. A step is null where either is (nothing pays under 8,192 columns on the staircase, no state pays in
the family), and `via` names what binds: the staircase's composition, or the family's (leaf, R₀, C₀, K, T).
Gate: every `padded_min_cols` point of free_family_compare_all.json is reproduced to 0.1 column (exit 3 otherwise).
    python3 padded_floor.py --family-dir DIR --ranks RANKS.json --staircase S.json --compare C.json --out OUT.json
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

LEAVES = {"fp16": ((16, 8, 16), 2.0), "tf32": ((16, 8, 8), 4.0)}
MS = np.arange(8, 8193, 8)


def family_states(fdir: Path, ranks: Path, leaf, price) -> np.ndarray:
    sys.path.insert(0, str(fdir))
    argv, sys.argv = sys.argv, ["pearlc_region_family.py", str(ranks)]
    try:
        import pearlc_region_family as F
    finally:
        sys.argv = argv
    _, best = F.search("all", leaf, price, 8192, 8192)
    return np.array([(a, b, c, math.exp(v)) for (a, b, c), v in best.items() if (a, b, c) != (1, 1, 1)], float)


def family_curve(st: np.ndarray, leaf, price, L: int) -> tuple[np.ndarray, list]:
    """N_min(L, M) at every M in MS (inf where nothing pays) and, per M, the binding (R₀, C₀, K, T)."""
    lm, ln, lk = leaf
    T = price * st[:, 3] * np.maximum(1.0, lk * st[:, 2] / (32 * L))
    keep = T < 1
    a, b, c, T = st[keep, 0], st[keep, 1], st[keep, 2], T[keep]
    best, arg = np.full(len(MS), np.inf), [None] * len(MS)
    for R0, C0 in ((lm * a, ln * b), (ln * b, lm * a)):
        # per R₀, only the states on the Pareto front of (T, C₀·T) can set N_min at some M
        V = C0 * T
        order = np.lexsort((V, T, R0))
        g = []
        for i0, i1 in itertools.pairwise([0, *np.flatnonzero(np.diff(R0[order])) + 1, len(order)]):
            run = order[i0:i1]
            g.extend(run[V[run] < np.minimum.accumulate(np.r_[np.inf, V[run][:-1]])].tolist())
        if not g:
            continue
        g = np.array(g)
        r0, c0, tg, kg = R0[g], C0[g], T[g], lk * c[g]
        t = tg[:, None] * np.maximum(1.0, r0[:, None] / MS[None, :])
        n = np.where(t < 1, c0[:, None] * t, np.inf)
        k = np.argmin(n, 0)
        v = n[k, np.arange(len(MS))]
        for j in np.flatnonzero(v < best):
            best[j] = v[j]
            arg[j] = (float(r0[k[j]]), float(c0[k[j]]), float(kg[k[j]]), round(float(tg[k[j]]), 5))
    return best, arg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-dir", type=Path, required=True, help="the directory holding pearlc_region_family.py")
    ap.add_argument("--ranks", type=Path, required=True)
    ap.add_argument("--staircase", type=Path, required=True)
    ap.add_argument("--compare", type=Path, required=True, help="free_family_compare_all.json, the gate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    stair = json.loads(args.staircase.read_text())["floors"]
    states = {k: family_states(args.family_dir, args.ranks, *v) for k, v in LEAVES.items()}
    cmp = json.loads(args.compare.read_text())["windows"]
    bad, floors, binds = [], {}, {"staircase": 0, "fp16": 0, "tf32": 0, "null": 0}
    for L in range(4, 257):
        curves = {k: family_curve(states[k], *LEAVES[k], L) for k in LEAVES}
        if str(L) in cmp:
            want = cmp[str(L)]
            for M, n in zip(want["rows"], want["padded_min_cols"]):
                got = curves["fp16"][0][M // 8 - 1] if M % 8 == 0 else None
                ok = (n is None and not np.isfinite(got)) or (n is not None and got is not None and abs(got - n) <= 0.1)
                if not ok:
                    bad.append({"L": L, "M": M, "want": n, "got": None if got is None or not np.isfinite(got) else got})
        ss = stair[str(L)]
        srows = [int(s["rows"]) for s in ss]
        steps, last = [], None
        for j, M in enumerate(MS):
            s = ss[bisect.bisect_right(srows, int(M)) - 1]
            w = math.inf if s["floor"] is None else float(s["floor"])
            fam = min(LEAVES, key=lambda k: curves[k][0][j])
            f = curves[fam][0][j]
            if not (np.isfinite(w) and np.isfinite(f)):
                val, via, by = None, None, "null"
            elif w >= f:
                val, via, by = w, {"binds": "staircase", "staircase_rows": s["rows"], **(s["via"] or {})}, "staircase"
            else:
                R0, C0, K, T = curves[fam][1][j]
                val, via, by = round(f, 1), {"binds": f"free-family-padded/{fam}", "R0": R0, "C0": C0, "K": K, "T": T,
                                             "staircase_floor": w, "rows_needed": None}, fam
            binds[by] += 1
            key = (val, via.get("binds") if via else None, json.dumps(via, sort_keys=True))
            if key != last:
                steps.append({"rows": int(M), "floor": val, "via": via})
                last = key
        floors[str(L)] = steps
    ident = {p.name: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in (args.staircase, args.ranks, args.compare, args.family_dir / "pearlc_region_family.py")}
    doc = {"table": "max(N_min(L, M), W*(L, M)): the padded free family (FP16 and TF32, all ranks) and the staircase, "
                    "L = 4..256 atoms, M = 8..8,192 rows at every multiple of 8",
           "how_to_read": "floors[L] is a list of steps {rows, floor, via}; a block of M rows is checked against the step "
                          "at or below M rounded up to a multiple of 8; floor null where either bound has nothing that pays",
           "inputs": ident, "points_by_binding": binds, "gate_mismatches": bad, "seconds": round(time.time() - t0, 1),
           "floors": floors}
    args.out.write_text(json.dumps(doc) + "\n")
    print(json.dumps({"points_by_binding": binds, "gate_mismatches": len(bad), "first": bad[:5],
                      "seconds": doc["seconds"]}))
    if bad:
        sys.exit(3)


if __name__ == "__main__":
    main()
