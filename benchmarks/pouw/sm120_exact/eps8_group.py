#!/usr/bin/env python3
"""ε₈ over the prover's own tiles (the assessor's 15:57Z ruling on `note:20261001T1541Z-…-eps8-reading-second-ask`): s is the
union of Δ ≠ 0 over a fragment's 16 derived A rows (8 B columns), and the prover chooses which row pairs share the fragment.
Does any choice of 16 (8) disjoint row pairs of a family's sampled rows make eps8_cost.py's block cost
min(8.46·d, 0.5 + 8.46·x, s) + 8.376/w fall below 1 on some window?  CPU only.

A pair (u, v) has Δ = v − c·u with c = ±2^e, one per pair.  The three terms separately:
- s: on a window of w positions, s·w + 8.376 < w exactly when the group's pairs share more than 8.376 positions where every
  Δ is 0, and a window holds at most the whole row's common zeros.  So some window profits from s iff some group shares at
  least 9 common zeros over the row.  The search: a beam over position sets J, where rows fall in classes equal on J up to
  ±2^e (each row's values on J divided by sign·2^e of its first nonzero there), and the disjoint pairs zero on all of J
  number Σ ⌊class size / 2⌋.  J grows while at least 16 (8) pairs remain; the largest J found is the group's common
  zeros, recounted exactly on the pairs taken from its classes, with that group's cheapest window under the full formula.
  This is a search, a lower bound on the adversary, like pairing.py's.
- d and x: Δ_j = 0 needs u_j and v_j zero together or of equal mantissa, for any c.  So per pair and atom, the mantissa
  matches bound the zeros above, d below, and the 2:4 excess below (an aligned 4-group with m matches keeps at least
  2 − m nonzeros past 2).  A window's d and x average its atoms, and a group's average its pairs, so the mean of the 16 (8)
  smallest per-pair minima over atoms bounds every group on every window: d's term is at least 8.46 times it, x's at
  least 0.5 + 8.46 times its.  These are bounds, not searches.
A family-side keeps the joint statement when no group found reaches 9 common zeros and both bounds are at least 1.  The
gate: the census's own tile (`--records`, eps8_cost.py's per-family JSONs) recomputes to its stored block cost and s.

    python benchmarks/pouw/sm120_exact/eps8_group.py [--records DIR] [--out DIR]
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
import pairing as PR  # noqa: E402
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from run import ATOM  # noqa: E402

FRAGMENTS = (("rank1", "A"), ("rank1", "B"), ("duplicated", "A"), ("coherent-gaussian", "A"), ("outliers-first", "A"))
COMMON_MIN = 9


def mantissa_class(V: np.ndarray) -> np.ndarray:
    """0 for a zero, else 1 + an index of |v|'s mantissa: equal for values that differ by ±2^e."""
    m, _ = np.frexp(np.abs(V))
    _, idx = np.unique(m, return_inverse=True)
    return np.where(V == 0, 0, 1 + idx.reshape(V.shape))


def excess_by_c(u: np.ndarray, v: np.ndarray) -> tuple[list[float], np.ndarray]:
    """A pair's 2:4 excess count per atom at each ±2^e ratio its values meet at (a ratio they never meet at is no better)."""
    cs = EC.ratios(u, v)
    T = len(u) // 32
    nz = (v[None, :] - np.array(cs)[:, None] * u[None, :]) != 0
    return cs, np.maximum(nz.reshape(len(cs), T, 8, 4).sum(3) - 2, 0).sum(2)


def bounds(V: np.ndarray, need: int, start: int = 64) -> dict:
    """Lower bounds on the d and x terms of every group of `need` disjoint pairs (disjointness dropped).

    d: per pair, its densest-zero atom by mantissa matches, so 8.46 times the mean of the `need` smallest bounds every
    window.  x: per window W, the mean of the `need` smallest per-pair excess densities on W gives
    0.5 + 8.46·x̄ + 8.376/w.  The mantissa counts give it for every pair; where that falls under 1, the exact excess (one c
    per pair, the best on W) replaces it for the pairs in front, extended until every pair left out has a mantissa bound
    at or above the `need`-th smallest exact value, so the window's bound is certified."""
    M = mantissa_class(V)
    R, k = V.shape
    T = k // 32
    npairs = R * (R - 1) // 2
    E = np.empty((npairs, T), np.uint8)
    idx = np.empty((npairs, 2), np.int32)
    dmin = np.empty(npairs)
    q = 0
    for u in range(R - 1):
        n = R - 1 - u
        eq = (M[u + 1:] == M[u]).reshape(n, T, 8, 4)
        E[q:q + n] = np.maximum(2 - eq.sum(3), 0).sum(2)
        dmin[q:q + n] = 1 - eq.sum((2, 3)).max(1) / 32.0
        idx[q:q + n, 0], idx[q:q + n, 1] = u, np.arange(u + 1, R)
        q += n
    ld = float(np.sort(dmin)[:need].mean())
    P = np.zeros((npairs, T + 1), np.int32)
    np.cumsum(E, 1, out=P[:, 1:])
    del E
    exact: dict[int, tuple] = {}
    worst = {"cost": np.inf}
    flagged = refined = 0
    for a in range(T):
        for b in range(a + 1, T + 1):
            w = 32 * (b - a)
            s = P[:, b] - P[:, a]
            front = np.argpartition(s, need - 1)[:need]
            lb = EC.SPARSE + EC.FFMA * float(s[front].mean()) / w + EC.FADD / w
            if lb >= 1:
                if lb < worst["cost"]:
                    worst = {"cost": lb, "window_atoms": [a, b], "bound": "mantissa"}
                continue
            flagged += 1
            N = start
            while True:
                order = np.argpartition(s, min(N, npairs - 1))[:N]
                vals = []
                for p in order:
                    if p not in exact:
                        cs, ex_ = excess_by_c(V[idx[p, 0]], V[idx[p, 1]])
                        exact[int(p)] = (cs, ex_.astype(np.int32))
                    ex = exact[int(p)][1]
                    vals.append(int(ex[:, a:b].sum(1).min()))
                vals = np.sort(np.array(vals))
                tau = vals[need - 1]
                rest = np.partition(s, N)[N] if N < npairs else np.inf
                if rest >= tau or N >= npairs:
                    break
                N *= 2
            refined = max(refined, N)
            lb = EC.SPARSE + EC.FFMA * float(vals[:need].mean()) / w + EC.FADD / w
            if lb < worst["cost"]:
                worst = {"cost": lb, "window_atoms": [a, b], "bound": "exact", "pairs_exact": int(N)}
    out = {"d_lower": round(ld, 6), "d_term_lower": round(EC.FFMA * ld, 6), "x_term_lower": round(float(worst["cost"]), 6),
           "x_worst_window": worst, "x_windows_refined": flagged, "x_most_pairs_exact": refined,
           "pairs_exact_total": len(exact), "pairs": npairs,
           "holds": bool(EC.FFMA * ld >= 1 and worst["cost"] >= 1)}
    if worst["cost"] < 1:
        a, b = worst["window_atoms"]
        s = P[:, b] - P[:, a]
        order = np.argsort(s, kind="stable")
        group, used, q = [], set(), 0
        while len(group) < need and q < npairs:
            for p in order[q:q + 256]:
                p = int(p)
                if p not in exact:
                    cs, ex_ = excess_by_c(V[idx[p, 0]], V[idx[p, 1]])
                    exact[p] = (cs, ex_.astype(np.int32))
            q += 256
            ranked = sorted(exact, key=lambda p: int(exact[p][1][:, a:b].sum(1).min()))
            group, used = [], set()
            for p in ranked:
                u, v = int(idx[p, 0]), int(idx[p, 1])
                if u in used or v in used:
                    continue
                cs, ex = exact[p]
                group.append((u, v, float(cs[int(ex[:, a:b].sum(1).argmin())])))
                used |= {u, v}
                if len(group) == need:
                    break
        out["x_group"] = group
    return out


def refine(V: np.ndarray, cls: np.ndarray, scale: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Classes and normalizers after adding position j (rows unset so far take their normalizer from j)."""
    v = V[:, j]
    unset = scale == 0
    e = np.where(v != 0, np.sign(v) * 2.0 ** np.floor(np.log2(np.abs(np.where(v == 0, 1.0, v)))), 0.0)
    sc = np.where(unset & (v != 0), e, scale)
    nv = np.where(sc != 0, v / np.where(sc == 0, 1.0, sc), 0.0)
    _, cid = np.unique(np.stack([cls.astype(np.float64), nv], 1), axis=0, return_inverse=True)
    return cid.reshape(-1), sc


def pair_counts(V: np.ndarray, cls: np.ndarray, scale: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """For each candidate column, Σ ⌊class size / 2⌋ after refining by it."""
    sub = V[:, cols]
    unset = (scale == 0)[:, None]
    e = np.where(sub != 0, np.sign(sub) * 2.0 ** np.floor(np.log2(np.abs(np.where(sub == 0, 1.0, sub)))), 0.0)
    sc = np.where(unset & (sub != 0), e, scale[:, None])
    nv = np.where(sc != 0, sub / np.where(sc == 0, 1.0, sc), 0.0)
    _, code = np.unique(nv, return_inverse=True)
    key = cls[:, None].astype(np.int64) * (code.max() + 1) + code.reshape(nv.shape)
    key = np.sort(key, 0)
    out = np.zeros(len(cols), np.int64)
    run = np.ones(len(cols), np.int64)
    for r in range(1, len(key)):
        same = key[r] == key[r - 1]
        out += np.where(same, 0, run // 2)
        run = np.where(same, run + 1, 1)
    return out + run // 2


def search(V: np.ndarray, need: int, beam: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """The beam over J on the whole row: the largest J found with at least `need` disjoint pairs zero on all of it."""
    R, k = V.shape
    cols = np.arange(k)
    states = [([], np.zeros(R, np.int64), np.zeros(R))]
    best = states[0]
    while True:
        cand = []
        for J, cls, sc in states:
            pc = pair_counts(V, cls, sc, cols)
            pc[J] = -1
            for j in np.argsort(-pc, kind="stable")[:beam]:
                if pc[j] >= need:
                    cand.append((int(pc[j]), J + [int(j)], cls, sc))
        if not cand:
            return best[0], best[1], best[2]
        cand.sort(key=lambda z: -z[0])
        seen, nxt = set(), []
        for p, J, cls, sc in cand:
            key = tuple(sorted(J))
            if key in seen:
                continue
            seen.add(key)
            c2, s2 = refine(V, cls, sc, J[-1])
            nxt.append((J, c2, s2))
            if len(nxt) >= beam:
                break
        states = nxt
        best = states[0]


def take_pairs(cls: np.ndarray, scale: np.ndarray, need: int) -> list[tuple[int, int, float]]:
    """`need` disjoint pairs from the classes (largest classes first), each with its c = scale_v / scale_u."""
    out = []
    for c in np.argsort(-np.bincount(cls), kind="stable"):
        rows = np.flatnonzero(cls == c)
        for q in range(0, len(rows) - 1, 2):
            u, v = int(rows[q]), int(rows[q + 1])
            out.append((u, v, float(scale[v] / scale[u]) if scale[u] and scale[v] else 1.0))
            if len(out) == need:
                return out
    return out


def group_cost(V: np.ndarray, pairs: list) -> dict:
    nz = np.stack([(V[v] - c * V[u]) != 0 for u, v, c in pairs])
    n, e = EC.atom_counts(nz)
    un = nz.any(0).reshape(-1, 32).sum(1)
    cost, d, x, s = EC.window_costs(n, e, un, len(pairs))
    out = EC.verdict(cost, d, x, s, np.unravel_index(int(np.argmin(cost)), cost.shape))
    common = ~nz.any(0)
    rows = [r for u, v, _ in pairs for r in (u, v)]
    out["common_zeros_row"] = int(common.sum())
    out["common_operand_zeros"] = int((common & (V[rows] == 0).all(0)).sum())
    return out


def pair_checks(X: np.ndarray, Y: np.ndarray, pairs: list, a: int, b: int) -> list[dict]:
    """eps8_fragments.py's rules on a group's pairs over atoms [a, b): the near-duplicate credit rule on codes, and the
    chain (v2, v1 from +0): exact for both words on every atom of the window in some column of the other side."""
    rows = sorted({r for u, v, _ in pairs for r in (u, v)})
    loc = {r: q for q, r in enumerate(rows)}
    fl = {ch: EF.flags(X[rows], Y, G) for ch, G in EF.CHAINS.items()}
    W = np.arange(a, b)
    Wg = np.unique(np.arange(32 * a, 32 * b) // 128)
    out = []
    for u, v, c in pairs:
        same, opp = EF.agree(X[u], X[v])
        row = max(int(same.sum()), int(opp.sum())) / X.shape[1]
        at, gr = EF.block_agree(same, opp, 32), EF.block_agree(same, opp, 128)
        out.append({"rows": [u, v], "c": c, "row_agreement": round(row, 6),
                    "uncredited": {"v2": bool(row >= EF.ROW_RULE or (at[W] > EF.D_ATOM).all()),
                                   "v1": bool(row >= EF.ROW_RULE or (gr[Wg] > EF.D_GROUP).all())},
                    "chain_exact": {ch: bool((fl[ch][loc[u]] & fl[ch][loc[v]])[:, a:b].all(1).any()) for ch in EF.CHAINS}})
    return out


def census_gate(fam: str, side: str, V: np.ndarray, records: Path | None) -> dict | None:
    if not records:
        return None
    frag = json.loads((records / f"{fam}.json").read_text())["fragments"][side]
    pairs = [(p["rows"][0], p["rows"][1], p["c_row"]) for p in frag["pairs"]]
    got = group_cost(V, pairs)
    return {"stored_cost": frag["block"]["cost"], "cost": got["cost"], "stored_s": frag["block"]["s"], "s": got["s"],
            "common_zeros_row": got["common_zeros_row"],
            "ok": bool(abs(got["cost"] - frag["block"]["cost"]) < 1e-6 and abs(got["s"] - frag["block"]["s"]) < 1e-6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", default=",".join(f"{f}/{s}" for f, s in FRAGMENTS))
    ap.add_argument("--records", default=None, help="eps8_cost.py's per-family JSONs, for the census-tile gate")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    census.set_atom(ATOM)
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = Path(args.records) if args.records else None
    items, ok = [], True
    cache: dict = {}
    for spec in args.fragments.split(","):
        fam, side = spec.split("/")
        t = time.time()
        if fam not in cache:
            cache[fam] = PR.item_sides(f"unit:{fam}", args.k, args.rows, None, args.seed)[0]
        X, okr = cache[fam][side]
        V = census.val(X)
        need = PR.NEED[side]
        gate = census_gate(fam, side, V, records)
        ok &= gate is None or gate["ok"]
        adm = np.flatnonzero(okr)
        Va = V[adm]
        J, cls, sc = search(Va, need, args.beam)
        pairs = [(int(adm[u]), int(adm[v]), c) for u, v, c in take_pairs(cls, sc, need)]
        grp = group_cost(V, pairs)
        bd = bounds(Va, need)
        xg = None
        if "x_group" in bd:
            xp = [(int(adm[u]), int(adm[v]), c) for u, v, c in bd.pop("x_group")]
            xc = group_cost(V, xp)
            a, b = xc["window_atoms"]
            checks = pair_checks(X, cache[fam]["B" if side == "A" else "A"][0], xp, a, b)
            live = {ch: sum(1 for c_ in checks if c_["chain_exact"][ch] and not c_["uncredited"][ch]) for ch in EF.CHAINS}
            xg = {"pairs": checks, **xc, "credited_chain_exact": live,
                  "reverts": {ch: bool(xc["cost"] < 1 and live[ch] > 0) for ch in EF.CHAINS}}
        s_ok = len(J) < COMMON_MIN and grp["common_zeros_row"] < COMMON_MIN
        x_verdict = ("certified" if bd["x_term_lower"] >= 1 else
                     "breaks" if xg and xg["cost"] < 1 else "inconclusive")
        rec = {"family": fam, "side": side, "need": need, "rows_admitted": int(len(adm)), "beam": args.beam,
               "gate": gate, "J": J, "J_size": len(J), "group": {"pairs": pairs, **grp}, "bounds": bd,
               "x_group": xg, "s_holds": bool(s_ok), "d_holds": bool(bd["d_term_lower"] >= 1), "x_verdict": x_verdict,
               "reverts": {ch: bool((not s_ok) or (xg is not None and xg["reverts"][ch])) for ch in EF.CHAINS},
               "holds": bool(s_ok and bd["holds"]), "seconds": round(time.time() - t, 1)}
        (out_dir / f"{fam}-{side}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k_: rec[k_] for k_ in ("family", "side", "J_size", "s_holds", "d_holds", "x_verdict", "reverts",
                                                 "holds", "seconds")}
                         | {"common": grp["common_zeros_row"], "cost": grp["cost"], "x_term_lower": bd["x_term_lower"],
                            "x_group_cost": xg and xg["cost"], "x_group_window": xg and xg["window_atoms"],
                            "x_group_live": xg and xg["credited_chain_exact"], "gate": gate and gate["ok"]}), flush=True)
        items.append(rec)
    summary = {"tool": "pouw_sm120_eps8_group", "gate_ok": ok, "common_min": COMMON_MIN,
               "fragments": [{k_: r[k_] for k_ in ("family", "side", "J_size", "s_holds", "d_holds", "x_verdict",
                                                   "reverts", "holds")}
                             | {"common": r["group"]["common_zeros_row"], "cost": r["group"]["cost"],
                                "d_term_lower": r["bounds"]["d_term_lower"], "x_term_lower": r["bounds"]["x_term_lower"],
                                "x_group_cost": r["x_group"] and r["x_group"]["cost"]}
                             for r in items],
               "holds": all(r["holds"] for r in items)}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    run_result.write(path, ok, "sm120-eps8-group",
                     f"{len(items)} family-sides; prover-grouped tiles {'keep' if summary['holds'] else 'break'} the joint "
                     f"statement (largest common zeros found {max(r['group']['common_zeros_row'] for r in items)}, "
                     f"profit needs {COMMON_MIN}); census-tile gate {'ok' if ok else 'FAILED'}",
                     [("family_sides", len(items), "count"),
                      ("max_common_zeros", max(r["group"]["common_zeros_row"] for r in items), "count"),
                      ("min_group_cost", min(r["group"]["cost"] for r in items), "per MAC"),
                      ("min_d_term_lower", min(r["bounds"]["d_term_lower"] for r in items), "per MAC"),
                      ("min_x_term_lower", min(r["bounds"]["x_term_lower"] for r in items), "per MAC")])
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
