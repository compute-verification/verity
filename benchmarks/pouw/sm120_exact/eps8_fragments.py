#!/usr/bin/env python3
"""ε₈'s one-sided check (server.md 14:07Z; the assessor's `red-team/ratings.md` 14:08Z line): on the census families where one
side alone reaches a Strassen fragment under the greedy adversarial pairing (Results 13: A on rank1, duplicated,
coherent-gaussian and outliers-first, B on rank1), is every fragment's row pair either uncredited as a near-duplicate or
chain-inexact on its window?  CPU only.

Per family, the A-only and B-only searches are replayed exactly as `pairing.py` ran them on node 2 (`fa74b9e3`, --k-search
8192 --rows 512, default seeds; the same rng stream, and each search's steps must equal the stored JSON's, --expect DIR),
now returning the fragment's row pairs and its matched k pairs.  The fragment's window: the atoms (and 128-groups) holding
its first 32 matched k pairs' 64 indices.  Per row pair (u, v) of a fragment (for B: two weight rows):
- near-duplicate, on codes up to sign (the approved-weights registration rule (a), `approved-weights-rows.md`: codes
  compared regardless of the row scale; ±0 equal): agreement over the whole row (the assessor's row-level rule: ≥ 15/16 of
  k), over each window atom (v2's d* = 29 of 32: a block agreeing on more is uncredited) and each window group (v1's d* =
  112 of 128); the pair counts as uncredited on its window when the whole row is past 15/16 or every window block is past
  d*;
- the codes' agreement on the fragment's 64 k indices and on its window atoms: the share of products a delta route
  C' = C + (v - u)·B could skip there;
- the chain, per word of the pair's two rows against every sampled row of the other side (512), from +0 on the sm_120
  unpromoted atom (`pearlc_census.step`, the census's chain): v2 (one chain over k) and v1 (G = 4: restarting each 128-deep
  group from +0).  `fixed`: the pair's two words are exact on every window atom; `best`: at least 32 of the fragment's
  matched k pairs lie in atoms exact for both words (the prover's best 32).  A pair is chain-inexact on its window when no
  column is `best`-exact (and so none `fixed`).
The fragment verdict per chain: every row pair uncredited or chain-inexact, or not (`neither` lists the pairs that are
neither).  +0 is the census's start; the v2-hot chain starts at H_i, which only lowered the per-atom exact share on node 2's
self-tests (hot.py: near-cap 0.903 against 0.959 from +0, rank1 0.963 against 0.978).

    python benchmarks/pouw/sm120_exact/eps8_fragments.py --expect DIR [--out DIR]
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
import pairing as PR  # noqa: E402
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from run import ATOM  # noqa: E402

FAMILIES = ("rank1", "duplicated", "coherent-gaussian", "outliers-first")
ROW_RULE = 15 / 16
D_ATOM, D_GROUP = 29, 112
CHAINS = {"v2": 0, "v1": 4}


def agree(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per index: u equals v, and u equals -v (±0 equal either way)."""
    zero = ((u & 0x7F) == 0) & ((v & 0x7F) == 0)
    return (u == v) | zero, (u == (v ^ 0x80)) | zero


def block_agree(same: np.ndarray, opp: np.ndarray, width: int) -> np.ndarray:
    return np.maximum(same.reshape(-1, width).sum(1), opp.reshape(-1, width).sum(1))


def flags(a: np.ndarray, b: np.ndarray, G: int) -> np.ndarray:
    """(R, S, T) per-atom exactness of the words a[r]·b[s] from +0 (restarting every G atoms if G)."""
    R, k = a.shape
    S, T = b.shape[0], k // 32
    out = np.zeros((T, R * S), bool)
    C = np.zeros(R * S)
    for t in range(T):
        if G and t % G == 0:
            C = np.zeros(R * S)
        C, out[t] = census.step(C, np.repeat(a[:, 32 * t:32 * t + 32], S, 0), np.tile(b[:, 32 * t:32 * t + 32], (R, 1)))
    return out.reshape(T, R, S).transpose(1, 2, 0)


def analyse(side: str, X: np.ndarray, Y: np.ndarray, pairs: list, kpairs: list) -> dict:
    """X: the fragment side's codes, Y: the other side's; pairs: the fragment's row pairs of X; kpairs: its matched k pairs."""
    k = X.shape[1]
    K = np.array(kpairs[:PR.KPAIRS]).reshape(-1)
    W = np.unique(K // 32)
    Wg = np.unique(K // 128)
    rows = sorted({r for p in pairs for r in p})
    loc = {r: q for q, r in enumerate(rows)}
    M = np.array(kpairs) // 32
    out = {"side": side, "row_pairs": len(pairs), "k_pairs_matched": len(kpairs), "window_atoms": W.tolist(),
           "window_groups": Wg.tolist(), "pairs": []}
    fl = {c: flags(X[rows], Y, G) for c, G in CHAINS.items()}
    for u_, v_ in pairs:
        u, v = X[u_], X[v_]
        same, opp = agree(u, v)
        row = max(int(same.sum()), int(opp.sum())) / k
        at, gr = block_agree(same, opp, 32), block_agree(same, opp, 128)
        win = max(int(same[K].sum()), int(opp[K].sum())) / len(K)
        idx = np.concatenate([np.arange(32 * t, 32 * t + 32) for t in W])
        winat = max(int(same[idx].sum()), int(opp[idx].sum())) / len(idx)
        rec = {"rows": [u_, v_], "row_agreement": round(row, 6), "row_rule": row >= ROW_RULE,
               "window_atom_agreement_min": int(at[W].min()), "window_atom_agreement_max": int(at[W].max()),
               "window_atoms_past_d": int((at[W] > D_ATOM).sum()),
               "window_group_agreement_min": int(gr[Wg].min()), "window_groups_past_d": int((gr[Wg] > D_GROUP).sum()),
               "agreement_on_fragment_k": round(win, 6), "agreement_on_window_atoms": round(winat, 6)}
        rec["uncredited"] = {"v2": bool(rec["row_rule"] or rec["window_atoms_past_d"] == len(W)),
                             "v1": bool(rec["row_rule"] or rec["window_groups_past_d"] == len(Wg))}
        for c in CHAINS:
            f = fl[c][loc[u_]] & fl[c][loc[v_]]                       # (S, T): both words exact at the atom
            fixed = f[:, W].all(1)
            best = (f[:, M[:, 0]] & f[:, M[:, 1]]).sum(1) >= PR.KPAIRS
            rec[c] = {"atom_exact_share": round(float(f.mean()), 6), "fixed_columns": int(fixed.sum()),
                      "best_columns": int(best.sum()), "columns": int(f.shape[0])}
            rec[c]["chain_inexact"] = rec[c]["best_columns"] == 0
        out["pairs"].append(rec)
    for c in CHAINS:
        f = np.ones((Y.shape[0], k // 32), bool)
        for r in rows:
            f &= fl[c][loc[r]]
        neither = [q for q, r in enumerate(out["pairs"]) if not r["uncredited"][c] and not r[c]["chain_inexact"]]
        out[c] = {"uncredited_pairs": sum(r["uncredited"][c] for r in out["pairs"]),
                  "chain_inexact_pairs": sum(r[c]["chain_inexact"] for r in out["pairs"]),
                  "neither": neither, "verdict": "opens nothing" if not neither else "neither on some pairs",
                  "fragment_fixed_columns": int(f[:, W].all(1).sum()),
                  "fragment_best_columns": int(((f[:, M[:, 0]] & f[:, M[:, 1]]).sum(1) >= PR.KPAIRS).sum())}
    return out


def replay(fam: str, expect: dict, args) -> tuple[dict, dict, dict]:
    item = f"unit:{fam}"
    sides, k, info = PR.item_sides(item, args.k, args.rows, None, args.seed)
    rng = census._rng("pairing-search", item, args.seed)
    ks = np.sort(rng.choice(k, min(k, args.k_search), replace=False))
    sim, first = PR.first_steps(sides, ks, args.cands, rng)
    res = {}
    for label, which in (("A_only", ("A",)), ("B_only", ("B",))):
        res[label] = PR.search(sides, ks, k, which, args.cands, args.seeds, rng, sim, first, "max")
    gate = {label: {"steps_equal": res[label]["steps"] == expect[label]["steps"],
                    "fragment": res[label]["fragment"], "expected_fragment": expect[label]["fragment"]}
            for label in res}
    return sides, res, {"item": item, **info, "gate": gate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", required=True, help="node 2's pairing_census JSONs (unit-<family>.json)")
    ap.add_argument("--families", default=",".join(FAMILIES))
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
        sides, res, rec = replay(fam, expect, args)
        rec["expect_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        ok &= all(g["steps_equal"] and g["fragment"] == g["expected_fragment"] for g in rec["gate"].values())
        rec["fragments"] = {}
        for label, s, o in (("A_only", "A", "B"), ("B_only", "B", "A")):
            if res[label]["fragment"]:
                rec["fragments"][s] = analyse(s, sides[s][0], sides[o][0], res[label]["fragment_pairs"][s],
                                              res[label]["fragment_k_pairs"])
                rec["fragments"][s]["k_pairs"] = res[label]["fragment_k_pairs"]
        rec["seconds"] = round(time.time() - t, 1)
        (out_dir / f"{fam}.json").write_text(json.dumps(rec, indent=1) + "\n")
        short = {s: {c: {x: f[c][x] for x in ("uncredited_pairs", "chain_inexact_pairs", "neither", "fragment_best_columns")}
                     for c in CHAINS} for s, f in rec["fragments"].items()}
        print(json.dumps({"family": fam, "gate": rec["gate"], "fragments": short, "seconds": rec["seconds"]}), flush=True)
        items.append(rec)
    frags = [(r["item"], s, f) for r in items for s, f in r["fragments"].items()]
    summary = {"tool": "pouw_sm120_eps8_fragments", "gate_ok": ok, "fragments": [
        {"item": i, "side": s, **{c: {x: f[c][x] for x in ("uncredited_pairs", "chain_inexact_pairs", "neither", "verdict",
                                                          "fragment_fixed_columns", "fragment_best_columns")}
                                  for c in CHAINS}} for i, s, f in frags]}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    bad = {c: sum(bool(f[c]["neither"]) for _, _, f in frags) for c in CHAINS}
    run_result.write(path, ok, "sm120-eps8-fragments",
                     f"{len(frags)} one-sided fragments replayed (gate {'ok' if ok else 'FAILED'}); fragments with a row pair "
                     f"neither uncredited nor chain-inexact: v2 {bad['v2']}, v1 {bad['v1']}",
                     [("fragments", len(frags), "count"), *((f"neither_{c}", bad[c], "count") for c in CHAINS)])
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
