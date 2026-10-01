#!/usr/bin/env python3
"""`preadd-floor/sm120`'s SWAR question (the assessor, 11:39 PM PDT): how much of v1's formed A′ can a packed integer add
pre-add?  CPU only.

One IADD3 on two biased-FP16 fields writes two exact FP16 sums (4.14 W1 per element, against the floor's 8): with
B = 1.5·2^e, enc(B + x) + enc(B + y) − enc(B) = enc(B + x + y) whenever x, y and x + y are multiples of q = 2^(e−10) and
below 2^9·q in magnitude.  B is common to a leaf row segment (one rank-1 correction per row and leaf block), so the
window q must be common to the segment's values in every term of the form.  For each census family at v1's forming
(pairing.py's `unit:<family>` sides: `form_s5` at delta 1, line norm 16, run 1's seeds; the domain-admitted A rows only),
each leaf inner dimension L (aligned segments of L k positions, inside one 128-deep promotion group for L ≤ 128) and
window width W, the share of segments that fit:

- `1`: one segment alone, the upper bound for any form;
- `2`: a two-term form, row i's segment j with row i + R/2's segment j ^ 1 (j for L = 128), as a 2 x 2 split pairs them;
- `3`: a three-term form, adding row i + R/4's segment j (IADD3's three operands);
- `4_bound`, `8_bound`: four and eight terms (rows i + m·R/8, segments j ^ m), only the necessary condition that every term
  fits one common q, so upper bounds; a leaf form of a depth-3 composition with 2-term forms at each level has 8 terms.

A segment set fits at W when, with q the smallest power of two dividing all its values (all-zero sets fit), every value
and every sum of the form (coefficients ±1, the sign the prover picks) is below 2^W·q in magnitude.  W = 9 is the
biased-FP16 field; 8 and 10 bracket it, and 7 is a signed int8 field (an int8 leaf).

    python benchmarks/pouw/sm120_exact/swar_window.py [--families F,...] [--out DIR]
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
import pairing as PR  # noqa: E402
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from run import ATOM, UNIT_FAMILIES  # noqa: E402

LS = (16, 32, 64, 128)
WS = (7, 8, 9, 10)


def units(codes: np.ndarray) -> np.ndarray:
    """E4M3 codes -> exact integers in units of 2^-9 (every E4M3 value is a multiple of it)."""
    v = census.val(codes)
    assert np.isfinite(v).all(), "a NaN code in the formed operand"
    n = np.rint(v * 512).astype(np.int64)
    assert (n == v * 512).all()
    return n


def lowbit(a: np.ndarray) -> np.ndarray:
    """Smallest power of two dividing each |a| (a huge sentinel for 0)."""
    a = np.abs(a)
    return np.where(a == 0, 1 << 40, a & -a)


def fits(terms: list[np.ndarray], W: int) -> np.ndarray:
    """terms: (S, L) integer segments, one per term -> (S,) whether a common q fits every value and the form's sums."""
    q = np.minimum.reduce([lowbit(t).min(1) for t in terms])
    lim = q.astype(np.float64) * (1 << W)
    ok = np.ones(len(q), bool)
    for t in terms:
        ok &= np.abs(t).max(1) < lim
    if len(terms) > 1:
        sums = np.zeros(len(q), bool)
        for signs in np.ndindex(*(2,) * (len(terms) - 1)):
            tot = terms[0] + sum((1 - 2 * s) * t for s, t in zip(signs, terms[1:]))
            sums |= np.abs(tot).max(1) < lim
        ok &= sums
    return ok | (q == 1 << 40)


def common(terms: list[np.ndarray], W: int) -> np.ndarray:
    """A necessary condition only: every term fits one common q at W, whatever the sums (the 4- and 8-term bounds)."""
    q = np.minimum.reduce([lowbit(t).min(1) for t in terms])
    lim = q.astype(np.float64) * (1 << W)
    return np.logical_and.reduce([np.abs(t).max(1) < lim for t in terms]) | (q == 1 << 40)


def segments(n: np.ndarray, L: int) -> np.ndarray:
    R, k = n.shape
    return n.reshape(R, k // L, L)


def measure(n: np.ndarray) -> dict:
    R, k = n.shape
    out = {}
    for L in LS:
        S = segments(n, L)
        j = np.arange(k // L)
        jx = j ^ 1 if L < 128 else j
        rows = np.arange(R)
        g = 128 // L
        t = [S[(rows + m * R // 8) % R][:, j ^ (m % g) if L < 128 else j].reshape(-1, L) for m in (0, 4, 2, 6, 1, 5, 3, 7)]
        t[1] = S[(rows + R // 2) % R][:, jx].reshape(-1, L)
        t[2] = S[(rows + R // 4) % R].reshape(-1, L)
        zero = float((np.abs(t[0]).max(1) == 0).mean())
        rec = {"segments": int(len(t[0])), "all_zero_share": round(zero, 6)}
        for W in WS:
            rec[f"W{W}"] = {"1": round(float(fits(t[:1], W).mean()), 6),
                            "2": round(float(fits(t[:2], W).mean()), 6),
                            "3": round(float(fits(t[:3], W).mean()), 6),
                            "4_bound": round(float(common(t[:4], W).mean()), 6),
                            "8_bound": round(float(common(t[:8], W).mean()), 6)}
        t1 = t[0]
        span = np.log2(np.abs(t1).max(1).clip(1)) + 1 - np.log2(np.minimum(lowbit(t1).min(1), 1 << 20).astype(np.float64))
        rec["span_bits_median"] = float(np.median(span))
        out[str(L)] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(UNIT_FAMILIES))
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    census.set_atom(ATOM)
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fams = {}
    for fam in args.families.split(","):
        t = time.time()
        sides, k, info = PR.item_sides(f"unit:{fam}", args.k, args.rows, None, args.seed)
        codes, ok = sides["A"]
        n = units(codes[ok]) if ok.any() else units(codes[:0])
        rec = {**info, "rows_used": int(len(n))}
        if len(n) >= 4:
            rec["L"] = measure(n)
        rec["seconds"] = round(time.time() - t, 1)
        fams[fam] = rec
        print(json.dumps({"family": fam, "rows_used": rec["rows_used"], "seconds": rec["seconds"],
                          **({"L16_W9": rec["L"]["16"]["W9"], "L32_W9": rec["L"]["32"]["W9"]} if "L" in rec else {})}), flush=True)
    worst = {}
    for L in LS:
        for W in WS:
            for m in ("1", "2", "3"):
                vals = [(r["L"][str(L)][f"W{W}"][m], f) for f, r in fams.items() if "L" in r]
                v, f = max(vals)
                worst[f"L{L}_W{W}_{m}"] = {"max": v, "family": f, "mean": round(float(np.mean([x for x, _ in vals])), 6)}
    summary = {"tool": "pouw_sm120_swar_window", "families": fams, "worst": worst}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    w = worst["L16_W9_1"]
    run_result.write(path, True, "sm120-swar-window",
                     f"{len(fams)} census families at v1's forming; the largest share of 16-wide leaf row segments fitting a "
                     f"common 9-bit biased-FP16 window is {w['max']:.4f} ({w['family']}), mean {w['mean']:.4f}",
                     [(f"max_share_{key}", v["max"], "share") for key, v in worst.items()])


if __name__ == "__main__":
    main()
