#!/usr/bin/env python3
"""R1-H's domain bound: while the accumulator stays in H's binade, C - H is a sum of per-atom roundings, exactly.

Let E = top bit of H (units of 2^-18), g = 2^(E - 25).  While every step's accumulator keeps top bit E (and the products
stay below it), each step truncates its products to g, the accumulator is a multiple of 4g (its own ulp), and T24 rounds
the result to 4g, toward zero.  So the step adds R_t = floor_4g(sum_p T_g(p)) for H > 0 (ceil_4g for H < 0): a function of
the atom's operands and E alone, whatever the chain's order or split.  The verifier's replay computes every accumulator, so
"every step stays in H's binade" is a check it can make on each replayed tile for free.

Per word, over the whole chain (all k / 32 atoms), this reports:
- `in_binade`: every step's result kept H's top bit.  For those words C == H + sum_t R_t is asserted bit for bit against
  the reference evaluator (`group_sum_total_batch`), with R_t computed per atom on its own.
- `max_ratio`: max over the chain of |acc - H| / |H|, and the share of words whose chain ever exceeds 2^-c |H|.
- `error`: U = C - H against the exact product, rms over the words, beside the exact product's own BF16 rounding; and the
  same with the floor's mean bias (1.5 g per atom, toward zero) folded into H.
Three starts: `p`, r1_census's separable H = fl(P_i Q_j) with keyed mantissas in [1, 1.5) each (so H's in [1, 2.25));
`c`, the centred separable start P_i = s_i 2^e_i, Q_j = t_j 1.5 2^f_j: H = +-1.5 2^E exactly, |H| / 3 from either edge;
and `cs`, the centred start at |H| >= 6 |a_i| |b_j|, in its binade for every word by Cauchy-Schwarz.

  python r1_binade.py --synthetic --out DIR [--adv flat]
  python r1_binade.py --acts DIR --weights SNAPSHOT --classes qkv-rot,o-vo-il,down-bh32 --layers 8 --out DIR [--adv flat]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import r1_census as rc
from r1_sdet import top_bit
from verity.ml.kernels import group_sum_total_batch

RATIOS = (2, 3, 4, 5, 6)


def centred_start(av: np.ndarray, bv: np.ndarray, h: int, g: np.random.Generator) -> np.ndarray:
    """H_ij = P_i Q_j with P_i = s_i 2^e_i and Q_j = t_j 1.5 2^f_j, the exponents as in r1_census.separable_start."""
    def side(x: np.ndarray, shift: float, mant: float) -> np.ndarray:
        n = np.linalg.norm(x, axis=1) * shift
        e = np.round(np.log2(np.where(n > 0, n, 1.0)))
        sign = np.where(g.integers(0, 2, len(x)) == 1, -1.0, 1.0)
        return (sign * np.exp2(e) * mant).astype(np.float32)
    p, q = side(av, 1.0, 1.0), side(bv, np.sqrt(32) / av.shape[1] * 2.0 ** h, 1.5)
    hv = np.outer(p, q).astype(np.float32)
    assert np.array_equal(hv.astype(np.float64) * rc.UNIT, np.floor(hv.astype(np.float64) * rc.UNIT)), "H below 2^-18"
    return hv.ravel().view(np.uint32)


def cs_start(av: np.ndarray, bv: np.ndarray, h: int, g: np.random.Generator) -> np.ndarray:
    """H_ij = P_i Q_j with P_i = s_i 2^ceil(log2 |a_i|) and Q_j = t_j 1.5 2^(ceil(log2 |b_j|) + 2): |H| >= 6 |a_i| |b_j|, so
    by Cauchy-Schwarz |C - H| <= |a_i| |b_j| + (truncation) < |H| / 3 and every step stays in H's binade (h is unused)."""
    def side(x: np.ndarray, extra: int, mant: float) -> np.ndarray:
        n = np.linalg.norm(x, axis=1)
        e = np.ceil(np.log2(np.where(n > 0, n, 1.0))) + extra
        sign = np.where(g.integers(0, 2, len(x)) == 1, -1.0, 1.0)
        return (sign * np.exp2(e) * mant).astype(np.float32)
    hv = np.outer(side(av, 0, 1.0), side(bv, 2, 1.5)).astype(np.float32)
    assert np.array_equal(hv.astype(np.float64) * rc.UNIT, np.floor(hv.astype(np.float64) * rc.UNIT)), "H below 2^-18"
    return hv.ravel().view(np.uint32)


STARTS = {"p": rc.separable_start, "c": centred_start, "cs": cs_start}


def bf16_err(x: np.ndarray) -> np.ndarray:
    y = (rc.f32_to_bf16_rn_batch(x.astype(np.float32).view(np.uint32))[0].astype(np.uint32) << 16).view(np.float32)
    return y.astype(np.float64) - x


def trunc(x: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """x truncated toward zero to a multiple of 2^bits (bits <= 0: x itself)."""
    b = np.maximum(bits, 0)
    return np.sign(x) * ((np.abs(x) >> b) << b)


def job(cls: str, layer: int, args, key: str, start_kind: str) -> dict:
    a, b = rc.operands(cls, layer, args, key)
    ac, bc = rc.quantize(a[:args.rows]), rc.quantize(b)
    av, bv = rc.values(ac), rc.values(bc)
    start = STARTS[start_kind](av, bv, args.h, rc.rng(key, "start", layer, cls))
    rows, cols = len(ac), len(bc)
    ii, jj = (x.ravel() for x in np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij"))
    ai, bi = (av * 512).astype(np.int64), (bv * 512).astype(np.int64)
    hu = rc.as_units(start)
    e = top_bit(hu)
    gb, ub = e - 25, e - 23
    pos = hu > 0
    acc = start.copy()
    inside = np.ones(len(hu), bool)
    ratio = np.zeros(len(hu))
    lemma = hu.copy()
    exact = np.zeros(len(hu), np.int64)
    atoms = ac.shape[1] // 32
    for t in range(atoms):
        s = slice(32 * t, 32 * t + 32)
        prod = ai[:, s][:, None, :] * bi[:, s][None, :, :]
        exact += prod.sum(axis=2).ravel()
        x = trunc(prod, gb.reshape(rows, cols)[:, :, None]).sum(axis=2).ravel()
        q = np.int64(1) << np.maximum(ub, 0)
        lemma += np.where(pos, np.floor_divide(x, q), -np.floor_divide(-x, q)) * q
        acc = group_sum_total_batch(rc.ATOM, acc, ac[ii, s], bc[jj, s])[0]
        now = rc.as_units(acc)
        inside &= (top_bit(now) == e) & (np.sign(now) == np.sign(hu)) & (top_bit(prod).max(axis=2).ravel() < e)
        ratio = np.maximum(ratio, np.abs(now - hu) / np.abs(hu))
    final = rc.as_units(acc)
    assert np.array_equal(final[inside], lemma[inside]), "an in-binade word that isn't H + sum_t R_t"
    # The floor to 4g loses 1.5 g per atom on average, toward zero (the products' own truncations are sign-symmetric on
    # generic rows).  H' = H - sign(H) 1.5 g atoms is exact in FP32, and fl(C - H') is still one FADD.
    truth = exact / rc.UNIT
    u = (final - hu) / rc.UNIT
    u_fix = u + np.sign(hu) * 1.5 * np.exp2(gb.astype(np.float64)) * atoms / rc.UNIT
    rms = np.sqrt(np.mean(truth ** 2))
    bf = np.sqrt(np.mean(bf16_err(truth) ** 2))
    err = {"rel_rms": float(np.sqrt(np.mean((u - truth) ** 2)) / rms),
           "rel_rms_bias_fixed": float(np.sqrt(np.mean((u_fix - truth) ** 2)) / rms),
           "bf16_rel_rms": float(bf / rms),
           "vs_bf16": float(np.sqrt(np.mean((u - truth) ** 2)) / bf),
           "vs_bf16_bias_fixed": float(np.sqrt(np.mean((u_fix - truth) ** 2)) / bf),
           "h_eff_p50": float(np.median(np.log2(np.abs(hu) / rc.UNIT * np.sqrt(atoms) / rms)))}
    return {"class": cls, "layer": layer, "written": args.adv, "start": start_kind, "h": args.h, "rows": rows,
            "cols": cols, "atoms": atoms, "words": len(hu), "in_binade": float(inside.mean()), "error": err,
            "lemma_checked_words": int(inside.sum()),
            "lemma_outside": float((final[~inside] == lemma[~inside]).mean()) if (~inside).any() else None,
            "max_ratio": {"max": float(ratio.max()), "p50": float(np.median(ratio)), "p99": float(np.quantile(ratio, 0.99))},
            "over": {f"2^-{c}": float((ratio > 2.0 ** -c).mean()) for c in RATIOS}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--classes", default="qkv-rot,gateup-rot,o-vo-il,down-bh32")
    p.add_argument("--layers", default="8")
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--cols", type=int, default=128)
    p.add_argument("--first-row", type=int, default=0)
    p.add_argument("--starts", default="p,c,cs")
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--adv", choices=sorted(rc.ADV_ZEROS))
    p.add_argument("--key", default="r1-binade-v0")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for layer in map(int, args.layers.split(",")):
        for cls in args.classes.split(","):
            for sk in args.starts.split(","):
                r = job(cls, layer, args, args.key, sk)
                name = f"L{layer:03d}-{cls}-{sk}{'-' + args.adv if args.adv else ''}.json"
                (args.out / name).write_text(json.dumps(r, indent=1) + "\n")
                print(f"L{layer:02d} {cls:10s} {args.adv or 'model':8s} h{args.h} start {sk}: in-binade {r['in_binade']:.4f} "
                      f"(lemma checked on {r['lemma_checked_words']} words; outside, lemma holds on {r['lemma_outside']}) "
                      f"max|C-H|/|H| p50 {r['max_ratio']['p50']:.4f} p99 {r['max_ratio']['p99']:.4f} max {r['max_ratio']['max']:.4f} "
                      f"over 2^-2 {r['over']['2^-2']:.4f} 2^-4 {r['over']['2^-4']:.4f} | h_eff {r['error']['h_eff_p50']:.1f} "
                      f"err/bf16 {r['error']['vs_bf16']:.2f} bias-fixed {r['error']['vs_bf16_bias_fixed']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
