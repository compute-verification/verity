#!/usr/bin/env python3
"""Where R1-H's hot-started step is a function of the atom's exact sum: the reach of the relation attack on written rows.

An sm_120 step is T24(acc + sum_p T_g(p)) (`GroupSum`, width 26): every term is truncated toward zero to g = 2^(E - 25), E
the top bit of the largest term, and the sum is truncated to 24 significant bits.  When acc and every product are multiples
of g the step is T24(acc + S), a function of acc and the atom's exact sum S alone (`sdet`).  A prover holding S for a row
it derived from others (row3 = (row1 + row2) / 2, written in the rotated basis) then derives that word's step with a few
scalar operations instead of a tensor-core step.  `cheap` is the part of `sdet` a prover can spot without forming the
products: the slice's least 2-adic valuation over A's row plus that over B's row reaches log2 g.  `cheap_tile` is the
share of mma.sync m16n8 tiles (16 consecutive rows by 8 consecutive columns) whose every step is `cheap`: the tensor core
steps a whole tile, so a word derived on the scalar pipes saves its tile's step only when the whole tile is.  `exact` is
r1_census's flag (the output equals acc + S).  Every `sdet` step is checked against T24(acc + S).

  python r1_sdet.py --synthetic --out DIR [--adv flat]
  python r1_sdet.py --acts DIR --weights SNAPSHOT --classes qkv-rot,o-vo-il,down-bh32 --layers 8 --out DIR [--adv flat]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import r1_census as rc
from verity.ml.kernels import group_sum_total_batch


def v2(x: np.ndarray) -> np.ndarray:
    """The 2-adic valuation of nonzero int64 entries; 1 << 40 where x == 0."""
    m = np.abs(x)
    low = m & -m
    return np.where(m == 0, 1 << 40, np.log2(np.where(low == 0, 1, low)).astype(np.int64))


def top_bit(x: np.ndarray) -> np.ndarray:
    m = np.abs(x)
    return np.where(m == 0, -(1 << 40), np.floor(np.log2(np.where(m == 0, 1, m))).astype(np.int64))


def t24(x: np.ndarray) -> np.ndarray:
    m = np.abs(x)
    shift = np.maximum(top_bit(x) + 1 - 24, 0)
    return np.sign(x) * ((m >> shift) << shift)


def job(cls: str, layer: int, args, key: str) -> dict:
    a, b = rc.operands(cls, layer, args, key)
    ac, bc = rc.quantize(a), rc.quantize(b)
    av, bv = rc.values(ac), rc.values(bc)
    start = rc.separable_start(av, bv, args.h, rc.rng(key, "start", layer, cls))
    rows, cols = len(ac), len(bc)
    ii, jj = (x.ravel() for x in np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij"))
    ai, bi = (av * 512).astype(np.int64), (bv * 512).astype(np.int64)
    acc, prev = start.copy(), rc.as_units(start)
    shares = {"sdet": [], "cheap": [], "exact": [], "cheap_tile": []}
    tr, tc = rows // 16 * 16, cols // 8 * 8
    checked = 0
    for t in range(args.atoms):
        s = slice(32 * t, 32 * t + 32)
        pa, pb = ai[:, s], bi[:, s]
        prod = pa[:, None, :] * pb[None, :, :]
        part = prod.sum(axis=2).ravel()
        e = np.maximum(top_bit(prev), top_bit(prod).max(axis=2).ravel())
        g = e - 25
        terms_ok = (v2(prod).min(axis=2).ravel() >= g) & ((prev == 0) | (v2(prev) >= g))
        sdet = terms_ok | (g <= 0)
        lo_a = v2(pa).min(axis=1)
        lo_b = v2(pb).min(axis=1)
        cheap = ((lo_a[ii] + lo_b[jj]) >= g) & ((prev == 0) | (v2(prev) >= g)) | (g <= 0)
        acc = group_sum_total_batch(rc.ATOM, acc, ac[ii, s], bc[jj, s])[0]
        now = rc.as_units(acc)
        assert np.array_equal(now[sdet], t24(prev + part)[sdet]), "an S-determined step that isn't T24(acc + S)"
        assert not (cheap & ~sdet).any()
        checked += int(sdet.sum())
        shares["sdet"].append(float(sdet.mean()))
        shares["cheap"].append(float(cheap.mean()))
        shares["exact"].append(float((now == prev + part).mean()))
        tiles = cheap.reshape(rows, cols)[:tr, :tc].reshape(tr // 16, 16, tc // 8, 8).all(axis=(1, 3))
        shares["cheap_tile"].append(float(tiles.mean()))
        prev = now
    return {"class": cls, "layer": layer, "written": args.adv, "rows": rows, "cols": cols, "atoms": args.atoms,
            "h": args.h, "checked_sdet_steps": checked,
            **{k: {"mean": float(np.mean(v)), "per_atom": v} for k, v in shares.items()}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--classes", default="qkv-rot,gateup-rot,o-vo-il,down-bh32")
    p.add_argument("--layers", default="8")
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--cols", type=int, default=256)
    p.add_argument("--first-row", type=int, default=0)
    p.add_argument("--atoms", type=int, default=16)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--adv", choices=sorted(rc.ADV_ZEROS))
    p.add_argument("--key", default="r1-sdet-v0")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for layer in map(int, args.layers.split(",")):
        for cls in args.classes.split(","):
            r = job(cls, layer, args, args.key)
            name = f"L{layer:03d}-{cls}{'-' + args.adv if args.adv else ''}.json"
            (args.out / name).write_text(json.dumps(r, indent=1) + "\n")
            print(f"L{layer:02d} {cls:10s} {args.adv or 'model':8s} sdet {r['sdet']['mean']:.3f} cheap {r['cheap']['mean']:.3f} "
                  f"exact {r['exact']['mean']:.3f} cheap-tile {r['cheap_tile']['mean']:.3f} (checked {r['checked_sdet_steps']})",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
