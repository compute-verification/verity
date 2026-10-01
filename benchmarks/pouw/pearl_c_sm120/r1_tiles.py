#!/usr/bin/env python3
"""The exact-window tile rule's thresholds: what a verifier tile of R1-H's chain looks like on model rows and on rows a
prover writes, under the centred start (`r1_binade.centred_start`).

Each word's chain runs atom by atom on the reference evaluator (`group_sum_total_batch`) over the first --atoms atoms, and
each step records whether it is exact (the output equals acc + S, S the atom's exact sum), S-determined (acc and every
product are multiples of g, so the step is T24(acc + S); `r1_sdet`) and cheaply spotted (`r1_sdet.cheap`).  Per tile of
T x T words, for T in --tiles, it reports:

  win<L>: the share of the tile's words exact over some L consecutive atoms, for L in WINDOWS;
  exact, sdet: the tile's shares of exact and S-determined steps;
  m16n8_sdet, m16n8_cheap: the share of the tile's (m16n8 sub-tile, atom) pairs whose every word is S-determined, or
    cheaply spotted (the tensor core steps a whole m16n8 tile, so only those let a derived word skip its step);
  binade: the share of the tile's words that keep H's top bit and sign at every step (the domain check), and ratio, the
    largest |C - H| / |H| over the chain.

On model rows a threshold's false-positive rate is the share of tiles above it; on written rows its catch rate is the share
above it.  Every tile's values are written with its first row and column, so the tiles holding row 0 of a capture (the
attention sink's position) can be read apart.

  python r1_tiles.py --acts DIR --weights SNAPSHOT --classes qkv-rot,o-vo-il --layers 0,8 --out DIR [--adv pow2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import r1_census as rc
from r1_binade import centred_start
from r1_sdet import t24, top_bit, v2
from verity.ml.kernels import group_sum_total_batch

WINDOWS = (4, 6, 8, 9, 12, 16)


def tiled(x: np.ndarray, tr: int, tc: int | None = None) -> np.ndarray:
    """x (rows, cols, ...) -> (rows // tr, cols // tc, tr * tc * ...), each tr x tc tile's entries flattened."""
    tc = tr if tc is None else tc
    r, c = x.shape[0] // tr * tr, x.shape[1] // tc * tc
    y = x[:r, :c].reshape(r // tr, tr, c // tc, tc, *x.shape[2:]).swapaxes(1, 2)
    return y.reshape(r // tr, c // tc, -1)


def job(cls: str, layer: int, args, key: str) -> dict:
    a, b = rc.operands(cls, layer, args, key)
    ac, bc = rc.quantize(a[:args.rows]), rc.quantize(b)
    av, bv = rc.values(ac), rc.values(bc)
    start = centred_start(av, bv, args.h, rc.rng(key, "start", layer, cls))
    rows, cols = len(ac), len(bc)
    ii, jj = (x.ravel() for x in np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij"))
    ai, bi = (av * 512).astype(np.int64), (bv * 512).astype(np.int64)
    acc, prev = start.copy(), rc.as_units(start)
    hu, e0 = prev.copy(), top_bit(prev)
    inside, ratio = np.ones(len(hu), bool), np.zeros(len(hu))
    run, best = np.zeros(len(hu), np.int64), np.zeros(len(hu), np.int64)
    exact_n, sdet_n = np.zeros(len(hu), np.int64), np.zeros(len(hu), np.int64)
    m_sdet, m_cheap = [], []
    checked = 0
    for t in range(args.atoms):
        s = slice(32 * t, 32 * t + 32)
        pa, pb = ai[:, s], bi[:, s]
        prod = pa[:, None, :] * pb[None, :, :]
        part = prod.sum(axis=2).ravel()
        ptop = top_bit(prod).max(axis=2).ravel()
        g = np.maximum(top_bit(prev), ptop) - 25
        prev_ok = (prev == 0) | (v2(prev) >= g)
        sdet = (v2(prod).min(axis=2).ravel() >= g) & prev_ok | (g <= 0)
        cheap = ((v2(pa).min(axis=1)[ii] + v2(pb).min(axis=1)[jj]) >= g) & prev_ok | (g <= 0)
        acc = group_sum_total_batch(rc.ATOM, acc, ac[ii, s], bc[jj, s])[0]
        now = rc.as_units(acc)
        assert np.array_equal(now[sdet], t24(prev + part)[sdet]), "an S-determined step that isn't T24(acc + S)"
        checked += int(sdet.sum())
        exact = now == prev + part
        run = (run + 1) * exact
        best = np.maximum(best, run)
        exact_n += exact
        sdet_n += sdet
        inside &= (top_bit(now) == e0) & (np.sign(now) == np.sign(hu)) & (ptop < e0)
        ratio = np.maximum(ratio, np.abs(now - hu) / np.abs(hu))
        m_sdet.append(sdet.reshape(rows // 16, 16, cols // 8, 8).all(axis=(1, 3)))
        m_cheap.append(cheap.reshape(rows // 16, 16, cols // 8, 8).all(axis=(1, 3)))
        prev = now
    grid = lambda v: v.reshape(rows, cols)
    out = {"class": cls, "layer": layer, "written": args.adv, "rows": rows, "cols": cols, "atoms": args.atoms,
           "h": args.h, "start": "centred", "checked_sdet_steps": checked, "tiles": {}}
    msd = np.stack(m_sdet, axis=-1)
    mch = np.stack(m_cheap, axis=-1)
    for T in args.tiles:
        if rows < T or cols < T or T % 16:
            continue
        per = {f"win{L}": tiled(grid(best >= L), T).mean(axis=2) for L in WINDOWS}
        per["exact"] = tiled(grid(exact_n), T).mean(axis=2) / args.atoms
        per["sdet"] = tiled(grid(sdet_n), T).mean(axis=2) / args.atoms
        per["m16n8_sdet"] = tiled(msd, T // 16, T // 8).mean(axis=2)
        per["m16n8_cheap"] = tiled(mch, T // 16, T // 8).mean(axis=2)
        per["binade"] = tiled(grid(inside), T).mean(axis=2)
        per["ratio"] = tiled(grid(ratio), T).max(axis=2)
        out["tiles"][str(T)] = {
            "shape": list(per["exact"].shape),
            "summary": {k: {"min": float(v.min()), "p50": float(np.median(v)), "p99": float(np.quantile(v, 0.99)),
                            "max": float(v.max()), "row0_tiles_max": float(v[0].max()),
                            "nonzero_tiles": float((v > 0).mean())} for k, v in per.items()},
            "per_tile": {k: np.round(v, 6).tolist() for k, v in per.items()},
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--classes", default="qkv-rot,gateup-rot,o-vo-il,down-bh32")
    p.add_argument("--layers", default="8")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--cols", type=int, default=512)
    p.add_argument("--first-row", type=int, default=0)
    p.add_argument("--atoms", type=int, default=16)
    p.add_argument("--h", type=int, default=10)
    p.add_argument("--tiles", type=lambda s: [int(x) for x in s.split(",")], default=[16, 64, 128])
    p.add_argument("--adv", choices=sorted(rc.ADV_ZEROS))
    p.add_argument("--key", default="r1-tiles-v0")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for layer in map(int, args.layers.split(",")):
        for cls in args.classes.split(","):
            r = job(cls, layer, args, args.key)
            name = f"L{layer:03d}-{cls}{'-' + args.adv if args.adv else ''}.json"
            (args.out / name).write_text(json.dumps(r) + "\n")
            for T, v in r["tiles"].items():
                s = v["summary"]
                print(f"L{layer:02d} {cls:10s} {args.adv or 'model':8s} T{T:>3s} max: " + " ".join(
                    f"{k} {s[k]['max']:.4f}" for k in s if k != "ratio") + f" ratio {s['ratio']['max']:.3f} | min: "
                    + " ".join(f"{k} {s[k]['min']:.4f}" for k in ("win9", "exact", "sdet", "m16n8_cheap") if k in s),
                    flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
