#!/usr/bin/env python3
"""How much of R1-H's per-atom-slice credit repeated and near-duplicate slices take, on a capture: per (layer, class), over
every A row and every atom, after the class's transform and the per-row quantizer (r1_census.py's `operands`).

A slice is a row's 32 codes at one atom, up to sign (the chain is odd).  Per class:
  repeat_share     the share of nonzero slices that repeat an earlier row's slice at the same atom exactly: what per-slice
                   credit removes
  near_share       the share of nonzero slices within d codes (default 3, the d* = 29/32 rule) of another row's slice at the
                   same atom, exact repeats included: an upper bound on what the near-duplicate rule removes
  near_only_share  near_share less the slices that repeat some other row exactly (first copies included)
  *_by_d           the same for every distance 0..d, so one run with --d 7 prices any d* from 25/32 to 32/32
Near-duplicates are found as the verifier would: two slices at most d codes apart agree on at least one of d + 1 disjoint
sub-keys, so only slices sharing a sub-key are compared.

  python r1_slices.py --acts DIR --weights SNAPSHOT --out FILE.json [--layers 0,8,16,28] [--classes ...] [--d 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from r1_census import operands, quantize

CLASSES = ("qkv-rot", "gateup-rot", "o-none", "o-vo-il", "down-none", "down-bh32")


def unsigned(s: np.ndarray) -> np.ndarray:
    """Slices (n, 32) with -0 as 0 and each slice's sign fixed by its first nonzero code."""
    s = np.where((s & 0x7F) == 0, 0, s).astype(np.uint8)
    first = np.take_along_axis(s, (s != 0).argmax(axis=1)[:, None], axis=1)
    return np.where((first >> 7) == 1, s ^ np.where(s != 0, np.uint8(0x80), np.uint8(0)), s)


def near_dist(u: np.ndarray, d: int, chunk: int = 512) -> np.ndarray:
    """For distinct slices u (n, 32): the distance in codes to the nearest other one, d + 1 where none is within d.  Two
    slices at most d codes apart agree on at least one of d + 1 disjoint sub-keys, so only slices sharing one are compared."""
    best = np.full(len(u), d + 1, np.int64)
    for cols in np.array_split(np.arange(32), d + 1):
        keys = np.ascontiguousarray(u[:, cols]).view(np.dtype((np.void, len(cols)))).ravel()
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        cuts = np.flatnonzero(ks[1:] != ks[:-1]) + 1
        for g in np.split(order, cuts):
            if len(g) < 2 or (best[g] == 1).all():
                continue
            s = u[g]
            for c0 in range(0, len(g), chunk):
                dist = (s[c0:c0 + chunk, None, :] != s[None, :, :]).sum(axis=2)
                dist[np.arange(min(chunk, len(g) - c0)), np.arange(c0, min(c0 + chunk, len(g)))] = 99
                best[g[c0:c0 + chunk]] = np.minimum(best[g[c0:c0 + chunk]], dist.min(axis=1))
    return best


def job(layer: int, cls: str, args) -> dict:
    t0 = time.monotonic()
    a, _ = operands(cls, layer, args, args.key)
    codes = quantize(a)
    rows, k = codes.shape
    total = rep = 0
    near = np.zeros(args.d + 1, np.int64)
    near_only = np.zeros(args.d + 1, np.int64)
    for t in range(k // 32):
        s = unsigned(codes[:, 32 * t:32 * t + 32])
        live = s.any(axis=1)
        _, first, inv, cnt = np.unique(s.view(np.dtype((np.void, 32))).ravel(), return_index=True, return_inverse=True,
                                       return_counts=True)
        inv = inv.ravel()
        is_first = np.zeros(rows, bool)
        is_first[first] = True
        dist = near_dist(s[first], args.d)[inv]
        total += int(live.sum())
        rep += int((live & ~is_first).sum())
        for e in range(args.d + 1):
            near[e] += int((live & ((dist <= e) | (cnt[inv] > 1))).sum())
            near_only[e] += int((live & (dist <= e) & (cnt[inv] == 1)).sum())
    return {"layer": layer, "class": cls, "rows": rows, "atoms": k // 32, "slices": total,
            "repeat_share": rep / total, "near_share": int(near[args.d]) / total,
            "near_only_share": int(near_only[args.d]) / total,
            "near_share_by_d": [int(x) / total for x in near], "near_only_share_by_d": [int(x) / total for x in near_only],
            "d": args.d, "seconds": round(time.monotonic() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--acts", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", default="0,8,16,28")
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--d", type=int, default=3, choices=range(8))
    ap.add_argument("--key", default="r1-census/v0")
    ap.add_argument("--first-row", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.synthetic, args.adv, args.cols = False, None, 1
    work = [(int(layer), cls) for cls in args.classes.split(",") for layer in args.layers.split(",")]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(work))) as pool:
        res = list(pool.map(job, *zip(*work), [args] * len(work)))
    for r in res:
        print(f'L{r["layer"]:02d} {r["class"]:10s} repeat {r["repeat_share"]:.4f} near {r["near_share"]:.4f} '
              f'near-only {r["near_only_share"]:.4f} by d {" ".join(f"{x:.4f}" for x in r["near_share_by_d"])} '
              f'({r["seconds"]}s)', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
