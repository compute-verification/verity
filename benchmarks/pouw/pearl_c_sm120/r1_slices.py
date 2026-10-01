#!/usr/bin/env python3
"""How much of R1-H's per-atom-slice credit repeated and near-duplicate slices take, on a capture: per (layer, class), over
every A row and every atom, after the class's transform and the per-row quantizer (r1_census.py's `operands`).

A slice is a row's 32 codes at one atom, up to sign (the chain is odd).  Per class:
  repeat_share     the share of nonzero slices that repeat an earlier row's slice at the same atom exactly: what per-slice
                   credit removes
  near_share       the share of nonzero slices within d codes (default 3, the d* = 29/32 rule) of another row's slice at the
                   same atom, exact repeats included: an upper bound on what the near-duplicate rule removes
  near_only_share  near_share less the slices that repeat some other row exactly (first copies included)
Near-duplicates are found as the verifier would: two slices at most 3 codes apart agree on at least one of four 8-code
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


def near_flags(u: np.ndarray, d: int, chunk: int = 512) -> np.ndarray:
    """For distinct slices u (n, 32): whether another one lies within d codes (d <= 3: four 8-code sub-keys)."""
    near = np.zeros(len(u), bool)
    for b in range(4):
        keys = np.ascontiguousarray(u[:, 8 * b:8 * b + 8]).view(np.uint64).ravel()
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        for g in np.split(order, cuts):
            if len(g) < 2 or near[g].all():
                continue
            s = u[g]
            for c0 in range(0, len(g), chunk):
                dist = (s[c0:c0 + chunk, None, :] != s[None, :, :]).sum(axis=2)
                dist[np.arange(min(chunk, len(g) - c0)), np.arange(c0, min(c0 + chunk, len(g)))] = 99
                near[g[c0:c0 + chunk]] |= (dist <= d).any(axis=1)
    return near


def job(layer: int, cls: str, args) -> dict:
    t0 = time.monotonic()
    a, _ = operands(cls, layer, args, args.key)
    codes = quantize(a)
    rows, k = codes.shape
    total = rep = near = near_only = 0
    for t in range(k // 32):
        s = unsigned(codes[:, 32 * t:32 * t + 32])
        live = s.any(axis=1)
        _, first, inv, cnt = np.unique(s.view(np.dtype((np.void, 32))).ravel(), return_index=True, return_inverse=True,
                                       return_counts=True)
        inv = inv.ravel()
        is_first = np.zeros(rows, bool)
        is_first[first] = True
        u = s[first]
        nf = near_flags(u, args.d)
        total += int(live.sum())
        rep += int((live & ~is_first).sum())
        near += int((live & (nf[inv] | (cnt[inv] > 1))).sum())
        near_only += int((live & nf[inv] & (cnt[inv] == 1)).sum())
    return {"layer": layer, "class": cls, "rows": rows, "atoms": k // 32, "slices": total,
            "repeat_share": rep / total, "near_share": near / total, "near_only_share": near_only / total,
            "d": args.d, "seconds": round(time.monotonic() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--acts", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", default="0,8,16,28")
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--d", type=int, default=3, choices=(0, 1, 2, 3))
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
              f'near-only {r["near_only_share"]:.4f} ({r["seconds"]}s)', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
