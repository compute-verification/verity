#!/usr/bin/env python3
"""The honest cost of crediting R1-H's residual-basis atom slices once across layers (the store's docs/pouw/new-designs.md
§3.7, the residual-state rule), on a capture.

The residual stream's keyed rotation Q is one rotation for every layer, so the A rows of every layer's qkv and gate/up
linears, (X / g) Q quantized per row, are codes of the same basis.  A residual the prover writes through a faulty glue unit,
with a norm large enough that later layers barely move it, would give the same codes at its position in every later layer.
The rule credits each such slice once across all those linears: per atom, a slice that repeats, or is within d codes of
(d* = 26/32 is d = 6), a slice of another layer or class earns nothing.  On honest rows that removes some credit beyond what
per-linear credit already removes; that increment is the rule's cost, measured here.

Per (layer, class), over every A row (row 0, the attention sink's position, apart) and every atom, the share of nonzero
slices
  within_share  a repeat of, or within d codes of, another slice of the same (layer, class): what per-linear credit removes
  pooled_share  the same against every slice of every captured layer and both classes
  cross_share   pooled_share less within_share: the rule's added cost (an upper bound, as r1_slices.py's: first copies count)
and the same for row 0 alone, and pooled_share against the other layers only (`other_layers_share`), by d = 0..D.

  python r1_xlayer.py --acts DIR --weights SNAPSHOT --out FILE.json [--layers 0,4,...,28] [--d 6] [--first-row R]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from r1_census import Safetensors, haar, quantize, rng
from r1_slices import near_dist, unsigned

CLASSES = {"qkv": "input_layernorm", "gateup": "post_attention_layernorm"}


def codes(layer: int, cls: str, args) -> np.ndarray:
    st = Safetensors(Path(args.weights))
    gamma = st.get(f"model.layers.{layer}.{CLASSES[cls]}.weight").astype(np.float64)
    x = np.asarray(np.load(Path(args.acts) / f"L{layer:03d}-{cls}.npy", mmap_mode="r")[args.first_row:], np.float64)
    q = haar(x.shape[1], rng(args.key, "residual-rotation"))
    return quantize((x / gamma) @ q)


def nearest(pool: np.ndarray, d: int) -> np.ndarray:
    """Per slice of pool (n, 32): the distance to the nearest other slice, 0 for an exact repeat, d + 1 beyond d."""
    _, first, inv, cnt = np.unique(pool.view(np.dtype((np.void, 32))).ravel(), return_index=True, return_inverse=True,
                                   return_counts=True)
    inv = inv.ravel()
    dist = near_dist(pool[first], d)[inv]
    return np.where(cnt[inv] > 1, 0, dist)


def across(slices: list[np.ndarray], layers: list[int], d: int, chunk: int = 512) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per set's rows: the distance to the nearest slice of another set, and of another layer (d + 1 beyond d), by
    near_dist's sub-keys over each set's distinct slices, pooled and labelled."""
    uniq, invs, set_of, layer_of = [], [], [], []
    for i, s in enumerate(slices):
        _, first, inv = np.unique(s.view(np.dtype((np.void, 32))).ravel(), return_index=True, return_inverse=True)
        uniq.append(s[first])
        invs.append(inv.ravel())
        set_of.append(np.full(len(first), i))
        layer_of.append(np.full(len(first), layers[i]))
    u, set_of, layer_of = np.concatenate(uniq), np.concatenate(set_of), np.concatenate(layer_of)
    best_set = np.full(len(u), d + 1, np.int64)
    best_layer = np.full(len(u), d + 1, np.int64)
    for cols in np.array_split(np.arange(32), d + 1):
        keys = np.ascontiguousarray(u[:, cols]).view(np.dtype((np.void, len(cols)))).ravel()
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        for g in np.split(order, np.flatnonzero(ks[1:] != ks[:-1]) + 1):
            if len(g) < 2 or ((best_set[g] == 0).all() and (best_layer[g] == 0).all()):
                continue
            s, sg, lg = u[g], set_of[g], layer_of[g]
            for c0 in range(0, len(g), chunk):
                c = slice(c0, min(c0 + chunk, len(g)))
                dist = (s[c, None, :] != s[None, :, :]).sum(axis=2)
                best_set[g[c]] = np.minimum(best_set[g[c]], np.where(sg[c, None] == sg[None, :], 99, dist).min(axis=1))
                best_layer[g[c]] = np.minimum(best_layer[g[c]], np.where(lg[c, None] == lg[None, :], 99, dist).min(axis=1))
    offs = np.cumsum([0] + [len(x) for x in uniq])
    return ([best_set[offs[i]:offs[i + 1]][invs[i]] for i in range(len(slices))],
            [best_layer[offs[i]:offs[i + 1]][invs[i]] for i in range(len(slices))])


def atom(t: int, sets: list[tuple[int, str]], path: str, rows: int, d: int) -> dict:
    allc = np.load(path, mmap_mode="r")
    slices = [unsigned(np.asarray(allc[i, :, 32 * t:32 * t + 32])) for i in range(len(sets))]
    live = [s.any(axis=1) for s in slices]
    to_set, to_layer = across(slices, [layer for layer, _ in sets], d)
    out = {}
    for i, (layer, cls) in enumerate(sets):
        within = nearest(slices[i], d)
        pooled = np.minimum(within, to_set[i])
        # other_layers counts a slice near a slice of another layer, or of its own (layer, class)
        other = np.minimum(within, to_layer[i])
        rec = {}
        for name, sel in (("rows", np.arange(rows) > 0), ("row0", np.arange(rows) == 0)):
            m = live[i] & sel
            rec[name] = {"slices": int(m.sum()),
                         "within": [int((m & (within <= e)).sum()) for e in range(d + 1)],
                         "pooled": [int((m & (pooled <= e)).sum()) for e in range(d + 1)],
                         "other_layers": [int((m & (other <= e)).sum()) for e in range(d + 1)]}
        out[f"{layer}/{cls}"] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--acts", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--d", type=int, default=6, choices=range(8))
    ap.add_argument("--key", default="r1-census/v0")
    ap.add_argument("--first-row", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scratch", required=True, help="a directory for the pooled codes (rows x 4,096 bytes per set)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.monotonic()
    sets = [(int(layer), cls) for layer in args.layers.split(",") for cls in CLASSES]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(sets))) as pool:
        cs = list(pool.map(codes, *zip(*sets), [args] * len(sets)))
    rows, k = cs[0].shape
    path = str(Path(args.scratch) / "xlayer-codes.npy")
    np.save(path, np.stack(cs))
    del cs
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        per_atom = list(pool.map(atom, range(k // 32), [sets] * (k // 32), [path] * (k // 32), [rows] * (k // 32),
                                 [args.d] * (k // 32)))
    res = []
    for layer, cls in sets:
        rec = {"layer": layer, "class": cls, "rows": rows, "atoms": k // 32, "d": args.d}
        for name in ("rows", "row0"):
            tot = {f: np.sum([pa[f"{layer}/{cls}"][name][f] for pa in per_atom], axis=0) for f in ("within", "pooled", "other_layers")}
            n = int(sum(pa[f"{layer}/{cls}"][name]["slices"] for pa in per_atom))
            share = {f: [float(x) / max(n, 1) for x in v] for f, v in tot.items()}
            rec[name] = {"slices": n, "within_share": share["within"], "pooled_share": share["pooled"],
                         "cross_share": [p - w for p, w in zip(share["pooled"], share["within"])],
                         "other_layers_share": share["other_layers"]}
        res.append(rec)
        r = rec["rows"]
        print(f"L{layer:02d} {cls:6s} within {r['within_share'][-1]:.4f} pooled {r['pooled_share'][-1]:.4f} "
              f"cross {r['cross_share'][-1]:.4f} (d<=3 {r['cross_share'][min(3, args.d)]:.4f}); row 0 cross "
              f"{rec['row0']['cross_share'][-1]:.4f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"seconds": round(time.monotonic() - t0, 1), "sets": res}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
