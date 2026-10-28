"""The orbit against the plain int8 GEMM on Llama 3.1 8B's int8 weights (weights.py), bit for bit, on three kinds of
activations, with the carry statistics a fused kernel's corrections depend on. CPU only.

Activations A [M, K] int8: `uniform` (uniform on [−128, 127]: a quarter of A's two-block forms carry), `gauss` (rows of
N(0, 1) quantized per row to absmax 127, the served rule's shape) and `bits` (0/1 entries: the A = 0 operand of an
un-noised dither check).

    python realw.py --weights DIR --layers 0,15 --m 256 --out OUT     # OUT/realw.json, OUT/realw.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orbit  # noqa: E402
import weights  # noqa: E402

TILE = 128


def activations(kind: str, m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "uniform":
        return rng.integers(-128, 128, (m, k), dtype=np.int8)
    if kind == "gauss":
        x = rng.standard_normal((m, k)).astype(np.float32)
        s = np.abs(x).max(axis=1, keepdims=True) / np.float32(127)
        return np.clip(np.rint(x / s), -127, 127).astype(np.int8)
    if kind == "bits":
        return rng.integers(0, 2, (m, k), dtype=np.int8)
    raise ValueError(kind)


def tile_stats(carry: np.ndarray) -> dict:
    """Carries per (slot, 128-row tile of the form) over all of K/2, and per (slot, row tile, 128-column K tile)."""
    s, n, h = carry.shape
    nt, kt = -(-n // TILE), -(-h // TILE)
    pad = np.zeros((s, nt * TILE, kt * TILE), bool)
    pad[:, :n, :h] = carry != 0
    per_tile_k = pad.reshape(s, nt, TILE, kt, TILE).sum(axis=(2, 4))
    per_tile = per_tile_k.sum(axis=2)
    distinct_k = pad.reshape(s, nt, TILE, kt * TILE).any(axis=2).sum(axis=2)
    two = [i for i, p in enumerate(orbit.ORDER) if len(orbit.B_FORMS[p]) == 2]
    return {"per_row_tile_mean": float(per_tile[two].mean()), "per_row_tile_max": int(per_tile.max()),
            "per_row_tile_k_tile_mean": float(per_tile_k[two].mean()), "per_row_tile_k_tile_max": int(per_tile_k.max()),
            "distinct_k_per_row_tile_mean": float(distinct_k[two].mean()), "half_k": h}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--layers", default="0,15")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--kinds", default="uniform,gauss,bits")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    ws = weights.load(a.weights, layers)
    rng = np.random.default_rng(20261028)
    rows, ok = [], True
    for (layer, key), w in sorted(ws.items()):
        t0 = time.time()
        pb = orbit.prepare_b(w)
        rate = pb.carry_rate()
        row = {"layer": layer, "linear": key, "n": int(w.shape[0]), "k": int(w.shape[1]),
               "b_carry_rate": {str(p): r for p, r in rate.items()},
               "b_carry_rate_mean": float(np.mean(list(rate.values()))), "b_tiles": tile_stats(pb.carry), "kinds": {}}
        for kind in a.kinds.split(","):
            x = activations(kind, a.m, w.shape[1], rng)
            _, a_carry, _ = orbit.forms_a(x)
            two = [i for i, p in enumerate(orbit.ORDER) if len(orbit.A_FORMS[p]) == 2]
            got = orbit.orbit_mm(x, pb)
            want = orbit.plain_mm(x, w)
            exact = bool(np.array_equal(got, want))
            uncorrected = orbit.orbit_mm(x, pb, corrected=False)
            row["kinds"][kind] = {"m": a.m, "exact": exact, "cells": int(want.size),
                                  "a_carry_rate_mean": float(np.count_nonzero(a_carry[two]) / a_carry[two].size),
                                  "cells_wrong_uncorrected": int(np.count_nonzero(uncorrected != want))}
            ok &= exact
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.out / "realw.json").write_text(json.dumps({"ok": ok, "rows": rows, "repo": weights.REPO,
                                                  "revision": weights.REVISION}, indent=1))
    lines = ["| layer | linear | N x K | B carry rate (two-block forms) | B carries / 128-row tile / 128-K tile (mean, max) | "
             "distinct k / 128-row tile of K/2 | exact: " + ", ".join(a.kinds.split(",")) + " | A carry rate: " +
             ", ".join(a.kinds.split(",")) + " |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["b_tiles"]
        lines.append(f"| {r['layer']} | {r['linear']} | {r['n']} x {r['k']} | {100 * r['b_carry_rate_mean']:.3f}% | "
                     f"{t['per_row_tile_k_tile_mean']:.1f}, {t['per_row_tile_k_tile_max']} | "
                     f"{t['distinct_k_per_row_tile_mean']:.0f} of {t['half_k']} | "
                     + ", ".join(str(v["exact"]) for v in r["kinds"].values()) + " | "
                     + ", ".join(f"{100 * v['a_carry_rate_mean']:.3f}%" for v in r["kinds"].values()) + " |")
    (a.out / "realw.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
