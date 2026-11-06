"""The lead's Python layout's test vectors, recorded for cpu_check.cc.

Runs the Python fold's own checks of blc4a with the de-noise (`exact_blc.fold_extreme`, `exact_blc.exact_full` at Llama's
K, both under `ROT_WEIGHTS` and `LIT_WEIGHTS`) and records, as they happen, every call of `operands` with the A·Bᵀ and
scales its stock epilogue model is called on: a layout vector a call, the Python's data, coins and private values (A, B,
a₁, a_t, u, G, the lane coins), its scales and A·Bᵀ, which ssn's lanes (layout.h's `lane_a`, `lane_b`) and fold words must
fold to 0x4B400000 + A·Bᵀ and serve as verity_int8's epilogue on it (layout.h's `int8_epilogue`) bit for bit.

--theory is the directory holding the Python fold (exact_blc.py, exact_sweep.py and noise_sweep.py). Writes index.txt (a
line a vector: kind, label, file, weights, 1), index.json (the checks' own reports) and one .bin a vector: records of
(u32 name length, name, dtype char, u32 ndim, u64 dims, little-endian data).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

CODES = {np.dtype(np.int8): "b", np.dtype(np.int32): "i", np.dtype(np.int64): "q", np.dtype(np.float32): "f",
         np.dtype(np.uint16): "H"}


def write(path: str, arrays: dict) -> None:
    with open(path, "wb") as f:
        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb + CODES[arr.dtype].encode() + struct.pack("<I", arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}Q", *arr.shape) + arr.astype(arr.dtype.newbyteorder("<")).tobytes())


def int8(x) -> np.ndarray:
    x = np.asarray(x)
    assert x.min() >= -128 and x.max() <= 127, (x.min(), x.max())
    return x.astype(np.int8)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--theory", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.abspath(args.theory))
    import exact_blc as eb
    import exact_sweep as es

    os.makedirs(args.out, exist_ok=True)
    stock0, operands0 = es.stock_epilogue, eb.operands
    calls: list[tuple] = []

    def stock(acc, s_col, s_row):
        out = stock0(acc, s_col, s_row)
        calls.append(("stock", (np.array(acc), s_col, s_row), out))
        return out

    def operands(a, b, s, e, tile, a1, at, u, g, coins, kappa, weights=eb.COIN_WEIGHTS):
        res = operands0(a, b, s, e, tile, a1, at, u, g, coins, kappa, weights)
        calls.append(("operands", (a, b, s, e, tile, a1, at, u, g, coins, kappa, weights), res))
        return res

    es.stock_epilogue, eb.operands = stock, operands
    index, reports = [], {}

    def layout(label: str, op, st, weights_name: str):
        a, b, s, e, tile, a1, at, u, g, coins, kappa, weights = op[1]
        p, s_col, s_row = st[1]
        write(os.path.join(args.out, f"layout_{label}.bin"), {
            "a": int8(a), "b": int8(b), "a1": int8(a1), "at": int8(at),
            "u0": np.asarray(u[0], np.int32).ravel(), "u1": np.asarray(u[1], np.int32).ravel(),
            "u2": np.asarray(u[2], np.int32).ravel(), "g": int8(g), "coins": int8(coins),
            "weights": np.asarray(eb.row_weights(weights, 2), np.int32),
            "meta": np.array([a.shape[1], s, e, tile], np.int64),
            "s_col": np.asarray(s_col, np.float32), "s_row": np.asarray(s_row, np.float32), "p": p.astype(np.int64)})
        index.append(("layout", label, f"layout_{label}.bin", weights_name, 1))

    def take(kind: str):
        return [c for c in calls if c[0] == kind]

    for wname, weights in (("rot", eb.ROT_WEIGHTS), ("lit", eb.LIT_WEIGHTS)):
        calls.clear()
        reports[f"fold_extreme_{wname}"] = eb.fold_extreme(weights=weights)
        (op,), (st,) = take("operands"), take("stock")
        layout(f"extreme_{wname}", op, st, wname)
        for k in eb.KS:
            calls.clear()
            reports[f"exact_full_{wname}_{k}"] = eb.exact_full(k, weights=weights)
            ops, sts = take("operands"), take("stock")
            assert len(ops) == len(sts)
            for q, (op, st) in enumerate(zip(ops, sts)):
                layout(f"full_{wname}_{k}_salt{q}", op, st, wname)

    with open(os.path.join(args.out, "index.txt"), "w") as f:
        for kind, label, fname, w, expect in index:
            f.write(f"{kind} {label} {fname} {w} {expect}\n")
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"vectors": [dict(zip(("kind", "label", "file", "weights", "expect"), v)) for v in index],
                   "reports": reports}, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps({"vectors": len(index), "layout": len(index)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
