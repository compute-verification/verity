"""N-EXACT's fold test vectors, recorded for cpu_check.cc.

Runs N-EXACT's own checks of blc4a with the de-noise (`exact_blc.fold_extreme`, `exact_blc.exact_full` at Llama's K,
both under `ROT_WEIGHTS` and `LIT_WEIGHTS`, and `exact_sweep.check_epilogue`) and records, as they happen, every call of
its epilogue models and of `operands`:

- a fold vector for each `magic_epilogue` call: its inputs (the final word D, ρ_i, the column term, s_j, s_i) and its
  bf16 outputs, which ssn's fold (layout.h's `xu_fold`) must reproduce bit for bit from D, 0x4B400000 − ρ_i and minus the
  column term, with A·Bᵀ and stock's outputs on it where the call has them; one more for `fold_extreme`'s mutation
  (D + 2 at an output of 2²² − 1, so A·Bᵀ + 2 is 2²² + 1), with A·Bᵀ + 2 and stock's outputs on it;
- a layout vector for each `operands` call: N-EXACT's data, coins and private values (A, B, a₁, a_t, u, G, the lane
  coins), its scales and stock's bf16 outputs on A·Bᵀ, which ssn's lanes (layout.h's `lane_a`, `lane_b`) must serve
  through the fold bit for bit.

--theory is the directory holding exact_blc.py, exact_sweep.py and noise_sweep.py (benchmarks/pouw/theory on N-EXACT's
branch). Writes index.txt (a line a vector: kind, label, file, weights, expectation on stock), index.json (the checks'
own reports) and one .bin a vector: records of (u32 name length, name, dtype char, u32 ndim, u64 dims, little-endian
data).
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
    magic0, stock0, operands0 = es.magic_epilogue, es.stock_epilogue, eb.operands
    calls: list[tuple] = []

    def magic(acc, r, c, s_col, s_row):
        out = magic0(acc, r, c, s_col, s_row)
        calls.append(("magic", (np.array(acc), np.array(r), np.broadcast_to(c, np.shape(acc)).copy(), s_col, s_row), out))
        return out

    def stock(acc, s_col, s_row):
        out = stock0(acc, s_col, s_row)
        calls.append(("stock", (np.array(acc), s_col, s_row), out))
        return out

    def operands(a, b, s, e, tile, a1, at, u, g, coins, kappa, weights=eb.COIN_WEIGHTS):
        res = operands0(a, b, s, e, tile, a1, at, u, g, coins, kappa, weights)
        calls.append(("operands", (a, b, s, e, tile, a1, at, u, g, coins, kappa, weights), res))
        return res

    es.magic_epilogue, es.stock_epilogue, eb.operands = magic, stock, operands
    index, reports = [], {}

    def fold(label: str, rec, expect: int = 0, stock=None):
        acc, r, c, s_col, s_row = rec[1]
        arrays = {"acc": acc.astype(np.int64), "rho": r.astype(np.int64), "col": c.astype(np.int64),
                  "s_col": np.asarray(s_col, np.float32), "s_row": np.asarray(s_row, np.float32),
                  "magic": np.asarray(rec[2], np.uint16)}
        if stock is not None:
            arrays["p"] = np.asarray(stock[1][0], np.int64)
            arrays["stock"] = np.asarray(stock[2], np.uint16)
        write(os.path.join(args.out, f"fold_{label}.bin"), arrays)
        index.append(("fold", label, f"fold_{label}.bin", "none", expect))

    def layout(label: str, op, st, weights_name: str):
        a, b, s, e, tile, a1, at, u, g, coins, kappa, weights = op[1]
        p, s_col, s_row = st[1]
        write(os.path.join(args.out, f"layout_{label}.bin"), {
            "a": int8(a), "b": int8(b), "a1": int8(a1), "at": int8(at),
            "u0": np.asarray(u[0], np.int32).ravel(), "u1": np.asarray(u[1], np.int32).ravel(),
            "u2": np.asarray(u[2], np.int32).ravel(), "g": int8(g), "coins": int8(coins),
            "weights": np.asarray(eb.row_weights(weights, 2), np.int32),
            "meta": np.array([a.shape[1], s, e, tile], np.int64),
            "s_col": np.asarray(s_col, np.float32), "s_row": np.asarray(s_row, np.float32),
            "p": p.astype(np.int64), "stock": np.asarray(st[2], np.uint16)})
        index.append(("layout", label, f"layout_{label}.bin", weights_name, 1))

    def take(kind: str):
        return [c for c in calls if c[0] == kind]

    for wname, weights in (("rot", eb.ROT_WEIGHTS), ("lit", eb.LIT_WEIGHTS)):
        calls.clear()
        rep = eb.fold_extreme(weights=weights)
        reports[f"fold_extreme_{wname}"] = rep
        (op,), (st,), (mg,) = take("operands"), take("stock"), take("magic")
        layout(f"extreme_{wname}", op, st, wname)
        fold(f"extreme_{wname}", mg, 1, st)
        acc, r, c, s_col, s_row = mg[1]
        mut, pm = acc.copy(), st[1][0].copy()
        mut[0, 0] += 2
        pm[0, 0] += 2
        fold(f"extreme_{wname}_mutation", ("magic", (mut, r, c, s_col, s_row), magic0(mut, r, c, s_col, s_row)), -1,
             ("stock", (pm, s_col, s_row), stock0(pm, s_col, s_row)))
        for k in eb.KS:
            calls.clear()
            rep = eb.exact_full(k, weights=weights)
            reports[f"exact_full_{wname}_{k}"] = rep
            ops, sts, mgs = take("operands"), take("stock"), take("magic")
            assert len(ops) == len(sts) == len(mgs)
            for q, (op, st, mg) in enumerate(zip(ops, sts, mgs)):
                layout(f"full_{wname}_{k}_salt{q}", op, st, wname)
                fold(f"full_{wname}_{k}_salt{q}", mg, 1, st)
    calls.clear()
    reports["check_epilogue"] = es.check_epilogue(np.random.default_rng(7))
    names = ["rounded_scales", "raw_scales", "out_of_range", "tile_sum"]
    for name, mg in zip(names, take("magic"), strict=True):
        fold(f"check_epilogue_{name}", mg)

    with open(os.path.join(args.out, "index.txt"), "w") as f:
        for kind, label, fname, w, expect in index:
            f.write(f"{kind} {label} {fname} {w} {expect}\n")
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"vectors": [dict(zip(("kind", "label", "file", "weights", "expect_stock"), v)) for v in index],
                   "reports": reports}, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps({"vectors": len(index), "fold": sum(v[0] == "fold" for v in index),
                      "layout": sum(v[0] == "layout" for v in index)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
