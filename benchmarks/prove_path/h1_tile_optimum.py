"""The PoUW H1 tile shape R × C against proving and committing, at δ = 2^-40 and ε = 1%, from `h1_tile_ands.py`'s measured
shapes at each contraction K of Llama-3.2-1B's int8 linears (2,048 and 8,192).

* A block at K is the tile (its ANDs a least-squares fit a·RC + b·R + c·C + d to K's measured shapes, each held to within
  0.1%) and its rows: R A rows and C B rows of K + 4 bytes, the zy record (6 RC bytes), the public row (200 bytes) and
  the receipt and ok rows (the unit's returned bits). Its openings are `class_size.opening_ands` of those rows, and its
  k_log `class_size.block_k_log` (an estimate, from C-Flock's ranges). ``--anchor``'s 4 × 8 tile, at the K past which
  `h1_prove.h1unit` says a lane fits no 2^27 block (K ≈ 5,041), checks that the model refuses it there.
* Proving a window is Λ/ε draws of units drawn by their work (compute ANDs), each costing its block: ANDs with openings,
  and bits (2^k_log).
* Committing a token: each tile's receipt and ok rows (the unit's returned bits at its K, the same at every shape) for
  every min(R, ``--rows``) tokens and C columns of a K's linears (`COLUMNS`), beside the zy records (6 bytes a cell, the
  same at every shape). ``--rows`` is the token rows a call holds (decode: its batch).
* Break-even: the prover rate (ANDs a second) at which a shape's receipts saved, hashed at `SHA_RATE`, pay for its extra
  proving against 4 × 8, for a window of ``--tokens`` real tokens. A slower prover keeps 4 × 8 the optimum.
* The optimum by window: the shape least in GPU-seconds a token, proving (Λ/ε draws a window, over its tokens, at
  ``--rate`` ANDs a second) plus hashing its receipts at `SHA_RATE`, for each of ``--windows`` tokens a window, on the
  code's grid and off it.
* Shapes off the code's grid (h1tile takes 4 | R and 8 | C, a whole number of 4 × 8 lanes) are the fit extrapolated.

    python h1_tile_optimum.py K2048.json K8192.json [--anchor K5056.json] [--rows 4] [--tokens 64] [--rate 1e9]
        [--windows 64,1e6] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import class_size as CS  # noqa: E402

DRAWS = math.log(2**40) / 0.01
#: the served int8 linears' output columns a token at Llama-3.2-1B by K: 16 layers of qkv 3,072, o 2,048 and gate_up
#: 16,384 at K 2,048, and down 2,048 at K 8,192
COLUMNS = {2048: 16 * (3072 + 2048 + 16384), 8192: 16 * 2048}
#: SHA-512 on the GPU at 4 KiB rows, bytes a second (the golden brief's measured figure)
SHA_RATE = 398e9
GRID = [(r, c) for r in (4, 8, 16, 32) for c in (8, 16, 32)]
OFF_GRID = [(1, 1), (2, 4), (4, 4)]


def fit(rows: list[dict]) -> np.ndarray:
    X = np.array([[r["tile"][0] * r["tile"][1], r["tile"][0], r["tile"][1], 1] for r in rows], float)
    y = np.array([r["ands"] for r in rows], float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    worst = max(abs(X[i] @ coef - y[i]) / y[i] for i in range(len(y)))
    if len(rows) < 5 or worst > 1e-3:
        raise ValueError(f"K={rows[0]['k']}: the shape fit is off by {worst:.2%} at a measured shape ({len(rows)} shapes; "
                         f"measure at least 5)")
    return coef


def tile_rows(k: int, r: int, c: int, returned_bits: int) -> list[int]:
    return [k + 4] * (r + c) + [6 * r * c, 200, math.ceil(returned_bits / 8), 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles", nargs="+")
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--rate", type=float, default=1e9)
    ap.add_argument("--windows", default="64,1e4,1e5,1e6,1e7,1e8,1e9")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    by_k = {}
    for f in a.tiles:
        rows = json.loads(Path(f).read_text())
        by_k[rows[0]["k"]] = {"coef": fit(rows), "returned": rows[0]["returned_bits"],
                              "measured": {tuple(r["tile"]): r["ands"] for r in rows}}
    if set(by_k) != set(COLUMNS):
        raise ValueError(f"measure every K of COLUMNS ({sorted(COLUMNS)}), not {sorted(by_k)}")
    check = None
    if a.anchor:
        an = next(r for r in json.loads(Path(a.anchor).read_text()) if r["tile"] == [4, 8])
        check = {"k": an["k"], "ands": an["ands"], "k_log_est": CS.block_k_log(an["ands"], tile_rows(an["k"], 4, 8, an["returned_bits"]))}
        if check["k_log_est"] <= CS.K_MAX:
            raise ValueError(f"block_k_log puts the 4 x 8 lane at K={an['k']} in 2^{check['k_log_est']} bits, but h1unit says no "
                             f"lane past K ≈ 5,041 fits 2^{CS.K_MAX}: recalibrate class_size.UNIT_BITS_PER_AND")

    def shape(r: int, c: int) -> dict:
        per_k = {}
        for k, m in by_k.items():
            compute = m["measured"].get((r, c)) or int(m["coef"] @ [r * c, r, c, 1])
            rows = tile_rows(k, r, c, m["returned"])
            ands = compute + sum(CS.opening_ands(b) for b in rows)
            per_k[k] = {"compute_ands": compute, "measured": (r, c) in m["measured"], "block_ands": ands,
                        "k_log_est": CS.block_k_log(compute, rows),
                        "tiles_per_token": COLUMNS[k] / (min(r, a.rows) * c),
                        "receipt_bytes_per_token": COLUMNS[k] / (min(r, a.rows) * c) * m["returned"] / 8}
        w = {k: COLUMNS[k] / (r * c) * p["compute_ands"] for k, p in per_k.items()}
        cost = sum(w[k] * p["block_ands"] for k, p in per_k.items()) / sum(w.values())
        bits = sum(w[k] * 2 ** p["k_log_est"] for k, p in per_k.items()) / sum(w.values())
        return {"tile": [r, c], "on_grid": r % 4 == 0 and c % 8 == 0, "by_k": per_k, "mean_block_ands": cost,
                "proved_ands_per_window": DRAWS * cost, "proved_bits_per_window": DRAWS * bits,
                "k_log_max_est": max(p["k_log_est"] for p in per_k.values()),
                "receipt_bytes_per_token": sum(p["receipt_bytes_per_token"] for p in per_k.values()),
                "padded_rows_share": max(0, r - a.rows) / r}

    shapes = [shape(r, c) for r, c in OFF_GRID + GRID]
    base = next(s for s in shapes if s["tile"] == [4, 8])
    for s in shapes:
        d_prove = s["proved_ands_per_window"] - base["proved_ands_per_window"]
        d_bytes = (base["receipt_bytes_per_token"] - s["receipt_bytes_per_token"]) * a.tokens
        s["breakeven_ands_per_s_vs_4x8"] = d_prove * SHA_RATE / d_bytes if d_bytes > 0 and d_prove > 0 else None
    zy = sum(COLUMNS.values()) * 6

    def seconds(s: dict, n: float) -> float:
        return s["proved_ands_per_window"] / (n * a.rate) + s["receipt_bytes_per_token"] / SHA_RATE

    optimum = []
    for n in (float(w) for w in a.windows.split(",")):
        grid = min((s for s in shapes if s["on_grid"]), key=lambda s: seconds(s, n))
        best = min(shapes, key=lambda s: seconds(s, n))
        optimum.append({"tokens_per_window": n, "on_grid": grid["tile"], "on_grid_k_log_max_est": grid["k_log_max_est"],
                        "on_grid_s_per_token": seconds(grid, n), "any": best["tile"], "any_s_per_token": seconds(best, n),
                        "s_per_token_4x8": seconds(base, n), "s_per_token_32x32": seconds(shapes[-1], n)})
    out = {"draws": DRAWS, "cap_log": CS.K_MAX, "anchor_check": check, "rows": a.rows, "tokens": a.tokens, "rate": a.rate,
           "sha_rate": SHA_RATE, "compression_ands": CS.COMPRESSION_ANDS, "zy_bytes_per_token": zy, "optimum": optimum,
           "fits": {k: {"per_cell": m["coef"][0], "per_row": m["coef"][1], "per_col": m["coef"][2], "fixed": m["coef"][3]}
                    for k, m in by_k.items()}, "shapes": shapes}
    print(f"draws {DRAWS:,.0f}; openings at {CS.COMPRESSION_ANDS:,} ANDs a compression; rows a call {a.rows}; "
          f"zy {zy / 1e6:.2f} MB/token at every shape"
          + (f"; check: 4x8 at K={check['k']} ({check['ands']:,} ANDs) is k{check['k_log_est']}" if check else ""))
    for k, m in by_k.items():
        c0 = m["coef"]
        print(f"  K={k}: tile ANDs = {c0[0]:,.0f}·RC + {c0[1]:,.0f}·R + {c0[2]:,.0f}·C + {c0[3]:,.0f}; receipt {m['returned']} bits")
    for s in shapes:
        ks = "  ".join(f"K{k} {p['block_ands'] / 1e6:6.1f}M k{p['k_log_est']}{'' if p['measured'] else '*'}" for k, p in s["by_k"].items())
        be = s["breakeven_ands_per_s_vs_4x8"]
        print(f"  {s['tile'][0]:>2}x{s['tile'][1]:<2}{' ' if s['on_grid'] else '!'} {ks}  proved/window {s['proved_ands_per_window'] / 1e9:7.1f} G"
              f" (2^{math.log2(s['proved_bits_per_window']):.2f} bits)  receipts {s['receipt_bytes_per_token'] / 1e6:7.2f} MB/token"
              f"  break-even vs 4x8 {'-' if be is None else f'{be:.1e} ANDs/s'}")
    print("  (* fit, not measured; ! off h1tile's grid, the fit extrapolated)")
    print(f"optimum by tokens a window, proving at {a.rate:.1e} ANDs/s and hashing at {SHA_RATE:.3e} B/s (GPU-seconds a token):")
    for o in optimum:
        print(f"  {o['tokens_per_window']:>8.0e} tokens: grid {o['on_grid'][0]}x{o['on_grid'][1]} (k{o['on_grid_k_log_max_est']} at its "
              f"largest K) {o['on_grid_s_per_token']:.2e} s; any {o['any'][0]}x{o['any'][1]} {o['any_s_per_token']:.2e} s; "
              f"4x8 {o['s_per_token_4x8']:.2e} s, 32x32 {o['s_per_token_32x32']:.2e} s")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
