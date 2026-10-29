"""COUNTER (LEAN program): N-PREM's statement of the lead's work guarantee, searched for strategies that falsify it.

The statement is `Pouw.Guarantees.Lead.G` with its layout, model, gap and premises (`Security/Definitions/Pouw/Lead/`,
`Assumptions/Lead.lean`, `Guarantees/Lead.lean`). This module ports the definitions line for line, in exact rationals
where the Lean is rational: `bal13`, `rowTag`, `colCode`, `rotWeight`, `bias`, `coinTerm`, `word`, `stepsFor`, `tiles`,
`tilePos` and every term of `Gap.lean` (`Commit.chunks` at `wordChunks`, its messages' credit at the formula's, so
`ShaWrites` charges a strategy exactly the honest compressions).

`G`'s acceptance asks only that each word's value be held at its tile position (`holds (tilePos i j) v`), and
`TileGrain` counts a value as held when the run's memory has it there at some address after some prefix, the initial
memory (the pre-coin tables and the stream) included. Two strategies meet every premise of `G` (with `Real` their
singleton, `can = Real`, `pr = h100Nominal`, κ = 0, the honest root and messages registered) and do less work than it
bounds:

- **table**: the pre-coin tables hold every integer at every tile position (`pre a p` an enumeration of ℤ), the run
  is empty, and the trace is the compressions' registers as `.other` at 32. Its work is the honest compressions'; it
  saves every word's register. A finite table needs the words' range at each position (reported, in bytes).
- **dedup**: empty tables, the honest stream; the run writes each distinct (tile position, value) pair of the call once,
  one `.other` a write. It saves the call's repeated pairs, counted exactly at sampled coins and bounded below at
  every coin value by pigeonhole on step 0's range.

Each is reported against `32 · (honestRegs − gapRegs)` at the same layout. `GapAt` and `GapAt320` are checked exactly
over a scan of n and K and by a closed-form bound for every n, K ≥ 4,096.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction as Q

import numpy as np

TILE_WORDS = 128
PRICE = 32


def bal13(z: int, q: int) -> int:
    return (z + 14280) // 13 ** q % 13 - 6


def row_tag(i: int) -> list[int]:
    z = i // 2 + 1 if i % 2 == 0 else -(i // 2 + 1)
    return [bal13(z, q) for q in range(4)]


def col_code(j: int) -> list[int]:
    return [bal13(j + 1, q) for q in range(4)]


def rot_weight(i: int, lane: int) -> int:
    return 127 if lane == i % 2 else 1


def bias(s: int, j: int) -> int:
    return (sum(col_code(j)) + 3 * (s - 1) + 1) // 2


def steps_for(k: int) -> int:
    return 1 + (max(k - 24, 0) + 27) // 28


def layout(t: int, m: int, n: int, k: int, cap: int = 32) -> dict:
    return {"T": t, "m": m, "n": n, "K": k, "S": steps_for(k), "tile": 64, "e": 7, "chunkWords": 65536, "cap": cap}


def tiles(lay: dict) -> int:
    return (lay["m"] + lay["tile"] - 1) // lay["tile"]


def word_chunks(lay: dict) -> int:
    return (lay["m"] * lay["n"] * lay["S"] + lay["chunkWords"] - 1) // lay["chunkWords"]


def gap(lay: dict) -> dict:
    """`Gap.lean` at `ChunksOf`, exact."""
    m, n, k, s, cap = lay["m"], lay["n"], lay["K"], lay["S"], lay["cap"]
    chunks = word_chunks(lay)
    stock = Q(m * n * k, 32)
    words = Q(m * n * s)
    sha = Q(351538, 100) * (4 * words / 128 + chunks) + Q(344905, 100) * Q(m * k, 128)
    terms = {"head": 23 * Q(chunks), "hiding": Q(16064, 1000000) * Q(n * k, 32), "coinBranch": Q(84, 100000) * stock,
             "pStar": Q(35, 1000000) * stock, "network": Q(36, 100000) * Q(cap, 32) * stock}
    g = sum(terms.values())
    return {"stock": stock, "words": words, "sha": sha, "honest": words + sha, "gap": g, "terms": terms,
            "gapShare": g / stock, "chunks": chunks}


def draw(lay: dict, rng: np.random.Generator) -> dict:
    """The coins `(b, a₁, g, u)`, uniform: `G = g − 3`, `s = 2u − (2^e − 1)`."""
    m, n, s = lay["m"], lay["n"], lay["S"]
    return {"b": rng.integers(0, 2, (m, s)), "a1": rng.integers(0, 2, (m, 4)),
            "G": rng.integers(0, 7, (4, n)) - 3,
            "s": (2 * rng.integers(0, 2 ** lay["e"], (s, tiles(lay), 3, n)) - (2 ** lay["e"] - 1)).astype(np.int16)}


def words_at(lay: dict, om: dict, rows: np.ndarray, cols: np.ndarray, kappa: int = 0) -> np.ndarray:
    """`word L κ ω i t j` for i in `rows`, every t, j in `cols` (κ constant), shape (rows, S, cols)."""
    tags = np.array([row_tag(int(i)) for i in rows], dtype=np.int64)
    code = np.array([col_code(int(j)) for j in cols], dtype=np.int64)
    beta = np.array([bias(lay["S"], int(j)) for j in cols], dtype=np.int64)
    step0 = tags @ om["G"][:, cols] + om["a1"][rows] @ code.T
    wts = np.array([[rot_weight(int(i), ell) for ell in range(3)] for i in rows], dtype=np.int64)
    tile_of = rows // lay["tile"]
    sv = om["s"][1:][:, tile_of][:, :, :, cols].astype(np.int64)
    later = (1 + om["b"][rows, 1:])[:, :, None] + np.einsum("il,tilj->itj", wts, sv)
    terms = np.concatenate([step0[:, None, :], later], axis=1) + kappa
    return np.cumsum(terms, axis=1) - beta[None, None, :]


def word_scalar(lay: dict, om: dict, i: int, t: int, j: int, kappa: int = 0) -> int:
    """`word` straight from the Lean, one entry, for checking `words_at`."""
    total = -bias(lay["S"], j)
    for tau in range(t + 1):
        if tau == 0:
            ct = sum(row_tag(i)[q] * int(om["G"][q, j]) for q in range(4)) + sum(
                int(om["a1"][i, q]) * col_code(j)[q] for q in range(4))
        else:
            k = i // lay["tile"]
            ct = 1 + int(om["b"][i, tau]) + sum(
                rot_weight(i, ell) * (int(om["s"][tau, k, ell, j]) if k < tiles(lay) else 0) for ell in range(3))
        total += kappa + ct
    return total


def check_port(seed: int = 3) -> dict:
    lay = layout(1, 80, 20, 24 + 28 * 4)
    om = draw(lay, np.random.default_rng(seed))
    rows, cols = np.arange(lay["m"]), np.arange(lay["n"])
    w = words_at(lay, om, rows, cols)
    rng = np.random.default_rng(seed + 1)
    picks = [(int(rng.integers(lay["m"])), int(rng.integers(lay["S"])), int(rng.integers(lay["n"])))
             for _ in range(400)]
    bad = [(i, t, j) for i, t, j in picks if word_scalar(lay, om, i, t, j) != int(w[i, t, j])]
    return {"layout": lay, "entries": len(picks), "mismatches": len(bad)}


def positions(lay: dict):
    for r in range(16):
        rows = np.arange(r, lay["m"], 16)
        for c in range(8):
            cols = np.arange(c, lay["n"], 8)
            if len(rows) and len(cols):
                yield r * 8 + c, rows, cols


def step0_bound(lay: dict) -> dict:
    """At every coin value, a position's step-0 words lie in an interval fixed by the layout, so the position's
    distinct step-0 values are at most its width, and the dedup strategy saves at least count − width there."""
    s = lay["S"]
    saved = 0
    widths = []
    for _, rows, cols in positions(lay):
        t1 = np.array([sum(abs(d) for d in row_tag(int(i))) for i in rows])
        code = np.array([col_code(int(j)) for j in cols])
        beta = np.array([bias(s, int(j)) for j in cols])
        lo = (-beta + np.minimum(code, 0).sum(axis=1)).min() - 3 * t1.max()
        hi = (-beta + np.maximum(code, 0).sum(axis=1)).max() + 3 * t1.max()
        width = int(hi - lo + 1)
        widths.append(width)
        saved += max(0, len(rows) * len(cols) - width)
    return {"saved_regs": saved, "width_max": max(widths), "per_position_step0": int(
        (lay["m"] // 16) * (lay["n"] // 8))}


def word_range(lay: dict) -> int:
    """A bound on |word| at κ = 0 over every coin value: β, step 0's terms and S − 1 later steps of at most
    2 + 129 · (2^e − 1)."""
    s = lay["S"]
    beta = max(abs(bias(s, j)) for j in range(lay["n"]))
    t1 = max(sum(abs(d) for d in row_tag(i)) for i in range(lay["m"]))
    c1 = max(sum(abs(d) for d in col_code(j)) for j in range(lay["n"]))
    return beta + 3 * t1 + c1 + (s - 1) * (2 + 129 * (2 ** lay["e"] - 1))


def dedup(lay: dict, seed: int) -> dict:
    """The dedup strategy's writes at one coin value: the call's distinct (tile position, value) pairs."""
    om = draw(lay, np.random.default_rng(seed))
    distinct = 0
    for _, rows, cols in positions(lay):
        distinct += len(np.unique(words_at(lay, om, rows, cols)))
    return {"seed": seed, "distinct_pairs": distinct}


def verdicts(lay: dict, saved_regs: Q) -> dict:
    """A strategy saving `saved_regs` registers against the honest work, at 32 a register, with the compressions
    charged as the honest's (rounded up to whole registers): G's bound fails when its work is below 32·(honest − gap).
    """
    g = gap(lay)
    work = PRICE * (g["honest"] - saved_regs + (math.ceil(g["sha"]) - g["sha"]))
    bound = PRICE * (g["honest"] - g["gap"])
    return {"work": float(work), "bound": float(bound), "breaks": bool(work < bound),
            "saved_share_of_stock": float(saved_regs / g["stock"]), "gap_share": float(g["gapShare"]),
            "excess_over_gap": float(saved_regs / g["gap"])}


def gap_at(bounds: dict, cap: int, ks: range, ns: tuple) -> dict:
    """`GapAt` (cap 32) or `GapAt320`: the largest gap share over the scan, and a closed form for every n, K ≥ 4096:
    head ≤ 23 · 32 / 65536 · S/K + 23 · 32 / (m n K) with S/K ≤ 1/28 + 31/(28 K)."""
    out = {}
    for name, (t, m, lim) in bounds.items():
        worst = max(gap(layout(t, m, n, k, cap))["gapShare"] for n in ns for k in ks)
        sk = Q(1, 28) + Q(31, 28 * 4096)
        closed = (Q(84, 100000) + Q(35, 1000000) + Q(36, 100000) * Q(cap, 32) + Q(16064, 1000000) / m
                  + Q(23 * 32, 65536) * sk + Q(23 * 32, m * 4096 * 4096))
        out[name] = {"claimed": float(lim), "scan_max": float(worst), "closed_form": float(closed),
                     "holds": bool(worst <= lim and closed <= lim), "margin": float(lim - max(worst, closed))}
    return out


POINTS = {"pt512x64": (512, 64), "pt512x512": (512, 512), "pt8192": (8192, 8192)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--dedup-points", nargs="*", default=["pt512x64", "pt512x512"])
    ap.add_argument("--small", action="store_true", help="also the smallest layouts each strategy breaks")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rep = {"port": check_port()}
    print("port:", rep["port"])
    assert rep["port"]["mismatches"] == 0
    for name, (t, m) in POINTS.items():
        lay = layout(t, m, args.n, args.k)
        g = gap(lay)
        b = word_range(lay)
        row = {"layout": lay, "gap": {key: float(v) for key, v in g.items() if key != "terms"},
               "gap_terms": {key: float(v) for key, v in g["terms"].items()},
               "table": verdicts(lay, g["words"]),
               "table_bytes": 4 * TILE_WORDS * (2 * b + 1), "word_bound": b}
        det = step0_bound(lay)
        row["dedup_step0"] = det | verdicts(lay, Q(det["saved_regs"]))
        if name in args.dedup_points:
            row["dedup"] = []
            for seed in args.seeds:
                d = dedup(lay, seed)
                row["dedup"].append(d | verdicts(lay, g["words"] - d["distinct_pairs"]))
        rep[name] = row
        print(f"{name}: gap {100 * float(g['gapShare']):.4f}% of stock; table saves "
              f"{100 * row['table']['saved_share_of_stock']:.2f}% (breaks={row['table']['breaks']}, "
              f"{row['table_bytes'] / 2 ** 30:.2f} GiB finite table); dedup step-0 bound saves "
              f"{100 * row['dedup_step0']['saved_share_of_stock']:.4f}% (breaks={row['dedup_step0']['breaks']}, "
              f"{row['dedup_step0']['excess_over_gap']:.2f}× the gap)", flush=True)
        for d in row.get("dedup", []):
            print(f"  dedup seed {d['seed']}: {d['distinct_pairs']} distinct pairs of {int(g['words'])} words, saves "
                  f"{100 * d['saved_share_of_stock']:.3f}% of stock, {d['excess_over_gap']:.1f}× the gap, "
                  f"breaks={d['breaks']}", flush=True)
    if args.small:
        small = {}
        for m, n, k in ((16, 8, 24), (16, 64, 24), (16, 512, 24), (64, 64, 24 + 28 * 7)):
            lay = layout(1, m, n, k)
            g = gap(lay)
            ds = [dedup(lay, seed) for seed in args.seeds]
            small[f"m{m}n{n}K{k}"] = {"table": verdicts(lay, g["words"]),
                                      "dedup": [d | verdicts(lay, g["words"] - d["distinct_pairs"]) for d in ds]}
            print(f"small m{m} n{n} K{k}: table breaks={small[f'm{m}n{n}K{k}']['table']['breaks']}; dedup "
                  + ", ".join(f"{d['distinct_pairs']}/{int(g['words'])} breaks={d['breaks']}"
                              for d in small[f"m{m}n{n}K{k}"]["dedup"]), flush=True)
        rep["small"] = small
    ks = range(4096, 4096 + 28 * 64)
    ns = (4096, 4097, 4099, 4100, 5000, 8192)
    rep["GapAt"] = gap_at({"pt512x64": (512, 64, Q(190, 100000)), "pt512x512": (512, 512, Q(168, 100000)),
                           "pt8192": (8192, 8192, Q(165, 100000))}, 32, ks, ns)
    rep["GapAt320"] = gap_at({"pt512x64": (512, 64, Q(514, 100000)), "pt512x512": (512, 512, Q(492, 100000)),
                              "pt8192": (8192, 8192, Q(489, 100000))}, 320, ks, ns)
    for key in ("GapAt", "GapAt320"):
        print(key, json.dumps(rep[key]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
