"""COUNTER (LEAN program): N-PREM's revised lead guarantee, at tile grain, searched for strategies that falsify it.

The revised statement (`cursor/compute-research-lean-statement-4457` at 0c9b58b686) holds whole MMA tiles: `holds` is a
predicate on `Fin tileWords → ℤ`, `Accepted` asks every committed tile `tileOf L κ ω rb cb t` held whole, a write sets a
whole tile (`OneInstructionMinimal` charges `tileWords` a write), `σ.pre` is zero past `L.mem = 2^34` tiles
(`pre_supp`), the stream is raw coins (`RawStream`), and the gap gains `tableGap`, `2 · mem · tiles · tileWords /
2^100`. This module reuses `counter_statement`'s port of `word` and the gap's first five terms and adds `tableGap`.

`tableGap` prices a pre-coin tile matching a committed tile at `2^-100`, which assumes every committed tile is spread
over far more than `mem` values. `tileOf` zeroes the positions outside the call, so a tile of the last row block and
last column block holds `min(16, m − 16 rb) · min(8, n − 8 cb)` live words. A tile with one live word takes at most
`2 · word_range + 1` values, a few million, well under `mem`. The **cover** strategy fills `σ.pre` with every such tile
(the live word's every value, zeros elsewhere), so each one-live-word committed tile is held at prefix 0 at every coin
value and never written. At m = n = 1 every tile is one: the strategy writes nothing, does the honest compressions, and
saves every word.

It meets every premise of the revised `G` with `Real` its singleton, `can = Real`, `pr = h100Nominal`, κ = 0, the honest
root and messages, egress 0 and a `RawStream` stream: `TileGrain` holds at prefix 0, `LaneSemantics` is vacuous (no
writes) and `OneInstructionMinimal` holds with the compressions' registers. It is reported against `32 · (honestRegs −
gapRegs)`, rounding the compressions up to whole registers. `block` counts committed tiles equal whole to an earlier
committed tile of the same call, at sampled coins: the only free copy the revised model leaves.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction as Q

import numpy as np

import counter_statement as cs

MEM = 2 ** 34


def blocks(lay: dict) -> tuple[int, int]:
    return (lay["m"] + 15) // 16, (lay["n"] + 7) // 8


def gap_tile(lay: dict) -> dict:
    """`Gap.lean` at 0c9b58b686: the five earlier terms and `tableGap`, exact."""
    g = cs.gap(lay)
    rb, cb = blocks(lay)
    table = Q(2 * MEM * rb * cb * lay["S"] * cs.TILE_WORDS, 2 ** 100)
    total = g["gap"] + table
    return g | {"gap": total, "gapShare": total / g["stock"], "table": table}


def live_words(lay: dict, rb: int, cb: int) -> int:
    return min(16, lay["m"] - 16 * rb) * min(8, lay["n"] - 8 * cb)


def cover(lay: dict) -> dict:
    """The cover strategy's saving: the live words of every tile whose values fit `mem`, held from `σ.pre`.

    A tile with `L` live words takes at most `(2 · word_range + 1)^L` values. The tables need at most the sum of that
    over the covered tiles' step-independent shapes, which must fit `mem`; tiles of one shape share their slots."""
    rb_n, cb_n = blocks(lay)
    vals = 2 * cs.word_range(lay) + 1
    shapes: dict[tuple[int, int], int] = {}
    for rb in range(rb_n):
        rows = min(16, lay["m"] - 16 * rb)
        for cb in range(cb_n):
            cols = min(8, lay["n"] - 8 * cb)
            shapes[(rows, cols)] = shapes.get((rows, cols), 0) + 1
    saved = 0
    slots = 0
    covered_tiles = 0
    for (rows, cols), count in shapes.items():
        live = rows * cols
        if live * math.log2(vals) <= math.log2(MEM):
            slots += vals ** live
            saved += live * count * lay["S"]
            covered_tiles += count * lay["S"]
    fits = slots <= MEM
    return {"value_count": vals, "slots": slots, "fits_mem": fits, "covered_tiles": covered_tiles,
            "saved_regs": saved if fits else 0}


def verdict(lay: dict, saved_regs: int) -> dict:
    g = gap_tile(lay)
    work = cs.PRICE * (g["honest"] - saved_regs + (math.ceil(g["sha"]) - g["sha"]))
    bound = cs.PRICE * (g["honest"] - g["gap"])
    return {"work": float(work), "bound": float(bound), "breaks": bool(work < bound),
            "saved_regs": saved_regs, "gap_regs": float(g["gap"]), "table_gap_regs": float(g["table"]),
            "words": int(g["words"]), "excess_over_gap": float(Q(saved_regs) / g["gap"])}


def block_dedup(lay: dict, seed: int) -> dict:
    """Committed tiles equal whole to another of the call, at one coin value, by an exact 128-word key."""
    om = cs.draw(lay, np.random.default_rng(seed))
    rb_n, cb_n = blocks(lay)
    rows = np.arange(lay["m"])
    seen: set[bytes] = set()
    repeats = 0
    total = 0
    for cb in range(cb_n):
        cols = np.arange(8 * cb, min(8 * cb + 8, lay["n"]))
        w = cs.words_at(lay, om, rows, cols)
        for rb in range(rb_n):
            block = np.zeros((lay["S"], 16, 8), dtype=np.int64)
            sub = w[16 * rb:16 * rb + 16]
            block[:, :sub.shape[0], :sub.shape[2]] = sub.transpose(1, 0, 2)
            for t in range(lay["S"]):
                key = block[t].tobytes()
                total += 1
                if key in seen:
                    repeats += 1
                else:
                    seen.add(key)
    return {"seed": seed, "tiles": total, "repeats": repeats}


def step0_bits(lay: dict, rb: int, cb: int) -> float:
    """log2 of the coin configurations a step-0 committed tile reads: `G[q, j]` for the block's columns where some row's
    tag digit `q` is nonzero, `a₁[i, q]` for its rows where some column's code digit `q` is nonzero. The coins are
    uniform, so each value the tile reaches has probability at least `2^-bits`."""
    rows = range(16 * rb, min(16 * rb + 16, lay["m"]))
    cols = range(8 * cb, min(8 * cb + 8, lay["n"]))
    g = sum(len(cols) for q in range(4) if any(cs.row_tag(i)[q] for i in rows))
    a = sum(len(rows) for q in range(4) if any(cs.col_code(j)[q] for j in cols))
    return g * math.log2(7) + a


def table_lemma(lay: dict) -> dict:
    """`tableGap` read as a lemma: a pre-coin table saves at most `tableGap` registers a coin value.

    Tile (0, 0) at step 0 reads `G[0:2, 0:8]` and `a₁[0:16, 0:2]` (rows 0–15 and columns 0–7 have tag and code digits 2
    and 3 zero), and determines `G[0, 0:8]`, `a₁[0:12, 0]` and `a₁[0:12, 1]` (row 10's tag is 6 and the bits are
    below it), so it reaches at least `7^8 · 2^24 ≈ 2^46.5` values, each with probability at least `2^-bits`. A table of
    `mem` of them holds the committed tile with probability at least `mem · 2^-bits`, saving 128 registers then."""
    rb_n, cb_n = blocks(lay)
    bits = {(rb, cb): step0_bits(lay, rb, cb) for rb in range(min(rb_n, 12)) for cb in range(min(cb_n, 12))}
    low = min(bits.values())
    hit = Q(MEM) * Q(1, 7 ** 16 * 2 ** 32) if math.isclose(low, 16 * math.log2(7) + 32) else None
    saved = Q(cs.TILE_WORDS) * hit if hit is not None else None
    g = gap_tile(lay)
    return {"min_bits": low, "tiles_below_100_bits": sum(b < 100 for b in bits.values()),
            "tiles_examined": len(bits), "hit_lower_bound": float(hit) if hit else None,
            "saved_regs_lower_bound": float(saved) if saved else None, "table_gap_regs": float(g["table"]),
            "lemma_breaks": bool(saved is not None and saved > g["table"]),
            "lemma_excess": float(saved / g["table"]) if saved else None,
            "G_gap_regs": float(g["gap"])}


def k_threshold(m: int, n: int, k_max: int = 20000) -> int | None:
    """The least K at which the cover strategy breaks `G` at m × n."""
    for k in range(24, k_max):
        lay = cs.layout(1, m, n, k)
        c = cover(lay)
        if c["fits_mem"] and verdict(lay, c["saved_regs"])["breaks"]:
            return k
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--block-points", nargs="*", default=["small"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rep: dict = {"port": cs.check_port()}
    print("port:", rep["port"])
    assert rep["port"]["mismatches"] == 0

    rep["cover"] = {}
    for m, n, k in ((1, 1, 24), (1, 1, 700), (1, 1, 4096), (1, 1, 8192), (1, 9, 4096), (2, 1, 4096),
                    (17, 4097, 4096), (16, 8, 4096)):
        lay = cs.layout(1, m, n, k)
        c = cover(lay)
        v = verdict(lay, c["saved_regs"])
        rep["cover"][f"m{m}n{n}K{k}"] = c | v
        print(f"cover m{m} n{n} K{k}: {c['covered_tiles']} tiles covered in {c['slots']} slots "
              f"(fits={c['fits_mem']}), saves {c['saved_regs']} of {v['words']} words, gap {v['gap_regs']:.3f} "
              f"(tableGap {v['table_gap_regs']:.3e}), breaks={v['breaks']}", flush=True)
    rep["k_threshold_m1n1"] = k_threshold(1, 1)
    print("least K breaking at m 1 n 1:", rep["k_threshold_m1n1"], flush=True)

    for name, (t, m) in cs.POINTS.items():
        for n in (4096, 4097):
            lay = cs.layout(t, m, n, 4096)
            c = cover(lay)
            v = verdict(lay, c["saved_regs"])
            g = gap_tile(lay)
            rep[f"{name}_n{n}"] = c | v | {"gapShare": float(g["gapShare"])}
            print(f"{name} n{n}: cover saves {c['saved_regs']} (tiles {c['covered_tiles']}), gap share "
                  f"{100 * float(g['gapShare']):.5f}%, breaks={v['breaks']}", flush=True)

    for name, (t, m) in cs.POINTS.items():
        lay = cs.layout(t, m, 4096, 4096)
        rep[f"table_lemma_{name}"] = table_lemma(lay)
        r = rep[f"table_lemma_{name}"]
        print(f"table lemma {name}: step-0 tiles below 100 bits {r['tiles_below_100_bits']} of "
              f"{r['tiles_examined']} examined, min {r['min_bits']:.2f} bits; table saves ≥ "
              f"{r['saved_regs_lower_bound']:.3e} regs a coin value vs tableGap {r['table_gap_regs']:.3e} "
              f"({r['lemma_excess']:.2f}×, lemma breaks={r['lemma_breaks']}); G's gap {r['G_gap_regs']:.0f}",
              flush=True)
    rng = np.random.default_rng(7)
    lay = cs.layout(1, 16, 8, 24)
    draws = 1 << 18
    om = {"b": np.zeros((16, 1), dtype=np.int64), "a1": None, "G": None, "s": np.zeros((1, 1, 3, 8), dtype=np.int16)}
    keys = set()
    for _ in range(draws):
        om["a1"] = rng.integers(0, 2, (16, 4))
        om["G"] = rng.integers(0, 7, (4, 8)) - 3
        keys.add(cs.words_at(lay, om, np.arange(16), np.arange(8))[:, 0, :].tobytes())
    rep["tile00_distinct"] = {"draws": draws, "distinct": len(keys)}
    print(f"tile (0, 0) at step 0: {len(keys)} distinct of {draws} draws", flush=True)

    rep["block"] = {}
    sizes = {"small": [(16, 8, 24 + 28 * 3), (64, 64, 24 + 28 * 7), (64, 512, 24 + 28 * 7), (512, 64, 24 + 28 * 7)],
             "pt512x64": [(64, 4096, 4096)], "pt512x512": [(512, 512, 4096)]}
    for point in args.block_points:
        for m, n, k in sizes[point]:
            lay = cs.layout(1, m, n, k)
            ds = [block_dedup(lay, s) for s in args.seeds]
            rep["block"][f"m{m}n{n}K{k}"] = ds
            print(f"block m{m} n{n} K{k}: " + ", ".join(f"{d['repeats']} repeats of {d['tiles']}" for d in ds),
                  flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
