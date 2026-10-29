"""COUNTER (LEAN program): copy-and-patch on the lead under measured prices instead of `TileGrain`.

`--head` picks the statement (`counter_pad.HEADS`): ebe1fcd968's step 0 carries the coin lanes, which rows of one
parity in one coin tile share, so the copies below keep their matches and lose step 0's cross-parity ones.

`TileGrain` charges a write to any word of a tile as the whole tile's 128 registers. At W1 a one-word patch is one
register. This module prices copy-and-patch on the statement's own words (narrow tag, `-rot`, `stepsFor`, whole
tiles, decode as 16 rows) the way the measurements price it, and reports the copy term `G` would need in place of
`coinBranchGap` at the three points and at decode.

The words are `nred_round4.lead_words` with the head's steps (`1 + ⌈(K − 21)/28⌉` at ebe1fcd968), checked against the Lean
port (`counter_statement`), on chosen data: identical rows with the cancel lane (`a2c`, `a2r`) or zero data (`zero`,
what decode's 15 padding rows carry). Rows of one parity in one 64-row coin tile share their coin lanes, so two of
them differ by `D0(i) − D0(i′)` plus their tag bits' walk: those are the only copies that match.

**Copy-and-patch at m16n8** (`lanes`). A tile (16 × 8 at one step) is copied whole from a source of the same columns
and step, its rows shifted by an even `δ` inside the coin tile (`--partial`: any even `δ` whose source rows meet the
coin tile, the rows outside it unmatched, which gives decode's one tile its in-tile sources), at 0.40 a word. Its lanes are the m16n8 fragment's
32 (rows g and g + 8, columns 2c and 2c + 1). A lane keeps its `k` matching words only under a decision of price
`p`, and patches the rest at `patch` a word, so it saves `max(0, k · patch − p)` over patching all four. The prover
knows every match for free (an upper bound for it). Skipping inside a `wgmma` block (64 rows × 64 columns, the
least full-rate shape) moves that block's other words off `wgmma`, to `alt` a word. Each block is copied only if
that pays. `r8` breaks 64 × 8 strips instead (`wgmma` runs at full rate at `N ≥ 64`, so B's columns can be regrouped
around a strip), and `r1` each tile alone at no cost to the rest, an upper bound on any arrangement.

- `uniform`: one source a tile (a warp-uniform choice, unpriced). `own`: each lane its own source, unpriced (a bound).
  `own+load`: each lane's source a data-dependent load at W1's `other` (32) on top of its decision.
- Price sets (units, 32 = a `wgmma` register):
  - `free`: `p = 0`, `patch = alt = 32`. This is model W with W1's one-register patch and no `TileGrain`: the
    choices are free because the trace depends on the coins.
  - `floor`: `p = 32` (`H32`'s floor for the decision's register), `patch = alt = 32`.
  - `user`: `p` measured, `patch = 32`, `alt = 48.78` (`mma.sync` m16n8k32 at 65.6% of `wgmma`).
  - `measured`: `p` measured, `patch = alt = 34.68`. Round 8's two-source IADD writes the patched words and the
    block's other words from positions (D0 + R + C), which is cheaper than `mma.sync`.
  - Measured decision prices: 61.8 and 71.7 a lane (N-FLOOR), 65.1 compare-and-select (Round 7, `h100Measured`),
    327 a divergent lane (N-W1).
- `p_star`: the least `p` at which `floor` saves nothing, and the least at which it saves at most `coinBranchGap`.

**Word grains** (`grains`): `nred_round4.shares` (literal P, and P′ with one decision a tile-column-step and one a
merging group) at the floor and the measured prices. `M′` adds Round 8's IADD price to the words P′ writes.

**Whole tiles** (`block`, `segment`): `counter_lead`'s whole-tile repeats and row-segment copies at 16 rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_lead as cl
import counter_pad as cp
import counter_statement as cs
import nred_attacks as na
import noise_sweep as ns
import nred_round4 as r4

LEAD = "blc4a-t64-e7-rot"
H32 = 32.0
COPY = r4.COPY_TILE_WORD
MMA_SYNC = 32 / 0.656
IADD = 34.68
DECISIONS = (61.8, 65.1, 71.7, 327.0)
OWN_LOAD = 32.0
GRID = (32.0, 40.0, 48.0, 56.0, 61.8, 64.0, 65.1, 71.7, 80.0, 96.0, 112.0, 127.0)
COIN_BRANCH = 84 / 100000

r4.PRICES = (32.0, *DECISIONS)
_lead_words, _lead_draws = r4.lead_words, r4.lead_draws


def _step0_lanes(cfg: str, m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    tile, e, lanes_n = r4.LAYOUTS[cfg]
    return rng.choice(np.arange(-(2 ** e - 1), 2 ** e, 2), (lanes_n, -(-m // tile), n))


def _draws(cfg: str, k: int, m: int, n: int, rng: np.random.Generator, tag: str) -> dict:
    dr = _lead_draws(cfg, k, m, n, rng, tag)
    return dr | {"coin0": _step0_lanes(cfg, m, n, rng)} if cp.HEAD["step0_lanes"] else dr


def _words(cfg: str, k: int, pattern: str, dr: dict, code: np.ndarray, honest_check: bool = True) -> dict:
    """`lead_words`, with `zero` (no cancel lane: κ = 0, what a padding row carries) shifted from `a2c` by `t`, and
    from ebe1fcd968 step 0's coin lanes, a constant of each word's row and column that every later step sums."""
    lw = _lead_words(cfg, k, "a2c" if pattern == "zero" else pattern, dr, code, honest_check)
    if pattern == "zero":
        t = np.arange(lw["w"].shape[1], dtype=np.int64)
        lw = lw | {"w": lw["w"] + t[None, :, None], "rpos": lw["rpos"] + t[None, :]}
    if "coin0" in dr:
        m = lw["w"].shape[0]
        lane0 = np.einsum("il,iln->in", lw["wsel"][lw["par"]], dr["coin0"][:, np.arange(m) // lw["tile"]]
                          .transpose(1, 0, 2).astype(np.int64))
        lw = lw | {"w": lw["w"] + lane0[:, None, :], "d0": lw["d0"] + lane0}
    return lw


COLUMNS: np.ndarray | None = None


def _col_code(n: int) -> np.ndarray:
    """The Lean's `colCode` (balanced base-13 digits of `j + 1`), shape (4, n), in place of the ssn code, at
    columns `0 .. n − 1` or at `COLUMNS`."""
    cols = range(n) if COLUMNS is None else COLUMNS[:n]
    assert len(cols) == n
    return np.array([cs.col_code(int(j)) for j in cols], dtype=np.int64).reshape(n, 4).T


def spread_columns(n: int, big_n: int) -> np.ndarray:
    """`n / 8` blocks of 8 columns spread evenly over `0 .. big_n − 1`. Low columns have small code digits and match
    more often, so the first `n` columns overstate a `big_n`-column call."""
    nb = n // 8
    starts = np.arange(nb) * (big_n // 8) // nb * 8
    return (starts[:, None] + np.arange(8)).ravel()


r4.lead_words, r4.lead_draws, na.code = _words, _draws, _col_code


def use_head(name: str) -> None:
    cp.use_head(name)
    ns.steps1l = cs.steps_for


def check_port(m: int = 128, n: int = 16, k: int = 300, seed: int = 5) -> dict:
    """`lead_words` against the Lean `word` (κ = 0 at step 0, κ = −c_t later), up to the column bias, whose rounding
    differs and which every word of a column shares."""
    rng = np.random.default_rng(seed)
    dr = r4.lead_draws(LEAD, k, m, n, rng, "narrow")
    code = na.code(n)
    tags_ok = all(list(dr["tags"][i]) == cs.row_tag(i) for i in range(m))
    code_ok = all(list(code[:, j]) == cs.col_code(j) for j in range(n))
    s = cs.steps_for(k)
    mism = 0
    for pattern in ("a2c", "a2r", "zero"):
        w = r4.lead_words(LEAD, k, pattern, dr, code)["w"]
        c = np.where((np.arange(s - 1) % 2 == 1) & (pattern == "a2r"), 2, 1) * (pattern != "zero")
        beta = na.bias(code, s, True)
        for i in range(0, m, 7):
            for j in range(n):
                acc = 0
                for t in range(s):
                    if t == 0:
                        acc += sum(cs.row_tag(i)[q] * int(dr["g"][q, j]) for q in range(4))
                        acc += sum(int(dr["a1"][i, q]) * cs.col_code(j)[q] for q in range(4))
                        if cp.HEAD["step0_lanes"]:
                            acc += sum(cs.rot_weight(i, ell) * int(dr["coin0"][ell, i // 64, j]) for ell in range(3))
                    else:
                        acc += -int(c[t - 1]) + 1 + int(dr["b"][i, t - 1]) + sum(
                            cs.rot_weight(i, ell) * int(dr["coin"][ell, i // 64, t - 1, j]) for ell in range(3))
                    mism += int(acc != w[i, t, j] + beta[j])
    return {"head": cp.HEAD["name"], "tags_equal": tags_ok, "code_equal": code_ok, "word_mismatches": mism,
            "steps": s}


PARTIAL = False


def _window(rb: int, m: int, tile: int = 64) -> tuple[int, int]:
    lo = (16 * rb // tile) * tile
    return lo, min(lo + tile, m)


def _offsets(rb: int, m: int, tile: int = 64) -> list[int]:
    """Even row shifts to a source inside the coin tile. `PARTIAL`: any shift whose source rows meet the coin tile,
    the rows outside it unmatched (copies inside one 16-row tile, the only ones decode has)."""
    lo, hi = _window(rb, m, tile)
    first, last = (lo - 16 * rb - 15, hi - 16 * rb) if PARTIAL else (lo - 16 * rb, hi - 16 * rb - 15)
    return [d for d in range(first, last) if d and d % 2 == 0]


def lane_counts(w: np.ndarray, rb: int, d: int) -> np.ndarray:
    """Matches a lane of the m16n8 fragment, tile rows 16 rb.. against rows shifted by d: (8 g, s, n/8, 4 c)."""
    rows = np.arange(16 * rb, 16 * rb + 16)
    lo, hi = _window(rb, w.shape[0])
    inside = (rows + d >= lo) & (rows + d < hi)
    eq = np.zeros((16, *w.shape[1:]), dtype=np.int8)
    eq[inside] = w[rows[inside]] == w[rows[inside] + d]
    m16, s, n = eq.shape
    pair = eq.reshape(16, s, n // 8, 4, 2).sum(axis=4)
    return pair[:8] + pair[8:]


def lanes(w: np.ndarray, k: int) -> dict:
    m, s, n = w.shape
    rb_n, cb_n = m // 16, n // 8
    prices = (0.0, *GRID, *DECISIONS, *(p + OWN_LOAD for p in DECISIONS))
    keys = sorted({(v, p, patch) for v in ("uniform", "own") for p in prices for patch in (H32, IADD)})
    ls = {key: np.zeros((rb_n, s, cb_n)) for key in keys}
    best_matches = np.zeros((rb_n, s, cb_n), dtype=np.int64)
    sources = 0
    for rb in range(rb_n):
        offs = _offsets(rb, m)
        sources += len(offs)
        if not offs:
            continue
        counts = np.stack([lane_counts(w, rb, d) for d in offs])
        tot = counts.sum(axis=(1, 4))
        best_matches[rb] = tot.max(axis=0)
        for v, p, patch in keys:
            gain = np.maximum(0.0, counts * patch - p)
            if v == "uniform":
                ls[(v, p, patch)][rb] = gain.sum(axis=(1, 4)).max(axis=0)
            else:
                ls[(v, p, patch)][rb] = gain.max(axis=0).sum(axis=(0, 3))
    stock = m * n * k

    def total(v: str, p: float, patch: float, alt: float, rc: int) -> float:
        """`rc` 64: a broken 64 × 64 `wgmma` block. 8: a broken 64 × 8 strip (B's columns regrouped, the rest at
        `N ≥ 64`). 1: each tile on its own with no cost to the rest, an upper bound on any arrangement."""
        lsv = ls[(v, p, patch)]
        copy_cost = 128 * COPY + 128 * patch - lsv
        broken = np.minimum(copy_cost, 128 * alt)
        gain = 128 * H32 - broken
        rr = 1 if rc == 1 else min(4, rb_n)
        cc = max(1, min(rc // 8, cb_n))
        g = gain[: (rb_n // rr) * rr, :, : (cb_n // cc) * cc]
        g = g.reshape(rb_n // rr, rr, s, cb_n // cc, cc).sum(axis=(1, 4))
        return float(np.maximum(0.0, g).sum() / stock)

    out = {"free/uniform": total("uniform", 0.0, H32, H32, 64), "free/own": total("own", 0.0, H32, H32, 64),
           "floor/uniform": total("uniform", 32.0, H32, H32, 64), "floor/own": total("own", 32.0, H32, H32, 64)}
    for p in DECISIONS:
        for v, pv in (("uniform", p), ("own", p), ("own+load", p + OWN_LOAD)):
            src = "own" if v.startswith("own") else v
            for rc in (64, 8, 1):
                out[f"user/{v}/p{p:g}/r{rc}"] = total(src, pv, H32, MMA_SYNC, rc)
                out[f"measured/{v}/p{p:g}/r{rc}"] = total(src, pv, IADD, IADD, rc)
    curve = {f"{v}/p{p:g}/r{rc}": total(v, p, H32, H32, rc) for v in ("uniform", "own") for p in GRID
             for rc in (64, 8, 1)}
    star = {}
    for v in ("uniform", "own"):
        for rc in (64, 8, 1):
            zero = [p for p in GRID if curve[f"{v}/p{p:g}/r{rc}"] <= 0]
            within = [p for p in GRID if curve[f"{v}/p{p:g}/r{rc}"] <= COIN_BRANCH]
            star[f"{v}/r{rc}"] = {"saves_nothing_from": min(zero) if zero else None,
                                  "within_coinBranchGap_from": min(within) if within else None}
    copy_term = {}
    for p in DECISIONS[:3]:
        realizable = [f"user/{v}/p{p:g}/r{rc}" for v in ("uniform", "own+load") for rc in (64, 8)]
        bound = [f"user/{v}/p{p:g}/r1" for v in ("uniform", "own+load")]
        copy_term[f"p{p:g}"] = {"achieved": max(out[c] for c in realizable),
                                "achieved_by": max(realizable, key=out.get),
                                "bound": max(out[c] for c in bound), "bound_unpriced_sources": out[f"user/own/p{p:g}/r1"]}
    hist = np.bincount(best_matches.ravel(), minlength=17)[:33]
    return {"shares": out, "copy_term": copy_term, "floor_curve": curve, "p_star": star,
            "sources_per_row_block": sources / max(rb_n, 1),
            "best_matches_hist": {int(i): int(c) for i, c in enumerate(hist) if c}, "tiles": int(rb_n * s * cb_n)}


def whole(w: np.ndarray, k: int) -> dict:
    m, s, n = w.shape
    blk = seg = 0
    for j0 in range(0, n, 8):
        sub = np.ascontiguousarray(w[:, :, j0:j0 + 8])
        blk += cl._block_repeats(sub, 16)
        seg += cl._segment_repeats(sub, 16)
    stock = m * n * k
    return {"block_repeats": blk, "segment_tiles": seg, "tiles": (m // 16) * s * (n // 8),
            "block_share": blk * 128 * (H32 - COPY) / stock,
            "segment_share": seg * (128 * (H32 - COPY) - 16 * H32) / stock}


def grains(m: int, n: int, k: int, seed: int, pattern: str, big_n: int) -> dict:
    tile = r4.LAYOUTS[LEAD]
    if m < tile[0]:
        r4.LAYOUTS[LEAD] = (m, tile[1], tile[2])
    try:
        inst = r4.lead_instance(LEAD, k, m, n, np.random.default_rng(seed), pattern, "narrow")
    finally:
        r4.LAYOUTS[LEAD] = tile
    sh = r4.shares(inst, k, big_n)
    s = inst["steps"]
    extra = (IADD - H32) / k
    for price in (32.0, *DECISIONS):
        for g in ("xrow", "tilecol", "column", "entry"):
            sh[f"M'{price:g}/{g}"] = sh[f"P'{price:g}/{g}"] - extra * (s - inst["saved"][g] + inst["c_track"]
                                                                      + (s - 1) / big_n)
    return {"equal_to_honest": inst["equal_to_honest"], "steps": s, "saved": inst["saved"], "shares": sh}


def instance(m: int, n: int, k: int, seed: int, pattern: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    row = r4.lead_draws(LEAD, k, m, 0, rng, "narrow")
    code = na.code(n)
    s = cs.steps_for(k)
    tile, e, lanes_n = r4.LAYOUTS[LEAD]
    ntiles = -(-m // tile)
    odd = np.arange(-(2 ** e - 1), 2 ** e, 2)
    w = np.empty((m, s, n), dtype=np.int64)
    eq = True
    for j0 in range(0, n, 8):
        g = rng.integers(-na.G_MAX, na.G_MAX + 1, (na.TAG_LANES, 8))
        coin = rng.choice(odd, (lanes_n, ntiles, s - 1, 8))
        dr = {"tags": row["tags"], "g": g, "a1": row["a1"], "b": row["b"], "coin": coin}
        if cp.HEAD["step0_lanes"]:
            dr["coin0"] = rng.choice(odd, (lanes_n, ntiles, 8))
        lw = r4.lead_words(LEAD, k, pattern, dr, code[:, j0:j0 + 8])
        eq &= bool(lw["equal_to_honest"])
        w[:, :, j0:j0 + 8] = lw["w"]
    assert eq
    return w


POINTS = {"decode": 16, "pt512x64": 64, "pt512x512": 512, "pt8192": 8192}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", nargs="+", default=list(POINTS))
    ap.add_argument("--ks", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--cols", type=int, default=256)
    ap.add_argument("--cols-8192", type=int, default=32)
    ap.add_argument("--k-8192", type=int, nargs="+", default=[4096])
    ap.add_argument("--patterns", nargs="+", default=["a2c", "a2r", "zero"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--big-n", type=int, default=4096)
    ap.add_argument("--no-grains", action="store_true")
    ap.add_argument("--head", choices=sorted(cp.HEADS), default="ebe1fcd968")
    ap.add_argument("--partial", action="store_true", help="sources that meet the coin tile in part (`_offsets`)")
    ap.add_argument("--spread", type=int, default=None, help="sample the columns across this many (`spread_columns`)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    global PARTIAL, COLUMNS
    PARTIAL = args.partial
    use_head(args.head)
    rep: dict = {"config": vars(args), "port": check_port()}
    print("port:", rep["port"], flush=True)
    assert rep["port"]["word_mismatches"] == 0 and rep["port"]["tags_equal"] and rep["port"]["code_equal"]
    for name in args.points:
        m = POINTS[name]
        n = args.cols_8192 if m == 8192 else args.cols
        COLUMNS = spread_columns(n, args.spread) if args.spread else None
        for k in (args.k_8192 if m == 8192 else args.ks):
            for pattern in args.patterns:
                for seed in args.seeds:
                    key = f"{name}/K{k}/{pattern}/s{seed}"
                    w = instance(m, n, k, seed, pattern)
                    r = {"m": m, "n": n, "lanes": lanes(w, k), "whole": whole(w, k)}
                    del w
                    if not args.no_grains:
                        r["grains"] = grains(m, n, k, seed + 1000, pattern, args.big_n)
                    rep[key] = r
                    lo, ct = r["lanes"]["shares"], r["lanes"]["copy_term"]["p61.8"]
                    best_meas = max(v for c, v in lo.items() if c.startswith("measured"))
                    print(f"{key}: free {100 * lo['free/uniform']:.4f}% / own {100 * lo['free/own']:.4f}%, floor "
                          f"{100 * lo['floor/uniform']:.4f}% / {100 * lo['floor/own']:.4f}%, user p61.8 achieved "
                          f"{100 * ct['achieved']:.5f}% ({ct['achieved_by']}) bound {100 * ct['bound']:.5f}% "
                          f"(unpriced sources {100 * ct['bound_unpriced_sources']:.5f}%), measured-IADD max "
                          f"{100 * best_meas:.5f}%, block {r['whole']['block_repeats']}, "
                          f"segment {r['whole']['segment_tiles']}, matches {r['lanes']['best_matches_hist']}",
                          flush=True)
                    if "grains" in r:
                        g = r["grains"]["shares"]
                        print("   grains: " + "  ".join(f"{c}={100 * g[c]:.3f}%" for c in
                                                       ("P/xrow", "P/column", "P'32/xrow", "P'61.8/xrow",
                                                        "P'65.1/xrow", "M'32/xrow", "M'65.1/xrow", "M/masked",
                                                        "P'32/column", "W/whole")), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)


if __name__ == "__main__":
    main()
