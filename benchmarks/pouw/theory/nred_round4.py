"""N-RED round 4 (final 2): the lead `ssn-1l-tag + blc4a-t64-e7` with coins after A_j, under model P read two ways;
R1's check (b) against copies across entries; and the order the coins need.

Two readings of model P (shares of stock, Llama-weighted, chosen data):

- **P, literal** (final 1's model P): moves, predicate-only writes and data-dependent branches are free. Detection
  is then free, so an adversary writes each distinct value of a group once, plus the positions it computes them from
  (one a tile-column-step, one a row-step).
- **P′** (P with N-W1's measured decision price, note 20261029T1045Z-report-np-w1 §1, r20261029-094825-8394): moves
  free, but every instruction that steers work costs at least `price` units a lane, 32 at the floor and 327 measured
  (divergent, if-converted). One decision may merge many words, so the adversary decides a group at once: one decision
  a tile-column-step to read the coins, and one a merging group (the rows that copy the same source). Paying only for
  productive groups makes this a lower bound on the adversary's cost. The prices include N-FLOOR's measured 61.8 and
  71.7 a lane (note 20261029T1037Z-report-np-floor).
- **M** (measured, for N-ADAPT's masked prover only): every copied word also pays N-FLOOR's measured selection and
  on-SM copy. A lag-1 revisit copies nothing, so it pays only the decision prices above.

The lead's words (`lead_instance`) on chosen data: step 1 carries 24 zero data lanes, the tags T(i) against G_j and the
coin bits a₁(i) against the column code C_j, from −β_j. A later step carries a cancel lane (A = 1, B = −c_t; c_t = 1 for
`a2c`, alternating 1, 2 for `a2r`), the row lane (1 + b_t(i))·1 and the coin lanes 1·s_t[tile, j] (odd, uniform in
±(2^e − 1)). The honest chain is the explicit lane product, tile by tile. The adversary's words come from positions:
D0(i, j) + R_t(i) + C_t(tile, j). They must be equal.

The groups whose distinct values are counted:

- `entry`: one entry's steps, R1 check (b)'s grain (final 1's unbounded column, now with the real step 1);
- `xrow`: a tile's rows at one step. The coin lanes are the tile's, so they cancel between its rows. Two rows meet
  when their walks differ by their step-1 difference D0(i, j) − D0(i′, j), which is small: |T·G| ≤ 72, |a₁·C| ≤ 24;
- `tilecol`: a tile's rows and steps (entry and xrow together, plus cross-step meetings);
- `column`: every row and step of one column.

`ordering` checks that the "coins s before A_j" attack of final 1 builds A without reading a coin: only B reads s, so
the requirement is that s and G come after root(B_j), and the row coins b after root(A_j).

The fix for the tile pools is the wide tag (`wide_tags`): balanced base-255 digits of z_i + `WIDE_OFFSET` against
G ∈ [−127, 127]. `blc4a-t64-e7-w127` is N-LIT's weighted lane (note 20261029T1108Z-report-np-lit §1): the same three
coin lanes against A′ = (1, 127, 1). `identical_columns` checks R1's check (b) against chosen identical weight columns.

`mirror_instance` is N-NOISE's frontier design `A/byte1/b13s4/mirror/s8/noc1/M0+tag+wv` (note
20261029T1106Z-report-np-noise §1): a coin byte u_t(i) a row a step against a drawn nonzero int8 window weight W_t[j],
replayed against −W_t[j] in order in the second half, so the final word is A·Bᵀ + code + tags. On chosen identical rows
every lane has cancelled at the last step, and a column's final words differ only by the narrow tag and code, so they
meet in a birthday pool of every row of the column. `salt` adds a per-row ρ_i (Q 2^18) to the last step, removed by
the stock epilogue's `IADD3` row source as N-EXACT's xu does; `salt_bigw` also draws the window weights with
|W| ∈ [64, 127]. `reversed` replays the lanes out and back instead of in order.

`W/…` reads N-FLOOR's `tile-grain` premise (note 20261029T1148Z-report-np-floor §2): an m16n8 MMA is skipped only
whole. A word is held when its value already occurs in its column (free search, as literal P). `W/tile` prices a
skipped tile at N-FLOOR's per-word figures (copy 0.40, the rest off the tensor pipe at 256); `W/whole` counts only
tiles whose 128 words are all held, which is N-FLOOR's own reading.

`mask_blocks` checks N-HIDE's remedy (b) mask on A's 1 KiB leaves against a V chosen before the salt.

`a5_lead` is round 2's A5 search (`nred_round2.a5_step_greedy`) on the lead: with s drawn per call after A_j a row's
step-t vector holds the call's coins for every t ≥ 2, so only step 1 (16 a₁ codes) can be tabled; with s drawn a
window, a row's step-t vectors are ssn_1l's 16·t.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_sweep as ns
import nred_attacks as na
import nred_egress as ne
import nred_final as nf
import nred_round2 as r2

H32 = ns.H32
LAYOUTS = {"blc4a-t64-e7": (64, 7, 3), "blc4a-t32-e7": (32, 7, 3), "blc7a-t64-e7": (64, 7, 6),
           "blc1a-t128-e4": (128, 4, 1), "blc4a-t64-e7-w127": (64, 7, 3)}
LANE_WEIGHTS = {"blc4a-t64-e7-w127": (1, 127, 1)}
PRICES = (32.0, 61.8, 71.7, 327.0)
SELECT_A_WORD, COPY_A_WORD = 71.7, 48.7
SHAPES = {"llama": ns.LLAMA, "v4_dense": ns.V4_DENSE}
STEP1_DATA = ns.DATA_PER_STEP - na.TAG_LANES


def _distinct(x: np.ndarray, axis: int) -> np.ndarray:
    srt = np.sort(x, axis=axis)
    return 1 + (np.diff(srt, axis=axis) != 0).sum(axis=axis)


def _repeated_values(x: np.ndarray, axis: int) -> np.ndarray:
    """How many values occur at least twice along `axis`: the merging groups."""
    srt = np.sort(x, axis=axis)
    eq = np.diff(srt, axis=axis) == 0
    head = np.concatenate([np.zeros_like(np.take(eq, [0], axis=axis)), eq], axis=axis)
    start = np.diff(head.astype(np.int8), axis=axis) == 1
    return start.sum(axis=axis)


def _merge_groups(v: np.ndarray, t: np.ndarray) -> tuple[int, int]:
    """Pools in the rows of v (G × P), each element at step t. An element merges when its value occurred before it in
    its pool (by step, then position); a group is one (pool, value, step) among the merged, which one decision copies.
    Returns (merged, groups)."""
    g_idx = np.repeat(np.arange(v.shape[0]), v.shape[1])
    vf, tf = v.ravel(), t.ravel()
    order = np.lexsort((tf, vf, g_idx))
    gs, vs, ts = g_idx[order], vf[order], tf[order]
    first = np.ones(len(vs), dtype=bool)
    first[1:] = (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])
    mg, mv, mt = gs[~first], vs[~first], ts[~first]
    if len(mg) == 0:
        return 0, 0
    new = np.ones(len(mg), dtype=bool)
    new[1:] = (mg[1:] != mg[:-1]) | (mv[1:] != mv[:-1]) | (mt[1:] != mt[:-1])
    return int((~first).sum()), int(new.sum())


TILE_M, TILE_N = 16, 8
COPY_TILE_WORD, OFF_TENSOR_WORD = 0.40, 256.0
COL_KEY = np.int64(1) << 40


def _tile_saving(held: np.ndarray) -> tuple[int, int, float]:
    """N-FLOOR's `tile-grain` (note 20261029T1148Z-report-np-floor §2): an m16n8 MMA (32 units a word) is skipped only
    whole, its held words copied at 0.40 and the rest produced off the tensor pipe at 256, so it pays from 113 of its
    128 words. `held` (m, n) at one step; returns (whole repeats, paying tiles, units saved)."""
    m, n = held.shape
    r = held.reshape(m // TILE_M, TILE_M, n // TILE_N, TILE_N).sum(axis=(1, 3))
    gain = r * (OFF_TENSOR_WORD - COPY_TILE_WORD) - TILE_M * TILE_N * (OFF_TENSOR_WORD - H32)
    return int((r == TILE_M * TILE_N).sum()), int((gain > 0).sum()), float(gain[gain > 0].sum())


def _whole_share(tg: dict, k: int) -> float:
    """Whole repeats only (N-FLOOR's "whole-tile repeats are the only copies that save"): each saves its 128 words
    at 32 − 0.40, over the instance's entries × k units of stock."""
    return tg["whole_repeats"] * TILE_M * TILE_N * (H32 - COPY_TILE_WORD) / (tg["entries"] * k)


def _isin_cols(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """q[i, j] occurs in x[:, j]."""
    off = np.arange(q.shape[1], dtype=np.int64) * COL_KEY
    xs = np.sort((x + off).ravel())
    qq = (q + off).ravel()
    pos = np.searchsorted(xs, qq).clip(max=len(xs) - 1)
    return (xs[pos] == qq).reshape(q.shape)


def _held_earlier_block(v: np.ndarray) -> np.ndarray:
    """v (m, n) at one step: the word's value occurs in its column in an earlier 16-row block."""
    m, n = v.shape
    blk = np.arange(m) // TILE_M
    _, first, inv = np.unique((v + np.arange(n, dtype=np.int64) * COL_KEY).T.ravel(), return_index=True,
                              return_inverse=True)
    return blk[(first % m)[inv].reshape(n, m).T] < blk[:, None]


def lead_tile_grain(w: np.ndarray) -> dict:
    """Tile grain on the lead's words w (m, s, n): a word is held when its value occurs in its column at an earlier
    step or an earlier 16-row block of its step (free search, as literal P)."""
    m, s, n = w.shape
    _, first, inv = np.unique((w + np.arange(n, dtype=np.int64)[None, None] * COL_KEY).transpose(2, 1, 0).ravel(),
                              return_index=True, return_inverse=True)
    p = first[inv].reshape(n, s, m).transpose(2, 1, 0)
    tp, ip = (p % (s * m)) // m, p % m
    now = np.arange(s)[None, :, None]
    blk = np.arange(m) // TILE_M
    held = (tp < now) | ((tp == now) & (blk[ip] < blk[:, None, None]))
    full = pay = 0
    units = 0.0
    for t in range(s):
        f, p_, u = _tile_saving(held[:, t])
        full, pay, units = full + f, pay + p_, units + u
    return {"tiles": (m // TILE_M) * (n // TILE_N) * s, "entries": m * n, "whole_repeats": full, "paying": pay,
            "held_words": float(held.mean()), "units_an_entry": units / (m * n)}


WIDE_OFFSET = sum(100 * 255 ** lane for lane in range(4))


def wide_tags(m: int) -> np.ndarray:
    """Balanced base-255 digits of z_i + WIDE_OFFSET, each in [−127, 127]: injective, with every lane in use (about
    100 each), against a full-range G. Without the offset a row with |z| ≤ 127 uses one lane, and T·G_j then takes
    only 255 values across columns."""
    i = np.arange(m)
    z = np.where(i % 2 == 0, i // 2 + 1, -(i // 2 + 1)).astype(np.int64) + WIDE_OFFSET
    out = []
    for _ in range(na.TAG_LANES):
        d = ((z + 127) % 255) - 127
        out.append(d)
        z = (z - d) // 255
    assert (z == 0).all()
    return np.stack(out, axis=-1)


def lead_instance(cfg: str, k: int, m: int, n: int, rng: np.random.Generator, pattern: str,
                  tag: str = "narrow") -> dict:
    """`tag` is the lead's (`narrow`: base-13 T against G ∈ [−3, 3]) or the fix (`wide`: base-255 T against
    G ∈ [−127, 127], the served offset removed in the epilogue)."""
    tile, e, lanes = LAYOUTS[cfg]
    s = ns.steps1l(k)
    ntiles = -(-m // tile)
    tile_of = np.arange(m) // tile
    if tag == "wide":
        tags, g = wide_tags(m), rng.integers(-127, 128, (na.TAG_LANES, n))
    else:
        tags, g = na.row_tags(m), rng.integers(-na.G_MAX, na.G_MAX + 1, (na.TAG_LANES, n))
    code = na.code(n)
    a1 = rng.integers(0, 2, (m, na.TAG_LANES))
    beta = na.bias(code, s, True)
    b = rng.integers(0, 2, (m, s - 1))
    odd = np.arange(-(2 ** e - 1), 2 ** e, 2)
    coin = rng.choice(odd, (lanes, ntiles, s - 1, n))
    c = np.where((np.arange(s - 1) % 2 == 1) & (pattern == "a2r"), 2, 1)
    wts = np.array(LANE_WEIGHTS.get(cfg, (1,) * lanes), dtype=np.int64)

    x1 = np.concatenate([np.zeros((m, STEP1_DATA), dtype=np.int64), tags, a1], axis=1)
    y1 = np.concatenate([np.zeros((STEP1_DATA, n), dtype=np.int64), g, code], axis=0)
    honest = np.empty((m, s, n), dtype=np.int64)
    honest[:, 0] = x1 @ y1 - beta[None]
    for tau in range(ntiles):
        rows = tile_of == tau
        xa = np.concatenate([np.ones((rows.sum(), s - 1, 1)), (1 + b[rows])[:, :, None],
                             np.broadcast_to(wts, (rows.sum(), s - 1, lanes))], axis=2).astype(np.int64)
        yb = np.concatenate([np.broadcast_to(-c[:, None, None], (s - 1, 1, n)), np.ones((s - 1, 1, n)),
                             coin[:, tau].transpose(1, 0, 2)], axis=1).astype(np.int64)
        honest[rows, 1:] = honest[rows, :1] + np.cumsum(np.einsum("itl,tln->itn", xa, yb), axis=1)
    assert np.abs(x1).max() <= 127 and np.abs(y1).max() <= 127 and np.abs(odd).max() <= 127 or e > 7

    d0 = tags @ g + a1 @ code - beta[None]
    rpos = np.concatenate([np.zeros((m, 1), dtype=np.int64), np.cumsum(1 + b - c[None], axis=1)], axis=1)
    cpos = np.concatenate([np.zeros((ntiles, 1, n), dtype=np.int64),
                           np.cumsum(np.einsum("l,ltsn->tsn", wts, coin), axis=1)], axis=1)
    w = d0[:, None, :] + rpos[:, :, None] + cpos[tile_of]
    per = m * n

    t4 = w.reshape(ntiles, tile, s, n)
    entry = float((s - _distinct(w, 1)).sum()) / per
    xrow = float((tile - _distinct(t4, 1)).sum()) / per
    tilecol = float((tile * s - _distinct(t4.reshape(ntiles, tile * s, n), 1)).sum()) / per
    column = float((m * s - _distinct(w.reshape(m * s, n), 0)).sum()) / per
    lag1 = float((np.diff(w, axis=1) == 0).sum()) / per
    xrow_groups = float(_repeated_values(t4, 1).sum()) / per
    steps_of = np.broadcast_to(np.arange(s)[None, :, None], (m, s, n))
    tc_v = t4.transpose(0, 3, 1, 2).reshape(ntiles * n, tile * s)
    tc_t = steps_of.reshape(ntiles, tile, s, n).transpose(0, 3, 1, 2).reshape(ntiles * n, tile * s)
    tc_merged, tc_groups = _merge_groups(tc_v, tc_t)
    col_merged, col_groups = _merge_groups(w.transpose(2, 0, 1).reshape(n, m * s),
                                           steps_of.transpose(2, 0, 1).reshape(n, m * s))
    assert tc_merged == round(tilecol * per) and col_merged == round(column * per)

    covered = np.zeros((m, s, n), dtype=bool)
    entry_groups = 0
    for d in range(1, s):
        new = (w[:, d:] == w[:, :-d]) & ~covered[:, d:]
        if new.any():
            covered[:, d:] |= new
            entry_groups += int(new.reshape(ntiles, tile, s - d, n).any(axis=1).sum())
    assert abs(float(covered.sum()) / per - entry) < 1e-9

    span = int(d0.max() - d0.min()) + 1
    key = (rpos[:, :, None] * span + (d0 - d0.min())[:, None, :]).reshape(ntiles, tile, s, n)
    masked = float((tile - _distinct(key, 1)).sum()) / per
    with_source = int(((tile - _distinct(rpos.reshape(ntiles, tile, s), 1))).sum())
    assert masked <= xrow + 1e-12

    return {"equal_to_honest": bool(np.array_equal(w, honest)), "steps": s, "tile": tile,
            "d0_range": [int(d0.min()), int(d0.max())], "masked_sources_a_row": with_source / m,
            "tile_grain": lead_tile_grain(w),
            "saved": {"entry": entry, "xrow": xrow, "tilecol": tilecol, "column": column, "lag1": lag1,
                      "masked": masked},
            "groups": {"entry": entry_groups / per, "xrow": xrow_groups, "tilecol": tc_groups / per,
                       "column": col_groups / per},
            "c_track": (s - 1) / tile}


def shares(inst: dict, k: int, big_n: int) -> dict:
    """Shares of one linear's stock under literal P and P′ at each price. `masked` is N-ADAPT's masked cross-row
    prover (note 20261029T1124Z-report-np-adapt §2): a word is copied from an earlier row of its tile at the same
    walk level whose step-1 word agrees in that column, at one source address a row-step. `M/masked` prices each
    copied word at N-FLOOR's measured selection and on-SM copy (71.7 and 48.7 units beside honest)."""
    s, tile, sv, gr = inst["steps"], inst["tile"], inst["saved"], inst["groups"]
    ptr = (s - 1) / big_n
    out = {}
    for grain in ("entry", "xrow", "tilecol", "column"):
        out[f"P/{grain}"] = H32 * (sv[grain] - inst["c_track"] - ptr) / k
    out["P/lag1"] = H32 * (sv["lag1"] - ptr) / k
    src = inst["masked_sources_a_row"] / big_n
    out["P/masked"] = H32 * (sv["masked"] - src) / k
    out["M/masked"] = H32 * (sv["masked"] * (1 - (SELECT_A_WORD + COPY_A_WORD) / H32) - src) / k
    out["W/tile"] = inst["tile_grain"]["units_an_entry"] / k
    out["W/whole"] = _whole_share(inst["tile_grain"], k)
    for price in PRICES:
        q = price / H32
        decide = (s - 1) / tile
        out[f"P'{price:g}/lag1"] = H32 * (sv["lag1"] - decide - q * decide - ptr) / k
        out[f"P'{price:g}/entry"] = H32 * (sv["entry"] - inst["c_track"] - ptr - q * (decide + gr["entry"])) / k
        for grain in ("xrow", "tilecol", "column"):
            out[f"P'{price:g}/{grain}"] = H32 * (sv[grain] - inst["c_track"] - ptr - q * (decide + gr[grain])) / k
    return out


def weighted(insts: dict, shape: dict) -> dict:
    stock = sum(nn * kk for nn, kk in shape.values())
    keys = next(iter(insts.values()))["share_keys"]
    return {key: sum(nn * kk * shares(insts[kk], kk, nn)[key] for nn, kk in shape.values()) / stock for key in keys}


def runs(seed: int = 97, m: int = 256, n: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    cases = [(cfg, "narrow") for cfg in LAYOUTS] + [(cfg, "wide") for cfg in
                                                     ("blc4a-t64-e7", "blc7a-t64-e7", "blc4a-t64-e7-w127")]
    v4_cfgs = {"blc4a-t64-e7", "blc7a-t64-e7", "blc4a-t64-e7-w127"}
    for cfg, tag in cases:
        for pattern in ("a2c", "a2r"):
            for shape_name, shape in SHAPES.items():
                if shape_name == "v4_dense" and cfg not in v4_cfgs:
                    continue
                insts = {}
                for kk in sorted({kk for _, kk in shape.values()}):
                    inst = lead_instance(cfg, kk, m, n, rng, pattern, tag)
                    inst["share_keys"] = list(shares(inst, kk, n).keys())
                    insts[kk] = inst
                row = {"equal_to_honest": all(v["equal_to_honest"] for v in insts.values()),
                       "shares": weighted(insts, shape),
                       "per_k": {f"k{kk}": {"saved": v["saved"], "groups": v["groups"], "c_track": v["c_track"],
                                            "steps": v["steps"], "d0_range": v["d0_range"],
                                            "tile_grain": v["tile_grain"],
                                            "shares": shares(v, kk, max(nn for nn, k2 in shape.values() if k2 == kk))}
                                 for kk, v in insts.items()}}
                out[f"{cfg}{'+wide' if tag == 'wide' else ''}/{pattern}/{shape_name}"] = row
    return out


def tile_rows(seed: int = 113) -> dict:
    """Tile grain against rows (A2r, Llama): a column's free-search pool fills as rows × steps approach the range its
    words span, so a fix that holds at 256 rows can fail at more."""
    rng = np.random.default_rng(seed)
    stock = sum(nn * kk for nn, kk in ns.LLAMA.values())
    out = {}
    for cfg, tag, m, n in (("blc4a-t64-e7", "wide", 1024, 64), ("blc4a-t64-e7-w127", "narrow", 1024, 64),
                           ("blc4a-t64-e7-w127", "wide", 1024, 64), ("blc4a-t64-e7-w127", "wide", 4096, 32)):
        per = {}
        for kk in sorted({kk for _, kk in ns.LLAMA.values()}):
            tg = lead_instance(cfg, kk, m, n, rng, "a2r", tag)["tile_grain"]
            per[f"k{kk}"] = {**tg, "W/whole": _whole_share(tg, kk), "W/tile": tg["units_an_entry"] / kk}
        out[f"{cfg}+{tag}/m{m}"] = {"per_k": per, "W/whole": sum(
            nn * kk * per[f"k{kk}"]["W/whole"] for nn, kk in ns.LLAMA.values()) / stock}
    return out


def scaling(seed: int = 98, m: int = 128, n: int = 32) -> dict:
    """The lead's per-K shares at one N (2,048) for growing K: the within-entry and cross-row terms grow with S."""
    rng = np.random.default_rng(seed)
    out = {}
    for kk in (2048, 4096, 8192, 16384):
        inst = lead_instance("blc4a-t64-e7", kk, m, n, rng, "a2r", "narrow")
        out[f"k{kk}"] = {"equal_to_honest": inst["equal_to_honest"], "saved": inst["saved"],
                         "groups": inst["groups"], "shares": shares(inst, kk, 2048)}
    return out


def ordering(seed: int = 99, m: int = 128, n: int = 16, k: int = 2048) -> dict:
    """Final 1's `before_s` adversary, rebuilt: A (the cancel lane and one tile indicator a tile) reads no coin, and B
    reads s. Its words equal the honest chain for two different coin draws with one A, so A can be committed first."""
    tile, e, lanes = nf.BLC["blc4a-t64-e7"]
    s = ns.steps1l(k)
    ntiles = -(-m // tile)
    tile_of = np.arange(m) // tile
    a = np.zeros((m, s - 1, 30), dtype=np.int64)
    a[:, :, 0] = 1
    for tau in range(ntiles):
        a[:, :, 1 + tau] = (tile_of == tau)[:, None]
    res = []
    rng = np.random.default_rng(seed)
    odd = np.arange(-(2 ** e - 1), 2 ** e, 2)
    for _ in range(2):
        coin = rng.choice(odd, (lanes, ntiles, s - 1, n)).sum(0)
        b = rng.integers(0, 2, (m, s - 1))
        bw = np.zeros((s - 1, 30, n), dtype=np.int64)
        bw[:, 0, :] = -1
        for tau in range(ntiles):
            bw[:, 1 + tau, :] = -coin[tau]
        inc = np.einsum("mtl,tln->mtn", a, bw) + coin[tile_of] + (1 + b)[:, :, None]
        words = np.concatenate([np.zeros((m, 1, n), dtype=np.int64), np.cumsum(inc, axis=1)], axis=1)
        rpos = np.concatenate([np.zeros((m, 1), dtype=np.int64), np.cumsum(b, axis=1)], axis=1)
        res.append({"equal": bool(np.array_equal(words, np.broadcast_to(rpos[:, :, None], words.shape))),
                    "repeat_share": float((np.diff(words, axis=1) == 0).mean())})
    return {"a_reads_no_coin": True, "draws": res}


def identical_columns(seed: int = 101, m: int = 64, n: int = 32, k: int = 2048, tag: str = "narrow") -> dict:
    """R1 check (b) is per entry (one (row, column)'s distinct values). Cross-column copies are a different grain:
    with every weight column equal, the data is column-constant, but the coin s_t[tile, j] and the tag G_j are per
    column, so a row's words still differ across columns. Count the cross-column merges at the entry grain (what (b)
    would over-credit) against what the coin and G leave."""
    rng = np.random.default_rng(seed)
    tile, e, lanes = LAYOUTS["blc4a-t64-e7"]
    s = ns.steps1l(k)
    ntiles = -(-m // tile)
    tile_of = np.arange(m) // tile
    if tag == "wide":
        tags, g = wide_tags(m), rng.integers(-127, 128, (na.TAG_LANES, n))
    else:
        tags, g = na.row_tags(m), rng.integers(-na.G_MAX, na.G_MAX + 1, (na.TAG_LANES, n))
    code = na.code(n)
    a1 = rng.integers(0, 2, (m, na.TAG_LANES))
    beta = na.bias(code, s, True)
    b = rng.integers(0, 2, (m, s - 1))
    odd = np.arange(-(2 ** e - 1), 2 ** e, 2)
    coin = rng.choice(odd, (lanes, ntiles, s - 1, n)).sum(0)
    d0 = tags @ g + a1 @ code - beta[None]
    rpos = np.concatenate([np.zeros((m, 1), dtype=np.int64), np.cumsum(1 + b, axis=1)], axis=1)
    cpos = np.concatenate([np.zeros((ntiles, 1, n), dtype=np.int64), np.cumsum(coin, axis=1)], axis=1)
    w = d0[:, None, :] + rpos[:, :, None] + cpos[tile_of]            # natural coins, every weight column equal
    dup = float((n - _distinct(w, 2)).sum()) / w.size
    step0 = float((n - _distinct(w[:, :1], 2)).sum()) / w[:, :1].size
    return {"cross_column_dup_share": dup, "step0_dup_share": step0,
            "tags_nonzero": bool((tags != 0).any(axis=1).all())}


MIRROR_POOL_TILE = 64
MIRROR_WALK_ROWS = 256


def mirror_instance(k: int, m: int, n: int, rng: np.random.Generator, variant: str = "narrow") -> dict:
    """Chosen identical rows (zero data) through mirror. Step t ≤ h = ⌈S/2⌉ adds u_t(i)·W_t[j]; step h + d replays
    lane d against −W_d[j], and for odd S the middle lane h is replayed in the last step as well. Returns the merges
    and merging groups a pool (a whole column; a tile of 64 rows), at the final step and at the steps before it,
    and A2r's within-entry revisits on the same words."""
    s = ns.steps1l(k)
    h = (s + 1) // 2
    if variant == "wide":
        tags, g = wide_tags(m), rng.integers(-127, 128, (na.TAG_LANES, n))
    else:
        tags, g = na.row_tags(m), rng.integers(-na.G_MAX, na.G_MAX + 1, (na.TAG_LANES, n))
    a1 = rng.integers(0, 2, (m, na.TAG_LANES))
    d0 = tags @ g + a1 @ na.code(n)
    u = rng.integers(-128, 128, (m, h))
    wv = rng.integers(64 if variant.endswith("bigw") else 1, 128, (h, n)) * rng.choice([-1, 1], (h, n))
    rho = rng.integers(-(1 << 17), 1 << 17, m) if variant.startswith("salt") else np.zeros(m, dtype=np.int64)
    word = d0.astype(np.int64)
    acc = {f"{p}/{when}": [0, 0] for p in ("column", "tile") for when in ("final", "interior")}
    entry_seen = []
    rev = variant == "reversed"
    target = d0 + rho[:, None]
    by_interior = np.zeros((m, n), dtype=bool)
    prev = None
    tg = {when: [0, 0, 0.0] for when in ("final", "interior")}
    for t in range(s):
        lane = t if t < h else (h - 1 - (t - h) if rev else t - h)
        word = word + (1 if t < h else -1) * u[:, lane, None] * wv[lane][None]
        if t == s - 1 and s % 2 == 1:
            mid = 0 if rev else h - 1
            word = word - u[:, mid, None] * wv[mid][None]
        out_word = word + rho[:, None] if t == s - 1 else word
        when = "final" if t == s - 1 else "interior"
        held = _held_earlier_block(out_word)
        if when == "interior":
            by_interior |= _isin_cols(target, out_word)
            if prev is not None:
                held |= _isin_cols(out_word, prev)
        else:
            held |= by_interior
        for slot, val in enumerate(_tile_saving(held)):
            tg[when][slot] += val
        prev = out_word
        tiles = out_word.reshape(m // MIRROR_POOL_TILE, MIRROR_POOL_TILE, n)
        acc[f"column/{when}"][0] += int((m - _distinct(out_word, 0)).sum())
        acc[f"column/{when}"][1] += int(_repeated_values(out_word, 0).sum())
        acc[f"tile/{when}"][0] += int((MIRROR_POOL_TILE - _distinct(tiles, 1)).sum())
        acc[f"tile/{when}"][1] += int(_repeated_values(tiles, 1).sum())
        entry_seen.append(out_word[:MIRROR_WALK_ROWS])
    final_clean = bool(np.array_equal(word, d0))
    walk = np.stack(entry_seen, axis=1)
    per = m * n
    return {"steps": s, "final_word_is_code_and_tags": final_clean,
            "saved": {key: v[0] / per for key, v in acc.items()}, "groups": {key: v[1] / per for key, v in acc.items()},
            "a2r_revisits": float((s - _distinct(walk, 1)).sum()) / (walk.shape[0] * n),
            "d0_distinct_a_column": float(_distinct(d0, 0).mean()),
            "tile_grain": {when: {"tiles": (m // TILE_M) * (n // TILE_N) * (1 if when == "final" else s - 1),
                                  "entries": per, "whole_repeats": v[0], "paying": v[1], "units_an_entry": v[2] / per}
                           for when, v in tg.items()}}


def mirror_shares(inst: dict, k: int, big_n: int, m: int) -> dict:
    """Literal P and P′ shares of one linear's stock, a pool kind and step class. A decision a column-step (pool) and a
    row pointer a row-step (big_n), as for the lead."""
    s, sv, gr = inst["steps"], inst["saved"], inst["groups"]
    out = {}
    for pool, rows in (("column", m), ("tile", MIRROR_POOL_TILE)):
        for when, nsteps in (("final", 1), ("interior", s - 1)):
            key = f"{pool}/{when}"
            base = sv[key] - nsteps / rows - nsteps / big_n
            out[f"P/{key}"] = H32 * base / k
            for price in PRICES:
                out[f"P'{price:g}/{key}"] = H32 * (base - price / H32 * (gr[key] + nsteps / rows)) / k
    out["P/a2r_entry"] = H32 * (inst["a2r_revisits"] - (s - 1) / big_n) / k
    for when in ("final", "interior"):
        out[f"W/tile/{when}"] = inst["tile_grain"][when]["units_an_entry"] / k
        out[f"W/whole/{when}"] = _whole_share(inst["tile_grain"][when], k)
    return out


def mirror_runs(seed: int = 103, n: int = 64) -> dict:
    """Mirror at Llama's shapes for N-CARD's row counts (64, 512 and 8,192 rows)."""
    rng = np.random.default_rng(seed)
    out = {}
    for variant in ("narrow", "wide", "salt", "salt_bigw", "reversed"):
        for m in {"reversed": (64,), "salt_bigw": (8192,)}.get(variant, (64, 512, 8192)):
            insts = {kk: mirror_instance(kk, m, n, rng, variant) for kk in sorted({kk for _, kk in ns.LLAMA.values()})}
            stock = sum(nn * kk for nn, kk in ns.LLAMA.values())
            keys = mirror_shares(next(iter(insts.values())), 2048, 2048, m).keys()
            sh = {key: sum(nn * kk * mirror_shares(insts[kk], kk, nn, m)[key] for nn, kk in ns.LLAMA.values()) / stock
                  for key in keys}
            out[f"{variant}/m{m}"] = {"shares": sh, "final_word_is_code_and_tags": all(
                v["final_word_is_code_and_tags"] for v in insts.values()),
                "per_k": {f"k{kk}": {key: v[key] for key in ("steps", "saved", "groups", "a2r_revisits",
                                                             "d0_distinct_a_column", "tile_grain")}
                          for kk, v in insts.items()}}
    return out


MASK_HEADER_BYTES = 173


def mask_blocks(seed: int = 107, salts: int = 4) -> dict:
    """N-HIDE's remedy (b) for A's 1 KiB pieces (note 20261029T1052Z-report-np-hide, 11:25Z): V' = V ⊕ Z, Z's u64
    word t = h1[0] ⊕ t, framed after the leaf's 173 header bytes. Z_t ⊕ Z_{t+16Δ} = t ⊕ (t + 16Δ) reads no salt, so V
    chosen before the salt as V_t = C_{t mod 16} ⊕ t gives V'_t = C_{t mod 16} ⊕ h1[0] for every salt: the leaf's
    value-only SHA-512 blocks are one message and share one schedule. `per_block` mixes word ⌊t/16⌋ of h1 instead
    (V has 8 blocks of 16 words, h1 has 8 words)."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 1 << 63, 16, dtype=np.uint64)
    t = np.arange(128, dtype=np.uint64)
    v = c[t % 16] ^ t
    out = {}
    for form in ("counter", "per_block"):
        distinct, full = set(), 0
        for _ in range(salts):
            h1 = rng.integers(0, 1 << 63, 8, dtype=np.uint64)
            z = (h1[0] if form == "counter" else h1[t // 16]) ^ t
            msg = rng.integers(0, 256, MASK_HEADER_BYTES, dtype=np.uint8).tobytes() + (v ^ z).astype(">u8").tobytes()
            blocks = [msg[128 * b:128 * b + 128] for b in range(len(msg) // 128) if 128 * b >= MASK_HEADER_BYTES]
            full = len(blocks)
            distinct.add(len(set(blocks)))
        out[form] = {"value_only_blocks_a_leaf": full, "distinct_messages_a_leaf": sorted(distinct)}
    return out


def a5_lead() -> dict:
    """Round 2's A5 search on the lead. `call`: s drawn per call after A_j, so a row's step-t vector holds the call's
    coins for t ≥ 2 and only step 1's 16 a₁ codes can be tabled. `window`: s drawn once a window, known to advice, so a
    row's step-t vectors are ssn_1l's 16·t."""
    call = r2.a5_step_greedy(designs=("lead_call_s",), vectors=lambda _d, t: 16 if t == 1 else float("inf"))
    window = r2.a5_step_greedy(designs=("ssn_1l",))
    return {"call": {key: v for key, v in call.items() if isinstance(v, dict)},
            "window": {key: v for key, v in window.items() if isinstance(v, dict)}}


def check(rep: dict) -> list[dict]:
    res = []

    def ok(name, cond, detail=""):
        res.append({"check": name, "pass": bool(cond), "detail": detail})

    r = rep["runs"]
    ok("every instance's words equal the honest lane product", all(v["equal_to_honest"] for v in r.values()))
    ok("scaling instances equal the honest lane product",
       all(v["equal_to_honest"] for v in rep["scaling"].values()))
    for p in ("a2c", "a2r"):
        sh = r[f"blc4a-t64-e7/{p}/llama"]["shares"]
        ok(f"{p}: literal P, cross-row meetings in a tile exceed 1% on the lead (the coin lanes cancel between "
           "a tile's rows, and T·G spans ~92 values for 64 rows)", sh["P/xrow"] > 0.01, sh["P/xrow"])
        ok(f"{p}: P′ at the floor (32), cross-row still exceeds 1% (one decision merges a group)",
           sh["P'32/xrow"] > 0.01, sh["P'32/xrow"])
        ok(f"{p}: literal P, within-entry revisits exceed 1% (final 1's unbounded column, with the real step 1)",
           sh["P/entry"] > 0.01, sh["P/entry"])
        ok(f"{p}: literal P, lag-1 alone stays under 1% (final 1's 0.28%)", sh["P/lag1"] < 0.01, sh["P/lag1"])
        ok(f"{p}: P′ at the floor, within-entry and lag-1 don't pay on Llama",
           max(sh["P'32/entry"], sh["P'32/lag1"]) <= 0, [sh["P'32/entry"], sh["P'32/lag1"]])
        v4 = r[f"blc4a-t64-e7/{p}/v4_dense"]["shares"]
        ok(f"{p}: P′ at the floor, within-entry pays above 1% on V4 dense (K 7,168: revisits grow as √S)",
           v4["P'32/entry"] > 0.01, v4["P'32/entry"])
        w = r[f"blc4a-t64-e7+wide/{p}/llama"]["shares"]
        ok(f"{p}: the wide tag (G ∈ [−127, 127]) closes cross-row under literal P (< 0.1%)",
           w["P/xrow"] < 0.001, w["P/xrow"])
        ok(f"{p}: the wide tag leaves within-entry unchanged (the tag is constant in t; check (b) closes it)",
           abs(w["P/entry"] - sh["P/entry"]) < 0.005, [w["P/entry"], sh["P/entry"]])
    wide_pp = [r[f"{c}+wide/{p}/{sh}"]["shares"][f"P'32/{g}"] for c in ("blc4a-t64-e7", "blc7a-t64-e7")
               for p in ("a2c", "a2r") for sh in SHAPES for g in ("xrow", "tilecol", "column")]
    ok("wide tag: P′ at the floor pays at no tile or column grain (blc4a and blc7a, Llama and V4 dense)",
       max(wide_pp) <= 0, max(wide_pp))
    b7 = [r[f"blc7a-t64-e7+wide/{p}/v4_dense"]["shares"]["P'32/entry"] for p in ("a2c", "a2r")]
    b4 = r["blc4a-t64-e7+wide/a2r/v4_dense"]["shares"]["P'32/entry"]
    ok("wide tag, V4 dense, P′ at the floor: blc7a's 6 coin lanes hold within-entry under 1% with no check (b); "
       "blc4a's 3 pay > 1%", max(b7) < 0.01 < b4, [b7, b4])
    for p in ("a2c", "a2r"):
        base, wl = r[f"blc4a-t64-e7/{p}/llama"]["shares"], r[f"blc4a-t64-e7-w127/{p}/llama"]["shares"]
        ok(f"{p}: N-LIT's weighted lane A′ = (1, 127, 1) leaves cross-row as it is (the coin still cancels between a "
           "tile's rows) and closes within-entry under literal P", abs(wl["P/xrow"] - base["P/xrow"]) < 0.02
           and wl["P/entry"] < 0 < base["P/entry"], [wl["P/xrow"], base["P/xrow"], wl["P/entry"]])
    ww = [r[f"blc4a-t64-e7-w127+wide/{p}/{sh}"]["shares"] for p in ("a2c", "a2r") for sh in SHAPES]
    ok("wide tag + weighted lane: P′ at the floor pays at no grain (Llama and V4 dense)",
       max(v[f"P'32/{g}"] for v in ww for g in ("lag1", "entry", "xrow", "tilecol", "column")) <= 0)
    ok("wide tag + weighted lane, literal P: lag-1, within-entry and cross-row close; the free-search tile-column "
       "pool still pays > 1% (N-LEAN §4)", max(v[f"P/{g}"] for v in ww for g in ("lag1", "entry", "xrow")) < 0.001
       and min(v["P/tilecol"] for v in ww) > 0.01, [[v["P/tilecol"], v["P/column"]] for v in ww])
    lead_m = [r[f"blc4a-t64-e7/{p}/llama"]["shares"] for p in ("a2c", "a2r")]
    wide_m = [r[f"blc4a-t64-e7+wide/{p}/llama"]["shares"] for p in ("a2c", "a2r")]
    ok("N-ADAPT's masked cross-row prover in this harness: > 5% of stock under bare P on the lead (its ≈ 8.9%), "
       "< 0 at N-FLOOR's measured selection and copy prices, < 0.1% under bare P with the wide tag",
       min(v["P/masked"] for v in lead_m) > 0.05 and max(v["M/masked"] for v in lead_m) < 0
       and max(v["P/masked"] for v in wide_m) < 0.001,
       [[v["P/masked"], v["M/masked"]] for v in lead_m] + [v["P/masked"] for v in wide_m])
    e4s = [r[f"blc1a-t128-e4/{p}/llama"]["shares"] for p in ("a2c", "a2r")]
    e7s = [r[f"blc4a-t64-e7/{p}/llama"]["shares"] for p in ("a2c", "a2r")]
    ok("measured decision prices (61.8, 71.7 a lane): e4's lag-1 pays under 1% but above 0.1%; e7's pays at no "
       "measured price", 0.001 < min(v[f"P'{q}/lag1"] for v in e4s for q in ("61.8", "71.7"))
       and max(v[f"P'{q}/lag1"] for v in e4s for q in ("61.8", "71.7")) < 0.01
       and max(v[f"P'{q}/lag1"] for v in e7s for q in ("61.8", "71.7")) < 0,
       [[v["P'61.8/lag1"], v["P'71.7/lag1"]] for v in e4s])
    e4 = r["blc1a-t128-e4/a2c/llama"]["shares"]
    ok("P′ at the floor: blc1a-t128-e4's lag-1 still pays (> 1%), so e7 is what closes lag-1",
       e4["P'32/lag1"] > 0.01, e4["P'32/lag1"])
    ok("the tilecol floor counts at least the entry and xrow merges together",
       all(v["per_k"][kk]["saved"]["tilecol"] + 1e-9 >= max(v["per_k"][kk]["saved"]["entry"],
                                                              v["per_k"][kk]["saved"]["xrow"])
           for v in r.values() for kk in v["per_k"]))
    ok("smaller tiles shrink cross-row but don't close it under literal P (t32 still > 1%)",
       0.01 < r["blc4a-t32-e7/a2r/llama"]["shares"]["P/xrow"] < r["blc4a-t64-e7/a2r/llama"]["shares"]["P/xrow"])
    o = rep["ordering"]
    ok("ordering: one coin-free A gives words equal to the honest chain for two coin draws (B reads s)",
       all(d["equal"] for d in o["draws"]))
    ic, icw = rep["identical_columns"]["narrow"], rep["identical_columns"]["wide"]
    ok("identical columns, narrow tag: > 1% of a row's words equal across columns, mostly early steps "
       "(check (b), per entry, doesn't see them)", ic["cross_column_dup_share"] > 0.01 and ic["tags_nonzero"],
       [ic["cross_column_dup_share"], ic["step0_dup_share"]])
    ok("identical columns, wide tag: < 0.1% equal across columns",
       icw["cross_column_dup_share"] < 0.001, icw["cross_column_dup_share"])
    sc = rep["scaling"]
    ok("scaling: within-entry grows with K under literal P and under P′ at the floor",
       sc["k16384"]["shares"]["P/entry"] > sc["k2048"]["shares"]["P/entry"]
       and sc["k16384"]["shares"]["P'32/entry"] > sc["k2048"]["shares"]["P'32/entry"])
    mr = rep["mirror"]
    ok("mirror: every final word is code + tags (all lanes cancel)",
       all(v["final_word_is_code_and_tags"] for v in mr.values()))
    nar = mr["narrow/m8192"]["shares"]
    ok("mirror, narrow tag, 8,192 rows: the final-step column pool pays > 1% under P′ at the floor (the tag spans "
       "~80 values for 8,192 rows; N-NOISE's adjacent-tag bound is 0.18%)", nar["P'32/column/final"] > 0.01,
       nar["P'32/column/final"])
    ok("mirror, narrow tag: the final-step leak needs the column pool; a 64-row tile pays < 1%",
       nar["P'32/tile/final"] < 0.01, nar["P'32/tile/final"])
    wid, sal = mr["wide/m8192"]["shares"], mr["salt/m8192"]["shares"]
    ok("mirror, wide tag, 8,192 rows: the final-step column pool < 1% under P′, below the narrow tag's",
       wid["P'32/column/final"] < min(0.01, nar["P'32/column/final"]), wid["P'32/column/final"])
    ok("mirror, per-row salt ρ_i in the final word: the final-step column pool ≤ 0.1% under P′",
       sal["P'32/column/final"] <= 0.001, sal["P'32/column/final"])
    big = mr["salt_bigw/m8192"]["shares"]
    ok("mirror, 8,192 rows, P′: the interior column pools pay > 0.1% with |W| ≥ 1 (a step whose pool holds one "
       "small-weight lane); drawing |W| ≥ 64 cuts them, and salt + |W| ≥ 64 + A2r's walk stays < 1%",
       sal["P'32/column/interior"] > 0.001 and big["P'32/column/interior"] < sal["P'32/column/interior"]
       and big["P'32/column/final"] + max(big["P'32/column/interior"], 0) + big["P/a2r_entry"] < 0.01,
       [sal["P'32/column/interior"], big["P'32/column/interior"], big["P/a2r_entry"]])
    ok("mirror's in-order replay: within-entry revisits < 1%; a reversed (out-and-back) replay retraces the path, "
       "> 10%", mr["narrow/m64"]["shares"]["P/a2r_entry"] < 0.01
       and mr["reversed/m64"]["shares"]["P/a2r_entry"] > 0.1,
       [mr["narrow/m64"]["shares"]["P/a2r_entry"], mr["reversed/m64"]["shares"]["P/a2r_entry"]])
    mk = rep["mask"]
    ok("N-HIDE's counter mask (Z_t = h1[0] ⊕ t): V chosen before the salt makes a leaf's 7 value-only blocks one "
       "SHA-512 message for every salt", mk["counter"]["value_only_blocks_a_leaf"] == 7
       and mk["counter"]["distinct_messages_a_leaf"] == [1], mk["counter"])
    ok("a per-block chaining word (Z_t = h1[⌊t/16⌋] ⊕ t) leaves the same V's 7 blocks 7 distinct messages",
       mk["per_block"]["distinct_messages_a_leaf"] == [7], mk["per_block"])
    a5 = rep["a5"]
    ok("A5 on the lead: s drawn per call after A_j saves 0 at every m",
       max(v["gap_share_of_stock"] for v in a5["call"].values()) == 0)
    ok("A5 on the lead: s drawn once a window reopens it (ssn_1l's 16·t vectors), > 1% at m 64",
       a5["window"]["ssn_1l/m64"]["gap_share_of_stock"] > 0.01, a5["window"]["ssn_1l/m64"]["gap_share_of_stock"])
    narrow_w = [r[f"{c}/{p}/llama"]["shares"] for c in ("blc4a-t64-e7", "blc1a-t128-e4") for p in ("a2c", "a2r")]
    wide_w = [r[f"{c}+wide/{p}/{sh}"]["shares"] for c in ("blc4a-t64-e7", "blc7a-t64-e7") for p in ("a2c", "a2r")
              for sh in SHAPES]
    ww_w = [r[f"blc4a-t64-e7-w127+wide/{p}/{sh}"]["shares"] for p in ("a2c", "a2r") for sh in SHAPES]
    ok("tile grain (N-FLOOR), whole repeats only: the narrow-tag lead still saves > 1% on Llama, e4 and e7 alike",
       min(v["W/whole"] for v in narrow_w) > 0.01, [v["W/whole"] for v in narrow_w])
    ok("tile grain, whole repeats only: the wide tag leaves e7 < 1% (blc4a, blc7a; Llama and V4 dense), and wide + "
       "(1, 127, 1) repeats no tile", max(v["W/whole"] for v in wide_w) < 0.01 and max(v["W/whole"] for v in ww_w) == 0,
       [v["W/whole"] for v in wide_w])
    ok("tile grain at N-FLOOR's per-word prices (a tile pays from 113 of 128 held): the narrow lead > 1%; wide + "
       "(1, 127, 1) pays no tile", min(v["W/tile"] for v in narrow_w) > 0.01 and max(v["W/tile"] for v in ww_w) == 0,
       [v["W/tile"] for v in narrow_w + wide_w])
    ok("mirror under tile grain, 8,192 rows: the final step's tiles repeat whole (> 1% of stock, narrow and wide "
       "tags); no interior tile repeats; the per-row salt takes the final step under 0.01%",
       min(nar["W/whole/final"], wid["W/whole/final"]) > 0.01
       and max(v["shares"]["W/whole/interior"] for v in mr.values()) == 0
       and max(sal["W/whole/final"], big["W/whole/final"]) < 1e-4,
       [nar["W/whole/final"], wid["W/whole/final"], sal["W/whole/final"]])
    tr = rep["tile_rows"]
    ok("tile grain against rows (free search): at 1,024 rows the wide tag alone repeats whole tiles (> 1% on Llama) "
       "and (1, 127, 1) with the narrow tag > 0.1%; wide + (1, 127, 1) stays < 0.01% at 1,024 and 4,096 rows",
       tr["blc4a-t64-e7+wide/m1024"]["W/whole"] > 0.01 and tr["blc4a-t64-e7-w127+narrow/m1024"]["W/whole"] > 0.001
       and max(tr[f"blc4a-t64-e7-w127+wide/m{m}"]["W/whole"] for m in (1024, 4096)) < 1e-4,
       {key: v["W/whole"] for key, v in tr.items()})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rep = {"runs": runs(), "scaling": scaling(), "ordering": ordering(),
           "identical_columns": {t: identical_columns(tag=t) for t in ("narrow", "wide")},
           "mirror": mirror_runs(), "a5": a5_lead(), "mask": mask_blocks(), "tile_rows": tile_rows()}
    lead = rep["runs"]["blc4a-t64-e7/a2r/llama"]["shares"]
    mir = rep["mirror"]["narrow/m8192"]["shares"]
    rep["periods"] = {"P/xrow": ne.periods(lead["P/xrow"]), "P'32/xrow": ne.periods(lead["P'32/xrow"]),
                      "P/entry": ne.periods(lead["P/entry"]),
                      "mirror_P'32/column/final_m8192": ne.periods(mir["P'32/column/final"]),
                      "W/whole": ne.periods(lead["W/whole"]),
                      "mirror_W/whole/final_m8192": ne.periods(mir["W/whole/final"])}
    rep["checks"] = check(rep)
    for key, v in rep["runs"].items():
        sh = v["shares"]
        print(f"{key:32s} eq={v['equal_to_honest']}  " + "  ".join(
            f"{g}={100 * sh[g]:.2f}%" for g in ("P/lag1", "P/entry", "P/xrow", "P/tilecol", "P/column",
                                                "P'32/lag1", "P'32/entry", "P'32/xrow", "P'32/tilecol",
                                                "P'32/column", "P'71.7/lag1", "P/masked", "M/masked", "W/tile",
                                                "W/whole")))
    for key, v in rep["mirror"].items():
        sh = v["shares"]
        print(f"mirror {key:16s} " + "  ".join(
            f"{g}={100 * sh[g]:.3f}%" for g in ("P/column/final", "P'32/column/final", "P'32/tile/final",
                                                "P/column/interior", "P'32/column/interior", "P/a2r_entry",
                                                "W/tile/final", "W/whole/final", "W/whole/interior")))
    for key, v in rep["tile_rows"].items():
        print(f"tile rows {key}: W/whole={100 * v['W/whole']:.4f}%  " + "  ".join(
            f"{kk}: {t['whole_repeats']}/{t['tiles']} held={t['held_words']:.3f}" for kk, t in v["per_k"].items()))
    print("A5: " + "  ".join(f"{k}={100 * v['gap_share_of_stock']:.2f}%"
                             for part in rep["a5"].values() for k, v in part.items()))
    for kk, v in rep["scaling"].items():
        print(f"scaling {kk}: " + "  ".join(f"{g}={100 * v['shares'][g]:.2f}%" for g in
                                            ("P/entry", "P/xrow", "P'32/entry", "P'32/xrow")))
    npass = sum(c["pass"] for c in rep["checks"])
    for c in rep["checks"]:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] != "" else ""))
    print(f"{npass}/{len(rep['checks'])} checks")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
