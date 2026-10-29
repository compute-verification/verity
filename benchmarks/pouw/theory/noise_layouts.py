"""Noise layouts whose zero-data words are not row-plus-column, against COUNTER's provers and their generalizations.

At zero data the lead's words are x + R_t(i) + C_t(class(i), j) (`hash_counter.separable`), so a credited function
that decomposes over those sums is cheap. This module builds lane-form instances (U, V, w0, as `hash_counter.Draw`) for
a grammar of layouts and runs every prover on them, bit for bit against the honest commitment:

- `lead`: the lead's lanes (COUNTER's `lead:rot`); `lead-salt` adds full-width salts ρ_i + γ_j; `lead-wide` adds
  TENSOR's wide rank-2 step-0 term U(i)·V(j) (16- and 15-bit live coins, removed exactly in the epilogue); `lead-tile`
  draws the column coins a τ-row tile (τ = m rows; below 16 rows a tile needs its own lanes, which the stock reads);
- `bilinear`: ℓ byte row coins a (row, step) times nonzero int8 column weights a (column, step) on the same lanes, so a
  step's noise has rank ℓ (coins that multiply); served, so it fails T3;
- `mirror`, `mirror-pi`: the second half replays the first half's coins negated, in order or in a coin-drawn order
  (`commit_space.free_order`), with r0 more lanes at step 0 removed at the last step; exact;
- `mirror-pat`: mirror-π over the lead's column coins (odd e7, three lanes), with each row's lane weights drawn a
  (row, step) from K public patterns (coins indexed by a row hash), so a step's noise has rank 3 from 4-bit row coins.

The commitments (`hash_counter.Rule`, all 32-bit words at every step): (a) the multiply–xor fold `mulxor` (from N₀ = 0)
and `mulxor:live` (from a live word N₀); (b) the thin fold `thin` (main's `thinFold`: XOR3 and IADD3 steps alternating
over word pairs, sealed with the final word); (c) the quadratic sketch `sq1` (Σ_t w_t², one value an output), the keyed
`sq` (Σ r_t w_t²), and NH: `nh:1` (Σ_k w_2k·w_2k+1 an output), `nh:16` and `nh2:8` (values over 16 entries of a row).

Provers beyond COUNTER's:
- `edge_any`: the multiply–xor fold's first step from free terms on any lane-form layout, N₁ = x·k₁ + Σ_l (U_l·k₁)·V_l,
  one IMMA (and an IADD3 for a per-cell offset); a live N₀ closes it.
- `decomp_sq`: a (keyed) quadratic sketch over separable words, Σ r(x + R + C)², with the cross term Σ_t (r_t C_t)·R_t
  as `hash_counter.digit_dot` chains.
- `pairsum`: the thin fold over separable words. An IADD3 step adds w_t + w_{t+1} = 2x + (R_t + R_{t+1}) + (C_t +
  C_{t+1}), two IADD3s of shared sums, so neither word is written; an XOR3 step forms its two words (an IADD3 each).
- `mma_pairsum`: the thin fold on any layout. At zero data a step's data lanes are free, so an IADD3 step is one MMA
  over its two steps' coin lanes (the first doubled) from σ + 2w_{t−1}, and neither word is written.
- `pack_bound` (xor and thin, any layout): packing L cells in b-bit lanes of g registers needs, a step, one IMAD a
  bilinear lane a packed register (U_l(i)·V′_l with V′ the packed column weights), so the words cost ≥ ℓ·g/L writes;
  with full-width salts b = 32 and L = g. A lower bound on the packed prover, from the instance's word span.
- `rank_dp` (sketches over non-separable words): the cheapest split of a value's steps into honest words, edges and
  middle blocks decomposed as Σ_k f_k(i)·g_k(group) (`commit_space.partition_search`, generalized to values over
  several entries), at ranks measured mod a prime on the layout's zero-data words. Features are free (the in-class
  bound) or tiled: a row feature is shared by a 64-column tile's values and a column feature by a 64-row tile.

`wire` gives each layout's coin entropy a token, n_max and network term (N-EGRESS's term scaled by n_max and W_mac,
as `commit_space.network_terms`), and T3. `feed` sets the deployed lead's coin feed (layout.h's u32 words, as the
certifier's `release_bytes` counts them) beside its coins' entropy and N-EGRESS's spread layout (`egress.py`'s
`lane_counts`, which coins only ⌈3·S·m/64⌉ lane slots of a tile under 64 rows), a token at each point; `lever_wire`
prices the wide step-0 term's coins a token. `gap_claimed` is the gap against credit = cost(H), `gap_stock` against credit =
stock, both over stock; `gap_claimed_tile_priced` prices COUNTER's `pack` with its column terms shared by a coin tile's
rows of a class (`pack@tile`) rather than the call's.

Run: `python3 -B noise_layouts.py OUT.json [--quick] [--part grid|ranks|all]` (from this directory; numpy only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exact_kit"))
import commit_space as cs  # noqa: E402
import hash_counter as hc  # noqa: E402
import noise_space as nsp  # noqa: E402
import noise_sweep as ns  # noqa: E402
from exact_kit.checks import Checks  # noqa: E402

M32 = hc.M32
PRIME = (1 << 31) - 1
#: The lead's column coins: three odd e7 lanes, weights by row parity (`hash_counter.Draw`, `lead:rot`).
ODD7 = np.arange(-127, 128, 2)
#: mirror-pat's K public row patterns over the three column lanes.
PATTERNS = 16
#: The fold-free honest cost a word beyond its MMA write, and the bits sealed an output.
AGG_OPS = {"mulxor": 2.0, "mulxor:live": 2.0, "thin": 0.5, "sq1": 1.0, "sq": 2.0, "nh:1": 0.5, "nh:16": 0.5,
           "nh2:8": 0.5, "xor": 0.5}
AGG_SEAL_BITS = {"mulxor": 64, "mulxor:live": 64, "thin": 64, "sq1": 32, "sq": 32, "nh:1": 32, "nh:16": 2, "nh2:8": 2,
                 "xor": 64}
#: The design h each commitment is priced at in its lane's note (this module doesn't change h: the coins enter by
#: cp.async and the salts by the accumulator's initial value).
AGG_H = {"mulxor": "1.57–2.94 (N-FLOOR)", "mulxor:live": "1.57–2.94 (N-FLOOR)", "thin": "2.31–3.50 (ASSUME)",
         "sq1": "2.37–4.55 (N-NOISE)", "sq": "≈ 2× sq1's commit ops", "nh:1": "1.84–3.52 (estimate)",
         "nh:16": "COUNTER's 0.72 harness basis", "nh2:8": "COUNTER's 0.71 harness basis", "xor": "0.6–4.1 (ASSUME)"}


@dataclass(frozen=True)
class Layout:
    family: str
    lanes: int = 4
    r0: int = 0
    salt: bool = False
    tile: int = 64

    def __str__(self):
        if self.family.startswith("lead"):
            return f"{self.family}" + (f"-{self.tile}" if self.family == "lead-tile" else "")
        return f"{self.family}-l{self.lanes}" + (f"-w{self.r0}" if self.r0 else "") + ("-salt" if self.salt else "")

    @property
    def separable(self) -> bool:
        return self.family.startswith("lead")

    @property
    def coin_lanes(self) -> int:
        """Lanes that carry a coin product a step (ℓ; three for mirror-pat; the lead's three column lanes)."""
        return 3 if self.family == "mirror-pat" or self.separable else self.lanes

    @property
    def q_eff(self) -> int:
        """Coin lanes a k32 step, as stock reads them: a tile under 16 rows (an MMA fragment's) needs its own lanes; the
        step-0 block's r0 lanes are two steps' and left out."""
        if self.separable:
            return 2 + 3 * max(1, 16 // self.tile)
        return self.coin_lanes


def _salts(m: int, n: int, g: np.random.Generator) -> np.ndarray:
    """ρ_i + γ_j as integers (the wrap mod 2³² is the word's; a rank over the integers needs them unwrapped)."""
    return g.integers(0, 1 << 32, m)[:, None] + g.integers(0, 1 << 32, n)[None, :]


def _wide(m: int, n: int, g: np.random.Generator) -> np.ndarray:
    """TENSOR's wide step-0 term U·V, rank 2: U 16-bit and V 15-bit signed live coins a row and a column."""
    return g.integers(-(1 << 15), 1 << 15, (m, 2)) @ g.integers(-(1 << 14), 1 << 14, (2, n))


def _finish(d, rule: hc.Rule, g: np.random.Generator, S: int):
    d.P = rule.positions(S)
    d.r = g.integers(0, 1 << 32, len(d.P), dtype=np.int64)
    d.n0 = int(g.integers(0, 1 << 32))
    d.sbox = g.permutation(256).astype(np.int64)
    c = np.einsum("itl,jtl->ijt", d.U, d.V) + np.einsum("itl,jtl->ijt", d.A, d.B)
    d.c = c & M32
    d.w = (d.w0[:, :, None] + np.cumsum(c, axis=2)) & M32
    d.rho = np.ones(S, np.int64)
    return d


def build(lay: Layout, m: int, n: int, S: int, rule: hc.Rule, seed: int):
    """(inst, d): a `hash_counter.Inst` and a Draw-like instance of the layout at zero data."""
    if lay.separable:
        m = lay.tile if lay.family == "lead-tile" else m
        inst = hc.Inst(m, n, S, 5, "zero", "add", seed, "lead:rot")
        d = hc.Draw(inst, rule)
        d.n0 = int(np.random.default_rng(seed + 977).integers(0, 1 << 32))
        if lay.family in ("lead-salt", "lead-wide"):
            g = np.random.default_rng(seed + 991)
            add = _salts(m, n, g) if lay.family == "lead-salt" else _wide(m, n, g)
            d.w0 = (d.w0 + add) & M32
            d.w = (d.w0[:, :, None] + np.cumsum(np.einsum("itl,jtl->ijt", d.U, d.V), axis=2)) & M32
        return inst, d
    g = np.random.default_rng(seed)
    U, V, w0 = lanes(lay, m, n, S, g)
    d = SimpleNamespace(U=U, V=V, w0=w0 & M32, A=np.zeros((m, S, 32 - U.shape[2]), np.int64),
                        B=np.zeros((n, S, 32 - U.shape[2]), np.int64))
    inst = hc.Inst(m, n, S, U.shape[2], "zero", "add", seed, str(lay))
    return inst, _finish(d, rule, g, S)


def lanes(lay: Layout, m: int, n: int, S: int, g: np.random.Generator):
    """A non-separable layout's coin lanes U (m, S, q), V (n, S, q) and salts w0 (m, n)."""
    ell = lay.coin_lanes
    q = ell + lay.r0
    U = np.zeros((m, S, q), np.int64)
    V = np.zeros((n, S, q), np.int64)
    half = S // 2 if lay.family != "bilinear" else S
    if lay.family == "mirror-pat":
        pats = np.random.default_rng(12345).integers(-127, 128, (PATTERNS, 3))
        a = pats[g.integers(0, PATTERNS, (m, half))]
        b = g.choice(ODD7, (n, half, 3))
    else:
        a = g.integers(-128, 128, (m, half, ell))
        b = g.choice(nsp.WIDE, (n, half, ell))
    U[:, :half, :ell], V[:, :half, :ell] = a, b
    if lay.family != "bilinear":
        order = np.arange(half) if lay.family == "mirror" else cs.free_order(half, g)
        U[:, half:2 * half, :ell], V[:, half:2 * half, :ell] = a[:, order], -b[:, order]
    if lay.r0:
        a0, b0 = g.integers(-128, 128, (m, lay.r0)), g.choice(nsp.WIDE, (n, lay.r0))
        U[:, 0, ell:], V[:, 0, ell:] = a0, b0
        U[:, S - 1, ell:], V[:, S - 1, ell:] = a0, -b0
    return U, V, (_salts(m, n, g) if lay.salt else np.zeros((m, n), np.int64))


# ---- the new commitments ---------------------------------------------------------------------------------------------

def thin_fold(fields: np.ndarray) -> np.ndarray:
    """main's `thinFold`: from σ = 0, the words two a step, XOR3 and IADD3 alternating from XOR3; sealed with the final
    word, so the commitment an output is (σ, O′)."""
    s = np.zeros(fields.shape[:2], np.int64)
    xor = True
    for t in range(0, fields.shape[2] - 1, 2):
        s = (s ^ fields[:, :, t] ^ fields[:, :, t + 1]) if xor else (s + fields[:, :, t] + fields[:, :, t + 1]) & M32
        xor = not xor
    return np.stack([s, fields[:, :, -1]], -1)


def aggregate(rule: hc.Rule, d, fields):
    if rule.agg == "thin":
        return thin_fold(fields)
    if rule.agg == "mulxor:live":
        N = np.full(fields.shape[:2], d.n0, np.int64)
        for t in range(fields.shape[2]):
            N = hc.mulmod(N ^ fields[:, :, t], d.r[t] | 1)
        return N
    if rule.agg == "sq1":
        return hc.mulmod(fields, fields).sum(axis=2) & M32
    return hc.aggregate(rule, d, fields)


def honest(rule, inst, d):
    fields = rule.field(d.w[:, :, d.P])
    return inst.S + AGG_OPS[rule.agg] * len(d.P), aggregate(rule, d, fields), "MMA every step"


def common_hash(rule: hc.Rule, nP: int) -> float:
    return AGG_SEAL_BITS[rule.agg] * hc.WB


# ---- provers beyond COUNTER's ----------------------------------------------------------------------------------------

def edge_any(rule, inst, d):
    """The multiply–xor fold's first step from free terms: N₁ = (x + Σ_l U_l·V_l)·k₁ = x·k₁ + Σ_l (U_l·k₁)·V_l, one
    IMMA over the 4q digit lanes of U·k₁ (q ≤ 8) and an IADD3 for a per-cell offset; then honest's word, LOP3 and IMUL
    a step. It needs N₀ = 0, so `mulxor:live` meets it only through `edge_forced` (a check that it fails there)."""
    if rule.agg != "mulxor" or inst.q > 8 or d.P[0] != 0:
        return None
    return edge_forced(rule, inst, d)


def edge_forced(rule, inst, d):
    k1 = int(d.r[0] | 1)
    N = hc.mulmod(d.w0 & M32, k1)
    for lane in range(d.U.shape[2]):
        N = N + hc.mulmod(((d.U[:, 0, lane] * k1) & M32)[:, None], d.V[:, 0, lane][None, :] & M32)
    N &= M32
    for idx in range(1, len(d.P)):
        N = hc.mulmod(N ^ d.w[:, :, d.P[idx]], d.r[idx] | 1)
    first = 1 + bool(np.any(d.w0))
    return inst.S + AGG_OPS[rule.agg] * len(d.P) - 3 + first, N, f"first step {first} write(s) from free terms"


@contextmanager
def coin_tile(rows: int):
    """COUNTER's column terms shared by the rows of a class in a `rows`-row coin tile (decomp's `TILE`; pack's
    `CALL_M` when `rows` is given for pack too)."""
    saved = hc.TILE, hc.CALL_M
    hc.TILE = rows
    try:
        yield
    finally:
        hc.TILE, hc.CALL_M = saved


def _sep(inst, d):
    parts = hc.separable(inst, d)
    if parts is None or inst.data != "zero":
        return None
    return parts


def decomp_sq(rule, inst, d, tile: int):
    """Σ_t r_t (x + R_t + C_t)² over separable words: x²·Σr + 2x·(Σ rR + Σ rC) + Σ rR² + Σ rC² + 2·Σ_t R_t·(r_t C_t),
    the last as u8-digit IMMA chains (R narrow on the row side, 2r·C on the column side). Column-only sums are shared by
    the rows of a class in a `tile`-row coin tile; row-only sums by the call's columns."""
    if rule.agg not in ("sq", "sq1") or rule.k != 32:
        return None
    parts = _sep(inst, d)
    if parts is None:
        return None
    R, C, cls, pats, rowlane = parts
    P, nP = d.P, len(d.P)
    r = d.r if rule.agg == "sq" else np.ones(nP, np.int64)
    x = d.w0 & M32
    Rp, Cp = R[:, P] & M32, C[:, :, P] & M32
    rR, rC = hc.mulmod(Rp, r[None, :]), hc.mulmod(Cp, r[None, None, :])
    sr = int(r.sum()) & M32
    rowA = (2 * rR.sum(axis=1)) & M32
    rowB = hc.mulmod(rR, Rp).sum(axis=1) & M32
    colA = (2 * rC.sum(axis=2)) & M32
    colB = hc.mulmod(rC, Cp).sum(axis=2) & M32
    nR = max(1, math.ceil(int(Rp.max(initial=0)).bit_length() / 8))
    cross, chains = hc.digit_dot(Rp, (2 * rC) & M32, cls, nR)
    t = (hc.mulmod(x, sr) + rowA[:, None] + colA[cls]) & M32
    s = (hc.mulmod(x, t) + rowB[:, None] + colB[cls] + cross) & M32
    col_ops = int((~rowlane).sum()) + 3
    cost = chains + 4 + col_ops * nP * len(pats) / tile + (nR + 4) * nP / hc.CALL_N
    return cost, s, f"row and column sums + {chains} IMMA writes an output for Σ R·(2rC) ({nR}×4 digits)"


def pairsum(rule, inst, d, tile: int):
    """The thin fold over separable words: an XOR3 step forms its two words (an IADD3 each) and one LOP3; an IADD3 step
    adds 2x + (R_t + R_{t+1}) + (C_t + C_{t+1}) in two IADD3s, writing neither word; O′ is one IADD3 when no XOR3 step
    formed it. Row pair sums are shared by the call's columns, column pair sums by a class's rows in the tile."""
    if rule.agg != "thin" or rule.k != 32 or rule.pos != "all":
        return None
    parts = _sep(inst, d)
    if parts is None:
        return None
    R, C, cls, pats, rowlane = parts
    S = inst.S
    x = d.w0 & M32
    word = lambda t: (x + R[:, t][:, None] + C[cls, :, t]) & M32  # noqa: E731
    s = np.zeros((inst.m, inst.n), np.int64)
    xor, n_xor, n_add, last_formed = True, 0, 0, False
    for t in range(0, S - 1, 2):
        if xor:
            s = s ^ word(t) ^ word(t + 1)
            n_xor += 1
            last_formed = t + 1 == S - 1
        else:
            s = (s + 2 * x + (R[:, t] + R[:, t + 1])[:, None] + (C[cls, :, t] + C[cls, :, t + 1])) & M32
            n_add += 1
        xor = not xor
    out = np.stack([s, word(S - 1)], -1)
    cost = 3 * n_xor + 2 * n_add + 1 + (0 if last_formed else 1)
    cost += (int(rowlane.sum()) + 2) * S / hc.CALL_N + (int((~rowlane).sum()) + 2) * S * len(pats) / tile
    return cost, out, f"{n_add} IADD3 steps from shared pair sums, {n_xor} XOR3 steps on formed words"


def mma_pairsum(rule, inst, d):
    """The thin fold on any lane-form layout at zero data, where a k32 step's data lanes are free for other steps'
    coin lanes. An IADD3 step is linear: σ + w_t + w_{t+1} = (σ + 2·w_{t−1}) + 2c_t + c_{t+1}, so one IADD3 forms the
    accumulator input and one MMA over step t's lanes twice and step t+1's once (3q ≤ 32 lanes) writes σ′; neither
    word is formed. An XOR3 step's first word is one MMA over the skipped steps' lanes from the last formed word, its
    second one MMA more, then the LOP3. O′ is one MMA when no XOR3 step formed it."""
    if rule.agg != "thin" or rule.k != 32 or rule.pos != "all" or inst.data != "zero" or 3 * d.U.shape[2] > 32:
        return None
    S = inst.S
    c = np.einsum("itl,jtl->ijt", d.U, d.V)
    s = np.zeros((inst.m, inst.n), np.int64)
    last, p = d.w0 & M32, -1
    xor, cost = True, 0
    for t in range(0, S - 1, 2):
        if xor:
            wt = (last + c[:, :, p + 1:t + 1].sum(axis=2)) & M32
            wt1 = (wt + c[:, :, t + 1]) & M32
            s = s ^ wt ^ wt1
            last, p = wt1, t + 1
            cost += 3
        else:
            acc = (s + 2 * last) & M32
            s = (acc + 2 * c[:, :, t] + c[:, :, t + 1]) & M32
            cost += 2
        xor = not xor
    final = (last + c[:, :, p + 1:S].sum(axis=2)) & M32
    cost += 0 if p == S - 1 else 1
    return cost, np.stack([s, final], -1), "IADD3 steps as one MMA each over the pair's lanes (step t's doubled)"


def word_span_bits(d) -> int:
    """The narrowest signed width holding every zero-data word (its residue mod 2³² read as signed)."""
    w = d.w.astype(np.int64)
    w = np.where(w >= 1 << 31, w - (1 << 32), w)
    return int(np.abs(w).max()).bit_length() + 1


def pack_bound(rule, inst, d, lay: Layout):
    """A lower bound on any packed prover for `xor` or `thin` on a non-separable layout: L = ⌊32g/b⌋ cells in b-bit
    lanes of g registers, a step one IMAD a bilinear lane a packed register (ℓ·g/L a word), the fold at its packed LOP3
    count (g a word pair, the cheapest fold step), and COUNTER's unpack (6 a cell). Returns None when no packing fits
    (L ≤ g, as with full-width salts)."""
    if rule.agg not in ("xor", "thin") or lay.separable:
        return None
    b = 32 if lay.salt else word_span_bits(d)
    best = None
    for g in (1, 2):
        L = 32 * g // b
        if L <= g:
            continue
        cost = (lay.coin_lanes * g / L + 0.5 * g / L) * inst.S + 6
        best = cost if best is None else min(best, cost)
    return None if best is None else (best, b)


COUNTER = ("honest", "pack", "decomp", "edge", "copy", "table")


def evaluate(lay: Layout, rule: hc.Rule, m: int, n: int, S: int, seed: int) -> dict:
    inst, d = build(lay, m, n, S, rule, seed)
    S = inst.S
    stock = (32 - lay.q_eff) * S / 32
    hcost, ref, _ = honest(rule, inst, d)
    rows = {}
    tile = inst.m if lay.separable else hc.CALL_M

    def add(name, res):
        if res is None:
            return
        cost, out, how = res
        rows[name] = {"writes": round(float(cost), 4), "exact": bool(np.array_equal(out, ref)), "how": how}

    with coin_tile(inst.m if lay.separable else hc.TILE):
        if rule.agg in ("hash", "lin", "sq", "xor", "mulxor") or hc.nh_kind(rule):
            for name in COUNTER[1:]:
                add(name, hc.STRATEGIES[name](rule, inst, d))
        if lay.separable and rule.agg.startswith("xor"):
            res = hc.pack(rule, inst, d)
            if res:
                L, g = (int(x) for x in re.match(r"(\d+) cells in \d+-bit lanes of (\d+)", res[2]).groups())
                parts = hc.separable(inst, d)
                npat, ncol = len(parts[3]), int((~parts[4]).sum())
                col = npat * (ncol + L) * g * S
                add("pack@tile", (res[0] - col / hc.CALL_M + col / (inst.m * L), res[1],
                                  res[2] + f", column terms a class shared by the {inst.m}-row tile's rows"))
    add("edge_any", edge_any(rule, inst, d))
    add("decomp_sq", decomp_sq(rule, inst, d, tile))
    add("pairsum", pairsum(rule, inst, d, tile))
    add("mma_pairsum", mma_pairsum(rule, inst, d))
    exact = {k: v for k, v in rows.items() if v["exact"]}
    best = min(exact, key=lambda k: exact[k]["writes"]) if exact else "honest"
    cheapest = exact[best]["writes"] if exact else hcost
    if cheapest >= hcost:
        best, cheapest = "honest", hcost
    tile_priced = [v["writes"] for k, v in exact.items() if not (k == "pack" and "pack@tile" in exact)]
    cheapest_tile = min([hcost] + tile_priced)
    pb = pack_bound(rule, inst, d, lay)
    if pb is None and rule.agg in ("xor", "thin") and not lay.separable:
        pb = (float("inf"), 32 if lay.salt else word_span_bits(d))
    return {
        "layout": str(lay), "rule": str(rule), "m": inst.m, "n": inst.n, "S": S, "seed": seed, "q_eff": lay.q_eff,
        "stock": stock, "honest": hcost, "common_hash": round(common_hash(rule, len(d.P)), 3),
        "strategies": rows, "cheapest": best, "cheapest_writes": cheapest,
        "gap_stock": round(max(0.0, (stock - cheapest) / stock), 6),
        "gap_claimed": round(max(0.0, (hcost - cheapest) / stock), 6),
        "gap_claimed_tile_priced": round(max(0.0, (hcost - cheapest_tile) / stock), 6),
        "pack_bound": None if pb is None else {"writes": round(pb[0], 3), "span_bits": pb[1],
                                                "below_honest": pb[0] < hcost},
        "span_bits": word_span_bits(d),
    }


# ---- rank-based decompositions of sketches over non-separable words --------------------------------------------------

#: A value's entries and the honest prover's cost a word beyond its MMA, for each sketch.
SKETCH = {"sq1": {"entries": 1, "op": 1.0, "mid": 3.0}, "nh:1": {"entries": 1, "op": 0.5, "mid": 3.0},
          "nh:16": {"entries": 16, "op": 0.5, "mid": 3.0}, "nh2:8": {"entries": 16, "op": 0.5, "mid": 2.5}}


def rank_mod_p(M: np.ndarray, p: int = PRIME) -> int:
    """The rank of an integer matrix mod p (≤ its rank over Q, equal except with probability ≤ rank/p)."""
    A = np.array(M % p, dtype=np.int64)
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = np.nonzero(A[r:, c])[0]
        if piv.size == 0:
            continue
        k = r + piv[0]
        A[[r, k]] = A[[k, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        f = A[:, c].copy()
        f[r] = 0
        nz = np.nonzero(f)[0]
        if nz.size:
            A[nz] = (A[nz] - (f[nz, None] * A[r][None, :]) % p) % p
        r += 1
    return r


def _walk(lay: Layout, m: int, n: int, S: int, seed: int, keep) -> dict:
    """The layout's zero-data words mod p at the steps in `keep`, from its lanes (BLAS over float64, exact: a step's
    noise is below 2^21)."""
    U, V, w0 = lanes(lay, m, n, S, np.random.default_rng(seed))
    w = (w0 % PRIME).astype(np.int64)
    out = {}
    for t in range(max(keep) + 1):
        step = (U[:, t, :].astype(np.float64) @ V[:, t, :].astype(np.float64).T).astype(np.int64)
        w = (w + step) % PRIME
        if t in keep:
            out[t] = w.copy()
    return out


def _value(agg: str, words: list, p: int) -> np.ndarray:
    """A sketch's contribution from a segment's words (each (m, n), integers mod p) as a (row, value) matrix."""
    if agg == "sq1":
        return sum((w * w) % p for w in words) % p
    if agg == "nh2:8":
        out = sum(((w[:, 0::2] * w[:, 1::2]) % p) for w in words) % p
        return out.reshape(out.shape[0], -1, 8).sum(axis=2) % p
    pairs = sum((words[k] * words[k + 1]) % p for k in range(0, len(words) - 1, 2)) % p
    return pairs if agg == "nh:1" else pairs.reshape(pairs.shape[0], -1, 16).sum(axis=2) % p


def rank_formula(lay: Layout, t: int, side: str, agg: str = "sq1") -> int | None:
    """F(F+1)/2, the degree-2 monomials in a segment's independent features: F = ℓ·t coin products, plus r0 (the
    step-0 block, which a suffix still carries until the last step) and 2 for salts (ρ_i, γ_j) on an edge. A middle
    block's increments carry neither. nh's pairs w_2k·w_2k+1 never square the segment's last step, which drops its
    ℓ(ℓ+1)/2 monomials, and its suffix ends a pair before the final word, so it carries that pair's step too. A served
    layout's suffix sees every step's coins (None: no decomposition)."""
    if side == "suffix" and lay.family == "bilinear":
        return None
    ell = lay.coin_lanes
    nh = agg.startswith("nh:")
    F = ell * (t + (1 if nh and side == "suffix" else 0))
    F += 0 if side == "block" else lay.r0 + (2 if lay.salt else 0)
    return F * (F + 1) // 2 - (ell * (ell + 1) // 2 if nh else 0)


def segment_ranks(lay: Layout, agg: str, S: int, t_max: int, size: int, seed: int = 3) -> dict:
    """Measured ranks mod p of a sketch's prefix (its first t words), suffix (the t words before the final one) and a
    middle block (a fresh walk's first t words, no salts or step-0 block), t ≤ t_max, on size rows × size values,
    beside `rank_formula` capped at size."""
    e = SKETCH[agg]["entries"]
    unit = 2 if agg.startswith("nh:") else 1
    ts = [t for t in range(unit, t_max + 1, unit)]
    n = size * e
    end = S - 1 - (unit - 1)
    main = _walk(lay, size, n, S, seed, set(range(t_max)) | set(range(end - t_max, end)))
    blk = _walk(replace(lay, r0=0, salt=False), size, n, S, seed + 1, set(range(t_max)))
    out = {"size": size, "measured": {"prefix": {}, "suffix": {}, "block": {}}, "formula": {}}
    for t in ts:
        out["measured"]["prefix"][t] = rank_mod_p(_value(agg, [main[u] for u in range(t)], PRIME))
        out["measured"]["suffix"][t] = rank_mod_p(_value(agg, [main[u] for u in range(end - t, end)], PRIME))
        out["measured"]["block"][t] = rank_mod_p(_value(agg, [blk[u] for u in range(t)], PRIME))
    for side in ("prefix", "suffix", "block"):
        out["formula"][side] = {t: (None if rank_formula(lay, t, side, agg) is None else
                                    min(rank_formula(lay, t, side, agg), size)) for t in ts}
    return out


def rank_dp(lay: Layout, agg: str, S: int, B: int, features: str) -> dict:
    """The cheapest split of a value's S − 1 words (the final word is M1's) into honest words, prefix and suffix edges
    (rank terms, or with the step-0 block and salts split off at 4 a cell plus a block's rank) and middle blocks from
    the honest word before them (`mid` a cell plus the block's rank), at `rank_formula`'s ranks for every length. A rank
    term costs B byte products, 32 a write, plus its features when `tiled`: a row feature (one IMAD) is shared by the
    values of a 64-column coin tile, and a column feature (one IMMA over the value's int8 weights) by a 64-row tile.
    Returns the cost and saving a value and the saving's share of stock. nh's segments are whole pairs."""
    sk = SKETCH[agg]
    e = sk["entries"]
    unit = 2 if agg.startswith("nh:") else 1
    word = e * (1 + sk["op"])
    per_term = B / 32 + ((e / 64 + 1 / 64) if features == "tiled" else 0.0)
    inf = float("inf")
    s = S - 1 - (unit - 1)
    lengths = range(unit, s + 1, unit)

    def costs(side: str, fixed: float) -> dict:
        out = {}
        for t in lengths:
            r = rank_formula(lay, t, side, agg)
            if r is not None and fixed + r * per_term < word * t:
                out[t] = fixed + r * per_term
        return out

    pre, suf = costs("prefix", 0.0), costs("suffix", 0.0)
    if lay.r0 or lay.salt:
        for t, c in costs("block", 4.0 * e).items():
            pre[t] = min(c, pre.get(t, inf))
            if lay.family != "bilinear":
                suf[t] = min(c, suf.get(t, inf))
    mid = costs("block", sk["mid"] * e)
    best = [inf] * (s + 1)
    best[0] = 0.0
    for t, c in pre.items():
        best[t] = min(best[t], c)
    for t in range(s):
        if best[t] == inf:
            continue
        nxt = min(s, t + unit)
        best[nxt] = min(best[nxt], best[t] + word * (nxt - t))
        if t > 0:
            for L, c in mid.items():
                if t + L <= s:
                    best[t + L] = min(best[t + L], best[t] + c)
    cost = min([best[s]] + [best[s - t] + c for t, c in suf.items() if t < s])
    saving = max(0.0, word * s - cost)
    return {"cost": round(cost, 3), "saving": round(saving, 3), "share": round(saving / (e * S * cs.STOCK_EQ), 6),
            "edges_used": sorted(pre)[:3], "blocks_used": sorted(mid)[:3]}


# ---- the wire: coin stream, n_max, network term, T3 ------------------------------------------------------------------

def _column_steps() -> float:
    """Column-steps a token over Llama-3.2-1B's linears (16 layers), Σ n·S."""
    return 16 * sum(nc * nsp.blc_steps(k) for nc, k in ns.LLAMA.values())


#: Each layout's coins: bits a (row, step, 64-column tile) αr and a (column, step, row tile) αc, the drawn share of
#: the steps (the replay halves it), and how its noise leaves the output (T3).
WIRE = {
    "lead": {"ar": 0.0, "ac": 21.0, "drawn": 1.0, "t3": "exact: the noise's total is row + column, folded by the epilogue"},
    "lead-salt": {"ar": 0.0, "ac": 21.0, "drawn": 1.0, "t3": "exact, as lead; the salts fold in the epilogue's two sources"},
    "lead-wide": {"ar": 0.0, "ac": 21.0, "drawn": 1.0,
                  "t3": "exact, as lead; U·V is removed mod 2³² in the epilogue (two IMADs a cell)"},
    "lead-tile": {"ar": 0.0, "ac": 21.0, "drawn": 1.0, "t3": "exact, as lead (a correction a tile and column)"},
    "bilinear": {"ar": 8.0, "ac": 8.0, "drawn": 1.0, "t3": "FAILS: the noise total has rank ℓS and is served"},
    "mirror": {"ar": 8.0, "ac": 8.0, "drawn": 0.5, "t3": "exact: the replay cancels the first half"},
    "mirror-pi": {"ar": 8.0, "ac": 8.0, "drawn": 0.5, "t3": "exact: the replay cancels in any order"},
    "mirror-pat": {"ar": 4.0, "ac": 21.0, "drawn": 0.5, "t3": "exact: the replay cancels"},
}


def wire(lay: Layout, agg: str, S: int = 74) -> dict:
    """Coin stream a token (MB, at ≥ 64 rows and at one row), n_max and the network term at caps 32 and 320 (floor /
    bar), scaled from N-EGRESS's lead term by n_max/85.3 and by the commitment's W_mac over the lead's."""
    w = WIRE[lay.family]
    lanes = 1 if lay.separable else (3 if lay.family == "mirror-pat" else lay.lanes)
    ar = w["ar"] * (lanes if lay.family in ("bilinear", "mirror", "mirror-pi") else 1)
    ac = w["ac"] * (lanes if lay.family in ("bilinear", "mirror", "mirror-pi") else 1)
    tr = lay.tile if lay.separable else 64
    cst = _column_steps()
    stream = lambda rows: cst * w["drawn"] * (ar / 64 + ac / min(tr, rows)) / 8 / 1e6  # noqa: E731
    data = 32 - lay.q_eff
    reads = 2 if w["drawn"] < 1 else 1
    n_row = reads * 64 * data / ar if ar else 0.0
    n_col = reads * tr * data / ac
    n_max = max(n_row, n_col)
    honest_word = 1 + AGG_OPS[agg] + AGG_SEAL_BITS[agg] / 32 * cs.SHA_WORD / S
    ratio = honest_word / (1 + cs.SHA_WORD)
    lead_n = 256 / 3
    terms = {f"cap{cap}": tuple(round(x * n_max / lead_n * ratio * cap / cs.NET_LEAD["cap"], 7)
                                for x in cs.NET_LEAD["term"]) for cap in (32, 320)}
    return {"coin_entropy_MB": {"64+": round(stream(64), 3), "1": round(stream(1), 3)}, "n_max": round(n_max, 1),
            "W_mac_over_lead": round(ratio, 5), **terms, "t3": w["t3"]}


#: Llama-3.2-1B's linears (K, n) and layers, and the rows of each point.
LINEARS = ((2048, 3072), (2048, 2048), (2048, 16384), (8192, 2048))
LAYERS = 16
POINTS = (1, 16, 64, 512, 8192)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _spread(steps: int, coined: int) -> list[bool]:
    return [t * coined // steps > (t - 1) * coined // steps for t in range(steps)]


def feed(m: int) -> dict:
    """Bytes a token at m rows, by part, for the deployed feed, its coins' entropy and N-EGRESS's spread layout.

    - `deployed`: m·H row words over 16⌈m/16⌉ rows, n G words, and a 64-row tile at a time S·n lane-coin words (three
      int8 coins and a public byte) and 2n column sources (layout.h; the certifier's `release_bytes`).
    - `entropy`: the same coins' bits: S + 3 a row, 4·log₂7 a column's G, 21 a (tile, step, column).
    - `spread`: `egress.py`'s blc4a-t64-e7-rot. Under 64 rows a tile coins ⌈S·m/64⌉ steps' three lanes, one lane a
      step, so 3⌈S·m/64⌉ steps carry a lane word and the rest an 8-bit row coin and the window coin; column sources a
      row class present."""
    out = {k: dict.fromkeys(("rows", "G", "lanes", "sources"), 0.0) for k in ("deployed", "entropy", "spread")}
    R, T = 16 * _cdiv(m, 16), _cdiv(m, 64)
    for K, n in LINEARS:
        S = 1 + _cdiv(K - 21, 28)
        H = _cdiv(S + 3, 32)
        dep, ent, spr = out["deployed"], out["entropy"], out["spread"]
        dep["rows"] += 4 * R * H
        dep["G"] += 4 * n
        dep["lanes"] += 4 * T * S * n
        dep["sources"] += 8 * T * n
        ent["rows"] += R * (S + 3) / 8
        ent["G"] += n * 4 * math.log2(7) / 8
        ent["lanes"] += T * S * n * 21 / 8
        slots = 3 * _cdiv(S * min(m, 64), 64)
        lane = [True] * S if slots >= S else _spread(S, slots)
        coined = sum(lane)
        later = sum(lane[1:])
        spr["rows"] += m * 32 * _cdiv(later + 8 * (S - 1 - later) + 4, 32) / 8
        spr["G"] += 4 * n
        spr["lanes"] += 4 * T * coined * n
        spr["sources"] += 4 * T * n * min(m, 64, 2) if coined > 1 else 0
    res = {}
    for k, parts in out.items():
        parts = {p: v * LAYERS / m for p, v in parts.items()}
        res[k] = {"total_bytes": round(sum(parts.values()), 1), "total_MB": round(sum(parts.values()) / 1e6, 4),
                  **{p: round(v / 1e6, 4) for p, v in parts.items()}}
    return res


def lever_wire(m: int) -> dict:
    """The wide step-0 term's coins a token at m rows: U, two 16-bit values a committed row (16⌈m/16⌉), and V, two
    15-bit values a column, a call; packed, and one value a 4-byte word as the feed carries its coins."""
    R = 16 * _cdiv(m, 16)
    bits = sum(2 * (16 * R + 15 * n) for _, n in LINEARS) * LAYERS / m
    words = sum(2 * (R + n) for _, n in LINEARS) * LAYERS / m
    dep = feed(m)["deployed"]["total_MB"]
    return {"packed_MB": round(bits / 8 / 1e6, 4), "u32_MB": round(4 * words / 1e6, 4),
            "share_of_feed": round(bits / 8 / 1e6 / dep, 5)}


ANALYTIC = [
    {"layout": "per-entry coins (an int8 a cell and step, streamed)",
     "stream": "8 bits a cell-step: 35 MB a token at ≥ 64 rows (Σ n·S × 8 bits), the same at one row",
     "n_max": "28/8 = 3.5 (no coin bit is shared)", "honest": "+1 ALU add a word (the MMA can't add a per-cell operand "
     "after step 0): the MMA's share of honest doubles", "T3": "exact: the coins' sum is subtracted (one more add)",
     "verdict": "words x + Σ e_u(i, j) have full rank, so decomp and rank_dp find nothing; but the prover pays the "
                "same add honest does and skips the MMA, so under credit = cost(H) it gains a write a word (≈ 114% "
                "of stock); credit = stock only. pack: narrow coins pack L cells an IADD3 unless salts are full width"},
    {"layout": "coins from a seed (hash-indexed, expanded on the GPU)",
     "stream": "a 512-bit seed a call", "n_max": "unbounded: an outside machine expands the same seed",
     "honest": "SHA-512 to expand: about 27 eq a coin byte, more than the MACs it gates (≥ 64·28/8)",
     "T3": "as the layout it expands to", "verdict": "fails the network term (or h, if expansion is charged); a seed "
                                                      "only moves the coins' bandwidth to compute"},
    {"layout": "coins indexed by a hash of (row, column) into M buckets", "stream": "M coins a step",
     "n_max": "≈ cells/M × 28 a coin bit", "honest": "a gather a cell and step (an ALU write)",
     "T3": "exact when the buckets' sums are removed per cell (a gather)",
     "verdict": "cells in one bucket share every word: `copy` computes M of them; needs M ≥ cells, i.e. per-entry"},
    {"layout": "multiplicative accumulator (w_t = ρ_t·w_{t−1} + c_t, public odd ρ)", "stream": "the lead's",
     "n_max": "the lead's", "honest": "+1 IMUL a word", "T3": "exact (the scale is public)",
     "verdict": "w_t = x·Πρ + R̃_t(i) + C̃_t(j) stays row-plus-column with public step weights, so decomp, decomp_sq "
                "and pairsum apply with those weights (argued); it only costs honest a write a word"},
]


# ---- the runs ---------------------------------------------------------------------------------------------------------

LAYOUTS = [Layout("lead"), Layout("lead-salt"), Layout("lead-wide"), Layout("lead-tile", tile=16),
           Layout("lead-tile", tile=8),
           Layout("bilinear", 1), Layout("bilinear", 4), Layout("bilinear", 4, salt=True),
           Layout("mirror", 4, 4, True), Layout("mirror-pi", 1, 0, True), Layout("mirror-pi", 2, 4, True),
           Layout("mirror-pi", 3, 4, True), Layout("mirror-pi", 4, 0, True), Layout("mirror-pi", 4, 4, True),
           Layout("mirror-pi", 4, 4, False), Layout("mirror-pat", 3, 4, True)]
AGGS = ["mulxor", "mulxor:live", "thin", "xor", "sq1", "sq", "nh:1", "nh:16", "nh2:8"]
RANK_LAYOUTS = [Layout("mirror-pi", 2, 4, True), Layout("mirror-pi", 3, 4, True), Layout("mirror-pi", 4, 0, True),
                Layout("mirror-pi", 4, 4, True), Layout("mirror-pi", 4, 8, True), Layout("mirror-pat", 3, 4, True),
                Layout("mirror-pat", 3, 8, True), Layout("bilinear", 4, 4, True)]


def mutations(checks: Checks):
    rule = hc.Rule("all", "32", "thin")
    inst, d = build(Layout("lead"), 64, 48, 74, rule, 7)
    _, ref, _ = honest(rule, inst, d)
    d.w0 = d.w0.copy()
    d.w0[2, 3] += 1
    _, out, _ = pairsum(rule, inst, d, inst.m)
    checks.mutation("pairsum with one cell's offset off by one", not np.array_equal(out, ref))
    rule = hc.Rule("all", "32", "thin")
    inst, d = build(Layout("mirror-pi", 4, 4, True), 64, 48, 74, rule, 7)
    _, ref, _ = honest(rule, inst, d)
    d.V = d.V.copy()
    d.V[:, 5, 1] = -d.V[:, 5, 1]
    _, out, _ = mma_pairsum(rule, inst, d)
    checks.mutation("mma_pairsum with one step's lane negated", not np.array_equal(out, ref))
    rule = hc.Rule("all", "32", "sq")
    inst, d = build(Layout("lead"), 64, 48, 74, rule, 7)
    _, ref, _ = honest(rule, inst, d)
    d.U = d.U.copy()
    d.U[1::2, :, 2:] = d.U[0::2, :, 2:]
    _, out, _ = decomp_sq(rule, inst, d, inst.m)
    checks.mutation("decomp_sq with the row-parity classes merged", not np.array_equal(out, ref))
    d = build(Layout("lead"), 64, 48, 74, rule, 7)[1]
    d.r = d.r.copy()
    d.r[5] ^= 1
    _, out, _ = decomp_sq(rule, inst, d, inst.m)
    checks.mutation("decomp_sq with one sketch coin flipped", not np.array_equal(out, ref))
    for agg, prover in (("thin", mma_pairsum), ("sq", lambda r, i, d: decomp_sq(r, i, d, i.m))):
        rule = hc.Rule("all", "32", agg)
        inst, d = build(Layout("lead-wide"), 64, 48, 74, rule, 7)
        _, ref, _ = honest(rule, inst, d)
        d.w0 = build(Layout("lead"), 64, 48, 74, rule, 7)[1].w0
        _, out, _ = prover(rule, inst, d)
        checks.mutation(f"{agg} on lead-wide with the wide term dropped from the start value",
                        not np.array_equal(out, ref))
    for lay in (Layout("lead"), Layout("mirror-pi", 4, 4, True)):
        rule = hc.Rule("all", "32", "mulxor:live")
        inst, d = build(lay, 64, 48, 74, rule, 7)
        _, ref, _ = honest(rule, inst, d)
        _, out, _ = edge_forced(rule, inst, d)
        checks.mutation(f"edge from N₀ = 0 against a live N₀ ({lay})", not np.array_equal(out, ref))


def run_grid(quick: bool, layouts: list[Layout] = LAYOUTS) -> list[dict]:
    shapes = [(64, 48, 74)] if quick else [(64, 48, 74), (64, 48, 292)]
    seeds = [0] if quick else [0, 1]
    out = []
    for lay in layouts:
        for (m, n, S), seed in [(sh, s) for sh in shapes for s in seeds]:
            for agg in AGGS:
                out.append(evaluate(lay, hc.Rule("all", "32", agg), m, n, S, seed))
    return out


def run_ranks(quick: bool) -> list[dict]:
    t_max, size = (2, 120) if quick else (4, 300)
    lays = RANK_LAYOUTS[:2] if quick else RANK_LAYOUTS
    out = []
    for lay in lays:
        for agg in SKETCH:
            for S in ((74,) if quick else (74, 292)):
                rk = segment_ranks(lay, agg, S, t_max, size)
                row = {"layout": str(lay), "agg": agg, "S": S, "ranks": rk, "dp": {}}
                for B in (2, 4):
                    for feat in ("free", "tiled"):
                        row["dp"][f"B{B}-{feat}"] = rank_dp(lay, agg, S, B, feat)
                out.append(row)
                print(f"ranks {str(lay):22s} {agg:6s} S{S}: measured {rk['measured']} "
                      f"dp {[(k, v['share']) for k, v in row['dp'].items()]}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part", choices=("grid", "lever", "ranks", "all"), default="all")
    args, rest = ap.parse_known_args()
    checks = Checks(rest + (["--mutations"] if "--mutations" not in rest else []))
    res: dict = {"schema": "noise-layouts/v1"}
    res["feed"] = {str(m): feed(m) for m in POINTS}
    res["lever_wire"] = {str(m): lever_wire(m) for m in POINTS}
    f1, f64 = res["feed"]["1"], res["feed"]["64"]
    checks.check("the deployed feed at one row is the certifier's release, 144,788,480 bytes a forward",
                 f1["deployed"]["total_bytes"] == 144_788_480, f1["deployed"])
    checks.check("the spread layout agrees with the deployed feed at 64 rows and above",
                 all(res["feed"][str(m)]["spread"]["total_bytes"] == res["feed"][str(m)]["deployed"]["total_bytes"]
                     for m in POINTS if m >= 64), f64["spread"])
    checks.check("the lead's coin entropy at one row is wire's figure (21 bits a tile, step and column)",
                 abs(f1["entropy"]["lanes"] - wire(Layout("lead"), "thin")["coin_entropy_MB"]["1"]) < 1e-3,
                 f1["entropy"])
    for m in POINTS:
        f, lw = res["feed"][str(m)], res["lever_wire"][str(m)]
        print(f"feed m {m:5d}: deployed {f['deployed']} | entropy {f['entropy']['total_MB']} | spread "
              f"{f['spread']['total_MB']} | lever {lw}", flush=True)
    if args.part in ("grid", "lever", "all"):
        grid = run_grid(args.quick, LAYOUTS if args.part != "lever" else
                        [Layout("lead"), Layout("lead-salt"), Layout("lead-wide")])
        res["grid"] = grid
        for name in ("pack", "pack@tile", "decomp", "edge", "edge_any", "decomp_sq", "pairsum", "mma_pairsum", "copy",
                     "table"):
            ran = [r for r in grid if name in r["strategies"]]
            bad = [f"{r['layout']} {r['rule']} S{r['S']}" for r in ran if not r["strategies"][name]["exact"]]
            checks.check(f"{name} matches the honest commitment bit for bit", not bad,
                         f"{len(ran) - len(bad)}/{len(ran)}" + (f"; first miss {bad[0]}" if bad else ""))
        live = [r for r in grid if r["rule"] == "all/32/mulxor:live"]
        checks.check("nothing beats honest on mulxor:live", all(r["cheapest"] == "honest" for r in live),
                     sorted({(r["layout"], r["cheapest"]) for r in live if r["cheapest"] != "honest"}))
        for r in grid:
            if r["gap_claimed"] > 0 or r["gap_stock"] > 0:
                print(f"{r['layout']:24s} {r['rule']:20s} S{r['S']:4d} s{r['seed']} {r['cheapest']:10s} "
                      f"gap_claimed {r['gap_claimed']:.4f} (tile-priced {r['gap_claimed_tile_priced']:.4f}) "
                      f"gap_stock {r['gap_stock']:.4f}")
        mutations(checks)
    if args.part in ("ranks", "all"):
        res["ranks"] = run_ranks(args.quick)
        for r in res["ranks"]:
            meas, form = r["ranks"]["measured"], r["ranks"]["formula"]
            bad = [(side, t, meas[side][t], form[side][t]) for side in meas for t in meas[side]
                   if form[side][t] is not None and meas[side][t] != form[side][t]]
            checks.check(f"measured ranks = rank_formula ({r['layout']}, {r['agg']}, S{r['S']})", not bad,
                         bad[:3] if bad else meas["prefix"])
    res["wire"] = {f"{lay}|{agg}": wire(lay, agg) for lay in LAYOUTS for agg in ("mulxor", "thin", "sq1", "nh2:8")}
    res["analytic"] = ANALYTIC
    res["checks"] = checks.verdict()
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
