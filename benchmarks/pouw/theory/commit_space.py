"""The joint noise × commitment space: which words a design commits, how many bits of each, and hash or sketch.

noise_space.py's grammar commits every k32 step's whole word. This module adds the commitment axes and scores each
noise family under each of them:

- which: `every` step, `every-k` (one word a segment of k steps, L = 28k data lanes), or `outputs` only;
- bits: a window [r, r+b) of each committed word, b in 32, 16, 8, 4, 2, 1 (32 is the whole word);
- kind: `hash` (the bits are SHA-512'd), `linear` (Σ r_t·w_t a chunk, then hashed) or `quad` (Σ_t w_t² mod 2^32 an
  output's steps, then hashed).

The noise families are the lead (`lead`: additive coin lanes, the tag and the column code), the grammar's one-lane-a-step
families with a row coin against coin-drawn column weights (`bilinear`, served), mirror (`mirror`: the second half
replays the first half's coins against negated weights) and `mirror-pi` (the replay in a coin-drawn order), each with
ℓ lanes a step, and N-DENSE's dense two-sided RFP-4 (`rfp4`, exact, X·Y = A·B).

All costs are in register-write equivalents a k32 step's word (eq; H32: 32 W1 units a register written, a MAC 1):
honest's MMA is 1 eq a step (32 lanes), stock `STOCK_EQ` (its 28 data lanes), and DistinctWrites credits 1 eq a
committed word. A design's h is its honest commitment cost over the lead's, times the lead's h. Its T1 is the honest
cost the cheapest attack below does not pay, over stock, beside the noise family's own T1 with whole words.

The attacks each axis meets, all at zero data (M0 frees salt-independent functions, and the prover chooses the data):

- Every committed word of the lead is X(i, j) + Y_t(i) + Z_t(par i, tile i, j), each term precomputable once a row,
  a column or an output (`lead_parts`). So a window bit is one 3-input carry, and a sketch polynomial in the words is
  a sum of products of row and column vectors, which tensor cores evaluate at MAC rate (`sketch_ranks`).
- Windows: under W1 as stated (predicate-only instructions free) IADD3's carry-out predicates give each window bit,
  and VOTE packs 32 a write. Under dataflow-W1 the adversary bit-slices the running state's low r+b bits, 32 columns
  a register, and its window planes are already the packed message words (`bitslice_window`, built and checked bit
  for bit). Honest pays the MMA's accumulator write and then the extraction.
- Sparse: covering credits one segment's distinct coin pairs; one-lane noise has k a segment, and zero-data packing puts
  them in one MMA. Dense noise (RFP-4) is creditable on covering, but N-SAKU's middle-segment skip (proved) costs L/K of
  stock in M0, and only M1, the final-word lemma (rank 3), closes it.
- Linear sketches fold (H12): Σ r_t·w_t = Σ_τ κ_τ·Σ_{t≥τ} r_t.
- Quadratic sketches: the decomposition costs rank × bytes MACs an output; mirror's replay gives the second half's
  squares from the first half's sums (`mirror_fold`); `mirror-pi` and the served bilinear family have no such identity.
  The edges still decompose: the first and last few words see few coin vectors (`edge_attack`), unless step 0 adds
  r0 more coin lanes, removed at the last step (`quad-lℓ-wr0`); and a middle block decomposes from the honest word
  before it (`edge_saves`). Both are priced against the byte products a rank term, and ℓ sets the margin.

`checks` runs the CPU checks and `score` the joint table; `--json` writes both.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_space as nsp
import noise_sweep as ns

#: A SHA-512 compression's registers over its 32 words (N-SHALB's R at 256 KiB chunks), a word.
SHA_WORD = 3530.8 / 32
#: A k32 step's stock, in eq: its 28 data MACs at 1 W1 unit against 32 a register written.
STOCK_EQ = 28 / 32
#: The lead's h (the brief's range) and its T2, sync and async, measured at the three points.
H_LEAD = (120.0, 230.0)
T2_LEAD = {"pt512x64": (10.9, 21.3), "pt512x512": (37.3, 65.1), "pt8192": (62.1, 74.6)}
#: The lead's noise lanes and tag over its data lanes: the MMA's part of honest cost, in stock.
LEAD_MMA = 32 / 28
#: Byte products the quadratic decomposition pays a rank term: two bytes a side, the adversary's cheapest split.
SKETCH_BYTES = 4
#: RFP-4's uncredited share of stock with the reuse lemma replaced by N-DENSE's argued projection bound.
RFP_REUSE_ARGUED = 0.35
TILE_WORDS = 128
LEAD_A = {0: (127, 1, 1), 1: (1, 127, 1)}
COMMIT_BITS = (32, 16, 8, 4, 2, 1)


def _balanced13(v: np.ndarray, digits: int = 4) -> np.ndarray:
    out, v = [], v.astype(np.int64).copy()
    for _ in range(digits):
        d = ((v + 6) % 13) - 6
        out.append(d)
        v = (v - d) // 13
    return np.stack(out)


def lead_parts(m: int, n: int, s: int, rng: np.random.Generator, data_sigma: float = 0.0) -> dict:
    """The lead's words at m rows, n columns and s steps, with their zero-data parts: X (m, n), the step-0 code, tag and
    bias; Y (s, m), the tag lane's prefix; Z (s, m, n), the coin lanes' prefix, a function of (par i, tile i, j); and
    D (s, m, n), the data's prefix (0 at zero data). Words are int64; their committed form is mod 2^32."""
    tag = _balanced13(np.where(np.arange(m) % 2 == 0, np.arange(m) // 2 + 1, -(np.arange(m) // 2 + 1)))
    code = _balanced13(np.arange(n))
    g = rng.integers(-3, 4, size=(4, n))
    a1 = rng.integers(0, 2, size=(4, m))
    beta = (code.sum(0) + 3 * (s - 1) + 1) // 2
    x = tag.T @ g + a1.T @ code - beta[None, :]
    b0, b1 = nsp.BLC_ROW["hi"]
    rowlane = np.where(rng.integers(0, 2, size=(s, m)) == 1, b1, b0)
    rowlane[0] = 0
    y = np.cumsum(rowlane, axis=0)
    tiles = (m + nsp.BLC_TILE - 1) // nsp.BLC_TILE
    sc = 2 * rng.integers(0, 128, size=(s, tiles, 3, n)) - 127
    lane = np.zeros((s, m, n), dtype=np.int64)
    for i in range(m):
        lane[:, i, :] = np.einsum("l,tln->tn", np.array(LEAD_A[i % 2]), sc[:, i // nsp.BLC_TILE])
    z = np.cumsum(lane, axis=0)
    d = np.zeros((s, m, n), dtype=np.int64)
    if data_sigma:
        lanes = np.array([nsp.BLC_DATA_LANES[0]] + [nsp.BLC_DATA_LANES[1]] * (s - 1))
        kappa = rng.normal(0, data_sigma ** 2, size=(s, m, n)) * np.sqrt(lanes)[:, None, None]
        d = np.cumsum(np.rint(kappa).astype(np.int64), axis=0)
    words = x[None] + y[:, :, None] + z + d
    return {"x": x, "y": y, "z": z, "d": d, "words": words}


def freshness(words: np.ndarray, bits: tuple = COMMIT_BITS) -> dict:
    """Each bit's flip rate step to step, and for each window width b at its best fresh position the chance that a
    whole 128-word tile's windows repeat the previous step's (a tile copy under tile grain)."""
    w = (words & 0xFFFFFFFF).astype(np.uint64)
    flips = [float(((w[1:] >> r) & 1 != (w[:-1] >> r) & 1).mean()) for r in range(32)]
    fresh = max([r for r in range(32) if flips[r] >= 0.45] or [-1])
    out = {"flip": flips, "r_fresh": fresh, "tile_repeat": {}}
    flat = w.reshape(w.shape[0], -1)
    tiles = flat.shape[1] // TILE_WORDS
    for b in bits:
        if b == 32:
            continue
        r = max(0, fresh - b + 1)
        win = (flat >> r) & ((1 << b) - 1)
        same = (win[1:] == win[:-1])[:, : tiles * TILE_WORDS].reshape(win.shape[0] - 1, tiles, TILE_WORDS)
        out["tile_repeat"][b] = {"r": r, "share": float(same.all(axis=2).mean())}
    return out


class _Ops:
    def __init__(self) -> None:
        self.n = 0

    def lop3(self, v):
        self.n += 1
        return v


def _planes(v: np.ndarray, nbits: int) -> np.ndarray:
    """(…, n) int64 → (nbits, …, n/32) uint32 bit planes, 32 columns a register."""
    u = (v & 0xFFFFFFFF).astype(np.uint64)
    sh = u.shape[:-1] + (u.shape[-1] // 32, 32)
    u = u.reshape(sh)
    weights = (np.uint64(1) << np.arange(32, dtype=np.uint64))
    return np.stack([(((u >> np.uint64(k)) & np.uint64(1)) * weights).sum(-1).astype(np.uint32) for k in range(nbits)])


def _add(ops: _Ops, state: list, addend: list, regs: int) -> list:
    """Ripple-carry add of bit planes, LOP3 for each sum (XOR3) and carry (MAJ); `regs` registers a plane."""
    carry, out = None, []
    for k, (s, a) in enumerate(zip(state, addend)):
        if carry is None:
            out.append(s ^ a)
            carry = s & a
            ops.n += 2 * regs
        else:
            out.append(s ^ a ^ carry)
            carry = (s & a) | (s & carry) | (a & carry)
            ops.n += 2 * regs
    return out


def bitslice_window(parts: dict, r: int, b: int, merged: bool = True) -> dict:
    """The dataflow-W1 adversary for a window [r, r+b) of every lead word at zero data: each (row, 32 columns) keeps
    the running word's low r+b bits as planes and adds a step's increment by ripple carry, XOR3 and MAJ a bit (2 LOP3).
    `merged`: the increment's planes are the coin lanes' plus the tag lane's row constant, formed once a (tile, parity,
    32 columns, step) for each of the constant's two values, so a row pays one ripple a step and its tile shares the
    forming (3 adds for 127·s₁ + s₂ + s₃ and one for the constant, both versions). Otherwise the constant is a second
    ripple. The window planes are the packed message words. Returns LOP3s a word and whether every window bit matches."""
    x, y, z, words = parts["x"], parts["y"], parts["z"], parts["words"]
    s, m, n = z.shape
    nbits = r + b
    regs = m * n // 32
    groups = 2 * ((m + nsp.BLC_TILE - 1) // nsp.BLC_TILE) * (n // 32)
    ops = _Ops()
    state = list(_planes(x, nbits))
    ok = True
    shared = 0
    ones = np.uint32(0xFFFFFFFF)
    for t in range(s):
        dz = z[t] - (z[t - 1] if t else 0)
        dy = y[t] - (y[t - 1] if t else 0)
        if merged:
            state = _add(ops, state, list(_planes(dz + dy[:, None], nbits)), regs)
            shared += 2 * (3 + 1) * 2 * nbits * groups
        else:
            state = _add(ops, state, list(_planes(dz, nbits)), regs)
            shared += 3 * 2 * nbits * groups
            cpl = [np.where(((dy >> k) & 1)[:, None] == 1, ones, np.uint32(0)) * np.ones((1, n // 32), dtype=np.uint32)
                   for k in range(nbits)]
            state = _add(ops, state, cpl, regs)
        want = _planes(words[t], nbits)[r:nbits]
        ok &= all(np.array_equal(state[r + q], want[q]) for q in range(b))
    per_word = (ops.n + shared) / (s * m * n)
    return {"r": r, "b": b, "merged": merged, "eq_a_word": per_word, "exact": bool(ok)}


def honest_extract(b: int, predicates: str) -> float:
    """Honest's extraction and packing of b-bit windows, eq a word beyond the MMA's accumulator write (estimate):
    b ≤ 4 by predicates and VOTE (a predicate-only instruction a bit, free under W1 as stated), bytes and halves by
    PRMT merges, other widths by a shift and LOP3 merges."""
    if b == 32:
        return 0.0
    pred = 0.0 if predicates == "w1" else 1.0
    ballot = b * (pred + 1 / 32)
    merge = (1 - b / 32) + (0.0 if b in (8, 16) else 1.0)
    return min(ballot, merge) if b <= 4 else merge


def adversary_window(b: int, predicates: str, bitslice_eq: float) -> float:
    """The window adversary's eq a word at zero data: carry-out predicates (two predicate-only instructions a bit)
    and VOTE, or the bit-sliced state; never more than honest's own route."""
    pred = 0.0 if predicates == "w1" else 1.0
    route = b * (2 * pred + 1 / 32)
    return min(route, bitslice_eq, 1 + honest_extract(b, predicates))


def _bilinear_noise(m: int, n: int, s: int, lanes: int, family: str, rng: np.random.Generator) -> np.ndarray:
    """(s, m, n) step noise: `lanes` byte coins a row and step against wide coin weights a column and step; `mirror`
    replays the first half's coins against negated weights, `mirror-pi` in a coin-drawn order."""
    half = s // 2 if family.startswith("mirror") else s
    a = rng.integers(-128, 128, size=(half, lanes, m))
    w = rng.choice(nsp.WIDE, size=(half, lanes, n))
    first = np.einsum("tlm,tln->tmn", a, w)
    if family == "bilinear":
        return first
    order = np.arange(half) if family == "mirror" else rng.permutation(half)
    return np.concatenate([first, -first[order]])


def sketch_ranks(s: int = 8, lanes: int = 1, size: int = 96, seed: int = 1) -> dict:
    """The rank over Q of the per-output quadratic sketch M(i, j) = Σ_t w_t(i, j)² at zero data, for each family: the
    inner dimension the tensor-core decomposition needs, an output (before its byte split). The lead's carries X, Y
    and Z; the bilinear families carry only their noise. Also checks mirror's replay fold exactly."""
    rng = np.random.default_rng(seed)
    out = {}
    lead = lead_parts(size, size, s, rng)
    out["lead"] = int(np.linalg.matrix_rank((lead["words"].astype(np.float64) ** 2).sum(0)))
    out["lead_tiles"] = (size + nsp.BLC_TILE - 1) // nsp.BLC_TILE
    for fam in ("bilinear", "mirror", "mirror-pi"):
        noise = _bilinear_noise(size, size, s, lanes, fam, rng)
        w = np.cumsum(noise, axis=0).astype(np.float64)
        out[fam] = int(np.linalg.matrix_rank((w ** 2).sum(0)))
    out["bound"] = {"bilinear": (lanes * s) * (lanes * s + 1) // 2,
                    "mirror": (lanes * s // 2) * (lanes * s // 2 + 1) // 2}
    out["mirror_fold"] = mirror_fold(s, lanes, rng)
    return out


def lead_rank_bound(s: int, tiles: int) -> int:
    """The lead's per-output quadratic sketch as a sum of row-by-column products: X has rank ≤ 9 (four tag lanes, four
    a₁ lanes, the bias), so X² ≤ 45; 2X·ΣY ≤ 9; 2X·ΣZ ≤ 9 a (tile, parity); ΣY² 1; ΣZ² 1 a (tile, parity); and
    2·Σ_t Y_t·Z_t ≤ s a (tile, parity)."""
    groups = 2 * tiles
    return 45 + 9 + 9 * groups + 1 + groups + s * groups


def mirror_fold(s: int, lanes: int, rng: np.random.Generator, size: int = 32) -> dict:
    """Whether the second half's squares follow from the first half's sums. Mirror's second-half words are
    W_{h-1} − W_u (u < h), so Σ_u (W_{h-1} − W_u)² = h·W_{h-1}² − 2·W_{h-1}·Σ_u W_u + Σ_u W_u², exact mod 2^32: a
    prover holding the first half's words and squares pays O(1) an output for the second half. A coin-drawn replay
    order breaks the identity."""
    res = {}
    h = s // 2
    for fam in ("mirror", "mirror-pi"):
        w = np.cumsum(_bilinear_noise(size, size, s, lanes, fam, rng), axis=0) % (1 << 32)
        first, last = w[:h], w[h - 1]
        direct = ((w[h:] ** 2) % (1 << 32)).sum(0)
        folded = h * last * last - 2 * last * first.sum(0) + (first ** 2).sum(0)
        res[fam] = bool(np.all((direct - folded) % (1 << 32) == 0))
    return res


def _edge_noise(m: int, n: int, s: int, lanes: int, r0: int, rng: np.random.Generator) -> np.ndarray:
    """mirror-pi's step noise with r0 more coin lanes at step 0, removed at the last step (r0 = 0 is mirror-pi)."""
    noise = _bilinear_noise(m, n, s, lanes, "mirror-pi", rng)
    if r0:
        wide = rng.integers(-128, 128, size=(r0, m)).T @ rng.choice(nsp.WIDE, size=(r0, n))
        noise[0] += wide
        noise[-1] -= wide
    return noise


def edge_attack(lanes: int, r0: int, t0_max: int = 12, size: int = 400, seed: int = 3) -> dict:
    """The edge decomposition of a quadratic sketch: its first (or last) t0 words see few coin vectors, so the sum of
    their squares has low rank and costs rank × bytes MACs an output in place of honest's 2 eq a word, and the prover
    goes on honestly from the word at t0, which one MMA gives. Measures that rank at zero data for each t0 (exact below
    `size`, which exceeds 32·t0_max, the rank past which an edge stops paying at 2 bytes; `edge_saves` skips a rank that
    reaches it). The final word is left out: M0 knows it, and M1 closes it for every exact design."""
    rng = np.random.default_rng(seed)
    s = 2 * t0_max + 2
    w = np.cumsum(_edge_noise(size, size, s, lanes, r0, rng), axis=0).astype(np.float64)
    out: dict = {"lanes": lanes, "r0": r0, "size": size, "prefix": {}, "suffix": {}}
    for t0 in range(1, t0_max + 1):
        out["prefix"][t0] = int(np.linalg.matrix_rank((w[:t0] ** 2).sum(0)))
        out["suffix"][t0] = int(np.linalg.matrix_rank((w[s - 1 - t0:s - 1] ** 2).sum(0)))
    return out


def edge_saves(edge: dict, block: dict, sketch_bytes: int = SKETCH_BYTES) -> dict:
    """The edge and middle-block attacks' savings at `sketch_bytes` byte products a rank term. A middle block of L
    words, from the honest word N before it, writes Σ (N + Δ_t)² = N·(L·N + 2·ΣΔ) + ΣΔ²: the words ΣΔ and
    L·N + 2·ΣΔ and the product, 3 eq, and ΣΔ²'s decomposition, whose rank is the r0 = 0 prefix's (`block`), against
    honest's 2L (the next word is one MMA for both): it saves the best (2L − 3 − rank·bytes/32)/L eq a word. An edge
    saves 2·t0 − rank·bytes/32 eq an output a side; with r0 > 0 the wide block R is constant until the last step, so
    it also separates like a block's N, at one more word (R): 2·t0 − 4 − rank₀·bytes/32."""
    def best(ranks: dict, size: int, fixed: float, per_word: bool) -> float:
        vals = [(2 * n - fixed - r * sketch_bytes / 32) / (n if per_word else 1) for n, r in ranks.items() if r < size]
        return max([0.0, *vals])
    out = {}
    for side in ("prefix", "suffix"):
        out[side] = max(best(edge[side], edge["size"], 0, False),
                        best(block[side], block["size"], 4, False) if edge["r0"] else 0.0)
    out["block_a_word"] = best(block["prefix"], block["size"], 3, True)
    return out


def rfp4_check(m: int = 8, k: int = 16, n: int = 8, seg: int = 4, seed: int = 2) -> dict:
    """RFP-4 on a small instance: X = [A+E | G−A | −E·π | −G·π′], Y = [B+F; F; π(B+F); π′F], coins in [−64, 64]. Checks
    X·Y = A·B exactly, that each block's words are coin-dependent at zero data (A = 0), and counts each committed
    word's distinct coin pairs (covering) and how many of them an earlier block repeats (the reuse lemma's)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-63, 64, size=(m, k))
    b = rng.integers(-63, 64, size=(k, n))
    e, g = rng.integers(-64, 65, size=(m, k)), rng.integers(-64, 65, size=(m, k))
    f = rng.integers(-64, 65, size=(k, n))
    p, q = np.eye(k, dtype=np.int64)[rng.permutation(k)], np.eye(k, dtype=np.int64)[rng.permutation(k)]
    x = np.hstack([a + e, g - a, -e @ p, -g @ q])
    yy = np.vstack([b + f, f, p.T @ (b + f), q.T @ f])
    exact = bool(np.array_equal(x @ yy, a @ b))
    x0 = np.hstack([e, g, -e @ p, -g @ q])
    y0 = np.vstack([f, f, p.T @ f, q.T @ f])
    src_p = [int(np.argmax(p[:, c])) for c in range(k)]
    src_q = [int(np.argmax(q[:, c])) for c in range(k)]
    names_x = [("e", r) for r in range(k)] + [("g", r) for r in range(k)] + [("e", r) for r in src_p] + \
        [("g", r) for r in src_q]
    names_y = [("f", r) for r in range(k)] * 2 + [("f", r) for r in src_p] + [("f", r) for r in src_q]
    steps = 4 * k // seg
    seen, distinct, repeated = set(), [], []
    for t in range(steps):
        new = {(names_x[c], names_y[c]) for c in range(t * seg, (t + 1) * seg)}
        repeated.append(len(new & seen))
        distinct.append(len(new))
        seen |= new
    zero_words = np.cumsum(np.stack([x0[:, c][:, None] * y0[c][None, :] for c in range(4 * k)]), axis=0)
    seg_words = zero_words[seg - 1::seg]
    coin_dependent = [bool(np.any(wd != 0)) for wd in seg_words[:-1]]
    return {"exact": exact, "segment_pairs": distinct, "repeated_pairs": repeated,
            "repeated_share": sum(repeated) / sum(distinct), "coin_dependent_before_end": all(coin_dependent),
            "final_known": bool(np.all(seg_words[-1] == 0))}


def _runs_saving(groups: np.ndarray) -> tuple[int, int]:
    """Over each row of `groups`, the values met k ≥ 2 times: Σ (k − 2)⁺, a sketch's saving, and Σ (k − 1), a hashed
    design's duplicates."""
    srt = np.sort(groups, axis=1)
    start = np.ones(srt.shape, dtype=bool)
    start[:, 1:] = srt[:, 1:] != srt[:, :-1]
    ids = np.cumsum(start.ravel()) - 1
    k = np.bincount(ids)
    return int(np.maximum(k - 2, 0).sum()), int((k - 1).sum())


def sketch_pools(m: int, n: int, s: int, lanes: int, r0: int, salt_bits: int, tile: int = 64, seed: int = 4) -> dict:
    """Equal words an adversary can share in mirror-π's sketch at zero data, on chosen identical rows (the data
    cancels; code and tags are left out, which only adds collisions). A pool of k equal values costs it one word, one
    square and k adds against honest's 2k, so it saves k − 2 eq; a hashed design's duplicates save k − 1 words. Counted
    within an output (A2r), within a (64-row tile, column) and a 64 × 64 block over every step, and within a whole
    column or row over every step. Salts ρ_i + c_j of `salt_bits` each enter at step 0 and are folded out with the
    output (the stock epilogue's two IADD3 sources). The final word is left out (M1). Shares are over stock."""
    rng = np.random.default_rng(seed)
    w = np.cumsum(_edge_noise(m, n, s, lanes, r0, rng), axis=0)[:-1]
    if salt_bits:
        w += rng.integers(0, 1 << salt_bits, size=m)[None, :, None]
        w += rng.integers(0, 1 << salt_bits, size=n)[None, None, :]
    w %= 1 << 32
    words = w.size
    pools = {"A2r": w.transpose(1, 2, 0).reshape(m * n, s - 1),
             "tile": w.reshape(s - 1, m // tile, tile, n).transpose(1, 3, 0, 2).reshape(m // tile * n, -1),
             "column": w.transpose(2, 0, 1).reshape(n, -1),
             "row": w.transpose(1, 0, 2).reshape(m, -1)}
    if n % tile == 0:
        pools["block"] = (w.reshape(s - 1, m // tile, tile, n // tile, tile).transpose(1, 3, 0, 2, 4)
                          .reshape(m // tile * (n // tile), -1))
    out: dict = {"m": m, "n": n, "s": s, "lanes": lanes, "r0": r0, "salt_bits": salt_bits}
    for name, g in pools.items():
        sk, hs = _runs_saving(g)
        out[name] = {"sketch": sk / (words * STOCK_EQ), "hash": hs / (words * STOCK_EQ)}
    return out


def partition_search(edge: dict, block: dict, s: int, sketch_bytes: int = SKETCH_BYTES) -> dict:
    """The cheapest split of one output's s words into honest words (2 eq), middle blocks decomposed from the word
    before them (3 eq + rank·bytes/32), and edges decomposed jointly (rank·bytes/32) or with the wide block split off
    (4 eq + rank₀·bytes/32), by dynamic programming over every partition the measured ranks cover. Returns the cost
    and the saving against honest's 2s, eq an output."""
    inf = float("inf")

    def seg(ranks: dict, size: int, fixed: float) -> dict:
        return {n: fixed + r * sketch_bytes / 32 for n, r in ranks.items() if r < size}

    def edge_costs(side: str) -> dict:
        costs = seg(edge[side], edge["size"], 0)
        if edge["r0"]:
            for n, c in seg(block[side], block["size"], 4).items():
                costs[n] = min(c, costs.get(n, inf))
        return costs
    pre, suf = edge_costs("prefix"), edge_costs("suffix")
    mid = seg(block["prefix"], block["size"], 3)
    best = [inf] * (s + 1)
    best[0] = 0.0
    for n, c in pre.items():
        if n <= s:
            best[n] = min(best[n], c)
    for t in range(s):
        if best[t] == inf:
            continue
        best[t + 1] = min(best[t + 1], best[t] + 2)
        if t > 0:
            for n, c in mid.items():
                if t + n <= s:
                    best[t + n] = min(best[t + n], best[t] + c)
    cost = min([best[s]] + [best[s - n] + c for n, c in suf.items() if n < s])
    return {"cost": cost, "saving": 2 * s - cost}


def pools_grid(shapes: tuple = ((512, 64), (8192, 64), (64, 8192)), salts: tuple = (0, 18, 24, 31)) -> list[dict]:
    """`sketch_pools` over rows × columns, one lane (mirror's) and four (with and without the step-0 block), and salt
    widths."""
    rows = []
    for m, n in shapes:
        for lanes, r0 in ((1, 0), (4, 0), (4, MIRROR_PI_STEP0)):
            for salt in salts:
                rows.append(sketch_pools(m, n, 74, lanes, r0, salt))
    return rows


#: N-EGRESS's network term for the lead at cap 32 (floor and bar W_mac), at its n_max.
NET_LEAD = {"term": (0.00037, 0.00072), "n_max": 85, "cap": 32}
MIRROR_PI_STEP0 = 4


def network_terms(lanes: int = 4, r0: int = MIRROR_PI_STEP0, coin_bits: int = 8, tile: int = 64,
                  rows: tuple = (1, 16, 64, 512, 8192)) -> dict:
    """N-EGRESS's network term β·E over stock for mirror-π's sketch, scaled from the lead's figure (checked arithmetic).
    β = n_max · W_mac. n_max is the most data MACs one exchanged coin bit carries: a row coin shared by `a_cols`
    columns, or a column weight shared by `b_rows` rows, each read at its step and again at its replay, over ℓ lanes
    of `coin_bits`. W_mac is the honest block's cost a MAC, so it scales by the honest eq a word, 2 + SHA_WORD/S,
    over the lead's 1 + SHA_WORD. Layouts: `rows` (a byte a row and step, a weight a column and step: mirror's),
    `a-tiled` (row coins a 64-column tile), and `tiled` (row coins a column tile, weights a row tile). Also returns
    the coin stream a token (bytes, Llama's linears, the r0 block included), and mirror with whole words hashed."""
    out: dict = {}
    calls = {name: (n, k) for name, (n, k) in ns.LLAMA.items()}
    layers = 16
    for layout in ("rows", "a-tiled", "tiled"):
        res = {}
        for t in rows:
            beta, stream = 0.0, 0.0
            for name, (ncols, k) in calls.items():
                s = nsp.blc_steps(k)
                a_cols = ncols if layout == "rows" else min(tile, ncols)
                b_rows = t if layout != "tiled" else min(tile, t)
                n_max = 2 * max(a_cols, b_rows) * nsp.BLC_DATA_LANES[1] / (lanes * coin_bits)
                ratio = (2 + SHA_WORD / s) / (1 + SHA_WORD)
                beta = max(beta, n_max * ratio)
                per_step = lanes * (s // 2) + r0
                stream += layers * per_step * (ncols / a_cols + ncols / b_rows) * coin_bits / 8
            res[t] = {"beta_over_lead": beta / NET_LEAD["n_max"], "stream_bytes_a_token": stream,
                      **{f"cap{cap}": tuple(x * beta / NET_LEAD["n_max"] * cap / NET_LEAD["cap"]
                                            for x in NET_LEAD["term"]) for cap in (32, 320)}}
        out[layout] = res
    n_gate_up = ns.LLAMA["gate_up"][0]
    out["mirror_hashed_check"] = NET_LEAD["term"][0] * (31 * n_gate_up / 8) / NET_LEAD["n_max"]
    return out


def checks(m: int = 128, n: int = 128, s: int = 74, seed: int = 0) -> dict:
    """The CPU checks this module's scores read."""
    rng = np.random.default_rng(seed)
    out = {"size": {"m": m, "n": n, "s": s, "seed": seed}}
    zero = lead_parts(m, n, s, rng)
    data = lead_parts(m, n, s, rng, data_sigma=float(ns.WEIGHT_SIGMA_INT8))
    out["fresh_zero"] = freshness(zero["words"])
    out["fresh_data"] = freshness(data["words"])
    rf = out["fresh_zero"]["r_fresh"]
    out["bitslice"] = [bitslice_window(zero, max(0, rf - b + 1), b) for b in (1, 2, 4, 8, 16)]
    out["bitslice_ripple2"] = [bitslice_window(zero, max(0, rf - b + 1), b, merged=False) for b in (1, 8)]
    out["bitslice_top"] = [bitslice_window(zero, 31, 1)]
    out["sketch"] = {f"s{st}_l{ln}": sketch_ranks(st, ln, size=sz)
                     for st, ln, sz in ((8, 1, 96), (12, 1, 128), (16, 1, 192), (8, 2, 192), (12, 2, 320))}
    out["lead_rank_within_bound"] = all(
        v["lead"] <= lead_rank_bound(int(key[1:].split("_")[0]), tiles=v["lead_tiles"]) for key, v in out["sketch"].items())
    out["rfp4"] = rfp4_check()
    out["edge"] = {f"l{ln}_r{r0}": edge_attack(ln, r0) for ln in (1, 2, 3, 4) for r0 in (0, 4, 8)}
    out["partition"] = {f"l{ln}_r{r0}_B{nb}": {st: partition_search(out["edge"][f"l{ln}_r{r0}"],
                                                                     out["edge"][f"l{ln}_r0"], st, nb)["saving"]
                                                 for st in (74, 293)}
                        for ln in (2, 3, 4) for r0 in (0, 4, 8) for nb in (2, 3, 4)}
    return out


def _llama_k_shares() -> dict:
    macs = {name: mn[0] * mn[1] for name, mn in ns.LLAMA.items()}
    tot = sum(macs.values())
    shares: dict = {}
    for name, (_, k) in ns.LLAMA.items():
        shares[k] = shares.get(k, 0) + macs[name] / tot
    return shares


def score(chk: dict, predicates: str = "dataflow", sketch_bytes: int = SKETCH_BYTES) -> list[dict]:
    """The joint table: one row a (family, which, bits, kind) design over Llama's linears, MAC-weighted by K.
    `predicates` reads predicate-only instructions as W1 states them (`w1`, free) or as dataflow-W1 prices them
    (`dataflow`, 1 eq); `sketch_bytes` is the byte products a quadratic decomposition pays a rank term."""
    se = STOCK_EQ
    shares = _llama_k_shares()
    steps = {k: nsp.blc_steps(k) for k in shares}
    rf = chk["fresh_zero"]["r_fresh"]
    bs = {row["b"]: row["eq_a_word"] for row in chk["bitslice"]}
    rows = []
    base = {"lead": {"t1": 0.00066, "t3": "-0.029% (served, measured)", "rank": 5},
            "mirror": {"t1": 0.0062, "t3": "code part served", "rank": 5},
            "mirror-pi": {"t1": 0.0062, "t3": "code part served (as mirror)", "rank": 5},
            "bilinear": {"t1": 0.0096, "t3": "fails: the lanes' noise is served", "rank": 5},
            "rfp4": {"t1": 0.0, "t3": "3.68% (int7 data)", "rank": 5}}

    def add(fam, which, bits, kind, h_ratio, t1_terms, premises, rank, attacks, t2_scale, mma=LEAD_MMA):
        t1 = sum(t1_terms.values())
        t2 = {p: tuple(round(mma + (v - LEAD_MMA) * t2_scale, 2) for v in vals) for p, vals in T2_LEAD.items()}
        rows.append({"design": f"{fam}/{which}/b{bits}/{kind}", "h_over_lead": h_ratio,
                     "h": tuple(round(h * h_ratio, 2) for h in H_LEAD), "t1": t1, "t1_terms": t1_terms,
                     "t2": t2, "t3": base[fam]["t3"], "premises": premises, "rank": rank, "attacks": attacks,
                     "passes": t1 <= 0.01 and h_ratio <= 0.1 and all(v[1] <= 100 for v in t2.values())
                     and not base[fam]["t3"].startswith("fails")})

    for fam in ("lead", "mirror", "bilinear"):
        add(fam, "every", 32, "hash", 1.0, {"base": base[fam]["t1"]}, ["DistinctWrites", "ShaWrites"],
            base[fam]["rank"], {"A1/A2/A2c/A2r": "the family's"}, 1.0)
    for b in COMMIT_BITS[1:]:
        e_h = honest_extract(b, predicates)
        r = max(0, rf - b + 1)
        adv = adversary_window(b, predicates, bs.get(b, 4 * (r + b) / 32))
        gap = max(0.0, 1 + e_h - adv) / se
        hr = (SHA_WORD * b / 32 + e_h) / SHA_WORD
        add("lead", "every", b, "hash", hr, {"base": 0.00066, "window": gap},
            ["WindowWrites (new)", "dataflow-W1" if predicates == "dataflow" else "W1 as stated"],
            "4a+", {"bit-slice / carry predicates": f"adversary {adv:.2f} eq a word vs honest {1 + e_h:.2f}",
                    "window": [r, r + b], "tile repeat": chk["fresh_zero"]["tile_repeat"].get(b, {})}, hr)
    for kseg in (2, 4, 8, 16, 64):
        hr = 1 / kseg
        add("lead", f"every-{kseg}", 32, "hash", hr, {"base": 0.00066, "coverage": (kseg - 1) / kseg / se},
            ["DistinctWrites", "CoverCount"], 5, {"zero-data packing": "a segment's k coin lanes in one MMA"}, hr)
    hr = sum(sh / steps[k] for k, sh in shares.items())
    add("lead", "every", 32, "linear", hr, {"fold": sum(sh * (1 - 1 / steps[k]) for k, sh in shares.items()) / se},
        ["none: folds (H12)"], 0, {"fold": "Σ r_t w_t = Σ_τ κ_τ Σ_{t≥τ} r_t"}, hr)
    quads = [("lead", 3, 0), ("mirror", 1, 0), ("mirror", 2, 0), ("bilinear", 1, 0)]
    quads += [("mirror-pi", ln, r0) for ln in (1, 2, 3, 4) for r0 in (0, 4, 8)]
    for fam, lanes, r0 in quads:
        gap, detail = 0.0, {}
        hr = 0.0
        sv = (edge_saves(chk["edge"][f"l{lanes}_r{r0}"], chk["edge"][f"l{lanes}_r0"], sketch_bytes)
              if fam == "mirror-pi" else None)
        for k, sh in shares.items():
            s = steps[k]
            if fam == "lead":
                rank = lead_rank_bound(s, tiles=1)
            elif fam == "bilinear":
                rank = (lanes * s) * (lanes * s + 1) // 2 + 9 * lanes * s
            else:
                rank = (lanes * s // 2) * (lanes * s // 2 + 1) // 2 + 9 * lanes * s // 2
            dec = rank * sketch_bytes / 32 / s
            replay = 1.0 if fam == "mirror" else 2.0
            whole = max(0.0, 2.0 - min(2.0, dec, replay))
            parts = (sv["prefix"] + sv["suffix"]) / s + sv["block_a_word"] if sv else 0.0
            gap += sh * min(2.0, max(whole, parts)) / se
            hr += sh * (1 + SHA_WORD / s) / SHA_WORD
            detail[f"K{k}"] = {"steps": s, "rank": rank, "decomposition_eq": round(dec, 3), "replay_eq": replay,
                               "edges_and_blocks_eq_a_word": round(parts, 4)}
        add(fam, "every", 32, f"quad-l{lanes}" + (f"-w{r0}" if r0 else ""), hr,
            {"base": base[fam]["t1"], "sketch": gap},
            ["SketchWrites2 (new)", "H32"] + (["mirror's"] if fam.startswith("mirror") else []), "4a+",
            {"decomposition": detail, "replay fold": fam == "mirror", "edges and blocks": sv,
             "sketch_bytes": sketch_bytes}, hr)
    for words, label in ((3, "blocks"), (7, "half-blocks")):
        hr = sum(sh * words / steps[k] for k, sh in shares.items())
        skip_m0 = sum(sh * (4 * k / (words + 1)) / k for k, sh in shares.items())
        for model, skip in (("M0", skip_m0), ("M1", 0.0)):
            for lemma, reuse in (("reuse", 0.0), ("argued", RFP_REUSE_ARGUED)):
                add("rfp4", f"{label}-{model}-{lemma}", 32, "hash", hr, {"skip": skip, "reuse": reuse},
                    ["CoverCountS8 (proved)", "WideNoGain (4a)"]
                    + (["reuse lemma (open)"] if lemma == "reuse" else ["projection bound (argued)"])
                    + (["M1 final-word lemma (3)"] if model == "M1" else []), 5 if lemma == "reuse" else "4a",
                    {"skip": "L/K in M0", "A1/A2": "dense coins", "zero-data b1": "coins span 129 values"}, hr,
                    mma=4.0)
    return rows


def table(rows: list[dict]) -> str:
    head = f"{'design':34} {'h/lead':>7} {'h':>13} {'T1':>8} {'T2 64r':>13} {'rank':>5} pass"
    lines = [head]
    for r in sorted(rows, key=lambda r: (not r["passes"], r["t1"], r["h_over_lead"])):
        lines.append(f"{r['design']:34} {r['h_over_lead']:7.4f} {str(r['h']):>13} {r['t1']:8.2%} "
                     f"{str(r['t2']['pt512x64']):>13} {str(r['rank']):>5} {'yes' if r['passes'] else 'no'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", help="write the checks and the scores here")
    ap.add_argument("--size", type=int, default=128, help="rows and columns of the lead's simulated words")
    ap.add_argument("--steps", type=int, default=74)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pools", help="write mirror-π's network terms and its sketch pools here, and stop")
    args = ap.parse_args()
    if args.pools:
        res = {"network": network_terms(), "pools": pools_grid()}
        for r in res["pools"]:
            print(r["m"], r["n"], f"l{r['lanes']}", f"r0={r['r0']}", f"salt={r['salt_bits']}",
                  {k: f"{v['sketch']:.4%} / {v['hash']:.3%}" for k, v in r.items() if isinstance(v, dict)}, flush=True)
        with open(args.pools, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        return
    chk = checks(args.size, args.size, args.steps, args.seed)
    scores = {pr: score(chk, pr) for pr in ("w1", "dataflow")}
    for pr, rows in scores.items():
        print(f"\npredicate-only instructions: {pr}\n{table(rows)}")
    scores["sketch_bytes"] = {nb: [r for r in score(chk, "dataflow", nb) if r["design"].startswith("mirror-pi")]
                              for nb in (2, 3, 4)}
    for nb, rows in scores["sketch_bytes"].items():
        print(f"\nmirror-pi at {nb} byte products a rank term\n{table(rows)}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"checks": chk, "scores": scores}, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
