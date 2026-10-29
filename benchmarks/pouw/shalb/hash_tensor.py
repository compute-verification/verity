"""HASH's FOLD/TENSOR lane: the tensor-core fold, whether it folds linearly, its zero-data adversaries, and its price.

    python3 hash_tensor.py LEAD.json [--verdict PATH] [--json OUT]

LEAD.json is hash_sha_lead.py's output. Units as hash_sha.py: W1, 1/32 of a register write; stock, 1 W1 a useful MAC;
h, honest's commitment W1 a stock MAC; a word, one k32 IMMA output (32 honest MACs, so r/32 words a stock MAC).

The fold. For each 64-row tile of BN words a row and each k32 step t, the step's words W_t enter as their four s8
bytes, through R (4·BN × 8, uniform s8) and the running state Z (16 bytes a row) through Q (16 × 8, uniform s8):
    Y_t = bytes(W_t)·R + Z_{t-1}·Q, exact in s32, and Z_t = σ(Y_t), Y_t's two low bytes an output, each read as s8
(two PRMTs a thread build the next k16 A fragment). R and Q are drawn by live coins after the operands' commitment and
before the call, one draw a call. The tile's final Z is sealed by SHA-512 under a hiding leaf. Honest writes the
fold's D registers (BN/8 + 1 n8 MMAs a row-tile step, 4 registers a thread each) and two PRMTs: 1 + 12/BN registers a
word, beside the stock IMMA's one.

- A. The fragment identity: a thread's s32 accumulators of one n8 slice are, register for register, the s8 A
  fragment of one m16n8k32 step (the PTX ISA's layouts, read), so honest feeds the in-place accumulators to the fold
  with no conversion; the order within the k32 is a fixed permutation of the slice's words, absorbed into R's rows.
- L. Linearity: a fold linear in the words folds through the increments (H12); bytes(W)·R doesn't, because the bytes
  carry; the carry route needs the bytes' corrections at every step; σ makes every step's words enter the state; a
  low-byte σ pushes through mod 256, bits 8..23 drop small changes, and a saturating σ keeps only signs.
- K. Low rank over GF(2^31 - 1): the zero-data byte matrices against the word matrices, and the incremental route for
  slowly changing bytes priced against the MMA slot it replaces.
- T. Tables: the distinct values each byte position takes.
- P. Pack: exact adversaries in byte-aligned lanes that the MMA reads directly, bit for bit against honest's state,
  with their writes modelled part by part, predicate writes free and priced: offset data in 24- or 16-bit lanes,
  the lead's signed data in a two's-complement stream, and the class-sharing prover (16-bit lanes, a row class's
  high bytes once a class).
- C. Credit, h, T1 (credit = cost(H) and credit = stock), T2 (two readings of the fold's tensor rate), the opening's
  AND gates, and the per-period figures.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))
from exact_kit.checks import Checks

M32 = (1 << 32) - 1
P31 = (1 << 31) - 1
W1_PER_REG = 32
BLOCK = 128
R_CHAIN = 3532
CHUNK_K, CHUNK_EXTRA = 2213, 5
ROWS, NOUT, ZB = 64, 8, 16
BN = 256
STEPS = {"k2048": 67, "k8192": 265}
POINTS = ("t512", "t512r512", "t8192")
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
H200_PEAK = 9.89e14
H200_MACS = H200_PEAK * 0.4
DC_GPUS = 100_000
WORLD_MACS_PER_YEAR = 4.68e30
SHA_AND_PER_BYTE = 454
AND_PER_MAC = 2079 / 16
# The fold's tensor rate, two readings: mma.sync m16n8k32 at half of the H200's int8 peak, and the n8 price measured
# for e4m3 wgmma (144 W1 a register against the model's 32).
FOLD_RATE = {"half-peak": 0.5, "n8-measured": 32 / 144}
OFFSET = {"rot": 1 << 20, "e7": 1 << 15}


def r_row(row: int) -> float:
    return R_CHAIN - CHUNK_K / (row / BLOCK + CHUNK_EXTRA)


def s8(x):
    x = np.asarray(x, dtype=np.int64) & 255
    return x - ((x & 128) << 1)


def cell_bytes(w):
    """(..., n) words in [0, 2^32) -> (..., n, 4) s8 bytes, byte 0 first."""
    return s8(np.asarray(w, dtype=np.int64)[..., None] >> (8 * np.arange(4)))


def row_bytes(w):
    b = cell_bytes(w)
    return b.reshape(*b.shape[:-2], -1)


def sigma(y):
    return np.concatenate([s8(y), s8(y >> 8)], axis=-1)


def sigma_mid(y):
    return np.concatenate([s8(y >> 8), s8(y >> 16)], axis=-1)


def sigma_lowbyte(y):
    return np.concatenate([s8(y), np.zeros_like(y)], axis=-1)


def mulmod32(a, b):
    a = np.asarray(a, dtype=np.int64) & M32
    b = np.asarray(b, dtype=np.int64) & M32
    return ((a * (b & 0xFFFF)) + (((a * (b >> 16)) & 0xFFFF) << 16)) & M32


def honest_fold_writes(bn: int = BN) -> float:
    return ((bn // 8 + 1) * 4 + 2) / (bn / 2)


def bal13(z: np.ndarray) -> np.ndarray:
    """(..., ) integers with |z| <= 14,280 -> (..., 4) balanced base-13 digits in [-6, 6]."""
    z = np.asarray(z, dtype=np.int64)[..., None] + 14280
    return z // 13 ** np.arange(4) % 13 - 6


def tag_d0(rng, m: int, n: int, S: int) -> np.ndarray:
    """The lead's step-0 word at zero data (Lead/Layout.lean's `word` at t = 0, κ = 0): -β_j + T(i)·G[:, j] +
    a₁[i]·C[:, j], with T(i) the digits of (⌊i/2⌋ + 1)·(-1)^i, C[:, j] those of j + 1, G in [-3, 3] and a₁ bits drawn
    after the data."""
    i = np.arange(m)
    T = bal13(np.where(i % 2 == 0, i // 2 + 1, -(i // 2 + 1)))
    C = bal13(np.arange(n) + 1)
    G = rng.integers(-3, 4, (n, 4), dtype=np.int64)
    a1 = rng.integers(0, 2, (m, 4), dtype=np.int64)
    beta = (C.sum(axis=1) + 3 * (S - 1) + 1) // 2
    return -beta[None, :] + T @ G.T + a1 @ C.T


def lead_words(rng, layout: str, m: int, n: int, S: int, base: str) -> dict:
    """COUNTER's zero-data form of the lead's words (N-RED's lead_instance as COUNTER reads it): c_t(i, j) =
    b_t(i) + Σ_l U_l(class i)·v_{t,l}(j), three odd s8 coin lanes, weights (127, 1, 1) / (1, 127, 1) by row parity at
    rot and 1 at e7, and a per-cell d0 in ±2S. `tag` takes d0 as the lead's own step-0 term (`tag_d0`), which makes
    every row distinct. `offset` adds a constant the prover's data reaches in its first steps (s8 data on 27 lanes adds
    up to 2^18.75 a step), so every word stays positive with constant high bytes; `uniform` adds a uniform 32-bit base
    a cell."""
    b = rng.integers(0, 2, (S, m), dtype=np.int64)
    v = rng.choice(np.arange(-127, 128, 2), (S, n, 3)).astype(np.int64)
    U = (np.array([(127, 1, 1), (1, 127, 1)])[np.arange(m) % 2] if layout == "rot"
         else np.ones((m, 3), np.int64)).astype(np.int64)
    cls = np.arange(m) % 2 if layout == "rot" else np.zeros(m, np.int64)
    col = np.einsum("pl,tjl->tpj", np.array([U[0], U[1 % m]]), v)
    c = b[:, :, None] + col[:, cls, :]
    d0 = (tag_d0(rng, m, n, S) if base == "tag"
          else rng.integers(-2 * S, 2 * S + 1, (m, n), dtype=np.int64))
    if base == "offset":
        d0 = d0 + OFFSET[layout]
    elif base == "uniform":
        d0 = d0 + rng.integers(0, 1 << 32, (m, n), dtype=np.int64)
    wint = d0[None] + np.cumsum(c, axis=0)
    return {"wint": wint, "w": wint & M32, "c": c, "d0": d0, "b": b, "col": col, "cls": cls}


def draw_coins(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    return (rng.integers(-128, 128, (4 * n, NOUT), dtype=np.int64),
            rng.integers(-128, 128, (ZB, NOUT), dtype=np.int64))


def chain(w, R, Q, sig=sigma, collect: bool = False):
    """Honest's fold over words w (S, m, n): the final state, and each step's Y if `collect`."""
    S, m, _ = w.shape
    Z = np.zeros((m, ZB), np.int64)
    ys = []
    for t in range(S):
        Y = row_bytes(w[t]) @ R + Z @ Q
        if collect:
            ys.append(Y)
        Z = sig(Y)
    return (Z, ys) if collect else Z


def seal(Z) -> str:
    return hashlib.sha512(np.asarray(Z, dtype=np.int8).tobytes()).hexdigest()


def rank_mod_p(M, p: int = P31) -> int:
    A = np.asarray(M, dtype=np.int64) % p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = np.nonzero(A[r:, c])[0]
        if len(piv) == 0:
            continue
        k = r + piv[0]
        A[[r, k]] = A[[k, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        f = A[:, c].copy()
        f[r] = 0
        nz = np.nonzero(f)[0]
        if len(nz):
            A[nz] = (A[nz] - (f[nz, None] * A[r][None, :]) % p) % p
        r += 1
    return r


def part_a(checks: Checks, rng) -> dict:
    W = rng.integers(0, 1 << 32, (ROWS, BN), dtype=np.int64)
    R, _ = draw_coins(rng, BN)
    Y = np.zeros((ROWS, NOUT), np.int64)
    used = np.zeros((ROWS, BN), np.int64)
    for ni in range(BN // 8):
        A = np.full((ROWS, 32), 1 << 40, np.int64)
        Rni = np.zeros((32, NOUT), np.int64)
        for q in range(4):
            for e in range(4):
                Rni[4 * q + e] = R[4 * (8 * ni + 2 * q) + e]
                Rni[16 + 4 * q + e] = R[4 * (8 * ni + 2 * q + 1) + e]
        for warp in range(4):
            for g in range(8):
                r0 = 16 * warp + g
                for q in range(4):
                    cols = (8 * ni + 2 * q, 8 * ni + 2 * q + 1)
                    d = [(r0, cols[0]), (r0, cols[1]), (r0 + 8, cols[0]), (r0 + 8, cols[1])]
                    a = [d[0], d[2], d[1], d[3]]
                    for reg, (wr, wc) in enumerate(a):
                        used[wr, wc] += 1
                        row = r0 + (8 if reg in (1, 3) else 0)
                        assert row == wr
                        for e in range(4):
                            A[row, 4 * q + e + (16 if reg >= 2 else 0)] = s8(W[wr, wc] >> (8 * e))
        Y += A @ Rni
    direct = row_bytes(W) @ R
    checks.check("A: a thread's s32 accumulators of an n8 slice, passed as (d0, d2, d1, d3), are the s8 A fragment of "
                 "one m16n8k32 step; the fold through the fragments equals bytes(W)·R with R's rows permuted",
                 np.array_equal(Y, direct) and np.all(used == 1),
                 f"64 x {BN} words, {BN // 8} slices, every accumulator used once")
    return {"ok": bool(np.array_equal(Y, direct))}


def part_l(checks: Checks, rng) -> dict:
    out = {}
    d = lead_words(rng, "rot", ROWS, BN, STEPS["k2048"], "small")
    w, c, d0 = d["w"], d["c"], d["d0"]
    S = w.shape[0]
    rho = rng.integers(0, 1 << 32, (S, ROWS, BN), dtype=np.int64)
    direct = np.zeros((ROWS, BN), np.int64)
    for t in range(S):
        direct = (direct + mulmod32(rho[t], w[t])) & M32
    suffix = np.cumsum(rho[::-1], axis=0)[::-1] & M32
    folded = mulmod32(d0, suffix[0])
    for t in range(S):
        folded = (folded + mulmod32(c[t], suffix[t])) & M32
    checks.check("L: a sketch linear in the words, Σ_t ρ_t·W_t mod 2^32, folds through the increments (H12): "
                 "d0·Σρ + Σ_τ c_τ·Σ_{t≥τ}ρ_t, and c_τ is a row term plus a column term at zero data",
                 np.array_equal(direct, folded), f"64 x {BN} cells, S {S}")
    Rs = rng.integers(-128, 128, (S, 4 * BN, NOUT), dtype=np.int64)
    honest = sum(row_bytes(w[t]) @ Rs[t] for t in range(S))
    suf = np.cumsum(Rs[::-1], axis=0)[::-1]
    fold = row_bytes(d0 & M32) @ suf[0] + sum(row_bytes(c[t] & M32) @ suf[t] for t in range(S))
    mis = float(np.mean(fold != honest))
    checks.check("L: the byte sketch Σ_t bytes(W_t)·R_t doesn't fold through the increments' bytes: the outputs "
                 "mismatch", mis > 0.99, f"{mis:.2%} of outputs differ")
    prev = cell_bytes(d0 & M32)
    dens = np.zeros(4)
    for t in range(S):
        cur = cell_bytes(w[t])
        dens += np.mean((cur - prev - cell_bytes(c[t] & M32)) != 0, axis=(0, 1))
        prev = cur
    dens /= S
    checks.check("L: the carry route needs a correction at every byte position on a quarter to a half of (word, "
                 "step): bytes(W_t) - bytes(W_t-1) - bytes(c_t) is nonzero at byte 0 when the s8 sum wraps, and at "
                 "bytes 1-3 when it wraps or a carry comes in",
                 dens[0] > 0.2 and min(dens[1:]) > 0.4, ", ".join(f"byte {i} {x:.1%}" for i, x in enumerate(dens)))
    out["carry_density"] = dens.tolist()

    S2 = 16
    w2 = w[:S2]
    R, Q = draw_coins(rng, BN)
    sigmas = {"bits 0..15": sigma, "bits 8..23": sigma_mid}
    sens = {}
    for name, sig in sigmas.items():
        Z0 = chain(w2, R, Q, sig)
        trials, one, pair = 300, 0, 0
        for _ in range(trials):
            t, i, j = rng.integers(0, S2), rng.integers(0, ROWS), rng.integers(0, BN)
            j2 = (j + rng.integers(1, BN)) % BN
            wa = w2.copy()
            wa[t, i, j] = (wa[t, i, j] + 1) & M32
            one += not np.array_equal(chain(wa, R, Q, sig), Z0)
            wa[t, i, j2] = (wa[t, i, j2] - 1) & M32
            pair += not np.array_equal(chain(wa, R, Q, sig), Z0)
        sens[name] = {"one": one / trials, "pair": pair / trials}
    checks.check("L: with σ = bits 0..15 every step, one word changed by 1 at any step changes the sealed state, and "
                 "so does a sum-preserving pair (+1, -1 in one row); bits 8..23 would drop such changes a third of "
                 "the time, since a small change of Y rarely carries past its low byte",
                 sens["bits 0..15"]["one"] > 0.98 and sens["bits 0..15"]["pair"] > 0.98,
                 "; ".join(f"{k}: one word {v['one']:.1%}, pair {v['pair']:.1%}" for k, v in sens.items())
                 + f", over S {S2}")
    out["sensitivity"] = sens

    A1 = rng.integers(-128, 128, (S2, ROWS, 4 * BN), dtype=np.int64)
    A2 = rng.integers(-128, 128, (S2, ROWS, 4 * BN), dtype=np.int64)

    def bchain(As, sig):
        Z = np.zeros((ROWS, ZB), np.int64)
        for t in range(S2):
            Z = sig(As[t] @ R + Z @ Q)
        return Z

    def additive(sig):
        d = (bchain(A1, sig) + bchain(A2, sig) - bchain(A1 + A2, sig)) % 256
        return float(np.mean(d[:, :NOUT] == 0)), float(np.mean(d[:, NOUT:] == 0))

    low = additive(sigma_lowbyte)[0]
    add = {k: additive(s) for k, s in sigmas.items()}
    _, ys = chain(w2, R, Q, collect=True)
    ylog = float(np.log2(np.percentile(np.abs(np.stack(ys)), 99)))
    checks.check("L: a low-byte σ pushes through: the chain is then additive in its input bytes mod 256 (a ring map "
                 "end to end), so a linear fold reaches the state; with σ = bits 0..15 neither half of the state is "
                 "additive (the carry into bit 8 is not a ring map); bits 8..23's top half is often additive, because "
                 "|Y| stays near 2^18 and its bits 16..23 carry about 3 bits",
                 low == 1.0 and max(add["bits 0..15"]) < 0.02,
                 f"additive state bytes: low byte {low:.2%}; " + "; ".join(
                     f"{k} {a:.2%} / {b:.2%}" for k, (a, b) in add.items()) + f"; |Y| p99 2^{ylog:.1f}")
    sat = float(np.mean(np.abs(np.stack(ys)) > 127))
    checks.check("L: a saturating σ (cvt.sat.s8) keeps only signs on these sums", sat > 0.99,
                 f"{sat:.2%} of Y saturate")
    out["sigma"] = {"low_additive": low, "additive": add, "y_p99_log2": ylog, "saturated": sat}
    return out


def part_k(checks: Checks, rng) -> dict:
    out = {}
    for base in ("small", "tag", "offset", "uniform"):
        d = lead_words(rng, "rot", ROWS, BN, STEPS["k2048"], base)
        t = d["w"].shape[0] - 1
        cb = cell_bytes(d["w"][t])
        ranks = [rank_mod_p(cb[:, :, b]) for b in range(4)]
        full = rank_mod_p(cb.reshape(ROWS, -1))
        lin = rank_mod_p(d["wint"][t] - d["d0"])
        partner = []
        for b in range(2):
            x = cb[:, :, b]
            dif = np.mean(x[:, None, :] != x[None, :, :], axis=2) + np.eye(ROWS)
            partner.append(float(np.mean(dif.min(axis=1))))
        out[base] = {"bytes": full, "per_byte": ranks, "words_minus_d0": lin, "d0": rank_mod_p(d["d0"]),
                     "words": rank_mod_p(d["wint"][t]), "distinct_rows": int(len(np.unique(d["w"][t], axis=0))),
                     "partner_differs": partner}
    sm, of, tg = out["small"], out["offset"], out["tag"]
    out["counting_floor"] = 0.25 + 0.25 + 8 * tg["partner_differs"][1] + 2 / ROWS + honest_fold_writes() - 1.0
    checks.check("K: at the lead's own zero data (its step-0 tag) the words are rank <= 13 as integers (the tag's 9, "
                 "the row term and two classes' column terms) and every row is distinct; bytes 2 and 3 are one "
                 "pattern a row class (rank 2), which the class-sharing prover (part P) takes; byte 0 can't be "
                 "shared (a row's best partner differs on most cells, 8 adds each), but byte 1 nearly can: its "
                 "corrections cost less than the MMA slot, so counting alone (byte 0's A register and slot, byte 1's "
                 "corrections, the class's bytes) floors a prover below stock; no prover reaching it is built",
                 tg["d0"] <= 9 and tg["words"] <= 13 and tg["distinct_rows"] == ROWS and max(tg["per_byte"][2:]) <= 2
                 and tg["partner_differs"][0] * 8 > 0.25 and out["counting_floor"] < 1.0,
                 f"d0 rank {tg['d0']}, words rank {tg['words']}, distinct rows {tg['distinct_rows']}/{ROWS}, "
                 f"per byte {tg['per_byte']}; the best partner row differs on "
                 + ", ".join(f"{x:.1%} (byte {b}, {8 * x:.2f} reg)" for b, x in enumerate(tg["partner_differs"]))
                 + f"; counting floor {out['counting_floor']:.3f} registers a word")
    checks.check("K: the zero-data words less d0 are rank <= 3 (a row term and a column term a class), but their byte "
                 "matrices are full rank at bytes 0 and 1, the bytes that move every step; bytes 2 and 3 are low rank",
                 sm["words_minus_d0"] <= 3 and sm["bytes"] == ROWS and min(sm["per_byte"][:2]) == ROWS
                 and max(sm["per_byte"][2:]) < 8,
                 f"words - d0 rank {sm['words_minus_d0']}; bytes rank {sm['bytes']}; per byte {sm['per_byte']}")
    checks.check("K: with the words kept positive (offset data) byte 3 is zero, rank 0: that slot carries nothing, and "
                 "pack drops it", of["per_byte"][3] == 0 and min(of["per_byte"][:2]) == ROWS,
                 f"per byte {of['per_byte']}")
    rates = {}
    for layout, base in (("rot", "tag"), ("rot", "small"), ("rot", "offset"), ("e7", "offset")):
        d = lead_words(rng, layout, ROWS, BN, STEPS["k8192"], base)
        cb = cell_bytes(d["w"])
        rates[f"{layout}/{base}"] = [float(np.mean(cb[1:, ..., b] != cb[:-1, ..., b])) for b in range(4)]
    out["change_rate"] = rates
    slot = 0.25
    moving = [x for v in rates.values() for x in v if x > 0]
    checks.check("K: the low-rank route is an incremental update, 8 adds (one R row) for each (word, byte) that "
                 "changes; every byte that isn't constant changes on more than 1/32 of (word, step), the sign byte of "
                 "signed zero data included, so the update costs more than the quarter register a word of the MMA "
                 "slot it replaces (estimate)",
                 min(moving) * 8 > slot,
                 "; ".join(f"{k}: " + ", ".join(f"byte {b} {x:.1%} ({8 * x:.2f} reg)" for b, x in enumerate(v))
                           for k, v in rates.items()))
    return out


def part_t(checks: Checks, rng) -> dict:
    out = {}
    for layout in ("rot", "e7"):
        for base in ("small", "tag", "offset", "uniform"):
            d = lead_words(rng, layout, ROWS, BN, STEPS["k8192"], base)
            cb = cell_bytes(d["w"])
            tile = [int(len(np.unique(cb[..., b]))) for b in range(4)]
            percell = [float(np.mean([len(np.unique(cb[:, i, j, b])) for i in range(0, ROWS, 8)
                                      for j in range(0, BN, 32)])) for b in range(4)]
            span = int((d["wint"].max(axis=0) - d["wint"].min(axis=0)).max())
            out[f"{layout}/{base}"] = {"tile": tile, "per_cell": percell, "span_bits": float(np.log2(span + 1)),
                                       "min": int(d["wint"].min()), "max": int(d["wint"].max())}
    so, eo = out["rot/offset"], out["e7/offset"]
    checks.check("T: offset data keeps every word in [0, 2^24) at rot and [0, 2^16) at e7, over K 8,192's 265 steps, "
                 "so bytes 3 (rot) and 2-3 (e7) are constant",
                 so["min"] >= 0 and so["max"] < 1 << 24 and so["tile"][3] == 1
                 and eo["min"] >= 0 and eo["max"] < 1 << 16 and eo["tile"][2] == 1 and eo["tile"][3] == 1,
                 f"rot [{so['min']}, {so['max']}], e7 [{eo['min']}, {eo['max']}]")
    checks.check("T: a table needs a value a (word, byte) entry and 8 outputs, 8 adds an entry against the MMA's "
                 "quarter register; at zero data bytes 2 and 3 take few values but bytes 0 and 1 take all 256",
                 out["rot/small"]["tile"][0] == 256 and out["rot/small"]["tile"][1] == 256,
                 "; ".join(f"{k}: tile {v['tile']}, a cell {[round(x, 1) for x in v['per_cell']]}, span 2^"
                           f"{v['span_bits']:.1f}" for k, v in out.items()))
    return out


def stream_limbs(lanes: np.ndarray, width: int, regs: int) -> np.ndarray:
    """Pack signed lane integers (..., L) as Σ_l 2^(width·l)·x_l mod 2^(32·regs), as `regs` 32-bit limbs."""
    L = lanes.shape[-1]
    shape = lanes.shape[:-1]
    flat = lanes.reshape(-1, L)
    out = np.zeros((flat.shape[0], regs), np.int64)
    mod = 1 << (32 * regs)
    for k in range(flat.shape[0]):
        x = sum(int(flat[k, l]) << (width * l) for l in range(L)) % mod
        for r in range(regs):
            out[k, r] = (x >> (32 * r)) & M32
    return out.reshape(*shape, regs)


def limb_add(a: np.ndarray, *incs: np.ndarray) -> tuple[np.ndarray, int]:
    """Multi-precision add of limb arrays (..., regs), the carries chained limb by limb, mod 2^(32·regs). Returns the
    sum and the number of limbs whose carry-out was nonzero (the IADD3 chain's carry predicates)."""
    regs = a.shape[-1]
    out = np.zeros_like(a)
    carry = np.zeros(a.shape[:-1], np.int64)
    nonzero = 0
    for r in range(regs):
        s = a[..., r] + carry + sum(x[..., r] for x in incs)
        out[..., r] = s & M32
        carry = s >> 32
        if r < regs - 1:
            nonzero += int(np.count_nonzero(carry))
    return out, nonzero


def limb_bytes(limbs: np.ndarray) -> np.ndarray:
    """(..., regs) 32-bit limbs -> (..., 4·regs) s8 bytes, low first."""
    b = s8(limbs[..., None] >> (8 * np.arange(4)))
    return b.reshape(*limbs.shape[:-1], -1)


def pack_offset(d: dict, R: np.ndarray, Q: np.ndarray, layout: str) -> tuple[np.ndarray, dict]:
    """The prover with positive words in byte-aligned lanes it updates in place and the MMA reads directly.

    rot: 24-bit lanes, four words in three registers (a 96-bit stream), one IADD3 chain a step (three writes, the
    carries between limbs as predicates); bytes 0-2 of a word are its lane's bytes and byte 3 is zero. e7: 16-bit
    lanes, two words a register, one IADD3 a step and no carries between lanes. The increment is the row term
    (b_t(i) replicated into the lanes, per row) plus the class's column term (per lane group, shared by the class's
    rows). The MMA reads 3 (rot) or 2 (e7) slots a word against honest's 4."""
    w0 = d["d0"] & M32
    S, m, n = d["w"].shape
    width, per, regs = (24, 4, 3) if layout == "rot" else (16, 2, 1)
    slots = width // 8
    g = n // per
    lanes = stream_limbs(w0.reshape(m, g, per), width, regs)
    rsel = np.zeros((g * per * slots, NOUT), np.int64)
    for j in range(n):
        for byte in range(slots):
            rsel[(j // per) * per * slots + (j % per) * slots + byte] = R[4 * j + byte]
    ones = stream_limbs(np.ones((1, per), np.int64), width, regs)[0]
    Z = np.zeros((m, ZB), np.int64)
    carries = 0
    for t in range(S):
        rowterm = (d["b"][t][:, None] * ones[None, :]) & M32
        colterm = stream_limbs(d["col"][t].reshape(d["col"].shape[1], g, per), width, regs)[d["cls"]]
        lanes, cz = limb_add(lanes, rowterm[:, None, :], colterm)
        carries += cz
        assert regs * 4 == per * slots
        Y = limb_bytes(lanes).reshape(m, g * per * slots) @ rsel + Z @ Q
        Z = sigma(Y)
    threads, rows_per_thread, words_per_thread = 128, 2, BN // 2
    row_w = rows_per_thread * regs / words_per_thread
    col_w = 2 * (n // per) * regs / threads / words_per_thread
    model = {
        "slots": slots, "lanes": regs / per, "row_term": row_w, "col_term": col_w, "mma": slots / 4,
        "predicates": (2 * (regs - 1)) / per, "carry_rate": carries / max(1, S * m * g * max(1, regs - 1)),
        "common": honest_fold_writes() - 1.0,
    }
    model["free"] = model["lanes"] + model["row_term"] + model["col_term"] + model["mma"] + model["common"]
    model["priced"] = model["free"] + model["predicates"]
    return Z, model


def pack_share(d: dict, R: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, dict]:
    """The prover that shares a row class's high bytes, at the lead's own signed zero data (`tag`).

    Within a class the words differ by a small δ(i, j) (the tag and the row term), so bytes 2 and 3 are the class
    representative's except where 2^16 falls between them. The prover keeps 16-bit lanes, two words a register, and
    adds c_lo + (c_hi - κ)·2^16 a step, κ the representative's carry out of its low word: the low lane's carry into
    the high lane cancels except at those straddling cells, which it fixes. The MMA reads bytes 0-1 (2 slots); the
    class's bytes 2-3 go through once a class; straddling cells get an 8-output correction a byte. It finds them by
    checking the rows of each column whose representative sits within the bound on |δ| of a 2^16 boundary."""
    wint, c, cls = d["wint"], d["c"], d["cls"]
    S, m, n = wint.shape
    reps = np.array([np.nonzero(cls == k)[0][0] for k in range(int(cls.max()) + 1)])
    rep_of = reps[cls]
    lo, hi = wint[..., 0::2], wint[..., 1::2]
    reg = ((lo[0] & 0xFFFF) + ((hi[0] & 0xFFFF) << 16)) & M32
    Rw = R.reshape(n, 4, NOUT)
    rsel = np.stack([Rw[0::2, 0], Rw[0::2, 1], Rw[1::2, 0], Rw[1::2, 1]], axis=1).reshape(2 * n, NOUT)
    R23 = Rw[:, 2:, :]
    rows_per_class = m // len(reps)
    dbound = int(np.abs(d["d0"] - d["d0"][rep_of]).max())
    Z = np.zeros((m, ZB), np.int64)
    lanes_ok, fixes, diffs, checked = True, 0, 0, 0
    for t in range(S):
        if t > 0:
            kap = (lo[t][rep_of] >> 16) - (lo[t - 1][rep_of] >> 16)
            act = (lo[t] >> 16) - (lo[t - 1] >> 16)
            reg = (reg + c[t][:, 0::2] + ((c[t][:, 1::2] - kap) << 16)) & M32
            bad = act != kap
            fixes += int(np.count_nonzero(bad))
            reg = (reg - ((act - kap) << 16) * bad) & M32
        lanes_ok &= bool(np.array_equal(reg & 0xFFFF, lo[t] & 0xFFFF) and np.array_equal(reg >> 16, hi[t] & 0xFFFF))
        rep_low = wint[t][reps] & 0xFFFF
        near = np.minimum(rep_low, (1 << 16) - rep_low) <= dbound + t + 1
        checked += int(np.count_nonzero(near)) * rows_per_class
        straddle = (wint[t] >> 16) != (wint[t][rep_of] >> 16)
        lanes_ok &= bool(np.all(near[cls][straddle]))
        hb_rep = cell_bytes(wint[t][rep_of] & M32)[..., 2:]
        diff = cell_bytes(wint[t] & M32)[..., 2:] - hb_rep
        diffs += int(np.count_nonzero(diff))
        Y = (limb_bytes(reg[..., None]).reshape(m, 2 * n) @ rsel + np.einsum("mjb,jbo->mo", hb_rep, R23)
             + np.einsum("mjb,jbo->mo", diff, R23) + Z @ Q)
        Z = sigma(Y)
    threads, rows_per_thread, words_per_thread = 128, 2, BN // 2
    words = S * m * n
    model = {"slots": 2, "lanes": 0.5, "row_term": rows_per_thread / words_per_thread,
             "col_term": 2 * (n // 2) / threads / words_per_thread, "mma": 0.5,
             "representative": 3 / m, "class_bytes": 2 / m,
             "fixes": fixes / words, "corrections": NOUT * diffs / words, "checks": checked / words,
             "predicates": 0.0, "common": honest_fold_writes() - 1.0, "lanes_ok": lanes_ok}
    model["free"] = sum(v for k, v in model.items() if k not in ("slots", "predicates", "lanes_ok"))
    model["priced"] = model["free"] + model["predicates"]
    return Z, model


def pack_zero(d: dict, R: np.ndarray) -> dict:
    """The M0 prover at the lead's own zero-data words (signed, small base): two's-complement 24-bit lanes, four words
    a 96-bit stream. A lane holds (w_l - β_l) mod 2^24, β_l the borrow of the lanes below; honest's byte 3 is -s_l.
    The MMA reads the stream's 3 slots, a sign slot (-ŝ_l, weight R_l3, ŝ the lane's top bit) and a borrow slot (ŝ_l-1,
    weight R_l0); the rest is a residual, nonzero only on rare words, fixed by an 8-output add each. Returns the
    identity's check and the counts, step by step on Y."""
    wint = d["wint"]
    S, m, n = wint.shape
    g = n // 4
    ok = True
    resid_words = 0
    clean_rows = 0
    for t in range(S):
        x = wint[t].reshape(m, g, 4)
        assert np.all(np.abs(x) < 1 << 23)
        lim = stream_limbs(x, 24, 3)
        sb = limb_bytes(lim).reshape(m, g, 4, 3)
        top = ((lim.reshape(m, g, 3)[..., None] >> np.arange(32)) & 1).reshape(m, g, 96)[:, :, [23, 47, 71, 95]]
        beta_hat = np.concatenate([np.zeros((m, g, 1), np.int64), top[:, :, :3]], axis=-1)
        honest_b = cell_bytes(wint[t] & M32).reshape(m, g, 4, 4)
        res = honest_b[..., :3] - sb
        res[..., 0] -= beta_hat
        sign_res = honest_b[..., 3] + top
        hit = np.any(res != 0, axis=-1) | (sign_res != 0)
        resid_words += int(np.count_nonzero(hit))
        Rg = R.reshape(g, 4, 4, NOUT)
        adv = (np.einsum("mglb,glbo->mo", sb, Rg[..., :3, :]) - np.einsum("mgl,glo->mo", top, Rg[..., 3, :])
               + np.einsum("mgl,glo->mo", beta_hat, Rg[..., 0, :]))
        hon = row_bytes(wint[t] & M32) @ R
        clean = ~np.any(hit, axis=(1, 2))
        clean_rows += int(np.count_nonzero(clean))
        fixed = (adv + np.einsum("mglb,glbo->mo", res, Rg[..., :3, :])
                 + np.einsum("mgl,glo->mo", sign_res, Rg[..., 3, :]))
        ok &= bool(np.array_equal(adv[clean], hon[clean]) and np.array_equal(fixed, hon))
    rate = resid_words / (S * m * n)
    model = {"slots": 5, "lanes": 0.75, "sign_register": 0.5, "mma": 5 / 4, "residual_rate": rate,
             "clean_rows": clean_rows / (S * m),
             "residual": rate * NOUT, "terms": 0.047 + 0.023, "predicates": 1.0,
             "common": honest_fold_writes() - 1.0}
    model["free"] = (model["lanes"] + model["sign_register"] + model["mma"] + model["residual"] + model["terms"]
                     + model["common"])
    model["priced"] = model["free"] + model["predicates"]
    return {"ok": ok, "model": model}


def part_p(checks: Checks, rng) -> dict:
    out = {}
    honest = 1.0 + honest_fold_writes()
    for layout in ("rot", "e7"):
        for k, S in STEPS.items():
            d = lead_words(rng, layout, ROWS, BN, S, "offset")
            R, Q = draw_coins(rng, BN)
            Zh = chain(d["w"], R, Q)
            Za, model = pack_offset(d, R, Q, layout)
            same = np.array_equal(Za, Zh)
            out[f"{layout}/{k}/offset"] = {"same": bool(same), "seal": seal(Zh)[:16], **model}
    for k, S in STEPS.items():
        d = lead_words(rng, "rot", ROWS, BN, S, "tag")
        R, Q = draw_coins(rng, BN)
        z = pack_zero(d, R)
        out[f"rot/{k}/zero"] = {"same": z["ok"], **z["model"]}
        Zs, ms = pack_share(d, R, Q)
        out[f"rot/{k}/share"] = {"same": bool(np.array_equal(Zs, chain(d["w"], R, Q))), **ms}
    sh = [out[f"rot/{k}/share"] for k in STEPS]
    checks.check("P: at the lead's own zero data, the class-sharing prover (16-bit lanes, two words a register, the "
                 "class's bytes 2-3 once a class, the low lane's carry cancelled by the representative's) matches "
                 "honest's sealed state bit for bit, its lanes exact and every straddling cell inside its checks",
                 all(v["same"] and v["lanes_ok"] for v in sh),
                 "; ".join(f"{k}: fixes {v['fixes']:.4f}, corrections {v['corrections']:.4f}, checks "
                           f"{v['checks']:.4f} a word, {v['free']:.3f} registers a word" for k, v in zip(STEPS, sh)))
    ro = [v for k, v in out.items() if k.startswith("rot/") and k.endswith("/offset")]
    eo = [v for k, v in out.items() if k.startswith("e7/")]
    zr = [v for k, v in out.items() if k.endswith("/zero")]
    checks.check("P: the packed prover's sealed state equals honest's bit for bit at rot (24-bit lanes, 3 slots a "
                 "word) and e7 (16-bit lanes, 2 slots), K 2,048 and 8,192",
                 all(v["same"] for v in ro + eo), f"{len(ro + eo)} instances, seals " +
                 ", ".join(v["seal"] for v in ro + eo))
    checks.check("P: at the lead's own zero data (signed words), the stream, a sign slot, a borrow slot and a rare "
                 "residual reproduce honest's Y at every step", all(v["same"] for v in zr),
                 ", ".join(f"residual {v['residual_rate']:.3%} of words" for v in zr))
    rot, e7, zero = ro[-1], eo[-1], zr[-1]
    checks.check("P: the packed prover beats honest's 1 + (1 + 12/BN) registers a word when predicate writes are "
                 "free (rot, offset) and with no predicates at all (e7); at rot with predicates priced it pays more",
                 rot["free"] < honest and e7["free"] < honest and rot["priced"] > honest,
                 f"honest {honest:.3f}; rot {rot['free']:.3f} free / {rot['priced']:.3f} priced "
                 f"(carry predicates set on {rot['carry_rate']:.1%} of limbs); e7 {e7['free']:.3f}; "
                 f"M0 rot {zero['free']:.3f} / {zero['priced']:.3f}")
    d21 = lead_words(rng, "rot", ROWS, 24, STEPS["k2048"], "offset")
    x = d21["wint"].reshape(-1, 3)[rng.permutation(STEPS["k2048"] * ROWS * 8)[:600]]
    feats = {"21-bit pair": limb_bytes(stream_limbs(x[:, None, :], 21, 2))[:, 0],
             "24-bit stream": limb_bytes(stream_limbs(np.concatenate([x, x[:, :1]], axis=1)[:, None, :], 24, 3))[:, 0]}
    target = cell_bytes(x & M32)[:, :, :3].reshape(len(x), -1)
    ranks = {}
    for name, F in feats.items():
        Fx = np.concatenate([F, np.ones((len(x), 1), np.int64)], axis=1)
        ranks[name] = (rank_mod_p(Fx), rank_mod_p(np.concatenate([Fx, target], axis=1)))
    checks.check("P: COUNTER's 21-bit lanes don't transfer to the MMA: the words' low bytes aren't affine in the "
                 "packed pair's bytes (rank grows when they're appended), while the byte-aligned stream's bytes are them",
                 ranks["21-bit pair"][1] > ranks["21-bit pair"][0] and ranks["24-bit stream"][1] == ranks["24-bit stream"][0],
                 "; ".join(f"{k}: rank {a} -> {b}" for k, (a, b) in ranks.items()))
    out["honest"] = honest
    out["lane_ranks"] = ranks
    return out


def naf_weight(x: int) -> int:
    x, w = abs(int(x)), 0
    while x:
        if x & 1:
            x -= 2 - (x & 3)
            w += 1
        x >>= 1
    return w


def part_c(checks: Checks, lead: dict, p: dict, rk: dict, rng) -> dict:
    honest = p["honest"]
    f = honest_fold_writes()
    adv = {"rot offset, predicates free": p["rot/k8192/offset"]["free"],
           "rot offset, predicates priced": p["rot/k8192/offset"]["priced"],
           "rot class-sharing (no predicates)": p["rot/k8192/share"]["free"],
           "e7 offset": p["e7/k8192/offset"]["free"],
           "rot M0, predicates free": p["rot/k8192/zero"]["free"],
           "rot M0, predicates priced": p["rot/k8192/zero"]["priced"]}
    R, _ = draw_coins(rng, BN)
    naf = float(np.mean([naf_weight(v) for v in R.ravel()]))
    fold_and_word = 4 * NOUT * (8 * naf + 1) * (1 + 8 / BN)
    out = {"f": f, "naf": naf, "fold_and_per_word": fold_and_word, "points": {}, "credit": {}}
    for pt in POINTS:
        q = lead["points"][pt]
        r = q["r"]
        h_words = r * 4 / 32 * r_row(256 * 1024) * W1_PER_REG / BLOCK
        h_a = q["h"] - h_words
        h_fold = r * f
        h_seal = r / 32 * ZB / (BN * STEPS["k2048"]) * r_row(256 * 1024) * W1_PER_REG / BLOCK
        h = h_a + h_fold + h_seal
        state_frac = ZB / (BN * STEPS["k2048"]) / 4
        fold_macs = q["stock_macs"] * r * (32 + 256 / BN) / 32
        parts = dict(q["extra_parts_ms"])
        rest = sum(parts.values()) - parts["export"] * (1 - state_frac) + q["hash_ms"] * state_frac
        t2 = {rd: {e: (s + rest + fold_macs / (H200_PEAK * rate) * 1e3) / s for e, s in q["stock_ms"].items()}
              for rd, rate in FOLD_RATE.items()}
        hiding = q["terms"].get("hiding", 0.0)
        cost_h = r * honest
        t1 = {name: {"credit_costH": hiding + max(0.0, cost_h - r * a), "credit_stock": hiding + max(0.0, 1 - r * a),
                     "c0_over_stock": r * a} for name, a in adv.items()}
        tilegrain = r * (ZB / 4) / (BN * STEPS["k2048"])
        per = {}
        for k, s in PERIODS.items():
            macs = {"h200": H200_MACS * s, "datacenter": H200_MACS * s * DC_GPUS,
                    "world": WORLD_MACS_PER_YEAR * s / PERIODS["year"]}
            per[k] = {scope: {"h": h * mm, "lead_h": q["h"] * mm, "lead_overcredit": q["t1"] * mm,
                              "tilegrain_credit": tilegrain * mm,
                              "overcredit": {name: {c: v[c] * mm for c in ("credit_costH", "credit_stock")}
                                             for name, v in t1.items()}}
                      for scope, mm in macs.items()}
        out["points"][pt] = {"r": r, "h_a": h_a, "h_fold": h_fold, "h_seal": h_seal, "h": h, "cut": q["h"] / h,
                             "t2": t2, "t1": t1, "lead_t1": q["t1"], "lead_t2": q["t2"], "tilegrain_credit": tilegrain,
                             "floor_over_stock": r * rk["counting_floor"], "per_period_w1": per}
    p8 = out["points"]["t8192"]
    checks.check("C: under H32 and TileGrain alone the fold earns only its hashed tiles, the sealed state's: TileGrain "
                 "counts a write step a committed tile, and the fold's Y and Z are never committed until the last Z, "
                 "so crediting the MMA work needs a premise about the fold itself",
                 p8["tilegrain_credit"] < 1e-3,
                 f"credit {p8['tilegrain_credit']:.2e} W1 a stock MAC ({p8['tilegrain_credit']:.4%} of stock), "
                 f"against honest's fold {p8['h_fold']:.3f}")
    checks.check("C: TENSOR's h is A's pre-noise rows plus r·(1 + 12/BN) plus the seal: about 1.5 against the lead's "
                 "127.5, a cut above 80x at every point",
                 all(v["cut"] > 80 for v in out["points"].values()),
                 "; ".join(f"{pt} h {v['h']:.3f} (A {v['h_a']:.3f}, fold {v['h_fold']:.3f}, seal {v['h_seal']:.4f}) "
                           f"cut {v['cut']:.1f}x" for pt, v in out["points"].items()))
    checks.check("C: crediting the fold (credit = cost(H)) fails T1 at the lead's layout whether predicate writes "
                 "are free or priced: the class-sharing prover needs none; the offset packer also fails it with "
                 "predicates free, and every packer at the narrower e7 layout",
                 p8["t1"]["rot offset, predicates free"]["credit_costH"] > 0.4
                 and p8["t1"]["e7 offset"]["credit_costH"] > 1.0
                 and p8["t1"]["rot class-sharing (no predicates)"]["credit_costH"] > 1.0,
                 "; ".join(f"{k} {v['credit_costH']:.1%}" for k, v in p8["t1"].items()))
    checks.check("C: crediting stock alone holds T1 against every prover found: at the lead's layout the cheapest "
                 "(class-sharing) pays 1.34x stock, at e7 1.24x; counting alone floors below stock, so the credit "
                 "rests on the premise, not on a count",
                 all(v["credit_stock"] < 0.01 for v in p8["t1"].values())
                 and min(v["c0_over_stock"] for v in p8["t1"].values()) > 1.15 and p8["floor_over_stock"] < 1.0,
                 "; ".join(f"{name} C0 {v['c0_over_stock']:.3f}x stock" for name, v in p8["t1"].items())
                 + f"; counting floor {p8['floor_over_stock']:.3f}x")
    checks.check("C: T2 stays under the lead's, engine by engine, at every point and both readings of the fold's "
                 "tensor rate", all(v["t2"][rd][e] < v["lead_t2"][e] for v in out["points"].values()
                                    for rd in FOLD_RATE for e in v["lead_t2"]),
                 "; ".join(f"{pt} " + ", ".join(f"{rd} {v['t2'][rd]['sync']:.2f}/{v['t2'][rd]['async']:.2f}"
                                                 for rd in FOLD_RATE) + f" (lead {v['lead_t2']['sync']:.2f}/"
                           f"{v['lead_t2']['async']:.2f})" for pt, v in out["points"].items()))
    lead_open = 32 * AND_PER_MAC + 4 * SHA_AND_PER_BYTE
    ten_open = 32 * AND_PER_MAC + fold_and_word
    out["opening"] = {"lead_per_word": lead_open, "tensor_per_word": ten_open, "ratio": ten_open / lead_open,
                      "fold_per_byte": fold_and_word / 4}
    checks.check("C: the opening checks the fold against public R by shift-and-add: fewer AND gates a byte than "
                 "SHA-512's, so a drawn unit's opening is no larger than the lead's (estimate)",
                 fold_and_word / 4 < SHA_AND_PER_BYTE and ten_open < lead_open,
                 f"R's mean NAF weight {naf:.2f}; fold {fold_and_word / 4:.0f} AND a byte (SHA-512 "
                 f"{SHA_AND_PER_BYTE}); a word's opening {ten_open:.0f} against the lead's {lead_open:.0f} "
                 f"({ten_open / lead_open:.2f}x)")
    return out


def main() -> int:
    checks = Checks()
    args = checks.args
    lead = json.loads(Path(args[0]).read_text())
    rng = np.random.default_rng(29)
    a = part_a(checks, rng)
    lin = part_l(checks, rng)
    k = part_k(checks, rng)
    t = part_t(checks, rng)
    p = part_p(checks, rng)
    c = part_c(checks, lead, p, k, rng)
    if "--json" in args:
        Path(args[args.index("--json") + 1]).write_text(json.dumps(
            {"fragment": a, "linearity": lin, "rank": k, "tables": t, "pack": p, "price": c},
            indent=1, sort_keys=True, default=float) + "\n")
    print("\npoint      h      cut    T2 half-peak s/a   T2 n8 s/a      lead T2 s/a")
    for pt, v in c["points"].items():
        print(f"{pt:<9} {v['h']:6.3f} {v['cut']:6.1f}x  " + "  ".join(
            f"{v['t2'][rd]['sync']:6.2f}/{v['t2'][rd]['async']:6.2f}" for rd in FOLD_RATE)
            + f"  {v['lead_t2']['sync']:6.2f}/{v['lead_t2']['async']:6.2f}")
    print("\nprover (t8192)                    regs/word  C0/stock  T1 credit=cost(H)  T1 credit=stock")
    for name, v in c["points"]["t8192"]["t1"].items():
        print(f"{name:<33} {v['c0_over_stock'] / c['points']['t8192']['r']:9.3f} {v['c0_over_stock']:9.3f} "
              f"{v['credit_costH']:17.2%} {v['credit_stock']:15.2%}")
    print(f"honest                            {p['honest']:9.3f} {c['points']['t8192']['r'] * p['honest']:9.3f}")
    print("\nW1 over a period (t8192): honest's commitment h against the lead's, and the overcredit of the "
          "class-sharing prover at the lead's layout")
    print("period  scope        h (TENSOR)   h (lead)     credit=cost(H)  credit=stock  TileGrain credit")
    for k, by in c["points"]["t8192"]["per_period_w1"].items():
        for scope, v in by.items():
            oc = v["overcredit"]["rot class-sharing (no predicates)"]
            print(f"{k:<7} {scope:<12} {v['h']:11.3e} {v['lead_h']:11.3e} {oc['credit_costH']:15.3e} "
                  f"{oc['credit_stock']:13.3e} {v['tilegrain_credit']:16.3e}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
