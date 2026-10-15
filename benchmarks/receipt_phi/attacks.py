"""Attacks on the receipt φ candidates of `twins.py`, at w = 32 with phi.cu's keys, and on 8-bit analogues.

    python attacks.py [--quick] > attacks.json

Each attack is a function returning JSON; `main` runs them all. The words p_t come from one of three distributions:
``u32`` (uniform), ``int26`` (uniform below 2^26: PoUW's INT32 prefix sums of int8 residues mod 127 at n = 16K stay
below 2^28 and average about 2^25) and ``fp32`` (FP32 patterns of exact sums of 64 products of nonzero FP4 codes, the
random-operand work's words).

- ``differential``: the best two-word cancellation. The prover changes word t by Δ1 (XOR e_j or ±2^j) and must fix
  the chain at word t+1. For every sample, φ_{t+1}(y', ·) is inverted at the honest state, which gives the one p'' that
  cancels, and the most frequent p'' ⊕ p or p'' − p is the best Δ2 for that Δ1. Its frequency is the probability that
  the pair (Δ1, Δ2) leaves the receipt unchanged.
- ``three_word``: the same search one word further, for the kinds without a high-probability two-word canceller.
- ``chain_end``: fixed differences in the block's last word and last two words, read through the receipt (the wide
  kinds' fold of their 64-bit state to 32 bits), by direct sampling.
- ``mum_lattice``: mum's state-independent additive cancellations Σ δ_t a_t ≡ 0 mod 2^64 for the real keys, with
  the bits a prover pays to reach them through the XOR of the state's halves.
- ``linearity``: the block receipt f(P) over b² = 256 words, tested for f(P ∘ Q) = f(P) ∘ f(Q) ∘ f(0)⁻¹ over
  Z/2^32, GF(2)^32 and Z/(2^32 − 1), with the share of exact hits and of output bits that are linear.
- ``degree2``: the receipt's third derivative over Z/2^32 and GF(2)^32, which vanishes for degree ≤ 2 (Σp²'s class).
- ``low_bits``: whether the receipt's low k bits depend only on the words' low k bits (a T-function chain's do).
- ``carries``: ra's chain is linear mod 2^32 − 1 except for one carry per wrapped addition; the number of carries per
  block, and the share of blocks with none, whose receipt the linear form gives exactly.
- ``anf`` and ``walsh``: the 8-bit analogue of two absorbed words (16 input bits, 8 output bits): the ANF degree of
  each output bit and the largest linear correlation, against a random function's.
- ``mad_sketch``: mad's receipts of every block of a small INT8 matmul, from the honest chain and from the cross-block
  linear sketch U = Xᵀ C in int8 digits, with both MAC counts.
- ``duo_g``: duo's two-word cancellations are exactly the collisions of g_t(u) = rotl(u, r_t) + a_t·u (word t+1
  cancels a change at word t only if z is unchanged): the share of u with a partner, on a 16-bit analogue and at
  w = 32 by solving for partners.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twins  # noqa: E402

M32 = np.uint64(0xFFFFFFFF)
U = np.uint64
B2 = 256
KINDS = twins.KINDS
PAIRED = twins.PAIRED


def keys(kseed: int = 0xC0FFEE11, n: int = B2):
    ka = np.array([twins.gen(kseed, 1, i, 0) | 1 for i in range(n)], dtype=np.uint64)
    kb = np.array([twins.gen(kseed, 2, i, 0) | 1 for i in range(n)], dtype=np.uint64)
    kk = np.array([twins.gen(kseed, 3, i, 0) for i in range(n)], dtype=np.uint64)
    return ka, kb, kk


KA, KB, KK = keys()
INV = {}


def inv32(a: int) -> int:
    return pow(int(a), -1, 1 << 32)


def inv64(a: int) -> int:
    return pow(int(a), -1, 1 << 64)


def rotl(x, r: int):
    r &= 31
    return ((x << U(r)) | (x >> U(32 - r))) & M32 if r else x


def rotr(x, r: int):
    return rotl(x, 32 - (r & 31))


def keypos(kind: str, t: int) -> int:
    return t - (t & 1) if kind in PAIRED else t


def step(kind: str, y, z, p, t: int):
    """One absorbed word at position t (a paired kind's even word is the pair's first half, its odd word the second)."""
    k = keypos(kind, t)
    a, b, c, r = KA[k], KB[k], KK[k], twins.rot_of(k)
    if kind == "mad":
        return (y * a + p) & M32, z
    if kind == "ra":
        return (rotl(y, r) + p) & M32, z
    if kind == "ram":
        return ((rotl(y, r) + p) & M32, z) if t % 2 == 0 else ((y * a + p) & M32, z)
    if kind == "xma":
        return (((y ^ p ^ c) * a) & M32, z) if t % 2 == 0 else ((y + p) & M32, z)
    if kind == "ar":
        return rotl((y + p + c) & M32, r), z
    if kind == "mum":
        u = y ^ z ^ p
        s = u * a + ((y << U(32)) | z)  # uint64 wraps mod 2^64
        return s >> U(32), s & M32
    if kind == "xsm":
        return ((y ^ (y >> U(16)) ^ p) * a) & M32, z
    if kind == "xsm4":
        v = y ^ p
        return ((v ^ (v >> U(16)) ^ c) * a) & M32, z
    if kind == "mxm":
        u = (y * a + p) & M32
        return ((u ^ (u >> U(16)) ^ c) * b) & M32, z
    if kind == "chi":
        u = y ^ p ^ c
        return u ^ ((~rotl(u, 1) & M32) & rotl(u, 2)), z
    if kind == "xsm2":
        u = ((y ^ (y >> U(16)) ^ p) * a) & M32
        return ((u ^ (u >> U(13)) ^ c) * b) & M32, z
    if kind == "duo":
        u = y ^ p ^ c
        y2 = (u * a + z) & M32
        return y2, (rotl(u, r) + y2) & M32
    raise ValueError(kind)


def fold(kind: str, y, z):
    """`twins.finish`: the block's receipt from the chain's final state."""
    return y ^ z if kind == "mum" else y ^ rotl(z, 16) if kind == "duo" else y


def unxs(w, s: int):
    """The inverse of v -> v ^ (v >> s) on 32 bits."""
    v = w.copy()
    sh = s
    while sh < 32:
        v ^= w >> U(sh)
        sh += s
    return v


def invert(kind: str, y, z, ty, tz, t: int):
    """The p with step(kind, y, z, p, t) == (ty, tz), and a mask of the samples where one exists."""
    k = keypos(kind, t)
    a, b, c, r = int(KA[k]), int(KB[k]), KK[k], twins.rot_of(k)
    ok = np.ones(y.shape, dtype=bool)
    if kind == "mad":
        p = (ty - y * U(a)) & M32
    elif kind == "ra":
        p = (ty - rotl(y, r)) & M32
    elif kind == "ram":
        p = (ty - rotl(y, r)) & M32 if t % 2 == 0 else (ty - y * U(a)) & M32
    elif kind == "xma":
        p = (((ty * U(inv32(a))) & M32) ^ y ^ c) if t % 2 == 0 else (ty - y) & M32
    elif kind == "ar":
        p = (rotr(ty, r) - y - c) & M32
    elif kind == "mum":
        s, st = (y << U(32)) | z, (ty << U(32)) | tz
        u = (st - s) * U(inv64(a))
        ok = u <= M32
        p = (u ^ y ^ z) & M32
    elif kind == "xsm":
        p = ((ty * U(inv32(a))) & M32) ^ y ^ (y >> U(16))
    elif kind == "xsm4":
        p = unxs(((ty * U(inv32(a))) & M32) ^ c, 16) ^ y
    elif kind == "mxm":
        u = unxs(((ty * U(inv32(b))) & M32) ^ c, 16)
        p = (u - y * U(a)) & M32
    elif kind == "chi":
        p = None  # chi isn't injective: p ^ y ^ y' is one canceller, set by the caller
    elif kind == "xsm2":
        u = unxs(((ty * U(inv32(b))) & M32) ^ c, 13)
        p = ((u * U(inv32(a))) & M32) ^ y ^ (y >> U(16))
    elif kind == "duo":
        u = (((ty - z) & M32) * U(inv32(a))) & M32
        ok = ((rotl(u, r) + ty) & M32) == tz
        p = u ^ y ^ c
    else:
        raise ValueError(kind)
    return p, ok


FP4 = np.array([0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def words(dist: str, rng: np.random.Generator, shape) -> np.ndarray:
    if dist == "u32":
        return rng.integers(0, 1 << 32, size=shape, dtype=np.uint64)
    if dist == "int26":
        return rng.integers(0, 1 << 26, size=shape, dtype=np.uint64)
    if dist == "fp32":
        n = int(np.prod(shape))
        x = FP4[rng.integers(0, 7, size=(n, 64))] * rng.choice([-1, 1], size=(n, 64)).astype(np.float32)
        w = FP4[rng.integers(0, 7, size=(n, 64))] * rng.choice([-1, 1], size=(n, 64)).astype(np.float32)
        v = (x * w).astype(np.float32).sum(axis=1, dtype=np.float32)
        return v.view(np.uint32).astype(np.uint64).reshape(shape)
    raise ValueError(dist)


def mode_share(x: np.ndarray, ok: np.ndarray):
    if not ok.any():
        return 0, 0.0
    v, n = np.unique(x[ok], return_counts=True)
    i = int(np.argmax(n))
    return int(v[i]), float(n[i]) / x.size


def differential(dist: str, samples: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    deltas = [("xor", 1 << j) for j in range(32)] + [("add", 1 << j) for j in range(32)] + \
             [("add", (1 << 32) - (1 << j)) for j in range(32)]
    out = {}
    for kind in KINDS:
        best = {"p": 0.0}
        for t in ((4, 5) if kind in PAIRED else (4,)):
            y = rng.integers(0, 1 << 32, size=samples, dtype=np.uint64)
            z = rng.integers(0, 1 << 32, size=samples, dtype=np.uint64)
            p0, p1 = words(dist, rng, samples), words(dist, rng, samples)
            s, sz = step(kind, y, z, p0, t)
            ty, tz = step(kind, s, sz, p1, t + 1)
            for op, d in deltas:
                q0 = p0 ^ U(d) if op == "xor" else (p0 + U(d)) & M32
                s2, sz2 = step(kind, y, z, q0, t)
                if kind == "chi":
                    q1, ok = p1 ^ s ^ s2, np.ones(samples, dtype=bool)
                else:
                    q1, ok = invert(kind, s2, sz2, ty, tz, t + 1)
                for rep, dd in (("xor", q1 ^ p1), ("add", (q1 - p1) & M32)):
                    v, share = mode_share(dd, ok)
                    if share > best["p"]:
                        best = {"p": share, "t": t, "d1": f"{op} {d:#010x}", "d2": f"{rep} {v:#010x}"}
        best["log2p"] = float(np.log2(best["p"])) if best["p"] > 0 else None
        best["resolution_log2"] = -float(np.log2(samples))
        out[kind] = best
    return out


DELTAS = [("xor", 1 << j) for j in range(32)] + [("add", 1 << j) for j in range(32)] + \
         [("add", (1 << 32) - (1 << j)) for j in range(32)]


def apply_delta(op: str, d: int, p):
    return p ^ U(d) if op == "xor" else (p + U(d)) & M32


def chain_end(dist: str, samples: int, seed: int = 11) -> dict:
    """Fixed differences at the end of the block, read through the receipt rather than the state: Δ in the last word
    alone, and (Δ1, Δ2) in the last two, by direct sampling over every pair. A wide kind's state carries 64 bits to the
    end, where its fold keeps 32: a fold that loses a difference there (duo's first fold, y ⊕ z with z = y + rotl(u, r),
    under +2^31 in both halves) is a cancellation no in-chain test sees."""
    rng = np.random.default_rng(seed)
    out = {}
    for kind in KINDS:
        t0 = B2 - (4 if kind in PAIRED else 2)
        y = rng.integers(0, 1 << 32, size=samples, dtype=np.uint64)
        z = rng.integers(0, 1 << 32, size=samples, dtype=np.uint64)
        P = [words(dist, rng, samples) for _ in range(B2 - t0)]

        def receipt(q, start, s):
            a, b = s
            for i in range(start, B2 - t0):
                a, b = step(kind, a, b, q[i], t0 + i)
            return fold(kind, a, b)

        r0 = receipt(P, 0, (y, z))
        last = len(P) - (2 if kind in PAIRED else 1)
        one, two = {"p": 0.0}, {"p": 0.0}
        for op, d in DELTAS:
            Q = list(P)
            Q[last] = apply_delta(op, d, P[last])
            share = float(np.mean(receipt(Q, 0, (y, z)) == r0))
            if share > one["p"]:
                one = {"p": share, "d": f"{op} {d:#010x}"}
        prev = last - (2 if kind in PAIRED else 1)
        for op1, d1 in DELTAS:
            Q = list(P)
            Q[prev] = apply_delta(op1, d1, P[prev])
            s = (y, z)
            for i in range(last):
                s = step(kind, *s, Q[i], t0 + i)
            for op2, d2 in DELTAS:
                Q2 = list(Q)
                Q2[last] = apply_delta(op2, d2, P[last])
                share = float(np.mean(receipt(Q2, last, s) == r0))
                if share > two["p"]:
                    two = {"p": share, "d1": f"{op1} {d1:#010x}", "d2": f"{op2} {d2:#010x}"}
        for r in (one, two):
            r["log2p"] = float(np.log2(r["p"])) if r["p"] > 0 else None
            r["resolution_log2"] = -float(np.log2(samples))
        out[kind] = {"last_word": one, "last_two_words": two}
    return out


def three_word(dist: str, samples: int, kinds=("mum", "xsm4", "xsm2", "duo"), seed: int = 6) -> dict:
    """``differential`` one word further: fixed Δ1 at word t and Δ2 at word t+1, the canceller at t+2 by inversion."""
    rng = np.random.default_rng(seed)
    deltas = [("xor", 1 << j) for j in range(32)] + [("add", 1 << j) for j in range(32)] + \
             [("add", (1 << 32) - (1 << j)) for j in range(32)]

    def apply(op, d, p):
        return p ^ U(d) if op == "xor" else (p + U(d)) & M32

    out = {}
    t = 4
    for kind in kinds:
        best = {"p": 0.0}
        y = rng.integers(0, 1 << 32, size=samples, dtype=np.uint64)
        z = rng.integers(0, 1 << 32, size=samples, dtype=np.uint64)
        p0, p1, p2 = (words(dist, rng, samples) for _ in range(3))
        s1, z1 = step(kind, y, z, p0, t)
        s2, z2 = step(kind, s1, z1, p1, t + 1)
        ty, tz = step(kind, s2, z2, p2, t + 2)
        for op1, d1 in deltas:
            a1, b1 = step(kind, y, z, apply(op1, d1, p0), t)
            for op2, d2 in deltas:
                a2, b2 = step(kind, a1, b1, apply(op2, d2, p1), t + 1)
                q2, ok = invert(kind, a2, b2, ty, tz, t + 2)
                for rep, dd in (("xor", q2 ^ p2), ("add", (q2 - p2) & M32)):
                    v, share = mode_share(dd, ok)
                    if share > best["p"]:
                        best = {"p": share, "d1": f"{op1} {d1:#010x}", "d2": f"{op2} {d2:#010x}",
                                "d3": f"{rep} {v:#010x}"}
        best["log2p"] = float(np.log2(best["p"])) if best["p"] > 0 else None
        best["resolution_log2"] = -float(np.log2(samples))
        out[kind] = best
    return out


def naf_weight(v: int) -> int:
    w = 0
    while v:
        if v & 1:
            v -= 2 - (v & 3)
            w += 1
        v >>= 1
    return w


def signed64(v: int) -> int:
    v %= 1 << 64
    return v - (1 << 64) if v >= 1 << 63 else v


def lll(B: list[list[int]], delta: float = 0.99) -> list[list[int]]:
    from fractions import Fraction
    B = [list(b) for b in B]
    n = len(B)

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v))

    def gso():
        Bs, mu = [], [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            v = [Fraction(x) for x in B[i]]
            for j in range(i):
                mu[i][j] = Fraction(dot(B[i], Bs[j])) / dot(Bs[j], Bs[j])
                v = [a - mu[i][j] * b for a, b in zip(v, Bs[j])]
            Bs.append(v)
        return Bs, mu

    k = 1
    Bs, mu = gso()
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                B[k] = [a - q * b for a, b in zip(B[k], B[j])]
                Bs, mu = gso()
        if dot(Bs[k], Bs[k]) >= (delta - mu[k][k - 1] ** 2) * dot(Bs[k - 1], Bs[k - 1]):
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu = gso()
            k = max(k - 1, 1)
    return B


def mum_lattice(t: int = 4) -> dict:
    """mum's state is S ← S + u_t·a_t mod 2^64 with u_t = hi(S) ⊕ lo(S) ⊕ p_t, so additive differences δ_t of u
    cancel, state-independently, whenever Σ δ_t a_t ≡ 0 mod 2^64 with |δ_t| < 2^32: a lattice of determinant 2^64.
    The prover reaches u only through p and the state's hi ⊕ lo, so a pattern costs about one bit per nonzero signed
    digit of each δ_t (setting it through the XOR) and of each intermediate state difference D (its carries). This
    gives, for the real keys, the cheapest such pattern found by LLL and small combinations of the reduced basis."""
    import itertools
    out = {}
    for m in (2, 3, 4):
        a = [int(KA[t + i]) for i in range(m)]
        inv = pow(a[-1], -1, 1 << 64)
        basis = [[1 if j == i else 0 for j in range(m - 1)] + [(-a[i] * inv) % (1 << 64)] for i in range(m - 1)]
        basis.append([0] * (m - 1) + [1 << 64])
        red = lll(basis)
        best = None
        for co in itertools.product(range(-2, 3), repeat=m):
            if not any(co):
                continue
            v = [sum(c * b[i] for c, b in zip(co, red)) for i in range(m)]
            if max(abs(x) for x in v) >= 1 << 32 or not any(v):
                continue
            D, cost, ds = 0, sum(naf_weight(abs(x)) for x in v), []
            for i in range(m - 1):
                D = signed64(D + v[i] * a[i])
                ds.append(D)
                cost += naf_weight(abs(D))
            if best is None or cost < best["cost_bits"]:
                best = {"deltas": v, "state_diffs": ds, "cost_bits": cost}
        out[f"{m}_words"] = best or {"cost_bits": None}
    return out


def receipts(kind: str, P: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Block receipts: P is (trials, 256) words, y0 the chain's start (z starts at y0 ^ 0x9E3779B9 for wide kinds)."""
    y, z = y0.copy(), y0 ^ U(0x9E3779B9)
    for t in range(P.shape[1]):
        y, z = step(kind, y, z, P[:, t], t)
    return fold(kind, y, z)


def linearity(dist: str, trials: int, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    m1 = (1 << 32) - 1
    for kind in KINDS:
        y0 = np.full(trials, 0x12345678, dtype=np.uint64)
        P, Q = words(dist, rng, (trials, B2)), words(dist, rng, (trials, B2))
        f0 = receipts(kind, np.zeros_like(P), y0)
        fp, fq = receipts(kind, P, y0), receipts(kind, Q, y0)
        r = {}
        e = (receipts(kind, (P + Q) & M32, y0) - fp - fq + f0) & M32
        low = 0
        while low < 32 and np.mean((e & U((1 << (low + 1)) - 1)) == 0) > 0.9:
            low += 1
        r["add"] = {"exact": float(np.mean(e == 0)), "low_bits_linear": low}
        e = receipts(kind, P ^ Q, y0) ^ fp ^ fq ^ f0
        bits = [float(np.mean((e >> U(i)) & U(1) == 0)) for i in range(32)]
        r["xor"] = {"exact": float(np.mean(e == 0)), "bits_linear": sum(b > 0.9 for b in bits),
                    "best_bit_bias": max(abs(b - 0.5) for b in bits)}
        s = P + Q
        s = (s & M32) + (s >> U(32))
        e = (receipts(kind, s & M32, y0).astype(object) - fp.astype(object) - fq.astype(object) +
             f0.astype(object)) % m1
        r["add_mod_2^32-1"] = {"exact": float(np.mean(e == 0))}
        out[kind] = r
    return out


def low_bits(dist: str, trials: int, seed: int = 12) -> dict:
    """R1's low-bits shortcut: redraw every word's bits from k up, keep its low k bits, and count the blocks whose
    receipt keeps its low k bits. A T-function chain keeps all of them (a k-bit matmul gives them); else about 2^-k."""
    rng = np.random.default_rng(seed)
    out = {}
    for kind in KINDS:
        y0 = np.full(trials, 0x12345678, dtype=np.uint64)
        P = words(dist, rng, (trials, B2))
        r0 = receipts(kind, P, y0)
        r = {}
        for k in (4, 8, 16):
            m = U((1 << k) - 1)
            Q = (P & m) | (rng.integers(0, 1 << 32, size=P.shape, dtype=np.uint64) & ~m & M32)
            r[f"k{k}"] = float(np.mean((receipts(kind, Q, y0) & m) == (r0 & m)))
        out[kind] = r
    return out


def degree2(dist: str, trials: int, seed: int = 8) -> dict:
    """The third derivative of the block receipt, Σ_{S ⊆ {P,Q,R}} (−1)^|S| f(ΣS), over Z/2^32 and GF(2)^32: zero for
    every degree-2 f, which is the class the Gram-matrix shortcut on Σp² batches."""
    rng = np.random.default_rng(seed)
    out = {}
    for kind in KINDS:
        y0 = np.full(trials, 0x12345678, dtype=np.uint64)
        P, Q, R = (words(dist, rng, (trials, B2)) for _ in range(3))
        r = {}
        for ring, op in (("add", lambda a, b: (a + b) & M32), ("xor", lambda a, b: a ^ b)):
            acc = np.zeros(trials, dtype=np.uint64)
            for mask in range(8):
                S = np.zeros_like(P)
                for i, M in enumerate((P, Q, R)):
                    if mask >> i & 1:
                        S = op(S, M)
                f = receipts(kind, S, y0)
                odd = bin(mask).count("1") % 2
                acc = (acc - f) & M32 if (odd and ring == "add") else (acc + f) & M32 if ring == "add" else acc ^ f
            r[ring] = {"exact": float(np.mean(acc == 0)),
                       "bits_zero": sum(float(np.mean((acc >> U(i)) & U(1) == 0)) > 0.9 for i in range(32))}
        out[kind] = r
    return out


def carries(dist: str, trials: int, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    P = words(dist, rng, (trials, B2))
    y = rng.integers(0, 1 << 32, size=trials, dtype=np.uint64)
    n = np.zeros(trials, dtype=np.int64)
    m1 = (1 << 32) - 1
    lin = [int(v) % m1 for v in y]
    for t in range(B2):
        r = twins.rot_of(t)
        s = rotl(y, r) + P[:, t]
        n += (s >> U(32)).astype(np.int64)
        y = s & M32
        lin = [(v * (1 << r) + int(p)) % m1 for v, p in zip(lin, P[:, t])]
    none = n == 0
    hit = np.array([int(v) % m1 == w for v, w in zip(y, lin)])
    return {"mean_carries_per_block": float(n.mean()), "blocks_with_none": float(np.mean(none)),
            "receipt_is_the_linear_form_mod_2^32-1": float(np.mean(hit)),
            "of_carry_free_blocks": float(np.mean(hit[none])) if none.any() else None}


# ------------------------------------------------------------------------------------------------- 8-bit analogues
M8 = 0xFF


def step8(kind: str, y, z, p, t: int, k8):
    a, b, c = k8["a"][t], k8["b"][t], k8["c"][t]
    r = (twins.rot_of(t) % 7) + 1

    def rl(x, s):
        return ((x << s) | (x >> (8 - s))) & M8

    if kind == "mad":
        return (y * a + p) & M8, z
    if kind == "ra":
        return (rl(y, r) + p) & M8, z
    if kind == "ram":
        return ((rl(y, r) + p) & M8, z) if t % 2 == 0 else ((y * k8["a"][t - 1] + p) & M8, z)
    if kind == "xma":
        return (((y ^ p ^ c) * a) & M8, z) if t % 2 == 0 else ((y + p) & M8, z)
    if kind == "ar":
        return rl((y + p + c) & M8, r), z
    if kind == "mum":
        u = y ^ z ^ p
        s = (u * a + ((y << 8) | z)) & 0xFFFF
        return s >> 8, s & M8
    if kind == "xsm":
        return ((y ^ (y >> 4) ^ p) * a) & M8, z
    if kind == "xsm4":
        v = y ^ p
        return ((v ^ (v >> 4) ^ c) * a) & M8, z
    if kind == "mxm":
        u = (y * a + p) & M8
        return ((u ^ (u >> 4) ^ c) * b) & M8, z
    if kind == "chi":
        u = y ^ p ^ c
        return u ^ ((~rl(u, 1) & M8) & rl(u, 2)), z
    if kind == "xsm2":
        u = ((y ^ (y >> 4) ^ p) * a) & M8
        return ((u ^ (u >> 3) ^ c) * b) & M8, z
    if kind == "duo":
        u = y ^ p ^ c
        y2 = (u * a + z) & M8
        return y2, (rl(u, r) + y2) & M8
    raise ValueError(kind)


def two_word_table(kind: str, y: int, z: int, k8) -> np.ndarray:
    """The 8-bit output of absorbing words (p_4, p_5) from state (y, z), for all 2^16 inputs (p_4 in the low byte)."""
    x = np.arange(1 << 16, dtype=np.int64)
    yy, zz = np.full_like(x, y), np.full_like(x, z)
    yy, zz = step8(kind, yy, zz, x & M8, 4, k8)
    yy, zz = step8(kind, yy, zz, x >> 8, 5, k8)
    return (yy ^ zz) if kind == "mum" else (yy ^ (((zz << 4) | (zz >> 4)) & M8)) if kind == "duo" else yy


def anf_degrees(tab: np.ndarray, nbits: int = 16) -> list[int]:
    deg = []
    pop = np.array([bin(i).count("1") for i in range(1 << nbits)])
    for bit in range(8):
        f = ((tab >> bit) & 1).astype(np.uint8)
        for i in range(nbits):
            f = f.reshape(-1, 2, 1 << i)
            f[:, 1, :] ^= f[:, 0, :]
            f = f.reshape(-1)
        deg.append(int(pop[f.astype(bool)].max()) if f.any() else 0)
    return deg


PAR8 = np.array([[bin(beta & v).count("1") & 1 for v in range(256)] for beta in range(1, 256)], dtype=np.int32)


def walsh_max(tab: np.ndarray, nbits: int = 16) -> float:
    """max over output masks β ≠ 0 and input masks α ≠ 0 of |corr(β·F(x), α·x)|."""
    f = 1 - 2 * PAR8[:, tab]
    for i in range(nbits):
        f = f.reshape(255, -1, 2, 1 << i)
        g = np.empty_like(f)
        g[:, :, 0, :] = f[:, :, 0, :] + f[:, :, 1, :]
        g[:, :, 1, :] = f[:, :, 0, :] - f[:, :, 1, :]
        f = g.reshape(255, -1)
    f[:, 0] = 0
    return float(np.abs(f).max()) / (1 << nbits)


def small(seed: int = 4, states: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    k8 = {"a": rng.integers(0, 128, size=16) * 2 + 1, "b": rng.integers(0, 128, size=16) * 2 + 1,
          "c": rng.integers(0, 256, size=16)}
    out = {}
    for kind in KINDS:
        degs, wal = [], []
        for _ in range(states):
            y, z = int(rng.integers(0, 256)), int(rng.integers(0, 256))
            tab = two_word_table(kind, y, z, k8)
            degs.append(anf_degrees(tab))
            wal.append(walsh_max(tab))
        d = np.array(degs)
        out[kind] = {"anf_degree_per_bit_min_over_states": d.min(axis=0).tolist(),
                     "anf_degree_max": int(d.max()), "walsh_max_corr": max(wal)}
    tab = rng.integers(0, 256, size=1 << 16)
    out["random"] = {"anf_degree_per_bit_min_over_states": anf_degrees(tab), "walsh_max_corr": walsh_max(tab)}
    return out


def duo_g(u, t: int):
    return (rotl(u, twins.rot_of(t)) + u * KA[t]) & M32


def duo_partners(u, t: int) -> list:
    """Every x ≠ u with g_t(x) = g_t(u), by guessing x's top r_t bits (rotl's low bits) and solving x bit by bit from
    the bottom (bit i of a_t·x is x_i plus lower bits, a_t odd): 2^r_t solves. One (x, found) pair per guess."""
    r, a, v = twins.rot_of(t), KA[t], duo_g(u, t)
    out = []
    for g in range(1 << r):
        x = np.full(u.shape, U(g) << U(32 - r), dtype=np.uint64)
        ok = np.ones(u.shape, dtype=bool)
        for i in range(32):
            low = x & U((1 << i) - 1)
            bit = (((v - rotl(x, r) - low * a) & M32) >> U(i)) & U(1)
            if i < 32 - r:
                x |= bit << U(i)
            else:
                ok &= bit == ((x >> U(i)) & U(1))
        out.append((x, ok & (x != u)))
    return out


def duo_adaptive(samples: int, t: int = 4, seed: int = 10) -> dict:
    """The state-dependent two-word cancellation `duo_g_image` counts, at w = 32: the share of states whose word t
    has a partner (a prover who knows the state changes p_t to reach it and fixes p_{t+1} by XOR)."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 1 << 32, size=samples, dtype=np.uint64)
    has = np.zeros(samples, dtype=bool)
    for _, ok in duo_partners(u, t):
        has |= ok
    return {"t": t, "r": twins.rot_of(t), "share_with_partner": float(np.mean(has)), "samples": samples}


def duo_g_image(bits: int = 16, keys_: int = 8, seed: int = 9) -> dict:
    """duo's word t changes the state's z exactly when g_t(u) = rotl(u, r_t) + a_t·u changes, and a change at word t is
    cancelled by word t+1 only if z is unchanged, so two-word cancellations are the collisions of g_t. The share of u
    that have a partner, on the `bits`-bit analogue over random odd a and every rotation."""
    rng = np.random.default_rng(seed)
    m = (1 << bits) - 1
    u = np.arange(1 << bits, dtype=np.int64)
    shares = []
    for _ in range(keys_):
        a = int(rng.integers(0, 1 << (bits - 1))) * 2 + 1
        for r in range(1, bits):
            g = (((u << r) | (u >> (bits - r))) + a * u) & m
            _, inv, n = np.unique(g, return_inverse=True, return_counts=True)
            shares.append(float(np.mean(n[inv] > 1)))
    return {"bits": bits, "mean_share_with_partner": float(np.mean(shares)), "min": min(shares), "max": max(shares),
            "random_function": 1 - float(np.exp(-1))}


def chi_image(bits: int = 16) -> float:
    m = (1 << bits) - 1
    u = np.arange(1 << bits, dtype=np.int64)

    def rl(x, s):
        return ((x << s) | (x >> (bits - s))) & m

    return float(np.unique(u ^ ((~rl(u, 1) & m) & rl(u, 2))).size) / (1 << bits)


# ------------------------------------------------------------------------------------------------------- mad sketch
def mad_sketch(n: int = 256, b: int = 16, L: int = 32, seed: int = 5) -> dict:
    """mad's receipts for every (block-row I, block-column J, chunk s) of P = XY over INT8 residues mod 127, honest and
    by the sketch: receipt = β·A + <C, P> with C_t = Π_{u>t} a_u, and <C, X_I Y_J> = <X_Iᵀ C, Y_J> with X_Iᵀ C formed
    once per (I, chunk) and split into four 8-bit digits, so every remaining product is int8 × int8 into INT32."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 127, size=(n, n), dtype=np.int64)
    Y = rng.integers(0, 127, size=(n, n), dtype=np.int64)
    nb, ns = n // b, n // L
    a = [int(v) for v in KA]
    C = np.zeros(B2, dtype=object)
    w = 1
    for t in reversed(range(B2)):
        C[t] = w
        w = w * a[t] % (1 << 32)
    Aall = w
    Cm = np.array([int(c) for c in C], dtype=np.int64).reshape(b, b)
    honest = np.zeros((nb, nb, ns), dtype=np.int64)
    for s in range(ns):
        P = X[:, : (s + 1) * L] @ Y[: (s + 1) * L, :]
        for i in range(nb):
            for j in range(nb):
                beta = (i * nb + j) * ns + s
                blk = P[i * b:(i + 1) * b, j * b:(j + 1) * b].reshape(-1)
                y = beta
                for t in range(B2):
                    y = (y * a[t] + int(blk[t])) % (1 << 32)
                honest[i, j, s] = y
    sk = np.zeros((nb, nb, ns), dtype=np.int64)
    macs = 0
    g = np.zeros((nb, nb, ns), dtype=np.int64)
    for s in range(ns):
        Ys = Y[s * L:(s + 1) * L, :]
        for i in range(nb):
            Xi = X[i * b:(i + 1) * b, s * L:(s + 1) * L]
            Uis = (Xi.T @ Cm) % (1 << 32)  # L x b; Cm's digits make this int8 x int8 too
            macs += 4 * L * b * b
            dig = [(Uis >> (8 * d)) & 0xFF for d in range(4)]
            for j in range(nb):
                Yj = Ys[:, j * b:(j + 1) * b]
                acc = 0
                for d in range(4):
                    v = int((dig[d] * Yj).sum())
                    assert v < (1 << 31)
                    acc += v << (8 * d)
                macs += 4 * L * b
                g[i, j, s] = acc % (1 << 32)
    for i in range(nb):
        for j in range(nb):
            run = 0
            for s in range(ns):
                run = (run + int(g[i, j, s])) % (1 << 32)
                beta = (i * nb + j) * ns + s
                sk[i, j, s] = (beta * Aall + run) % (1 << 32)
    return {"n": n, "b": b, "L": L, "blocks": nb * nb * ns, "receipts_equal": bool((honest == sk).all()),
            "honest_macs": n ** 3, "sketch_macs": macs, "ratio": macs / n ** 3,
            "ratio_formula": "4/b + 4b/n (digits x (n^3/b + n^2 b))"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--dist", default="u32,int26,fp32")
    ap.add_argument("--samples-log2", type=int, default=0)
    ap.add_argument("--kinds", default="", help="restrict the per-kind attacks to these candidates")
    args = ap.parse_args()
    global KINDS
    if args.kinds:
        KINDS = tuple(args.kinds.split(","))
    ns, nt = (1 << 12, 1 << 9) if args.quick else (1 << 16, 1 << 12)
    if args.samples_log2:
        ns = 1 << args.samples_log2
    every = ["differential", "chain_end", "three_word", "linearity", "degree2", "low_bits", "carries", "small",
             "mum_lattice", "mad_sketch", "duo_g"]
    todo = args.only.split(",") if args.only else every
    res = {}
    for dist in args.dist.split(","):
        if "differential" in todo:
            res.setdefault("differential", {})[dist] = differential(dist, ns)
        if "chain_end" in todo:
            res.setdefault("chain_end", {})[dist] = chain_end(dist, ns >> 2)
        if "three_word" in todo:
            three = [k for k in ("mum", "xsm4", "xsm2", "duo") if k in KINDS]
            res.setdefault("three_word", {})[dist] = three_word(dist, ns >> 2, kinds=three)
        if "linearity" in todo:
            res.setdefault("linearity", {})[dist] = linearity(dist, nt)
        if "degree2" in todo:
            res.setdefault("degree2", {})[dist] = degree2(dist, nt >> 3)
        if "low_bits" in todo:
            res.setdefault("low_bits", {})[dist] = low_bits(dist, nt >> 2)
        if "carries" in todo:
            res.setdefault("carries_ra", {})[dist] = carries(dist, nt)
    if "small" in todo:
        res["small_8bit"] = small()
        res["chi_image_share_16bit"] = chi_image(16)
    if "mum_lattice" in todo:
        res["mum_lattice"] = mum_lattice()
    if "mad_sketch" in todo:
        res["mad_sketch"] = mad_sketch(n=128 if args.quick else 256)
    if "duo_g" in todo:
        res["duo_g_image"] = duo_g_image()
        res["duo_adaptive"] = duo_adaptive(ns >> 4)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
