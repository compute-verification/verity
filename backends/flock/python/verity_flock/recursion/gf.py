"""Flock's fields (PROTOCOL.md §2.1, §2.2, §2.4): F = GF(2^128) in GHASH form and K = F[u]/(u^2 + u + x^-1).

Scalars are Python ints (bit i is the coefficient of x^i); K elements are pairs (c0, c1). Batches are numpy arrays of
shape (..., 2) and dtype uint64 holding (lo, hi).
"""
from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
MOD = (1 << 128) | 0x87
XINV = (0x8000000000000000 << 64) | 0x43
PHI8_GEN = (0x0DCB364640A222FE << 64) | 0x6B8330483C2E9849


def mul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 128:
            a ^= MOD
    return r


def inv(a: int) -> int:
    """a^(2^128 - 2); inv(0) = 0."""
    acc = 1
    for _ in range(127):
        acc = mul(mul(acc, acc), a)
    return mul(acc, acc)


def kmul(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    p0 = mul(a[0], b[0])
    p1 = mul(a[1], b[1])
    p2 = mul(a[0] ^ a[1], b[0] ^ b[1])
    return (p0 ^ mul(XINV, p1), p2 ^ p0)


def kadd(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] ^ b[0], a[1] ^ b[1])


def ksmul(a: tuple[int, int], s: int) -> tuple[int, int]:
    return (mul(a[0], s), mul(a[1], s))


def to_bytes(a: int) -> bytes:
    return (a & MASK64).to_bytes(8, "little") + (a >> 64).to_bytes(8, "little")


def from_bytes(b: bytes) -> int:
    return int.from_bytes(b[:8], "little") | (int.from_bytes(b[8:16], "little") << 64)


def from_hex(h: str) -> int:
    """Record hex: `{hi:016x}{lo:016x}` is the 128-bit integer."""
    return int(h, 16)


def _phi8_table() -> list[int]:
    pows, p = [], 1
    for _ in range(8):
        pows.append(p)
        p = mul(p, PHI8_GEN)
    out = []
    for v in range(256):
        acc = 0
        for i in range(8):
            if (v >> i) & 1:
                acc ^= pows[i]
        out.append(acc)
    return out


PHI8 = _phi8_table()
NODES_S = PHI8[:64]
NODES_L = PHI8[64:128]


def _den(d: int) -> int:
    acc = 1
    for i in range(2**d - 1):
        acc = mul(acc, PHI8[i + 1])
    return acc


DEN6_INV = inv(_den(6))
DEN7_INV = inv(_den(7))


def eq_table(r: list[int]) -> list[int]:
    """eq(r, u) for u < 2^|r|, index bit j <-> r[j] (LSB first)."""
    return from_arr(eq_arr(r))


def eq_arr(r: list[int]) -> np.ndarray:
    t = to_arr([1])
    for rj in r:
        c = to_arr([rj ^ 1, rj])
        t = np.concatenate([vmul(t, c[0]), vmul(t, c[1])])
    return t


def eq_table_k(r: list[tuple[int, int]]) -> list[tuple[int, int]]:
    t = [(1, 0)]
    for rj in r:
        a = (rj[0] ^ 1, rj[1])
        t = [kmul(x, a) for x in t] + [kmul(x, rj) for x in t]
    return t


def lagrange(nodes: list[int], den_inv: int, z: int) -> list[int]:
    if z in nodes:
        i = nodes.index(z)
        return [1 if j == i else 0 for j in range(len(nodes))]
    zn = 1
    for n in nodes:
        zn = mul(zn, z ^ n)
    c = mul(zn, den_inv)
    return [mul(c, inv(z ^ n)) for n in nodes]


def skip_weights(z: int) -> list[int]:
    return lagrange(NODES_S, DEN6_INV, z)


def _frac(y: int) -> int:
    return mul(y, inv(y ^ 1))


def _pinned() -> list[int]:
    x = 2
    x2 = mul(x, x)
    x4 = mul(x2, x2)
    x8 = mul(x4, x4)
    return [PHI8[0xF7], PHI8[0x53], PHI8[0xB5], _frac(x), _frac(x2), _frac(x4), _frac(x8)]


PINNED = _pinned()


# ---------------------------------------------------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------------------------------------------------

def to_arr(xs) -> np.ndarray:
    xs = list(xs)
    out = np.empty((len(xs), 2), dtype=np.uint64)
    for i, x in enumerate(xs):
        out[i, 0] = x & MASK64
        out[i, 1] = x >> 64
    return out


def from_arr(a: np.ndarray) -> list[int]:
    a = np.asarray(a).reshape(-1, 2)
    return [int(lo) | (int(hi) << 64) for lo, hi in a]


_ONE = np.uint64(1)
_ALL = np.uint64(MASK64)


def _clmul64(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(np.broadcast(a, b).shape, dtype=np.uint64)
    hi = np.zeros_like(lo)
    for i in range(64):
        m = ((b >> np.uint64(i)) & _ONE) * _ALL
        lo ^= (a << np.uint64(i)) & m
        if i:
            hi ^= (a >> np.uint64(64 - i)) & m
    return lo, hi


def vmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise product of (..., 2) uint64 batches (broadcasting)."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    al, ah, bl, bh = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    p0l, p0h = _clmul64(al, bl)
    p2l, p2h = _clmul64(ah, bh)
    ml, mh = _clmul64(al ^ ah, bl ^ bh)
    m1l = ml ^ p0l ^ p2l
    m1h = mh ^ p0h ^ p2h
    r0, r1, r2, r3 = p0l, p0h ^ m1l, p2l ^ m1h, p2h
    s1, s2, s7 = np.uint64(1), np.uint64(2), np.uint64(7)
    tl = r2 ^ (r2 << s1) ^ (r2 << s2) ^ (r2 << s7)
    th = r3 ^ ((r3 << s1) | (r2 >> np.uint64(63))) ^ ((r3 << s2) | (r2 >> np.uint64(62))) ^ ((r3 << s7) | (r2 >> np.uint64(57)))
    o = (r3 >> np.uint64(63)) ^ (r3 >> np.uint64(62)) ^ (r3 >> np.uint64(57))
    out = np.empty(np.broadcast(al, bl).shape + (2,), dtype=np.uint64)
    out[..., 0] = r0 ^ tl ^ o ^ (o << s1) ^ (o << s2) ^ (o << s7)
    out[..., 1] = r1 ^ th
    return out


def vkmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise K products of (..., 2, 2) batches: [..., 0, :] is c0, [..., 1, :] is c1."""
    p0 = vmul(a[..., 0, :], b[..., 0, :])
    p1 = vmul(a[..., 1, :], b[..., 1, :])
    p2 = vmul(a[..., 0, :] ^ a[..., 1, :], b[..., 0, :] ^ b[..., 1, :])
    out = np.empty(p0.shape[:-1] + (2, 2), dtype=np.uint64)
    out[..., 0, :] = p0 ^ vmul(to_arr([XINV])[0], p1)
    out[..., 1, :] = p2 ^ p0
    return out
