"""The Goldilocks field F (p = 2^64 - 2^32 + 1) and its cubic extension E = F[θ]/(θ^3 - 7), where the coins live.

F's elements are ints in [0, p). E's are triples (c0, c1, c2) for c0 + c1·θ + c2·θ². 7 generates F's multiplicative group,
so it is not a cube (p ≡ 1 mod 3) and θ^3 - 7 is irreducible. |E| = p^3 ≈ 2^192 bounds every Schwartz–Zippel term.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

P = 2**64 - 2**32 + 1
GENERATOR = 7
TWO_ADICITY = 32
CUBIC = 7

Ext = tuple[int, int, int]
ZERO: Ext = (0, 0, 0)
ONE: Ext = (1, 0, 0)


def embed(x: int) -> Ext:
    return (x % P, 0, 0)


def add(a: Ext, b: Ext) -> Ext:
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P, (a[2] + b[2]) % P)


def sub(a: Ext, b: Ext) -> Ext:
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P, (a[2] - b[2]) % P)


def scale(a: Ext, c: int) -> Ext:
    return (a[0] * c % P, a[1] * c % P, a[2] * c % P)


def mul(a: Ext, b: Ext) -> Ext:
    a0, a1, a2 = a
    b0, b1, b2 = b
    return ((a0 * b0 + CUBIC * (a1 * b2 + a2 * b1)) % P,
            (a0 * b1 + a1 * b0 + CUBIC * a2 * b2) % P,
            (a0 * b2 + a1 * b1 + a2 * b0) % P)


def power(a: Ext, k: int) -> Ext:
    out, base = ONE, a
    while k:
        if k & 1:
            out = mul(out, base)
        base = mul(base, base)
        k >>= 1
    return out


def inv(a: Ext) -> Ext:
    if a == ZERO:
        raise ZeroDivisionError("0 has no inverse in E")
    return power(a, P**3 - 2)


def total(xs: Iterable[Ext]) -> Ext:
    s0 = s1 = s2 = 0
    for x in xs:
        s0 += x[0]
        s1 += x[1]
        s2 += x[2]
    return (s0 % P, s1 % P, s2 % P)


def dot(xs: Sequence[Ext], ys: Sequence[Ext]) -> Ext:
    return total(mul(x, y) for x, y in zip(xs, ys, strict=True))


def dot_f(xs: Sequence[Ext], ys: Sequence[int]) -> Ext:
    """⟨xs, ys⟩ with ys in F."""
    s0 = s1 = s2 = 0
    for x, y in zip(xs, ys, strict=True):
        s0 += x[0] * y
        s1 += x[1] * y
        s2 += x[2] * y
    return (s0 % P, s1 % P, s2 % P)


def powers(b: Ext, n: int) -> list[Ext]:
    out, cur = [], ONE
    for _ in range(n):
        out.append(cur)
        cur = mul(cur, b)
    return out


def eq_table(point: Sequence[Ext]) -> list[Ext]:
    """eq(point, x) for every x in {0,1}^m, x's bit i the coordinate point[i] reads: entry Σ_i x_i 2^i."""
    out = [ONE]
    for r in point:
        one_minus = sub(ONE, r)
        out = [mul(v, one_minus) for v in out] + [mul(v, r) for v in out]
    return out


def to_bytes(x: int) -> bytes:
    return x.to_bytes(8, "little")


def ext_bytes(a: Ext) -> bytes:
    return b"".join(to_bytes(c) for c in a)


def from_bytes(data: bytes) -> list[int] | None:
    """F elements from 8-byte little-endian words; None if a word is not below p (one encoding per element)."""
    if len(data) % 8:
        return None
    xs = [int.from_bytes(data[i:i + 8], "little") for i in range(0, len(data), 8)]
    return xs if all(x < P for x in xs) else None


def ext_from_bytes(data: bytes) -> list[Ext] | None:
    xs = from_bytes(data)
    if xs is None or len(xs) % 3:
        return None
    return [(xs[i], xs[i + 1], xs[i + 2]) for i in range(0, len(xs), 3)]
