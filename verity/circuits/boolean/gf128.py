"""GF(2^128) = GF(2)[x]/(x^128 + x^7 + x^2 + x + 1), C-Flock's F, as integers and over `forms`: the field every fingerprint
under a live coin works in, here so that each unit that may import `verity.circuits` (the verifiers' circuits in
`verity.core` among them) reaches one product.

An element is an int below 2^128 whose bit i is the coefficient of x^i; over forms it is 128 forms, bit i first. `mul` is the
reference. `mul_forms` is Karatsuba down to single bits (`karatsuba`): ``kara(a, b)`` on 2^n coefficients is one AND for
n = 0, else, split at h = 2^(n-1) into (lo, hi), p0 = kara(lo_a, lo_b), then p2 = kara(hi_a, hi_b), then p1 = kara(lo_a +
hi_a, lo_b + hi_b), and the product p0 + X^h (p1 + p0 + p2) + X^2h p2; then the reduction (`reduce_forms`): from
coefficient 254 down to 128, coefficient i is added into i - 128, i - 127, i - 126 and i - 121 and dropped. XORs only, so
3^7 = 2,187 ANDs, one per base product, in that order. `verity.definitions.ops.boolean.gf2k` registers it as
``Gf128Mul_v1``.
"""
from __future__ import annotations

__all__ = ["MASK", "clmul", "karatsuba", "mul", "mul_forms", "reduce", "reduce_forms"]

MASK = (1 << 128) - 1


def clmul(a: int, b: int) -> int:
    """The carry-less product of two polynomials over GF(2), four bits of ``b`` at a time."""
    a2 = a << 1
    a4 = a << 2
    a8 = a << 3
    t = [0, a, a2, a2 ^ a, a4, a4 ^ a, a4 ^ a2, a4 ^ a2 ^ a]
    t += [a8 ^ x for x in t]
    r = 0
    s = 0
    while b:
        r ^= t[b & 15] << s
        b >>= 4
        s += 4
    return r


def reduce(r: int) -> int:
    """``r`` (degree below 255) modulo x^128 + x^7 + x^2 + x + 1."""
    for _ in range(2):
        h = r >> 128
        r = (r & MASK) ^ h ^ (h << 1) ^ (h << 2) ^ (h << 7)
    return r


def mul(a: int, b: int) -> int:
    return reduce(clmul(a, b))


def karatsuba(C, a: list, b: list) -> list:
    """The carry-less product of two forms vectors of equal power-of-two length n, LSB first: 2n - 1 forms, 3^log2(n) ANDs."""
    n = len(a)
    if n == 1:
        return [C.AND(a[0], b[0])]
    h = n // 2
    p0 = karatsuba(C, a[:h], b[:h])
    p2 = karatsuba(C, a[h:], b[h:])
    pm = karatsuba(C, [x ^ y for x, y in zip(a[:h], a[h:])], [x ^ y for x, y in zip(b[:h], b[h:])])
    out = p0 + [None] + p2
    for i in range(2 * h - 1):
        mid = pm[i] ^ p0[i] ^ p2[i]
        j = i + h
        out[j] = mid if out[j] is None else out[j] ^ mid
    return out


def reduce_forms(r: list) -> list:
    """The 255 product coefficients modulo the GHASH polynomial: x^k for k >= 128 is x^(k-128)·(x^7 + x^2 + x + 1)."""
    r = list(r)
    for k in range(len(r) - 1, 127, -1):
        for t in (0, 1, 2, 7):
            r[k - 128 + t] = r[k - 128 + t] ^ r[k]
    return r[:128]


def mul_forms(C, a: list, b: list) -> list:
    return reduce_forms(karatsuba(C, a, b))
