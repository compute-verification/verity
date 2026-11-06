"""The Reed–Solomon code: a row of `width` elements is a polynomial's coefficients, and its codeword is the polynomial's values
at the n-th roots of unity, n = width · blowup, both powers of two. The map is F-linear, so a row over E encodes
coordinate by coordinate. Its distance is d = n - width + 1."""
from __future__ import annotations

from collections.abc import Sequence

from .field import GENERATOR, P, TWO_ADICITY, Ext


def root_of_unity(n: int) -> int:
    if n < 1 or n & (n - 1) or n > 1 << TWO_ADICITY:
        raise ValueError(f"no {n}-th root of unity of order a power of two in F")
    return pow(GENERATOR, (P - 1) // n, P)


def ntt(coeffs: Sequence[int], n: int) -> list[int]:
    """The values at ω^0 .. ω^(n-1), ω = root_of_unity(n), of the polynomial with these coefficients (len(coeffs) ≤ n)."""
    if len(coeffs) > n:
        raise ValueError(f"{len(coeffs)} coefficients exceed the domain of {n} points")
    a = [c % P for c in coeffs] + [0] * (n - len(coeffs))
    bits = n.bit_length() - 1
    for i in range(n):
        j = int(format(i, f"0{bits}b")[::-1], 2) if bits else 0
        if i < j:
            a[i], a[j] = a[j], a[i]
    size = 2
    while size <= n:
        w = root_of_unity(size)
        half = size // 2
        for start in range(0, n, size):
            t = 1
            for k in range(start, start + half):
                u, v = a[k], a[k + half] * t % P
                a[k], a[k + half] = (u + v) % P, (u - v) % P
                t = t * w % P
        size *= 2
    return a


def encode(row: Sequence[int], n: int) -> list[int]:
    return ntt(row, n)


def encode_ext(row: Sequence[Ext], n: int) -> list[Ext]:
    c0, c1, c2 = (ntt([x[i] for x in row], n) for i in range(3))
    return list(zip(c0, c1, c2))


def distance(width: int, blowup: int) -> int:
    return width * blowup - width + 1
