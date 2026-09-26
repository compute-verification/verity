"""BabyBear^6 reference arithmetic and a naive degree-3 sumcheck round (pure Python integers).

Field: BabyBear ``p = 2^31 - 2^27 + 1``; challenge field ``F_{p^6} = F_p[X] / (X^6 - W)`` with ``W`` the smallest
non-square non-cube (``x^6 - a`` is irreducible over ``F_p`` iff ``a`` is neither a square nor a cube, since
``2, 3 | p - 1`` and ``4`` does not divide ``6``).  Elements are 6-tuples of ints in ``[0, p)``; a base-field element
is the tuple ``(a, 0, 0, 0, 0, 0)``.

The round: three tables ``f, g, h`` of ``t`` (even) elements, ``f_k(x) = (1 - x) f[2k] + x f[2k + 1]`` and the round
polynomial ``S(x) = sum_k f_k(x) g_k(x) h_k(x)`` (degree 3, sent as ``S(0), S(1), S(2), S(3)``); the fold with the
verifier's ``r`` is ``f'[k] = (1 - r) f[2k] + r f[2k + 1]``.  This is the shape of A-GKR's GKR c-rounds
(PROTOCOL.md 4.1: degree 3 in the copy variable; ``eq``, ``V_l``, ``V_r`` as the three factors).  Everything here is
the oracle the INT8 kernel (``sumcheck_int8.py``) must agree with bit-for-bit.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

P = 2**31 - 2**27 + 1
DEGREE = 6
LIMB_BITS = 8
LIMBS_PER_BASE = 4
LIMBS_PER_EXT = DEGREE * LIMBS_PER_BASE

Ext = tuple[int, int, int, int, int, int]


def _find_w() -> int:
    for a in range(2, P):
        if pow(a, (P - 1) // 2, P) != 1 and pow(a, (P - 1) // 3, P) != 1:
            return a
    raise RuntimeError("no non-residue found")


#: ``X^6 - W`` irreducible: W is the smallest non-square non-cube of BabyBear (31).
W = _find_w()
ZERO: Ext = (0,) * DEGREE
ONE: Ext = (1,) + (0,) * (DEGREE - 1)


def base(a: int) -> Ext:
    return (a % P,) + (0,) * (DEGREE - 1)


def add(a: Ext, b: Ext) -> Ext:
    return tuple((x + y) % P for x, y in zip(a, b))


def sub(a: Ext, b: Ext) -> Ext:
    return tuple((x - y) % P for x, y in zip(a, b))


def scale(a: Ext, c: int) -> Ext:
    return tuple(x * c % P for x in a)


def mul(a: Ext, b: Ext) -> Ext:
    acc = [0] * (2 * DEGREE - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                acc[i + j] += x * y
    out = [acc[d] + W * acc[d + DEGREE] for d in range(DEGREE - 1)] + [acc[DEGREE - 1]]
    return tuple(v % P for v in out)


def is_base(a: Ext) -> bool:
    return all(v == 0 for v in a[1:])


def random_ext(rng: random.Random) -> Ext:
    return tuple(rng.randrange(P) for _ in range(DEGREE))


def random_base(rng: random.Random) -> Ext:
    return base(rng.randrange(P))


def eval_at(even: Ext, odd: Ext, x: int) -> Ext:
    """``(1 - x) even + x odd`` for a small integer ``x``."""
    return add(scale(even, (1 - x) % P), scale(odd, x % P))


def round_polynomial(f: Sequence[Ext], g: Sequence[Ext], h: Sequence[Ext], points: Sequence[int] = (0, 1, 2, 3)) -> list[Ext]:
    """``S(x)`` at ``points``: ``sum_k f_k(x) g_k(x) h_k(x)``."""
    t = len(f)
    if t % 2 or len(g) != t or len(h) != t:
        raise ValueError("tables must have equal even length")
    out = []
    for x in points:
        s = ZERO
        for k in range(t // 2):
            fk = eval_at(f[2 * k], f[2 * k + 1], x)
            gk = eval_at(g[2 * k], g[2 * k + 1], x)
            hk = eval_at(h[2 * k], h[2 * k + 1], x)
            s = add(s, mul(mul(fk, gk), hk))
        out.append(s)
    return out


def fold(table: Sequence[Ext], r: Ext) -> list[Ext]:
    """``table'[k] = (1 - r) table[2k] + r table[2k + 1]``."""
    one_minus_r = sub(ONE, r)
    return [add(mul(one_minus_r, table[2 * k]), mul(r, table[2 * k + 1])) for k in range(len(table) // 2)]


def naive_round(f: Sequence[Ext], g: Sequence[Ext], h: Sequence[Ext], r: Ext) -> tuple[list[Ext], list[Ext], list[Ext], list[Ext]]:
    """One round: the four round-polynomial values, then the three folded tables."""
    return round_polynomial(f, g, h), fold(f, r), fold(g, r), fold(h, r)


def limbs_balanced(a: int, n: int = LIMBS_PER_BASE) -> list[int]:
    """``n`` signed 8-bit limbs ``l_j in [-128, 128)`` with ``a = sum l_j 2^(8j)`` (the INT8 layout the kernel uses)."""
    out = []
    for _ in range(n):
        l = ((a + 128) & 255) - 128
        out.append(l)
        a = (a - l) >> LIMB_BITS
    if a != 0:
        raise ValueError("value does not fit the limb count")
    return out


def from_limbs(ls: Sequence[int]) -> int:
    return sum(l << (LIMB_BITS * j) for j, l in enumerate(ls))
