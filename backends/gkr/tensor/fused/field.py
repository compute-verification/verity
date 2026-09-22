"""BabyBear Montgomery arithmetic as the fused kernels perform it, emulated in Python with uint32 semantics.

The kernels keep table values *plain* (in ``[0, p)``); only the challenge is pre-scaled by ``R = 2^32`` on the host so
that one Montgomery product ``mmul(r R, x) = r x`` is plain again.  The round polynomial values come out of two
Montgomery products (``f g R^-1``, then ``. h R^-1``) and are fixed by ``R^2`` on the host (``fix_s``).  Everything
here is the oracle for ``tests/`` (the CPU side) and the source of the kernel constants.
"""
from __future__ import annotations

from .. import reference as ref

P = ref.P                              # 0x78000001
W = ref.W                              # 31
R = 2**32
MU = pow(P, -1, R)                     # 0x88000001 = p^-1 mod 2^32 (subtractive Montgomery: hi(T) - hi(m p))
MU_I32 = MU - R                        # the same 32-bit pattern as a signed int32 constant for Triton
R_MOD_P = R % P
R2_MOD_P = R * R % P
WR = W * R % P                         # W in Montgomery form: mmul(x, WR) = W x plain
MASK32 = R - 1


def mmul(a: int, b: int) -> int:
    """``a b R^-1 mod p`` for ``a, b in [0, p)``, exactly as the kernel computes it (uint32 mul.lo / mul.hi)."""
    t = a * b
    lo, hi = t & MASK32, t >> 32
    m = (lo * MU) & MASK32
    u = (m * P) >> 32
    r = hi - u
    return r + P if r < 0 else r


def madd(a: int, b: int) -> int:
    s = a - P + b
    return s + P if s < 0 else s


def msub(a: int, b: int) -> int:
    s = a - b
    return s + P if s < 0 else s


def ext_mmul(a: ref.Ext, b: ref.Ext) -> ref.Ext:
    """``a b R^-1`` in BabyBear^6 (schoolbook + ``X^6 = W`` through ``W b_j``), coefficientwise as the kernel."""
    w = [None] + [mmul(b[j], WR) for j in range(1, 6)]
    out = []
    for d in range(6):
        c = None
        for i in range(6):
            j = d - i
            if 0 <= j < 6:
                t = mmul(a[i], b[j])
                c = t if c is None else madd(c, t)
        for i in range(6):
            j = d + 6 - i
            if 1 <= j < 6:
                c = madd(c, mmul(a[i], w[j]))
        out.append(c)
    return tuple(out)


def to_mont(x: ref.Ext) -> ref.Ext:
    """``x R mod p`` coefficientwise (the pre-scaled challenge the kernel loads)."""
    return tuple(v * R_MOD_P % P for v in x)


def fix_s(v: int) -> int:
    """Undo the two Montgomery factors of a round-polynomial sum: ``v R^2 mod p``."""
    return v % P * R2_MOD_P % P


def fold_pair(e: ref.Ext, o: ref.Ext, r_mont: ref.Ext) -> ref.Ext:
    """``e + r (o - e)`` as the kernel does it (one extension product)."""
    d = tuple(msub(o[c], e[c]) for c in range(6))
    t = ext_mmul(r_mont, d)
    return tuple(madd(e[c], t[c]) for c in range(6))


def fold_pair_base(e: int, o: int, r_mont: ref.Ext) -> ref.Ext:
    d = msub(o, e)
    out = [mmul(r_mont[c], d) for c in range(6)]
    out[0] = madd(e, out[0])
    return tuple(out)


def points(a: ref.Ext, b: ref.Ext) -> tuple[ref.Ext, ref.Ext, ref.Ext, ref.Ext]:
    """``x(0) = a, x(1) = b, x(2) = 2b - a, x(3) = 3b - 2a`` via ``d = b - a``."""
    d = tuple(msub(b[c], a[c]) for c in range(6))
    x2 = tuple(madd(b[c], d[c]) for c in range(6))
    x3 = tuple(madd(x2[c], d[c]) for c in range(6))
    return a, b, x2, x3
