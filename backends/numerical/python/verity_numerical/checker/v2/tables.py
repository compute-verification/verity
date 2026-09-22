"""The v2 fixed tables (``params.py`` of this package lists them).  Every table carries its integer function
``fn``, its (possibly sparse) domain and the same function on z3 expressions (``sym``), so that the z3 searches at
the real widths reason about a 3.4-million-row table symbolically instead of enumerating it.

Keys are packed mixed-radix: a key ``a * R + b`` with ``b`` known (by an earlier lookup or a Boolean) to lie in
``[0, R)`` decomposes uniquely, so a lookup on such a key binds ``a`` and ``b`` to the row's components without a
further identity (the soundness argument of every ALIGN4 / SHIFT / TNORM lookup; z3 checks it per gadget)."""
from __future__ import annotations

from functools import cache

from ..ctx import Table
from ..params import Params
from .params import V2, check_widths

__all__ = [
    "align4_table",
    "all_tables",
    "hdr_table",
    "lead_table",
    "leadnorm_table",
    "op_table",
    "shift_table",
    "sshift_hi_table",
    "sshift_lo_table",
    "tnorm_table",
]


def _ifchain(z3, key, cases: list[tuple[int, object]], default):
    out = default
    for k, v in reversed(cases):
        out = z3.If(key == k, v, out)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------------------------------------------------

@cache
def op_table(p: Params) -> Table:
    """T_OP: operand word w -> (s, m, e_op); finite words only (the all-ones exponent field has no row).
    ``e_op = e_zero`` for a zero operand (``m = 0``), else ``max(t, 1) - bias``."""
    check_widths(p)
    ez = V2.e_zero(p)
    fmask, emask = 2**p.frac - 1, 2**p.exp - 1

    def fn(w: int) -> tuple[int, int, int]:
        s, t, f = w >> (p.word_bits - 1), (w >> p.frac) & emask, w & fmask
        m = f + (2**p.frac if t else 0)
        return s, m, (ez if m == 0 else max(t, 1) - p.bias)

    def dom(w: int) -> bool:
        return (w >> p.frac) & emask != p.t_finite

    def sym(z3, w):
        s = w / 2 ** (p.word_bits - 1)
        t = (w / 2**p.frac) % 2**p.exp
        f = w % 2**p.frac
        m = f + z3.If(t == 0, 0, 2**p.frac)
        e = z3.If(m == 0, ez, z3.If(t == 0, 1, t) - p.bias)
        return t != p.t_finite, (s, m, e)

    return Table("T_OP", 0, 2**p.word_bits, ("s", "m", "e"), fn, domain=dom, sym=sym,
                 size=2**p.word_bits - 2 * 2**p.frac)


@cache
def hdr_table(p: Params) -> Table:
    """T_HDR: the accumulator word's header h = s 2^exp + t (finite t) -> (s, nz, e = max(t, 1) - bias)."""
    emask = 2**p.exp - 1

    def fn(h: int) -> tuple[int, int, int]:
        s, t = h >> p.exp, h & emask
        return s, int(t != 0), max(t, 1) - p.bias

    def dom(h: int) -> bool:
        return h & emask != p.t_finite

    def sym(z3, h):
        s, t = h / 2**p.exp, h % 2**p.exp
        return t != p.t_finite, (s, z3.If(t == 0, 0, 1), z3.If(t == 0, 1, t) - p.bias)

    return Table("T_HDR", 0, V2.hdr_rows(p), ("s", "nz", "e"), fn, domain=dom, sym=sym, size=V2.hdr_rows(p) - 2)


# ---------------------------------------------------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------------------------------------------------

@cache
def align4_table(p: Params) -> Table:
    """ALIGN4: key 4 d + s4 (d the shift, s4 = s_a + 2 s_b) -> (d~ = min(d, DT), s_p = s_a xor s_b, isz = [d = 0])."""
    DT = V2.dt(p)

    def fn(k: int) -> tuple[int, int, int]:
        d, s4 = k >> 2, k & 3
        return min(d, DT), int(s4 in (1, 2)), int(d == 0)

    def sym(z3, k):
        d, s4 = k / 4, k % 4
        return True, (z3.If(d < DT, d, DT), z3.If(z3.Or(s4 == 1, s4 == 2), 1, 0), z3.If(d == 0, 1, 0))

    return Table("ALIGN4", 0, 4 * V2.shift_rows(p), ("dt", "sp", "isz"), fn, sym=sym)


def _shift_table(name: str, p: Params, bits: int, scale: int) -> Table:
    """(x, d~, s) -> (1 - 2 s) floor(x 2^scale / 2^d~) for x < 2^bits; key x (2 DT_rows) + 2 d~ + s."""
    R = 2 * V2.dt_rows(p)
    DT = V2.dt(p)

    def fn(k: int) -> tuple[int]:
        x, rem = divmod(k, R)
        dt, s = divmod(rem, 2)
        v = (x << scale) >> dt
        return (-v if s else v,)

    def sym(z3, k):
        x, rem = k / R, k % R
        dt, s = rem / 2, rem % 2
        cases = [(d, x * 2 ** (scale - d) if d <= scale else x / 2 ** (d - scale)) for d in range(DT + 1)]
        v = _ifchain(z3, dt, cases, z3.IntVal(0))
        return True, (z3.If(s == 1, -v, v),)

    return Table(name, 0, 2**bits * R, ("term",), fn, sym=sym)


@cache
def shift_table(p: Params) -> Table:
    """SHIFT: the signed aligned product term from (u, d~, s_p)."""
    return _shift_table("SHIFT", p, 2 * p.sig, p.prod_shift)


@cache
def sshift_hi_table(p: Params) -> Table:
    """SSHIFT_HI: the state term's high half, (M_hi, d~, s) -> (1 - 2 s) floor(M_hi 2^(mb + acc_shift) / 2^d~)."""
    return _shift_table("SSHIFT_HI", p, V2.mh(p), V2.mb(p) + p.acc_shift)


@cache
def sshift_lo_table(p: Params) -> Table:
    """SSHIFT_LO: the state term's low half, (M_lo, d~, s) -> (1 - 2 s) floor(M_lo 2^acc_shift / 2^d~)."""
    return _shift_table("SSHIFT_LO", p, V2.mb(p), p.acc_shift)


# ---------------------------------------------------------------------------------------------------------------------
# Leading bit and normalisation
# ---------------------------------------------------------------------------------------------------------------------

@cache
def lead_table(p: Params) -> Table:
    """LEAD: x < 2^vb -> (nz = [x != 0], l = bit_length(x))."""
    vb = V2.vb(p)

    def fn(x: int) -> tuple[int, int]:
        return int(x != 0), x.bit_length()

    def sym(z3, x):
        out = z3.IntVal(vb)
        for l in range(vb - 1, -1, -1):
            out = z3.If(x < 2**l, l, out)
        return True, (z3.If(x == 0, 0, 1), out)

    return Table("LEAD", 0, 2**vb, ("nz", "l"), fn, sym=sym)


@cache
def leadnorm_table(p: Params) -> Table:
    """LEADNORM: key l + (l_hi + 1)(E - floor) -> (koff = k - k_lo, A = 2^(vb + k), e_hat, z, ovf_eff, isfloor)
    with k = min(acc_sig - l, E - e_min + acc_sig - W) (the clamp as a minimum), e1 = E - k + acc_sig - W,
    z = [l = 0 or l + k <= 0] (= [floor(v 2^k) = 0] for 2^(l-1) <= v < 2^l), e_hat = floor if z else e1,
    ovf_eff = [not z and e1 > e_max], isfloor = [E = floor]."""
    check_widths(p)
    L = p.l_hi + 1
    vb = V2.vb(p)
    cl = p.acc_sig - p.width

    def fn(key: int) -> tuple[int, ...]:
        E, l = divmod(key, L)
        E += p.floor
        k = min(p.acc_sig - l, E - p.e_min + cl)
        e1 = E - k + cl
        z = l == 0 or l + k <= 0
        return k - p.k_lo, 2 ** (vb + k), (p.floor if z else e1), int(z), int(not z and e1 > p.e_max), int(E == p.floor)

    def sym(z3, key):
        l, E = key % L, key / L + p.floor
        ku, kc = p.acc_sig - l, E - p.e_min + cl
        k = z3.If(ku < kc, ku, kc)
        e1 = E - k + cl
        z = z3.Or(l == 0, l + k <= 0)
        A = _ifchain(z3, k, [(kk, z3.IntVal(2 ** (vb + kk))) for kk in range(p.k_lo, p.k_hi + 1)], z3.IntVal(0))
        return True, (k - p.k_lo, A, z3.If(z, p.floor, e1), z3.If(z, 1, 0),
                      z3.If(z3.And(z3.Not(z), e1 > p.e_max), 1, 0), z3.If(E == p.floor, 1, 0))

    return Table("LEADNORM", 0, V2.leadnorm_rows(p), ("koff", "A", "e_hat", "z", "ovf", "isfloor"), fn, sym=sym)


@cache
def tnorm_table(p: Params) -> Table:
    """TNORM: key v_lo + 2^vb koff -> floor(v_lo 2^k), k = koff + k_lo."""
    vb = V2.vb(p)

    def fn(key: int) -> tuple[int]:
        koff, v_lo = divmod(key, 2**vb)
        k = koff + p.k_lo
        return ((v_lo << k) if k >= 0 else (v_lo >> -k),)

    def sym(z3, key):
        koff, v_lo = key / 2**vb, key % 2**vb
        cases = [(ko, v_lo * 2 ** (ko + p.k_lo) if ko + p.k_lo >= 0 else v_lo / 2 ** (-(ko + p.k_lo)))
                 for ko in range(V2.k_rows(p))]
        return True, (_ifchain(z3, koff, cases, z3.IntVal(0)),)

    return Table("TNORM", 0, 2**vb * V2.k_rows(p), ("t",), fn, sym=sym)


def all_tables(p: Params) -> list[Table]:
    return [op_table(p), hdr_table(p), align4_table(p), shift_table(p), sshift_hi_table(p), sshift_lo_table(p),
            lead_table(p), leadnorm_table(p), tnorm_table(p)]
