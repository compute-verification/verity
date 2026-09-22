"""The v2 arithmetisation's derived widths, as functions of the frozen :class:`Params` (nothing in ``params.py``
changes: the relation and its domain are v1's; only the witness layout differs).

v2 replaces hint + range-check gadgets by lookups into wider fixed tables (see ``gadgets.py``):

    T_OP        operand word -> (s, m, e_op)            e_op = E_ZERO for a zero operand: no zero flag, no exponent mask
    T_HDR       state header (s, t) -> (s, nz, e)        the accumulator word's top 9 bits
    ALIGN4      4 d + s4 -> (d~, s_p, isz)               d~ = min(d, DT); s4 = s_a + 2 s_b; isz = [d = 0]
    SHIFT       (u, d~, s_p) -> +- floor(u 2^prod_shift / 2^d~)          the aligned, signed product term
    SSHIFT_HI   (M_hi, d~, s) -> +- floor(M_hi 2^(mb + acc_shift) / 2^d~)   the state term, high half
    SSHIFT_LO   (M_lo, d~, s) -> +- floor(M_lo 2^acc_shift / 2^d~)          the state term, low half
    LEAD        x < 2^vb -> (nz, bit_length(x))          both halves of the sum magnitude v = v_hi 2^vb + v_lo
    LEADNORM    (l, E) -> (koff, A, e_hat, z, ovf_eff, isfloor)   everything normalisation needs from (l, E)
    TNORM       (v_lo, k) -> floor(v_lo 2^k)             the low half of M' = floor(v 2^k) = v_hi A(k) + TNORM

E_ZERO.  A zero operand is given the exponent ``e_zero = floor - e_max - 1`` so that any product with a zero
operand has ``e_a + e_b <= e_zero + e_max < floor``: it never raises the group maximum, exactly as the v1 mask
``e_hat = e_p + z (floor - e_p)`` never does.  The shift of such a product is larger (up to ``last - 2 e_zero``),
so the ALIGN4 table has ``shift_rows2 = last - 2 e_zero + 1`` shifts (780 at REAL, 512 in v1); its aligned term is
0 either way (``u = 0``).

Split sizes.  ``mb = acc_sig // 2`` bits for the low half of the state magnitude; ``vb = ceil(s_bits / 2)`` for the
low half of the sum magnitude.  The two-half shift is exact because the low half times its scale is below the
high half's scale (``params.py``-style bounds, checked in :func:`check_widths`).
"""
from __future__ import annotations

from ..params import Params

__all__ = ["V2", "check_widths"]


class V2:
    """Namespace of the derived widths (all static functions of ``p``)."""

    @staticmethod
    def e_zero(p: Params) -> int:
        return p.floor - p.e_max - 1

    @staticmethod
    def last(p: Params) -> int:                    # largest group maximum anywhere
        return p.e_hi(len(p.groups) - 1)

    @staticmethod
    def cand_lo(p: Params) -> int:                 # smallest product exponent: two zero operands
        return 2 * V2.e_zero(p)

    @staticmethod
    def shift_rows(p: Params) -> int:              # d = E - e in [0, shift_rows)
        return V2.last(p) - V2.cand_lo(p) + 1

    @staticmethod
    def dt(p: Params) -> int:                      # d~ = min(d, DT): every term is 0 once d >= DT
        return max(p.prod_bits, p.acc_bits)

    @staticmethod
    def dt_rows(p: Params) -> int:
        return V2.dt(p) + 1

    @staticmethod
    def mb(p: Params) -> int:                      # low-half bits of the state magnitude
        return p.acc_sig // 2

    @staticmethod
    def mh(p: Params) -> int:                      # high-half bits
        return p.acc_sig - V2.mb(p)

    @staticmethod
    def vb(p: Params) -> int:                      # low-half bits of the sum magnitude (also the LEAD table width)
        return (p.s_bits + 1) // 2

    @staticmethod
    def vh(p: Params) -> int:
        return p.s_bits - V2.vb(p)

    @staticmethod
    def k_of(p: Params, l: int, E: int) -> int:    # M' = floor(v 2^k): unclamped acc_sig - l, clamped by e_min
        return min(p.acc_sig - l, E - p.e_min + p.acc_sig - p.width)

    @staticmethod
    def e1_of(p: Params, l: int, E: int) -> int:
        return max(E + l - p.width, p.e_min)

    @staticmethod
    def k_rows(p: Params) -> int:
        return p.k_hi - p.k_lo + 1

    @staticmethod
    def leadnorm_rows(p: Params) -> int:
        return (p.l_hi + 1) * (V2.last(p) - p.floor + 1)

    @staticmethod
    def leadnorm_key(p: Params, l, E):
        return l + (p.l_hi + 1) * (E - p.floor)

    @staticmethod
    def hdr_rows(p: Params) -> int:                # state header s 2^exp + t, t finite
        return 2**(p.exp + 1)


def check_widths(p: Params) -> None:
    """The bounds the v2 tables rely on; raised at construction, never silently."""
    if -p.k_lo > V2.vb(p):
        raise ValueError("the normalisation right shift must not exceed the low half of v")
    if V2.vb(p) + p.k_lo < 0:
        raise ValueError("A(k) = 2^(vb + k) must be an integer")
    if V2.mb(p) < 1 or V2.vb(p) < 1:
        raise ValueError("degenerate halves")
    if V2.e_zero(p) + p.e_max >= p.floor:
        raise ValueError("a zero operand's exponent must keep every product below the floor")
