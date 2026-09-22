"""The fixed tables the gadgets look up.  Each is a function of one small integer key; a proof backend realises it
as a lookup argument (the key's membership in the table's domain is part of what the lookup proves)."""
from __future__ import annotations

from functools import lru_cache

from .ctx import Table
from .params import Params

__all__ = ["align_table", "lead_table", "norm_table", "pow_table"]


@lru_cache(maxsize=None)
def pow_table(p: Params) -> Table:
    """POW: shift d -> (2^min(d, W), 2^(W - min(d, W))), W = acc_bits.  Row 0 is (1, 2^W): a zero shift leaves the
    remainder no room (r * 2^W < 2^W => r = 0).  Rows d >= W are (2^W, 1): the quotient has no room (q = 0)."""
    W = p.acc_bits

    def fn(d: int) -> tuple[int, int]:
        dd = min(d, W)
        return 2**dd, 2 ** (W - dd)

    return Table("POW", 0, p.shift_rows, ("pow", "ipow"), fn)


@lru_cache(maxsize=None)
def align_table(p: Params) -> Table:
    """ALIGN: shift d -> (sl, sr, isr) for the product P = u * 2^prod_shift, u < 2^(2 sig):

        d <= prod_shift          floor(P / 2^d) = u * 2^(prod_shift - d)        sl = 2^(prod_shift-d), sr = 1
        prod_shift < d < prod_bits  floor(P / 2^d) = floor(u / 2^(d - prod_shift))  sl = 1, sr = 2^(d - prod_shift)
        d >= prod_bits           floor(P / 2^d) = 0                             sl = 1, sr = 2^(2 sig) (q < 1)

    ``isr = 2^(2 sig) / sr`` bounds the remainder: r * isr < 2^(2 sig) <=> r < sr."""
    U = 2 * p.sig

    def fn(d: int) -> tuple[int, int, int]:
        if d <= p.prod_shift:
            return 2 ** (p.prod_shift - d), 1, 2**U
        if d < p.prod_bits:
            return 1, 2 ** (d - p.prod_shift), 2 ** (p.prod_bits - d)
        return 1, 2**U, 1

    return Table("ALIGN", 0, p.shift_rows, ("sl", "sr", "isr"), fn)


@lru_cache(maxsize=None)
def lead_table(p: Params) -> Table:
    """LEAD: leading-bit position l -> (lo, ispan) with lo = 2^(l-1) and the span 2^(l-1) of [2^(l-1), 2^l), so that
    (v - lo) * ispan in [0, 2^s_bits) <=> 2^(l-1) <= v < 2^l.  Row 0 is (0, 2^s_bits): v * 2^s_bits < 2^s_bits <=> v = 0."""
    B = p.s_bits

    def fn(l: int) -> tuple[int, int]:
        if l == 0:
            return 0, 2**B
        return 2 ** (l - 1), 2 ** (B - (l - 1))

    return Table("LEAD", 0, p.l_hi + 1, ("lo", "ispan"), fn)


@lru_cache(maxsize=None)
def norm_table(p: Params) -> Table:
    """NORM: k -> (mul, div, idiv) realising M' = floor(v * 2^k) as div * M' + r = v * mul with r < div:
    mul = 2^max(k, 0), div = 2^max(-k, 0), idiv = 2^(norm_rem_bits - max(-k, 0)) (r * idiv < 2^norm_rem_bits <=> r < div)."""
    R = p.norm_rem_bits

    def fn(k: int) -> tuple[int, int, int]:
        neg = max(-k, 0)
        return 2 ** max(k, 0), 2**neg, 2 ** (R - neg)

    return Table("NORM", p.k_lo, p.k_hi + 1, ("mul", "div", "idiv"), fn)
