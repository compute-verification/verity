"""``verity.ml.tc.silicon`` generalised to :class:`Params`, kept in its successive-shift form.

At ``REAL`` this must agree with ``silicon.tc_dot`` word for word (``tests/checker/test_differential.py`` checks
it); at reduced widths it is the function the exhaustive searches compare the checker against.  Nothing here is
shared with the gadgets: the reference shifts, the checker divides.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .params import Params, Reject

__all__ = ["Term", "decode_acc", "dot", "group_sum", "pack", "product", "to_bf16", "vu"]


@dataclass(frozen=True)
class Term:
    negative: bool
    exponent: int          # exponent of bit acc_frac of the magnitude
    magnitude: int         # 0 is the unsigned zero


ZERO = Term(False, 0, 0)


def product(p: Params, a: int, b: int) -> Term:
    ta, tb = (a >> p.frac) & (2**p.exp - 1), (b >> p.frac) & (2**p.exp - 1)
    if ta == p.t_finite or tb == p.t_finite:
        raise Reject("non-finite operand")
    ma = (a & (2**p.frac - 1)) | (2**p.frac if ta else 0)
    mb = (b & (2**p.frac - 1)) | (2**p.frac if tb else 0)
    magnitude = (ma * mb) << (p.acc_frac - 2 * p.frac)
    if magnitude == 0:
        return ZERO
    return Term(((a ^ b) >> (p.word_bits - 1)) & 1 == 1, (ta or 1) + (tb or 1) - 2 * p.bias, magnitude)


def decode_acc(p: Params, c: int) -> Term:
    t = (c >> p.acc_frac) & (2**p.exp - 1)
    if t == p.t_finite:
        raise Reject("non-finite accumulator")
    f = c & (2**p.acc_frac - 1)
    negative = c >> (p.acc_word_bits - 1) == 1
    if t == 0:
        return ZERO if f == 0 else Term(negative, p.e_min, f)
    return Term(negative, t - p.bias, (1 << p.acc_frac) | f)


def pack(p: Params, term: Term) -> int:
    if term.magnitude == 0:
        return 0
    if term.exponent > p.e_max:
        return (int(term.negative) << (p.acc_word_bits - 1)) | (p.t_finite << p.acc_frac)
    field = term.exponent + p.bias
    if term.magnitude & (1 << p.acc_frac) == 0:
        field -= 1
    return (int(term.negative) << (p.acc_word_bits - 1)) | (field << p.acc_frac) | (term.magnitude & (2**p.acc_frac - 1))


def group_sum(p: Params, terms: Sequence[Term]) -> Term:
    rescale = p.width - p.acc_sig
    max_exponent = p.floor
    for term in terms:
        if term.magnitude and term.exponent > max_exponent:
            max_exponent = term.exponent
    total = 0
    for term in terms:
        if not term.magnitude:
            continue
        aligned = (term.magnitude << rescale) >> (max_exponent - term.exponent)
        total += -aligned if term.negative else aligned
    if total == 0:
        return ZERO
    negative = total < 0
    magnitude = -total if negative else total
    bit_length = magnitude.bit_length()
    exponent = max_exponent + bit_length - p.width
    if bit_length > p.width:
        magnitude >>= bit_length - p.width
    else:
        magnitude <<= p.width - bit_length
    if exponent < p.e_min:
        magnitude >>= p.e_min - exponent
        exponent = p.e_min
    magnitude >>= rescale
    if magnitude == 0:
        return ZERO
    return Term(negative, exponent, magnitude)


def dot(p: Params, c: int, a: Sequence[int], b: Sequence[int]) -> int:
    k = sum(p.groups)
    if len(a) != k or len(b) != k:
        raise Reject(f"{k} operand pairs expected")
    acc = decode_acc(p, c)
    start = 0
    for size in p.groups:
        acc = group_sum(p, [acc, *(product(p, a[i], b[i]) for i in range(start, start + size))])
        if acc.magnitude and acc.exponent > p.e_max:
            return pack(p, acc)
        start += size
    return pack(p, acc)


def to_bf16(p: Params, y32: int) -> int:
    """Round to nearest even to the operand width (finite and infinite words; NaN is outside the domain)."""
    L = p.acc_frac - p.frac
    lsb = (y32 >> L) & 1
    return ((y32 + (2 ** (L - 1) - 1) + lsb) >> L) & (2**p.word_bits - 1)


def vu(p: Params, a: Sequence[int], b: Sequence[int]) -> int:
    k = sum(p.groups)
    c = 0
    for start in range(0, len(a), k):
        c = dot(p, c, a[start:start + k], b[start:start + k])
    return to_bf16(p, c)
