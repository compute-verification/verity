"""``verity.ml.circuits.silicon.silicon`` generalised to :class:`Params`, kept in its successive-shift form.

At ``REAL`` this must agree with ``silicon.tc_dot`` word for word (``tests/checker/test_differential.py`` checks
it), at ``Params.from_model(HOPPER_E4M3_K32)`` / ``ADA_E4M3_M16N8K32`` with ``model.step`` (``test_fp8.py``); at
reduced widths it is the function the exhaustive searches compare the checker against.  Nothing here is shared
with the gadgets: the reference shifts, the checker divides.
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


def _finite_operand(p: Params, w: int) -> bool:
    t = (w >> p.frac) & (2**p.exp - 1)
    if p.op_nan_rule == "e4m3":      # only the all-ones exponent WITH the all-ones fraction is NaN; no infinities
        return not (t == p.t_finite and (w & (2**p.frac - 1)) == 2**p.frac - 1)
    return t != p.t_finite


def product(p: Params, a: int, b: int) -> Term:
    if not (_finite_operand(p, a) and _finite_operand(p, b)):
        raise Reject("non-finite operand")
    ta, tb = (a >> p.frac) & (2**p.exp - 1), (b >> p.frac) & (2**p.exp - 1)
    ma = (a & (2**p.frac - 1)) | (2**p.frac if ta else 0)
    mb = (b & (2**p.frac - 1)) | (2**p.frac if tb else 0)
    magnitude = (ma * mb) << (p.acc_frac - 2 * p.frac)
    if magnitude == 0:
        return ZERO
    return Term(((a ^ b) >> (p.word_bits - 1)) & 1 == 1, (ta or 1) + (tb or 1) - 2 * p.bias, magnitude)


def decode_acc(p: Params, c: int) -> Term:
    t = (c >> p.acc_frac) & (2**p.acc_exp - 1)
    if t == p.acc_t_finite:
        raise Reject("non-finite accumulator")
    f = c & (2**p.acc_frac - 1)
    negative = c >> (p.acc_word_bits - 1) == 1
    if t == 0:
        return ZERO if f == 0 else Term(negative, p.acc_e_min, f)
    return Term(negative, t - p.acc_bias, (1 << p.acc_frac) | f)


def pack(p: Params, term: Term) -> int:
    if term.magnitude == 0:
        return 0
    if term.exponent > p.acc_e_max:
        return (int(term.negative) << (p.acc_word_bits - 1)) | (p.acc_t_finite << p.acc_frac)
    field = term.exponent + p.acc_bias
    if term.magnitude & (1 << p.acc_frac) == 0:
        field -= 1
    return (int(term.negative) << (p.acc_word_bits - 1)) | (field << p.acc_frac) | (term.magnitude & (2**p.acc_frac - 1))


def group_sum(p: Params, terms: Sequence[Term]) -> Term:
    """One grouped, truncating step; ``rescale < 0`` (lossy) truncates every term -- the state included -- to W bits."""
    rescale = p.width - p.acc_sig
    max_exponent = p.floor
    for term in terms:
        if term.magnitude and term.exponent > max_exponent:
            max_exponent = term.exponent
    total = 0
    for term in terms:
        if not term.magnitude:
            continue
        scaled = term.magnitude << rescale if rescale >= 0 else term.magnitude >> -rescale
        aligned = scaled >> (max_exponent - term.exponent)
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
    if exponent < p.acc_e_min:
        magnitude >>= p.acc_e_min - exponent
        exponent = p.acc_e_min
    magnitude = magnitude >> rescale if rescale >= 0 else magnitude << -rescale
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
        if acc.magnitude and acc.exponent > p.acc_e_max:
            return pack(p, acc)
        start += size
    return pack(p, acc)


def to_bf16(p: Params, y32: int) -> int:
    """Round to nearest even to the operand width (finite and infinite words; NaN is outside the domain).

    Only meaningful when the operand and accumulator formats share their exponent width (BF16 <- FP32)."""
    if p.acc_exp != p.exp:
        raise Reject("the epilogue cast is defined for operand formats sharing the accumulator's exponent width")
    L = p.acc_frac - p.frac
    lsb = (y32 >> L) & 1
    return ((y32 + (2 ** (L - 1) - 1) + lsb) >> L) & (2**p.word_bits - 1)


def vu(p: Params, a: Sequence[int], b: Sequence[int]) -> int:
    k = sum(p.groups)
    c = 0
    for start in range(0, len(a), k):
        c = dot(p, c, a[start:start + k], b[start:start + k])
    return to_bf16(p, c)
