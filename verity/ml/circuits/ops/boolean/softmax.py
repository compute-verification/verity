"""Attention's softmax arithmetic as circuits over `forms` (formerly Flock's `tail_pieces`): the FTZ f32 operations, the
strict compare, ``F32Max`` and ``GuardNegInfZero``. Each is its word primitive (`verity.ml.circuits.ops.fp32`) bit for bit. The MUFU
operations around them read tables and stay in Flock's `tail_pieces`.
"""
from __future__ import annotations

from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, and_reduce, const_bits, mux, or_reduce, sub

#: numpy float32's NaN of an invalid operation
DEFAULT_NAN = 0xFFC00000


def _eq_const(C, bits, value: int):
    return and_reduce(C, [b if (value >> i) & 1 else NOT(b) for i, b in enumerate(bits)])


def _is_nan(C, w):
    return C.AND(and_reduce(C, w[23:31]), or_reduce(C, w[:23]))


def ftz(C, w):
    """A subnormal word flushed to its signed zero."""
    e_nz = or_reduce(C, w[23:31])
    return [C.AND(e_nz, m) for m in w[:23]] + list(w[23:])


def f32_add_ftz(C, a, b):
    """``F32AddFtz_v1``: ftz(RN(ftz(a) + ftz(b)))."""
    return ftz(C, fp.f32_add(C, ftz(C, a), ftz(C, b), ir_nan=True))


def f32_sub(C, a, b):
    """numpy float32's ``a - b``: RN; a's NaN quieted when a is a NaN, else b's (its own sign), else 0xFFC00000 for an invalid
    operation (inf - inf of one sign)."""
    r = fp.f32_add(C, a, fp.f32_neg(b))
    A, B = fp.unpack(C, a), fp.unpack(C, b)
    r = mux(C, C.AND(C.AND(A.inf, B.inf), NOT(A.s ^ B.s)), const_bits(DEFAULT_NAN, 32), r)
    src = mux(C, A.nan, a[:22] + a[23:], b[:22] + b[23:])
    return mux(C, OR(C, A.nan, B.nan), src[:22] + [ONE] + src[22:], r)


def f32_sub_ftz(C, a, b):
    """``F32SubFtz_v1``: ftz(RN(ftz(a) - ftz(b)))."""
    return ftz(C, f32_sub(C, ftz(C, a), ftz(C, b)))


def f32_mul_ftz(C, a, b):
    """``F32MulFtz_v1``: ftz(RN(ftz(a) * ftz(b)))."""
    return ftz(C, fp.f32_mul(C, ftz(C, a), ftz(C, b), ir_nan=True))


def f32_fma_ftz(C, a, b, c):
    """``F32FmaFtz_v1``: ftz(RN(ftz(a) * ftz(b) + ftz(c)))."""
    return ftz(C, fp.f32_fma(C, ftz(C, a), ftz(C, b), ftz(C, c)))


def f32_fma_sub_ftz(C, a, b, c):
    """``F32FmaSubFtz_v1``: ftz(RN(ftz(a) * ftz(b) - ftz(c)))."""
    return ftz(C, fp.f32_fma(C, ftz(C, a), ftz(C, b), fp.f32_neg(ftz(C, c))))


def f32_add_ftz_gpu(C, a, b):
    """``F32AddFtz_v2``: ftz(RN(ftz(a) + ftz(b))), every NaN result `F32_NAN_PTX`."""
    return ftz(C, fp.f32_add(C, ftz(C, a), ftz(C, b), nan_word=fp.F32_NAN_PTX))


def f32_sub_ftz_gpu(C, a, b):
    """``F32SubFtz_v2``: ftz(RN(ftz(a) - ftz(b))), every NaN result `F32_NAN_PTX` (so no NaN's payload or sign survives)."""
    return ftz(C, fp.f32_add(C, ftz(C, a), fp.f32_neg(ftz(C, b)), nan_word=fp.F32_NAN_PTX))


def f32_mul_ftz_gpu(C, a, b):
    """``F32MulFtz_v2``: ftz(RN(ftz(a) * ftz(b))), every NaN result `F32_NAN_PTX`."""
    return ftz(C, fp.f32_mul(C, ftz(C, a), ftz(C, b), nan_word=fp.F32_NAN_PTX))


def f32_fma_ftz_gpu(C, a, b, c):
    """``F32FmaFtz_v2``: ftz(RN(ftz(a) * ftz(b) + ftz(c))), every NaN result `F32_NAN_PTX`."""
    return ftz(C, fp.f32_fma(C, ftz(C, a), ftz(C, b), ftz(C, c), nan_word=fp.F32_NAN_PTX))


def f32_fma_sub_ftz_gpu(C, a, b, c):
    """``F32FmaSubFtz_v2``: ftz(RN(ftz(a) * ftz(b) - ftz(c))), every NaN result `F32_NAN_PTX`."""
    return ftz(C, fp.f32_fma(C, ftz(C, a), ftz(C, b), fp.f32_neg(ftz(C, c)), nan_word=fp.F32_NAN_PTX))


def f32_gt(C, a, b):
    """``F32GtStrict_v1``: a > b on f32 values (NaN false, +0 == -0)."""
    _, mag_gt = sub(C, b[:31], a[:31], 31)
    _, mag_lt = sub(C, a[:31], b[:31], 31)
    both_zero = NOT(OR(C, or_reduce(C, a[:31]), or_reduce(C, b[:31])))
    sa, sb = a[31], b[31]
    gt = C.AND(C.AND(NOT(sa), NOT(sb)), mag_gt) ^ C.AND(C.AND(NOT(sa), sb), NOT(both_zero)) ^ C.AND(C.AND(sa, sb), mag_lt)
    return [C.AND(C.AND(gt, NOT(_is_nan(C, a))), NOT(_is_nan(C, b)))]


def f32_max(C, a, b):
    """``F32Max_v1``: ``x > y ? x : y`` on the flushed operands (FA2's ``MaxOp``)."""
    x, y = ftz(C, a), ftz(C, b)
    return mux(C, f32_gt(C, x, y)[0], x, y)


def guard_neg_inf_zero(C, m):
    """``GuardNegInfZero_v1``: ``m == -inf ? 0 : m``."""
    keep = NOT(_eq_const(C, m, 0xFF800000))
    return [C.AND(keep, b) for b in m]
