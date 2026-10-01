"""IEEE f32 arithmetic, PTX max / min and the E4M3 cast as Boolean Definitions: the next version of each word primitive, the same
function on bits (the scalar operations of the served FP8 linear, around its tensor-core steps).

    F32Add_v3, F32Mul_v3, F32Div_v3    (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>      word view: their _v1
    F32Fma_v3                          (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>   word view: F32Fma_v1
    F32Fabs_v2                         (x: Array<32, Value<1>>) -> Array<32, Value<1>>         word view: F32Fabs_v1
    F32Fmaxf_v2, F32Fminf_v2           (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>      word view: their _v1
    Bf16ToF32_v2                       (a: Array<16, Value<1>>) -> Array<32, Value<1>>         word view: Bf16ToF32_v1
    F32ToE4m3Sat_v2                    (x: Array<32, Value<1>>) -> Array<8, Value<1>>          word view: F32ToE4m3Sat_v1

The arithmetic is `fp`'s, with the IR primitives' NaN words (``ir_nan``; ``F32Fma_v1``'s NaN is 0x7FC00000); max / min and the
cast are written here (formerly C-Flock's `tail_pieces`, which imports them). Each is traced from its builder
(`trace.definition`), the circuit C-Flock's lowering runs for its word primitive.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.types import Array

from .. import fp32 as W
from .. import prims as P
from .. import scalar as S
from . import fp
from .forms import NOT, OR, ZERO, add, const_bits, mux, or_reduce, sub
from .softmax import _eq_const, _is_nan, f32_gt
from .trace import definition

__all__ = ["Bf16ToF32", "F32Add", "F32Div", "F32Fabs", "F32Fma", "F32Fmaxf", "F32Fminf", "F32Mul", "F32ToE4m3Sat"]

BIT = BL.BIT
F32 = Array(32, BIT)
BF16 = Array(16, BIT)
E4M3 = Array(8, BIT)
_AB = (("a", F32), ("b", F32))


def _ult_const(C, w, k: int):
    """``w < k`` for an unsigned word and a constant."""
    return sub(C, w, const_bits(k, len(w)), len(w))[1]


def _fields(C, x):
    s, e, m = x[31], list(x[23:31]), list(x[:23])
    return s, e, m, _eq_const(C, e, 0xFF), NOT(or_reduce(C, e)), or_reduce(C, m)


def fmaxmin(C, a, b, want_max: bool):
    """PTX ``max.f32`` / ``min.f32``: a NaN operand yields the other, two NaNs 0x7FFFFFFF; ``max(+0, -0) = +0``,
    ``min(+0, -0) = -0``."""
    na, nb = _is_nan(C, a), _is_nan(C, b)
    r = mux(C, f32_gt(C, b, a)[0] if want_max else f32_gt(C, a, b)[0], b, a)
    both_zero = NOT(OR(C, or_reduce(C, a[:31]), or_reduce(C, b[:31])))
    zs = C.AND(a[31], b[31]) if want_max else OR(C, a[31], b[31])
    r = mux(C, both_zero, [ZERO] * 31 + [zs], r)
    r = mux(C, na, b, mux(C, nb, a, r))
    return mux(C, C.AND(na, nb), fp.f32_const(0x7FFFFFFF), r)


def f32_to_e4m3_sat(C, x):
    """``F32ToE4m3Sat_v1`` (``cvt.rn.satfinite.e4m3x2.f32``): RN-even onto the E4M3 grid (three fraction bits from 2^-6, steps of
    2^-9 below), ±448 (0x7E) past it and for the infinities, 0x7F for NaN."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    sig = m + [NOT(e_zero)]
    normal = NOT(_ult_const(C, e, 121))                       # |x| >= 2^-6
    amt = mux(C, normal, const_bits(19, 5),
              mux(C, _ult_const(C, e, 116), const_bits(24, 5), sub(C, const_bits(140, 8), e, 8)[0][:5]))
    y, sticky = fp.shr_sticky(C, sig, amt)                    # the kept bits and a guard bit
    n = add(C, y[1:5] + [ZERO], [C.AND(y[0], OR(C, sticky, y[1]))], width=5)
    base = [C.AND(normal, b) for b in sub(C, e, const_bits(121, 8), 8)[0][:4]]
    enc = add(C, [ZERO] * 3 + base, n, width=8)
    sat = OR(C, NOT(_ult_const(C, e, 136)), NOT(_ult_const(C, enc, 0x7F)))
    out = mux(C, sat, const_bits(0x7E, 7), enc[:7]) + [s]
    return mux(C, C.AND(e_max, m_nz), const_bits(0x7F, 8), out)


F32Add = definition("F32Add", 3, lambda C, a, b: fp.f32_add(C, a, b, ir_nan=True), _AB, F32, word=W.F32Add,
                    doc="FADD on bits: F32Add_v1, RN(a + b)")
F32Mul = definition("F32Mul", 3, lambda C, a, b: fp.f32_mul(C, a, b, ir_nan=True), _AB, F32, word=W.F32Mul,
                    doc="FMUL on bits: F32Mul_v1, RN(a * b)")
F32Fma = definition("F32Fma", 3, fp.f32_fma, (("a", F32), ("b", F32), ("c", F32)), F32, word=W.F32Fma,
                    doc="FFMA on bits: F32Fma_v1, RN(a * b + c), one rounding")
F32Div = definition("F32Div", 3, lambda C, a, b: fp.f32_div(C, a, b, ir_nan=True), _AB, F32, word=W.F32Div,
                    doc="div.rn.f32 on bits: F32Div_v1, RN(a / b)")
F32Fabs = definition("F32Fabs", 2, lambda C, x: list(x[:31]) + [ZERO], (("x", F32),), F32, word=S.F32Fabs,
                     doc="fabsf on bits: F32Fabs_v1, the sign bit cleared")
F32Fmaxf = definition("F32Fmaxf", 2, lambda C, a, b: fmaxmin(C, a, b, True), _AB, F32, word=S.F32Fmaxf,
                      doc="max.f32 on bits: F32Fmaxf_v1, NaN yields the other operand, max(+0, -0) = +0")
F32Fminf = definition("F32Fminf", 2, lambda C, a, b: fmaxmin(C, a, b, False), _AB, F32, word=S.F32Fminf,
                      doc="min.f32 on bits: F32Fminf_v1, NaN yields the other operand, min(+0, -0) = -0")
Bf16ToF32 = definition("Bf16ToF32", 2, lambda C, a: fp.bf16_to_f32(a), (("a", BF16),), F32, word=P.Bf16ToF32,
                       doc="bf16 -> f32 on bits: Bf16ToF32_v1, sixteen zero bits below the word")
F32ToE4m3Sat = definition("F32ToE4m3Sat", 2, f32_to_e4m3_sat, (("x", F32),), E4M3, word=P.F32ToE4m3Sat,
                          doc="cvt.rn.satfinite.e4m3x2.f32 on bits: F32ToE4m3Sat_v1")
