"""A bf16 row's softmax, and RMSNorm's output, as Boolean Definitions split so that each outputs at most 16 bits: the row
reductions of CMP-92's transformer, whose every operation is a proof unit.

    SoftmaxMax_v1{N}          (x: Array<N, Array<16, Value<1>>>) -> Array<16, Value<1>>
    SoftmaxRecip_v1{N, HALF}  (x: Array<N, Array<16, Value<1>>>, m: Array<16, Value<1>>) -> Array<16, Value<1>>
    SoftmaxOut_v1             (x, m: Array<16, Value<1>>, r: Array<32, Value<1>>) -> Array<16, Value<1>>
    RmsNormOut_v1             (x: Array<16, Value<1>>, rs: Array<32, Value<1>>, w: Array<16, Value<1>>) -> Array<16, Value<1>>

Softmax over x_0 .. x_{N-1} is `SoftmaxRef_v1{N, bf16}`'s dataflow (`verity_vllm.program.registry.ref_prims`) on the GPU's
operations, so a kernel matches it bit for bit with ``add.rn.f32``, ``mul.rn.f32``, ``div.rn.f32``, ``ex2.approx.ftz.f32``
and ``cvt.rn.bf16.f32``:

- ``m`` is the first greatest x_i (+0 and -0 one value, so the earlier is kept, as `F32MaxRef_v1`'s fold keeps it), or 0x7FFF
  when any x_i is a NaN. It is one of the x_i, so `SoftmaxMax` outputs it exactly, in 16 bits.
- ``e_j = MUFU.EX2.FTZ(RN(RN(f32(x_j) - f32(m)) * L))``, L = RN32(log2 e) = 0x3FB8AA3B (`MufuEx2Ftz_v2`'s builder): the
  reference's `F32ExpRn_v1` has no Boolean form, and MUFU.EX2 is what an sm_120 kernel computes.
- ``s = (((+0 + e_0) + e_1) + ...) + e_{N-1}``, in ascending order, and ``r = RN(1 / s)``. `SoftmaxRecip` outputs r's low
  (HALF = 0) or high (HALF = 1) 16 bits.
- ``p_j = bf16_rn(RN(e_j * r))``. `SoftmaxOut` recomputes e_j from x_j and m, so it reads x_j, m and r's two halves.

RMSNorm's output is `RMSNormRef_v1{N, EPS, bf16}`'s: ``out_j = bf16_rn(RN(f32(w_j) * f32(bf16_rn(RN(f32(x_j) * rs)))))``,
which is `Bf16MulBf16_v2` (w_j, `Bf16MulF32_v2` (x_j, rs)), rs the row's reciprocal root.

Every f32 operation's NaN is the GPU's 0x7FFFFFFF and every bf16 rounding F2FP's (NaN 0x7FFF), as in `elementwise`'s `_v4`s;
none is flushed but MUFU.EX2's operand. An f32 sum is not associative, so a split of a longer row is a new Definition that
states its order, never a regrouping of this one.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import OR, const_bits, mux
from verity.ml.circuits.boolean.trace import definition

from .. import fp32 as W
from .elementwise import bf16_mul_f32, specialized
from .mufu import mufu_ex2
from .scalar import bf16_key, gt_u

__all__ = ["LOG2E", "RmsNormOut", "SoftmaxMax", "SoftmaxOut", "SoftmaxRecip", "rms_norm_out", "softmax_exp", "softmax_max",
           "softmax_out", "softmax_recip"]

BF16 = U16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
#: RN32(log2 e)
LOG2E = 0x3FB8AA3B
ONE_F = 0x3F800000
_NAN = fp.F32_NAN_PTX


def softmax_max(C, xs):
    """``m``: the first greatest of the bf16 words ``xs`` in `bf16_key`'s order (+0 and -0 one key), or 0x7FFF when any is a
    NaN. The running word and its key are both selected, so each word's key is made once."""
    nan, key = bf16_key(C, xs[0])
    m = list(xs[0])
    for x in xs[1:]:
        n, k = bf16_key(C, x)
        take = gt_u(C, k, key)
        m, key, nan = mux(C, take, x, m), mux(C, take, k, key), OR(C, nan, n)
    return mux(C, nan, const_bits(W.BF16_NAN_F2FP, 16), m)


def softmax_exp(C, x, m):
    """``e = MUFU.EX2.FTZ(RN(RN(f32(x) - f32(m)) * L))`` of bf16 words x and m."""
    d = fp.f32_add(C, fp.bf16_to_f32(x), fp.f32_neg(fp.bf16_to_f32(m)), nan_word=_NAN)
    return mufu_ex2(C, fp.f32_mul(C, d, fp.f32_const(LOG2E), nan_word=_NAN))


def softmax_recip(C, xs, m):
    """``r = RN(1 / s)``, ``s`` the row's `softmax_exp` words summed from +0 in ascending order."""
    s = fp.f32_const(0)
    for x in xs:
        s = fp.f32_add(C, s, softmax_exp(C, x, m), nan_word=_NAN)
    return fp.f32_div(C, fp.f32_const(ONE_F), s, nan_word=_NAN)


def softmax_out(C, x, m, r):
    """``p = bf16_rn(RN(e * r))``, e = `softmax_exp` (x, m)."""
    return fp.f32_to_bf16(C, fp.f32_mul(C, softmax_exp(C, x, m), r, nan_word=_NAN), W.BF16_NAN_F2FP)


def rms_norm_out(C, x, rs, w):
    """``bf16_rn(RN(f32(w) * f32(bf16_rn(RN(f32(x) * rs)))))``: `Bf16MulBf16_v2` (w, `Bf16MulF32_v2` (x, rs))."""
    return bf16_mul_f32(C, w, fp.bf16_to_f32(bf16_mul_f32(C, x, rs)))


def _row(S: Any) -> int:
    if not isinstance(S.N, int) or S.N < 1:
        raise ValueError(f"N is the row's length, at least 1, not {S.N!r}")
    return S.N


def _half(S: Any) -> int:
    if S.HALF not in (0, 1):
        raise ValueError(f"HALF is 0 (r's low 16 bits) or 1 (its high 16 bits), not {S.HALF!r}")
    return S.HALF


def _recip(S: Any):
    lo = 16 * _half(S)
    return lambda C, *e: softmax_recip(C, e[:-1], e[-1])[lo:lo + 16]


def _recip_signature(S: Any):
    _half(S)
    return (("x", Array(_row(S), BF16)), ("m", BF16)), U16


SoftmaxMax = specialized("SoftmaxMax", 1, ("N",), lambda S: lambda C, *xs: softmax_max(C, xs),
                         lambda S: ((("x", Array(_row(S), BF16)),), BF16),
                         doc="a bf16 row's softmax max on bits: the first greatest x_i (+0 == -0), 0x7FFF if any is a NaN")
SoftmaxRecip = specialized("SoftmaxRecip", 1, ("N", "HALF"), _recip, _recip_signature,
                           doc="a bf16 row's softmax reciprocal on bits: 16 bits (HALF) of RN(1 / s), s the sum from +0 of "
                               "MUFU.EX2.FTZ(RN(RN(x_j - m) * RN32(log2 e))), every NaN 0x7FFFFFFF")
SoftmaxOut = definition("SoftmaxOut", 1, softmax_out, (("x", BF16), ("m", BF16), ("r", F32)), BF16,
                        doc="one softmax output on bits: bf16_rn(RN(MUFU.EX2.FTZ(RN(RN(x - m) * RN32(log2 e))) * r)), NaN 0x7FFF")
RmsNormOut = definition("RmsNormOut", 1, rms_norm_out, (("x", BF16), ("rs", F32), ("w", BF16)), BF16,
                        doc="one RMSNorm output on bits: bf16_rn(RN(f32(w) * f32(bf16_rn(RN(f32(x) * rs))))), NaN 0x7FFF")
