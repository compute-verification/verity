"""A head's attention scores and their softmax as one Boolean Definition, one proof unit a head writing its probability row:
the attention of CMP-92's transformer (`sampled_proofs.one_stage.partition`'s size classes: T bf16 outputs, class 16·T rounded
up).

    QkSoftmaxRow_v1{T, D, DOT}   (q: Array<D, Array<16, Value<1>>>, k: Array<T, Array<D, Array<16, Value<1>>>>)
                                 -> Array<T, Array<16, Value<1>>>
    SoftmaxRow_v1{N}             (x: Array<N, Array<16, Value<1>>>) -> Array<N, Array<16, Value<1>>>

`QkSoftmaxRow` is `SoftmaxRow{N=T}` of the T scores ``x_j = GemmCoordinate_v3{K=D, DOT}(q, k_j)`` (`gemm.Gemm_v3` over the key
rows; `gemm.GEMM_HOPPER` binds the H200's step). Its q row is 16·D bits and each key row as long, so at D = 64 every row it
reads is over C-Flock's 111-byte floor, which a unit of short rows alone (a softmax row at T <= 55) doesn't reach. A score
scale is folded into q's producer, as a power of two.

`SoftmaxRow` is `SoftmaxRef_v1{N, bf16}`'s dataflow (`verity_vllm.program.registry.ref_prims`) on the GPU's operations, so a kernel
matches it bit for bit with ``add.rn.f32``, ``mul.rn.f32``, ``div.rn.f32``, ``ex2.approx.ftz.f32`` and ``cvt.rn.bf16.f32``:

- ``m`` is the first greatest x_i (+0 and -0 one value, so the earlier is kept, as `F32MaxRef_v1`'s fold keeps it), or 0x7FFF
  when any x_i is a NaN.
- ``e_j = MUFU.EX2.FTZ(RN(RN(f32(x_j) - f32(m)) * L))``, L = RN32(log2 e) = 0x3FB8AA3B (`MufuEx2Ftz_v2`'s builder), made once
  for each j. The reference's `F32ExpRn_v1` has no Boolean form, and MUFU.EX2 is what an sm_90 kernel computes, read from the
  sm_89 table as the H200 Definitions read it (`verity_vllm.program.kernels.fa3_model`: assumed identical on sm_90).
- ``s = (((+0 + e_0) + e_1) + ...) + e_{N-1}``, in ascending order, and ``r = RN(1 / s)``.
- ``p_j = bf16_rn(RN(e_j * r))``.

Every f32 operation's NaN is the GPU's 0x7FFFFFFF and every bf16 rounding F2FP's (NaN 0x7FFF), as in `elementwise`'s `_v4`s;
none is flushed but MUFU.EX2's operand. An f32 sum is not associative, so a split of a longer row is a new Definition that
states its order, never a regrouping of this one.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.types import Array
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import OR, const_bits, mux

from .. import fp32 as W
from . import gemm as BG
from .elementwise import specialized
from .mufu import mufu_ex2
from .scalar import bf16_key, gt_u

__all__ = ["LOG2E", "QkSoftmaxRow", "SoftmaxRow", "softmax_exp", "softmax_max", "softmax_row"]

BF16 = Array(16, BL.BIT)
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


def softmax_row(C, xs):
    """``p_j = bf16_rn(RN(e_j * r))`` for each j, ``r = RN(1 / s)`` and ``s`` the e_j summed from +0 in ascending order."""
    m = softmax_max(C, xs)
    es = [softmax_exp(C, x, m) for x in xs]
    s = fp.f32_const(0)
    for e in es:
        s = fp.f32_add(C, s, e, nan_word=_NAN)
    r = fp.f32_div(C, fp.f32_const(ONE_F), s, nan_word=_NAN)
    return [fp.f32_to_bf16(C, fp.f32_mul(C, e, r, nan_word=_NAN), W.BF16_NAN_F2FP) for e in es]


def _row(S: Any) -> Array:
    if not isinstance(S.N, int) or S.N < 1:
        raise ValueError(f"N is the row's length, at least 1, not {S.N!r}")
    return Array(S.N, BF16)


SoftmaxRow = specialized("SoftmaxRow", 1, ("N",), lambda S: lambda C, *xs: softmax_row(C, xs),
                         lambda S: ((("x", _row(S)),), _row(S)),
                         doc="a bf16 row's softmax on bits: bf16_rn(RN(e_j * RN(1 / s))), e_j = MUFU.EX2.FTZ(RN(RN(x_j - m) * "
                             "RN32(log2 e))), m the first greatest x_i (0x7FFF on a NaN), s the e_j summed from +0 in order, "
                             "every f32 NaN 0x7FFFFFFF")


def _keys(S: Any) -> int:
    if not isinstance(S.T, int) or S.T < 1:
        raise ValueError(f"T is the number of key rows, at least 1, not {S.T!r}")
    return S.T


@composite("QkSoftmaxRow", 1, ["T", "D", "DOT"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(_keys(S), Array(S.D, BF16)))), Array(_keys(S), BF16)),
           conformance="exact-model-tested", refuse=lambda b: BG.Gemm.refuse({"K": b.get("D")}))
def QkSoftmaxRow(B, S, q, k):
    """A head's probability row on bits: `SoftmaxRow_v1{N=T}` of ``GemmCoordinate_v3{K=D, DOT}(q, k_j)`` for j < T."""
    return B.call(bind(SoftmaxRow, N=S.T), B.call(bind(BG.Gemm, K=S.D, N=S.T, DOT=S.DOT), q, k))
