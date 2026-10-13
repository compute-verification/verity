"""`deepgemm`'s sm_90 block-FP8 linear and grouped einsum on bits: version 2 of each Definition, its word body line for line over
Boolean Definitions (the bodies are `deepgemm`'s, written once over `deepgemm.Ops` and its signatures over the word types), so each
one's word view is its `_v1` at the same statics.

    F32LtFtz_v2                                    traced: setp.lt.ftz.f32 (`f32_gt` of the flushed operands, swapped)
    Fp8Ue8m0Ceil_v2                                one traced circuit (`boolean_dsa.flat`): its three constant words folded
    Fp8GroupScaleUe8m0_v2{G}, Fp8GroupQuantUe8m0Group_v2{G}, FlashInferFp8Quant1x128Group_v2
                                                   one traced circuit each: a group's words share their scale's and their own
                                                   decoding (the tracer's common ANDs once, its dead gates none)
    Fp8GroupQuantUe8m0_v2{K,G}, FlashInferFp8Quant1x128_v2{K}
    DeepGemmFp8Linear_v2{K,N}, FlashInferFp8Linear_v2{K,N}, ScaledMmFp8BlockGrouped_v2{H,K,N,G}

The callees are core's Boolean FP32 pieces (`verity.ml.circuits.ops.boolean`'s `scalar`, `attention`, `mufu`, `gemm`, `cast`,
`elementwise`), `boolean_dsa`'s `U32And_v2`, and `boolean_fp8`'s `Fp8GroupScale_v2` and `ScaledMmFp8BlockSharedScale_v2` (the GEMM, over
`HopperE4m3QgmmaDot32_v2`); each is its word Definition on every word, NaN payloads included.  No table or lookup gate.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.circuits.ops.boolean.softmax import f32_gt, ftz
from verity_vllm.program.registry import boolean_dsa as BD
from verity_vllm.program.registry import boolean_fp8 as BF8
from verity_vllm.program.registry import deepgemm as DG
from verity_vllm.program.registry.boolean_attention import _Bits
from verity_vllm.program.registry.fp8 import FP8_BLOCK

BIT, BF16, F32, E4M3 = BD.BIT, BD.BF16, BD.F32, BD.E4M3
T = (BF16, E4M3, F32)

F32LtFtzV2 = definition("F32LtFtz", 2, lambda C, a, b: f32_gt(C, ftz(C, b), ftz(C, a)), (("a", F32), ("b", F32)), BIT,
                        word=DG.F32LtFtz, doc="setp.lt.ftz.f32 on bits: F32LtFtz_v1, ftz(b) > ftz(a)")

BITS = dict(BD.BITS, fminf=SC.F32Fminf, lt_ftz=F32LtFtzV2, mul_ftz=BA.F32MulFtz, rcp_ftz=BM.MufuRcpFtz, bf16_gt=SC.Bf16GtStrict,
            f2bf16=BG.F2fpBf16)


def bit_ops(B) -> DG.Ops:
    c = _Bits(B)
    return DG.Ops(B, BITS, lambda v: c.word(v, 32), BD.bit_ops(B).neg, lambda v: c.word(v, 16))


def _ceil(B, S, y):
    """`Fp8Ue8m0Ceil_v1`'s body on bits."""
    return DG.ue8m0_ceil(bit_ops(B), y)


Fp8Ue8m0CeilV2 = BD.flat("Fp8Ue8m0Ceil", 2, [], lambda S: ((("y", F32),), F32), _ceil, DG.Fp8Ue8m0Ceil)


def _group_scale(B, S, x):
    """`Fp8GroupScaleUe8m0_v1`'s body on bits."""
    return B.call(Fp8Ue8m0CeilV2, B.call(bind(BF8.Fp8GroupScaleV2, G=S.G), x))


Fp8GroupScaleUe8m0V2 = BD.flat("Fp8GroupScaleUe8m0", 2, ["G"], lambda S: ((("x", Array(S.G, BF16)),), F32), _group_scale,
                               DG.Fp8GroupScaleUe8m0)


def _group_quant_group(B, S, x):
    """`Fp8GroupQuantUe8m0Group_v1`'s body on bits."""
    return DG.group_quant(bit_ops(B), S.G, x, bind(Fp8GroupScaleUe8m0V2, G=S.G), T)


Fp8GroupQuantUe8m0GroupV2 = BD.flat("Fp8GroupQuantUe8m0Group", 2, ["G"], lambda S: DG.group_sig(S.G, T), _group_quant_group,
                                    DG.Fp8GroupQuantUe8m0Group)


def _group_quant(B, S, x):
    """`Fp8GroupQuantUe8m0_v1`'s body on bits."""
    return DG.group_row(B, "Fp8GroupQuantUe8m0", S.K, S.G, x, bind(Fp8GroupQuantUe8m0GroupV2, G=S.G), T)


Fp8GroupQuantUe8m0V2 = EW.on_bits("Fp8GroupQuantUe8m0", 2, ["K", "G"], lambda S: DG.quant_sig(S, S.G, T), _group_quant,
                                  DG.Fp8GroupQuantUe8m0)


def _fi_group(B, S, x):
    """`FlashInferFp8Quant1x128Group_v1`'s body on bits."""
    return DG.fi_group(bit_ops(B), x, T)


FlashInferFp8Quant1x128GroupV2 = BD.flat("FlashInferFp8Quant1x128Group", 2, [], lambda S: DG.fi_group_sig(S, T), _fi_group,
                                         DG.FlashInferFp8Quant1x128Group)


def _fi(B, S, x):
    """`FlashInferFp8Quant1x128_v1`'s body on bits."""
    return DG.group_row(B, "FlashInferFp8Quant1x128", S.K, FP8_BLOCK, x, FlashInferFp8Quant1x128GroupV2, T)


FlashInferFp8Quant1x128V2 = EW.on_bits("FlashInferFp8Quant1x128", 2, ["K"], lambda S: DG.quant_sig(S, FP8_BLOCK, T), _fi,
                                       DG.FlashInferFp8Quant1x128)


def _gemm(S):
    return bind(BF8.ScaledMmFp8BlockSharedScaleV2, K=S.K, N=S.N, G=FP8_BLOCK)


def _deepgemm_linear(B, S, x, w, ws):
    """`DeepGemmFp8Linear_v1`'s body on bits."""
    return DG.linear(B, bind(Fp8GroupQuantUe8m0V2, K=S.K, G=FP8_BLOCK), _gemm(S), x, w, ws, True)


DeepGemmFp8LinearV2 = EW.on_bits("DeepGemmFp8Linear", 2, ["K", "N"], lambda S: DG.linear_sig(S, True, T), _deepgemm_linear,
                                 DG.DeepGemmFp8Linear)


def _flashinfer_linear(B, S, x, w, ws):
    """`FlashInferFp8Linear_v1`'s body on bits."""
    return DG.linear(B, bind(FlashInferFp8Quant1x128V2, K=S.K), _gemm(S), x, w, ws, False)


FlashInferFp8LinearV2 = EW.on_bits("FlashInferFp8Linear", 2, ["K", "N"], lambda S: DG.linear_sig(S, False, T), _flashinfer_linear,
                                   DG.FlashInferFp8Linear)


def _grouped(B, S, xq, sx, w, ws):
    """`ScaledMmFp8BlockGrouped_v1`'s body on bits."""
    return B.batch(bind(BF8.ScaledMmFp8BlockSharedScaleV2, K=S.K, N=S.N, G=S.G), xq, sx, w, ws, axes=(0, 0, 0, 0))


ScaledMmFp8BlockGroupedV2 = EW.on_bits("ScaledMmFp8BlockGrouped", 2, ["H", "K", "N", "G"], lambda S: DG.grouped_sig(S, T), _grouped,
                                       DG.ScaledMmFp8BlockGrouped)

BOOLEAN_DEEPGEMM = [F32LtFtzV2, Fp8Ue8m0CeilV2, Fp8GroupScaleUe8m0V2, Fp8GroupQuantUe8m0GroupV2, Fp8GroupQuantUe8m0V2,
                    FlashInferFp8Quant1x128GroupV2, FlashInferFp8Quant1x128V2, DeepGemmFp8LinearV2, FlashInferFp8LinearV2,
                    ScaledMmFp8BlockGroupedV2]
