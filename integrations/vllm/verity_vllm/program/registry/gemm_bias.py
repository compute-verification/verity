"""vLLM's batch-invariant linear with a bias as ONE Definition: `GemmBias_v1{K,N}(x, W, b)` (cc 8.x) and
`GemmBiasF32Epilogue_v1{K,N,DOT}(x, W, b)` (cuBLASLt targets).

`linear_batch_invariant` runs the persistent matmul (`matmul_kernel_persistent`, `bias_ptr` unset: `Gemm_v1`'s coordinate, the f32
accumulator rounded once to bf16) and then `output + bias` in bf16 (`BiasAdd_v1`: one f32 add of the two bf16 words, RNE).  Written as
two Definitions, the pre-bias word is a Call boundary: a value the proof must read as committed, which serving never commits and no
host source can recover from the sum (bf16 addition is not invertible).  Here the add sits inside each output word's coordinate, so the
pre-bias word is interior to that coordinate's unit.  Bit-equal to `BiasAdd_v1(Gemm_v1(x, W), b)` by construction: the same two bodies.

Off cc 8.x the pinned vLLM installs no Triton override and `F.linear(x, W, b)` is cuBLASLt's GEMM with the bias in its FP32 epilogue:
the ascending chain of the target's k16 step (`DotBf16_v2{K, DOT}`), the bias widened and added in FP32, one rounding to bf16 -- a
different word from `GemmBias_v1`'s whenever the two roundings disagree.  `GemmBiasF32Epilogue_v1` is that coordinate.  At M = 1 cuBLAS
runs a gemv kernel instead, which is not this chain.
"""
from __future__ import annotations

from verity.ir.defs import bind, composite
from verity.ir.types import Array
from verity.ml import gemm as core_gemm
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16


@composite("GemmBiasCoordinate", 1, ["K"], lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16)), ("b", BF16)), BF16),
           conformance="exact-model-tested: GemmCoordinate_v1 then Bf16Add, the bodies of Gemm_v1 and BiasAdd_v1")
def GemmBiasCoordinate(B, S, x, wrow, b):
    """One output word: `GemmCoordinate_v1(x, W[j, :])`, then `Bf16Add` of that word and `b[j]`."""
    return B.call(P.Bf16Add, B.call(bind(b1.GemmCoordinate, K=S.K), x, wrow), b)


@composite("GemmBias", 1, ["K", "N"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16))), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested: bit-equal to BiasAdd_v1{N}(Gemm_v1{K,N}(x, w), b) by construction")
def GemmBias(B, S, x, w, b):
    """`x[K] @ W[N,K]^T + b[N]`: one `GemmBiasCoordinate_v1` per output word."""
    return B.batch(bind(GemmBiasCoordinate, K=S.K), x, w, b, axes=(None, 0, 0))


@composite("GemmBiasF32EpilogueCoordinate", 1, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16)), ("b", BF16)), BF16),
           conformance="exact-model-tested: DotBf16_v2{K,DOT}, then F32Add of the widened bias, then F2fpBf16")
def GemmBiasF32EpilogueCoordinate(B, S, x, wrow, b):
    """One output word: the FP32 accumulator of `DotBf16_v2{K, DOT}(x, W[j, :])`, plus `b[j]` widened to FP32 (one rounding), then
    `cvt.rn.bf16.f32`."""
    acc = B.call(bind(core_gemm.DotBf16, K=S.K, DOT=S.DOT), x, wrow)
    return B.call(P.F2fpBf16, B.call(P.F32Add, acc, B.call(P.Bf16ToF32, b)))


@composite("GemmBiasF32Epilogue", 1, ["K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16))), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested: DotBf16_v2{K,DOT} per coordinate, the bias added in the FP32 epilogue")
def GemmBiasF32Epilogue(B, S, x, w, b):
    """`x[K] @ W[N,K]^T + b[N]` with the bias in the FP32 epilogue: one `GemmBiasF32EpilogueCoordinate_v1` per output word."""
    return B.batch(bind(GemmBiasF32EpilogueCoordinate, K=S.K, DOT=S.DOT), x, w, b, axes=(None, 0, 0))
