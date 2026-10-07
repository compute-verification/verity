"""vLLM's batch-invariant linear with a bias as ONE Definition: `GemmBias_v1{K,N}(x, W, b)`.

`linear_batch_invariant` runs the persistent matmul (`matmul_kernel_persistent`, `bias_ptr` unset: `Gemm_v1`'s coordinate, the f32
accumulator rounded once to bf16) and then `output + bias` in bf16 (`BiasAdd_v1`: one f32 add of the two bf16 words, RNE).  Written as
two Definitions, the pre-bias word is a Call boundary: a value the proof must read as committed, which serving never commits and no
host source can recover from the sum (bf16 addition is not invertible).  Here the add sits inside each output word's coordinate, so the
pre-bias word is interior to that coordinate's unit.  Bit-equal to `BiasAdd_v1(Gemm_v1(x, W), b)` by construction: the same two bodies.

`GemmBias_v2{K,N,DOT}` is the same pair on a target whose Triton `tl.dot` lowers to another k16 step: `Gemm_v2{K,N,DOT}`'s coordinate,
then the same `Bf16Add`, bit-equal to `BiasAdd_v1(Gemm_v2(x, W), b)`.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.gemm import GemmCoordinate as GemmCoordinateV2
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


@composite("GemmBiasCoordinate", 2, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16)), ("b", BF16)), BF16),
           conformance="exact-model-tested: GemmCoordinate_v2{K,DOT} then Bf16Add, the bodies of Gemm_v2 and BiasAdd_v1")
def GemmBiasCoordinateV2(B, S, x, wrow, b):
    """One output word: `GemmCoordinate_v2{K,DOT}(x, W[j, :])`, then `Bf16Add` of that word and `b[j]`."""
    return B.call(P.Bf16Add, B.call(bind(GemmCoordinateV2, K=S.K, DOT=S.DOT), x, wrow), b)


@composite("GemmBias", 2, ["K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16))), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested: bit-equal to BiasAdd_v1{N}(Gemm_v2{K,N,DOT}(x, w), b) by construction")
def GemmBiasV2(B, S, x, w, b):
    """`x[K] @ W[N,K]^T + b[N]` with the k16 step `DOT`: one `GemmBiasCoordinate_v2` per output word."""
    return B.batch(bind(GemmBiasCoordinateV2, K=S.K, DOT=S.DOT), x, w, b, axes=(None, 0, 0))
