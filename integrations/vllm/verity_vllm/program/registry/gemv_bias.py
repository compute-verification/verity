"""cuBLAS's single-row linear with a bias as ONE Definition: `GemvBiasF32_v1{K,N,V,T}(x, W, b)` (cuBLASLt targets, M = 1).

Off cc 8.x the pinned vLLM keeps `F.linear(x, W, b)` on cuBLAS, and at M = 1 with a bias cuBLAS runs its `gemvx` kernel, not the
tensor-core GEMM.  Per output word, T threads each own the V-wide chunks c of k with c % T == t and sum them in ascending k, one FFMA
of the widened operands per k into an FP32 accumulator from +0; the T partials then reduce by a halving tree (offsets T/2, ..., 1: the
warp butterfly), and the bias joins in FP32 before the rounding to bf16.  V and T are cuBLAS's choice per (K, N) on one GPU and
cuBLAS version: the target's table (`gemm_targets.GEMV_BIAS_TABLES`).
"""
from __future__ import annotations

from verity.ir.defs import bind, composite
from verity.ir.types import Array
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16


@composite("GemvBiasF32Coordinate", 1, ["K", "V", "T"],
           lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16)), ("b", BF16)), BF16),
           conformance="exact-model-tested: T strided FFMA chains from +0, a halving tree of F32Add, F32Add of the widened bias, F2fpBf16")
def GemvBiasF32Coordinate(B, S, x, wrow, b):
    """One output word of cuBLAS gemvx with a bias (module docstring)."""
    K, V, T = S.K, S.V, S.T
    if K % V or T < 1 or T & (T - 1) or K // V < T:
        raise ValueError(f"GemvBiasF32: K={K} must split into V={V}-wide chunks over T={T} (a power of two) threads")
    parts = []
    for t in range(T):
        acc = B.call(b1.ZERO32)
        for c in range(t, K // V, T):
            for k in range(c * V, c * V + V):
                acc = B.call(P.F32Fma, B.call(P.Bf16ToF32, x[k]), B.call(P.Bf16ToF32, wrow[k]), acc)
        parts.append(acc)
    off = T // 2
    while off:
        parts = [B.call(P.F32Add, parts[t], parts[t + off]) for t in range(off)]
        off //= 2
    return B.call(P.F2fpBf16, B.call(P.F32Add, parts[0], B.call(P.Bf16ToF32, b)))


@composite("GemvBiasF32", 1, ["K", "N", "V", "T"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16))), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested: one GemvBiasF32Coordinate_v1 per output word")
def GemvBiasF32(B, S, x, w, b):
    """`x[K] @ W[N,K]^T + b[N]` as cuBLAS gemvx computes it: one `GemvBiasF32Coordinate_v1` per output word."""
    return B.batch(bind(GemvBiasF32Coordinate, K=S.K, V=S.V, T=S.T), x, w, b, axes=(None, 0, 0))
