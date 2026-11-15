"""`cuda_rmsnorm`'s two CUDA RMSNorm kernels cut where they broadcast their scale, so that no unit returns a whole row.

A sum-of-squares unit reads the row and returns one F32 word, the kernel's ``rsqrtf(var / N + EPS)``. An apply unit
reads G elements of the row, their G weights and that scale, and returns those G elements of each output. The sum of
squares is the kernel's own reduction (`cuda_rmsnorm._pure_scale`, and for the fused kernel `_fused_scale`, the served
v3's batches, v1's words) and an apply unit is the kernel's own
epilogue (`cuda_rmsnorm._out`) on its elements, so N / G apply units return the kernel's outputs word for word. The fused
apply redoes the residual add on its G elements, so the cut adds N residual adds to the fused kernel's gates and none to
the pure kernel's (`verity/definitions/tests/test_cuda_rmsnorm_split.py`). Their Boolean versions are
`boolean_norms_split`'s units of the same names, `_v2`.
"""
from __future__ import annotations

from verity.circuits.defs import composite
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import boolean_norms as BN
from verity.definitions.registry import cuda_rmsnorm as C
from verity.definitions.registry import prims as P
from verity.definitions.registry.prims import BF16, F32

_EXACT = ("exact-model-tested: `RMSNormCudaVector8_v1` and `RMSNormFusedCudaVector8_v1{N, EPS, BLOCK}` cut at their scale; "
          "the sum of squares and N / G apply units return their outputs word for word "
          "(`verity/definitions/tests/test_cuda_rmsnorm_split.py`)")


@composite("RMSNormSumSqCudaVector8", 1, ["N", "EPS", "BLOCK"],
           lambda S: ((("x", Array(S.N, BF16)),), F32), conformance=_EXACT)
def RMSNormSumSqCudaVector8(B, S, x):
    """`RMSNormCudaVector8_v1{N, EPS, BLOCK}` up to the scale its threads broadcast: ``rsqrtf(var / N + EPS)``."""
    return C._pure_scale(B, S, x)


@composite("RMSNormApplyCudaVector8", 1, ["G"],
           lambda S: ((("x", Array(S.G, BF16)), ("w", Array(S.G, BF16)), ("scale", F32)), Array(S.G, BF16)),
           conformance=_EXACT)
def RMSNormApplyCudaVector8(B, S, x, w, scale):
    """`RMSNormCudaVector8_v1`'s epilogue on G elements: ``bf16_rn((f32(x) * scale) * f32(w))``."""
    return C._out(B, x, w, scale)


@composite("RMSNormFusedSumSqCudaVector8", 1, ["N", "EPS", "BLOCK"],
           lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16))), F32), conformance=_EXACT)
def RMSNormFusedSumSqCudaVector8(B, S, x, res):
    """`RMSNormFusedCudaVector8_v1{N, EPS, BLOCK}` up to the scale its threads broadcast: the residual add, then
    ``rsqrtf(var / N + EPS)`` over the sums."""
    C._check(S, fused=True)
    return C._fused_scale(B, S, B.batch(P.Bf16AddF2fp, x, res, axes=(0, 0)))


@composite("RMSNormFusedApplyCudaVector8", 1, ["G"],
           lambda S: ((("x", Array(S.G, BF16)), ("res", Array(S.G, BF16)), ("w", Array(S.G, BF16)), ("scale", F32)),
                      Tuple(Array(S.G, BF16), Array(S.G, BF16))),
           conformance=_EXACT)
def RMSNormFusedApplyCudaVector8(B, S, x, res, w, scale):
    """`RMSNormFusedCudaVector8_v1`'s residual add and epilogue on G elements: (the normed elements, ``res'``)."""
    r = B.batch(P.Bf16AddF2fp, x, res, axes=(0, 0))
    return tuple_of(C._out(B, r, w, scale), r)


BN._set_cuda_vector8_words(sumsq=RMSNormSumSqCudaVector8, fused_sumsq=RMSNormFusedSumSqCudaVector8,
                           apply=RMSNormApplyCudaVector8, fused_apply=RMSNormFusedApplyCudaVector8)

__all__ = ["RMSNormApplyCudaVector8", "RMSNormFusedApplyCudaVector8", "RMSNormFusedSumSqCudaVector8",
           "RMSNormSumSqCudaVector8"]
