"""The GEMM coordinate as an IR composite, parameterised by the k16 tensor-core step.

    DotBf16_v2{K, DOT}          f32 accumulator after K/16 `DOT` steps in ascending k from a zero accumulator
    GemmCoordinate_v2{K, DOT}   bf16 word: cvt.rn.bf16.f32(DotBf16)   (Triton `acc.to(bf16)`; the same on sm_80/sm_90)
    Gemm_v2{K, N, DOT}          x[K] @ W[N,K]^T -> [N], no epilogue

`DOT` is a static naming the step: `GEMM_AMPERE` binds `AmpereBF16TcDot16_v2`, `GEMM_HOPPER` binds
`HopperBF16WgmmaDot16_v1`.  A verification unit for the FLOP proof is one `GemmCoordinate_v2{K, DOT}` call: K/16
transitions of `DOT` followed by one `F2fpBf16`.

Provenance: veritor `verity.ir/registry/b1.py` `DotBf16V2` / `GemmCoordinateV2` / `GemmV2` and the binding records
`GEMM_AMPERE` (b1.py) / `GEMM_HOPPER` (hopper.py).  The `_v1` forms, which named the Ampere step in their bodies, were not
carried over: the static form is the definition the proof work targets, and their bodies would now name a different
primitive id (see `prims.AmpereBF16TcDot16`).
"""
from __future__ import annotations

from verity.ir.defs import bind, composite
from verity.ir.types import Array

from . import prims as P
from .prims import BF16, F32, ZERO32

__all__ = ["DotBf16", "GEMM_AMPERE", "GEMM_HOPPER", "Gemm", "GemmCoordinate"]


@composite("DotBf16", 2, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.K, BF16))), F32),
           conformance="exact-model-tested")
def DotBf16(B, S, x, w):
    """f32 accumulator after K/16 `DOT` steps in ascending k from a zero accumulator (the Triton
    `matmul_kernel_persistent` k-loop on either architecture)."""
    acc = B.call(ZERO32)
    for s in range(S.K // 16):
        acc = B.call(S.DOT, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
    return acc


@composite("GemmCoordinate", 2, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16))), BF16),
           conformance="exact-model-tested")
def GemmCoordinate(B, S, x, wrow):
    """One output word of `x @ W^T`: bf16_rne(DotBf16{K, DOT}(x, W[j, :])) (`cvt.rn.bf16.f32`)."""
    return B.call(P.F2fpBf16, B.call(bind(DotBf16, K=S.K, DOT=S.DOT), x, wrow))


@composite("Gemm", 2, ["K", "N", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def Gemm(B, S, x, w):
    """`x[K] @ W[N,K]^T -> [N]`, no epilogue (a bias is a separate node)."""
    return B.batch(bind(GemmCoordinate, K=S.K, DOT=S.DOT), x, w, axes=(None, 0))


#: Binding records for the two measured architectures.
GEMM_AMPERE: dict = {"DOT": P.AmpereBF16TcDot16}
GEMM_HOPPER: dict = {"DOT": P.HopperBF16WgmmaDot16}


def _register_kernels() -> None:
    from verity.evaluation import register_kernel

    from . import kernels as K
    from .tc.silicon import AMPERE_BF16_M16N8K16
    from .tc.total import HOPPER_BF16_WGMMA_K16

    steps = {P.AmpereBF16TcDot16: AMPERE_BF16_M16N8K16, P.HopperBF16WgmmaDot16: HOPPER_BF16_WGMMA_K16}
    domain = "every encoding; DOT one of the core BF16 steps (other steps fall back to the reference)"
    register_kernel(DotBf16, K.dot_kernel(steps), domain=domain, sample=K.sample_tc)
    register_kernel(GemmCoordinate, K.gemm_coordinate_kernel(steps), domain=domain, sample=K.sample_tc)
    register_kernel(Gemm, K.gemm_kernel(steps), domain=domain)


_register_kernels()
