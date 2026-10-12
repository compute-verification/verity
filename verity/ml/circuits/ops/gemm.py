"""The GEMM coordinate as an IR composite, parameterised by the k16 tensor-core step.

    DotBf16_v2{K, DOT}          f32 accumulator after K/16 `DOT` steps in ascending k from a zero accumulator
    GemmCoordinate_v2{K, DOT}   bf16 word: cvt.rn.bf16.f32(DotBf16)   (Triton `acc.to(bf16)`; the same on sm_80/sm_90)
    Gemm_v2{K, N, DOT}          x[K] @ W[N,K]^T -> [N], no epilogue

`DOT` is a static naming the step: `GEMM_AMPERE` binds `AmpereBF16TcDot16_v2`, `GEMM_HOPPER` binds
`HopperBF16WgmmaDot16_v1`.  A proof unit for the FLOP proof is one `GemmCoordinate_v2{K, DOT}` call: K/16
transitions of `DOT` followed by one `F2fpBf16`.

The same coordinate over the sm_120 low-precision steps, each K/k `DOT` steps in ascending k from a zero accumulator and
one `F2fpBf16`:

    GemmCoordinateE4m3_v1{K, DOT}   (x, wrow: e4m3[K]) -> bf16; k = 32   GEMM_SM120_E4M3: BlackwellE4m3QmmaDot32_v1
    GemmCoordinateNvf4_v1{K, DOT}   (x, wrow: e2m1[K], sx, sw: ue4m3[K/16]) -> bf16; k = 64   GEMM_SM120_NVF4: BlackwellNvf4OmmaDot64_v1
    GemmCoordinateMxf4_v1{K, DOT}   (x, wrow: e2m1[K], sx, sw: ue8m0[K/32]) -> bf16; k = 64   GEMM_SM120_MXF4: BlackwellMxf4OmmaDot64_v1

Provenance: veritor `verity.ir/registry/b1.py` `DotBf16V2` / `GemmCoordinateV2` / `GemmV2` and the binding records
`GEMM_AMPERE` (b1.py) / `GEMM_HOPPER` (hopper.py).  The `_v1` forms, which named the Ampere step in their bodies, were not
carried over: the static form is the definition the proof work targets, and their bodies would now name a different
primitive id (see `prims.AmpereBF16TcDot16`).
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.types import Array, Value

from . import prims as P
from .prims import BF16, E2M1, E4M3, F32, ZERO32

__all__ = ["DotBf16", "GEMM_AMPERE", "GEMM_HOPPER", "GEMM_SM120_E4M3", "GEMM_SM120_MXF4", "GEMM_SM120_NVF4", "Gemm",
           "GemmCoordinate", "GemmCoordinateE4m3", "GemmCoordinateMxf4", "GemmCoordinateNvf4"]

_SCALE = Value(8)


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


@composite("GemmCoordinateE4m3", 1, ["K", "DOT"], lambda S: ((("x", Array(S.K, E4M3)), ("wrow", Array(S.K, E4M3))), BF16),
           conformance="exact-model-tested")
def GemmCoordinateE4m3(B, S, x, wrow):
    """One output word of an E4M3 `x @ W^T`: K/32 `DOT` steps in ascending k from a zero accumulator, then `cvt.rn.bf16.f32`."""
    acc = B.call(ZERO32)
    for s in range(S.K // 32):
        acc = B.call(S.DOT, acc, x[32 * s:32 * s + 32], wrow[32 * s:32 * s + 32])
    return B.call(P.F2fpBf16, acc)


def _block_scaled_coordinate(B, S, x, wrow, sx, sw, ns: int):
    acc = B.call(ZERO32)
    for s in range(S.K // 64):
        acc = B.call(S.DOT, acc, x[64 * s:64 * s + 64], wrow[64 * s:64 * s + 64], sx[ns * s:ns * s + ns], sw[ns * s:ns * s + ns])
    return B.call(P.F2fpBf16, acc)


@composite("GemmCoordinateNvf4", 1, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, E2M1)), ("wrow", Array(S.K, E2M1)), ("sx", Array(S.K // 16, _SCALE)),
                       ("sw", Array(S.K // 16, _SCALE))), BF16),
           conformance="exact-model-tested")
def GemmCoordinateNvf4(B, S, x, wrow, sx, sw):
    """One output word of an NVFP4 `x @ W^T` (one UE4M3 scale byte per 16 codes): K/64 `DOT` steps in ascending k from a zero
    accumulator, then `cvt.rn.bf16.f32`."""
    return _block_scaled_coordinate(B, S, x, wrow, sx, sw, 4)


@composite("GemmCoordinateMxf4", 1, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, E2M1)), ("wrow", Array(S.K, E2M1)), ("sx", Array(S.K // 32, _SCALE)),
                       ("sw", Array(S.K // 32, _SCALE))), BF16),
           conformance="exact-model-tested")
def GemmCoordinateMxf4(B, S, x, wrow, sx, sw):
    """One output word of an MXFP4 `x @ W^T` (one UE8M0 scale byte per 32 codes): K/64 `DOT` steps in ascending k from a zero
    accumulator, then `cvt.rn.bf16.f32`."""
    return _block_scaled_coordinate(B, S, x, wrow, sx, sw, 2)


#: Binding records for the measured architectures.
GEMM_AMPERE: dict = {"DOT": P.AmpereBF16TcDot16}
GEMM_HOPPER: dict = {"DOT": P.HopperBF16WgmmaDot16}
GEMM_SM120_E4M3: dict = {"DOT": P.BlackwellE4m3QmmaDot32}
GEMM_SM120_NVF4: dict = {"DOT": P.BlackwellNvf4OmmaDot64}
GEMM_SM120_MXF4: dict = {"DOT": P.BlackwellMxf4OmmaDot64}


def _register_kernels() -> None:
    from verity.ml.circuits.evaluation import register_kernel

    from verity.ml.kernels import tc as K
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32, BLACKWELL_SM120_MXF4, BLACKWELL_SM120_NVF4
    from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16
    from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16
    from verity.ml.circuits.silicon.total_fp8 import HOPPER_E4M3_WGMMA_K32

    steps = {P.AmpereBF16TcDot16: AMPERE_BF16_M16N8K16, P.HopperBF16WgmmaDot16: HOPPER_BF16_WGMMA_K16}
    domain = "every encoding; DOT one of the core BF16 steps (other steps fall back to the reference)"
    register_kernel(DotBf16, K.dot_kernel(steps), domain=domain, sample=K.sample_tc)
    register_kernel(GemmCoordinate, K.gemm_coordinate_kernel(steps), domain=domain, sample=K.sample_tc)
    register_kernel(Gemm, K.gemm_kernel(steps), domain=domain)
    e4m3 = {P.BlackwellE4m3QmmaDot32: BLACKWELL_SM120_E4M3_M16N8K32, P.HopperE4m3QgmmaDot32: HOPPER_E4M3_WGMMA_K32}
    register_kernel(GemmCoordinateE4m3, K.gemm_coordinate_kernel(e4m3), sample=K.sample_tc,
                    domain="every encoding; DOT one of the core E4M3 steps (other steps fall back to the reference)")
    for coord, dot, model, fire in ((GemmCoordinateNvf4, P.BlackwellNvf4OmmaDot64, BLACKWELL_SM120_NVF4, K.nvf4_step_total_batch),
                                    (GemmCoordinateMxf4, P.BlackwellMxf4OmmaDot64, BLACKWELL_SM120_MXF4, K.mxf4_step_total_batch)):
        register_kernel(coord, K.block_scaled_coordinate_kernel({dot: (model, fire)}), sample=K._sample_block_scaled(model),
                        domain=f"every encoding; DOT {dot.id} (other steps fall back to the reference)")


_register_kernels()
