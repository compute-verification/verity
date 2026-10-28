"""NVFP4 (W4A4) linear of the pinned vLLM (d9105ea80) where it runs `CutlassNvFp4LinearKernel`: cc 12.0, both checkpoint formats
(`modelopt_fp4` and compressed-tensors `W4A4Fp4`), under `VLLM_BATCH_INVARIANT=1`.

Each linear launches two kernels (as captured on the device):

1. `vllm::cvt_fp16_to_fp4<bf16>` (`scaled_fp4_quant`, csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu and
   nvfp4_utils.cuh `cvt_warp_fp16_to_fp4`, UE4M3 path), per 16-element block of the bf16 row, with `g` the fp32 global scale
   (`input_global_scale_inv`):
   `vecMax = max |f32(x)|` (`__habs2` / `__hmax2`: NaN-ignoring), `SF = cvt.rn.satfinite.e4m3(g * (vecMax * rcp.approx.ftz(6)))`,
   `outputScale = SFValue != 0 ? rcp.approx.ftz(SFValue * rcp.approx.ftz(g)) : 0` with `SFValue = f32(SF)`, and every element
   `cvt.rn.satfinite.e2m1(f32(x) * outputScale)`.  The FMULs are IEEE (the file is not in the `--use_fast_math` set).
2. The CUTLASS sm120 block-scaled GEMM (nvfp4_scaled_mm_sm120_kernels.cu; batch invariance selects `sm120_fp4_config_default`, a
   256x128x128 persistent tile for every M): per output word the ascending chain of K/64 `BlackwellNvf4OmmaDot64` steps (two per
   128-wide k-tile) from +0, then the default linear-combination epilogue with `alpha_ptr` and no source: `bf16(alpha * acc)`.

`g` and `alpha` are derived Inputs formed at load (`alpha = input_global_scale * weight_global_scale`): compressed-tensors takes
`g` as the checkpoint's `input_global_scale` and `alpha = f32(1 / g) * f32(1 / weight_global_scale)`; modelopt takes
`input_scale` and `weight_scale_2` as they are, `g = f32(1 / input_scale)`.  A linear's bias is a separate bf16 add after the
GEMM (`out + bias`).

The device layouts the Commit opens (`sf_offset`, `unpack_codes`) are `kernels.nvfp4_twins`.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll as _Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.prims import E4m3ToF32, F32ToE2m1Sat, F32ToE4m3Sat
from verity.ml.circuits.ops.scalar import F32Eq, F32Fabs, F32Fmaxf, SelectF32
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16, F32

E2M1 = Value(4)
UE4M3 = Value(8)
BLOCK = 16                 # elements per UE4M3 scale
STEP_K = 64                # elements per mma.sync kind::mxf4nvf4 step
C6 = b1.f32c(6.0)

_QUANT_CONF = "exact-model-tested; see tests/program/test_nvfp4.py and the PRO 6000 capture its docstring cites"
_GEMM_CONF = "exact-model-tested; see tests/program/test_nvfp4.py and the PRO 6000 capture its docstring cites"


@composite("Nvfp4QuantBlock", 1, [], lambda S: ((("x", Array(BLOCK, BF16)), ("g", F32)), Tuple(Array(BLOCK, E2M1), UE4M3)),
           conformance=_QUANT_CONF)
def Nvfp4QuantBlock(B, S, x, g):
    """One 16-element block of `cvt_fp16_to_fp4` (UE4M3): (the 16 E2M1 codes, the block's UE4M3 scale byte)."""
    m = B.call(F32Fabs, B.call(P.Bf16ToF32, x[0]))
    for i in range(1, BLOCK):
        m = B.call(F32Fmaxf, m, B.call(F32Fabs, B.call(P.Bf16ToF32, x[i])))
    sf = B.call(F32ToE4m3Sat, B.call(P.F32Mul, g, B.call(P.F32Mul, m, B.call(P.MufuRcpFtz, B.call(C6)))))
    sfv = B.call(E4m3ToF32, sf)
    inv = B.call(P.MufuRcpFtz, B.call(P.F32Mul, sfv, B.call(P.MufuRcpFtz, g)))
    scale = B.call(SelectF32, B.call(F32Eq, sfv, B.call(b1.ZERO32)), B.call(b1.ZERO32), inv)
    q = [B.call(F32ToE2m1Sat, B.call(P.F32Mul, B.call(P.Bf16ToF32, x[i]), scale)) for i in range(BLOCK)]
    return tuple_of(_Coll(Array(BLOCK, E2M1), concat([w.refs for w in q])), sf)


@composite("Nvfp4ActQuant", 1, ["K"], lambda S: ((("x", Array(S.K, BF16)), ("g", F32)), Tuple(Array(S.K, E2M1), Array(S.K // BLOCK, UE4M3))),
           conformance=_QUANT_CONF)
def Nvfp4ActQuant(B, S, x, g):
    """`scaled_fp4_quant` on one bf16 row: (x_q[K] E2M1, x_s[K/16] UE4M3), the two values the quant kernel materialises."""
    assert S.K % BLOCK == 0, S.K
    qs, ss = [], []
    for b in range(S.K // BLOCK):
        r = B.call(Nvfp4QuantBlock, x[BLOCK * b:BLOCK * (b + 1)], g)
        qs.append(r[0])
        ss.append(r[1])
    return tuple_of(_Coll(Array(S.K, E2M1), concat([w.refs for w in qs])), _Coll(Array(S.K // BLOCK, UE4M3), concat([w.refs for w in ss])))


@composite("DotNvfp4", 1, ["K", "DOT"],
           lambda S: ((("xq", Array(S.K, E2M1)), ("xs", Array(S.K // BLOCK, UE4M3)), ("wq", Array(S.K, E2M1)),
                       ("ws", Array(S.K // BLOCK, UE4M3))), F32),
           conformance=_GEMM_CONF)
def DotNvfp4(B, S, xq, xs, wq, ws):
    """f32 accumulator after K/64 `DOT` steps (a block-scaled k64 step primitive) in ascending k from +0; step s takes codes
    [64s, 64s + 64) and scales [4s, 4s + 4) of both operands, activation first."""
    assert S.K % STEP_K == 0, S.K
    acc = B.call(b1.ZERO32)
    per = STEP_K // BLOCK
    for s in range(S.K // STEP_K):
        acc = B.call(S.DOT, acc, xq[STEP_K * s:STEP_K * (s + 1)], wq[STEP_K * s:STEP_K * (s + 1)], xs[per * s:per * (s + 1)],
                     ws[per * s:per * (s + 1)])
    return acc


@composite("ScaledMmNvfp4Coordinate", 1, ["K", "DOT"],
           lambda S: ((("xq", Array(S.K, E2M1)), ("xs", Array(S.K // BLOCK, UE4M3)), ("wrow", Array(S.K, E2M1)),
                       ("wsrow", Array(S.K // BLOCK, UE4M3)), ("alpha", F32)), BF16),
           conformance=_GEMM_CONF)
def ScaledMmNvfp4Coordinate(B, S, xq, xs, wrow, wsrow, alpha):
    """One output word of `cutlass_scaled_fp4_mm`: `bf16(alpha * DotNvfp4{K,DOT})` (`cvt.rn.bf16.f32`)."""
    acc = B.call(bind(DotNvfp4, K=S.K, DOT=S.DOT), xq, xs, wrow, wsrow)
    return B.call(P.F2fpBf16, B.call(P.F32Mul, alpha, acc))


@composite("ScaledMmNvfp4", 1, ["K", "N", "DOT"],
           lambda S: ((("xq", Array(S.K, E2M1)), ("xs", Array(S.K // BLOCK, UE4M3)), ("w", Array(S.N, Array(S.K, E2M1))),
                       ("ws", Array(S.N, Array(S.K // BLOCK, UE4M3))), ("alpha", F32)), Array(S.N, BF16)),
           conformance=_GEMM_CONF)
def ScaledMmNvfp4(B, S, xq, xs, w, ws, alpha):
    """`cutlass_scaled_fp4_mm` on one token row: the N output coordinates (`ScaledMmNvfp4Coordinate`)."""
    coord = bind(ScaledMmNvfp4Coordinate, K=S.K, DOT=S.DOT)
    return B.batch(coord, xq, xs, w, ws, alpha, axes=(None, None, 0, 0, None))


@composite("QuantLinearNvfp4", 1, ["K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("g", F32), ("w", Array(S.N, Array(S.K, E2M1))), ("ws", Array(S.N, Array(S.K // BLOCK, UE4M3))),
                       ("alpha", F32)), Tuple(Array(S.K, E2M1), Array(S.K // BLOCK, UE4M3), Array(S.N, BF16))),
           conformance=_GEMM_CONF)
def QuantLinearNvfp4(B, S, x, g, w, ws, alpha):
    """The NVFP4 linear on one token row: `Nvfp4ActQuant` then `ScaledMmNvfp4`; returns (x_q, x_s, y), what production
    materialises."""
    q = B.call(bind(Nvfp4ActQuant, K=S.K), x, g)
    y = B.call(bind(ScaledMmNvfp4, K=S.K, N=S.N, DOT=S.DOT), q[0], q[1], w, ws, alpha)
    return tuple_of(q[0], q[1], y)


__all__ = ["BLOCK", "DotNvfp4", "E2M1", "Nvfp4ActQuant", "Nvfp4QuantBlock", "QuantLinearNvfp4", "STEP_K", "ScaledMmNvfp4",
           "ScaledMmNvfp4Coordinate", "UE4M3"]
