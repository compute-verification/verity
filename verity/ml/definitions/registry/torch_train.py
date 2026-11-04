"""torch 2.14's training-step kernels on CUDA as Boolean Definitions: the transpose a transposed GEMM reads, and HF
`LlamaRMSNorm` and SiLU forward and backward, each as PyTorch eager computes it in bf16 with f32 opmath.

    Transpose_v1{R, C}                      (x: Array<R, Array<C, U16>>) -> Array<C, Array<R, U16>>
    RmsNormTorch_v1{H, LANES, EPS, RSQRT}   (x, w: Array<H, BF16>) -> Tuple<y: Array<H, BF16>, hb: Array<H, BF16>, r: F32>
      RmsNormTorchIn_v1{H, LANES, EPS}      (x: Array<H, BF16>) -> F32
      RmsNormTorchOut_v1{H}                 (x, w: Array<H, BF16>, r: F32) -> Tuple<Array<H, BF16>, Array<H, BF16>>
    RmsNormTorchBackward_v1{H, LANES}       (dy, x, w: Array<H, BF16>, r: F32) -> Array<H, BF16>
    RmsNormTorchWeightGrad_v1{M, H}         (dy, hb: Array<M, BF16>) -> BF16
    SiluTorch_v1{N, EX2}                    (x: Array<N, BF16>) -> Array<N, BF16>
    SiluTorchBackward_v1{N, EX2, CONTRACT}  (dy, x: Array<N, BF16>) -> Array<N, BF16>

(U16 and BF16 are ``Array<16, Value<1>>``, F32 ``Array<32, Value<1>>``.)

`Transpose_v1` is ``out[c][r] = x[r][c]``, wiring with no gate. Values are committed token-major, so a GEMM that reads a column
(dX = dY·W reads Wᵀ, dW = dYᵀ·X reads Xᵀ and dYᵀ, attention's backward Pᵀ, dSᵀ, Kᵀ, Qᵀ and dOᵀ) reads it from a transposed copy,
and the transposed GEMMs are then today's GEMM Definitions on its rows. At R = C = 128 a tile is 128 rows of 128 bf16 words in
and out, each row one 256 B commitment row (262,144 bits each way); the 1 KiB layout's 512×512 tile is past K_MAX.

`RmsNormTorch_v1` is one token of ``w * (x.float() * rsqrt(x.float().pow(2).mean(-1) + eps)).to(bf16)``, each step one ATen
kernel, every f32 NaN 0x7FFFFFFF and every bf16 NaN 0x7FFF:

    s  = Σ RN(f32(x) * f32(x))                   ``pow(2)`` is ``base * base``; the sum is `Reduce.cuh`'s, as ``gr`` below
    v  = RN(RN(s * RN(1 / H)) + f32(EPS))        `MeanOps`' ``acc * factor``, factor ``(float)num_out / numel`` = RN(1 / H)
                                                 while M·H < 2^24; then the add of the f32 eps word EPS
    r  = RSQRT(v)                                ``::rsqrt``, `rsqrt.approx.f32`
    hb = bf16(RN(f32(x) * r))                    the multiply, then `_to_copy` (cvt.rn.bf16.f32)
    y  = bf16(RN(f32(w) * f32(hb)))              the weight's bf16 multiply

RSQRT is a static, the Definition of ``rsqrt.approx.f32``: `mufu.RsqrtApprox_v2`, MUFU.RSQ by the library's ``rsq`` table
with the subnormal range scaling. `RmsNormTorchIn_v1` is the kernels up to v and `RmsNormTorchOut_v1` those after r, so
RSQRT is one call between them; r is an output, which the backward reads.

`RmsNormTorchBackward_v1` is one token's input gradient through the autograd chain of
``w * (x.float() * rsqrt(x.float().pow(2).mean(-1) + eps)).to(bf16)``, r the saved ``rsqrt`` output, each line one ATen kernel
(derivatives.yaml's formulas, `FunctionsManual.cpp`), every NaN result 0x7FFF:

    g  = f32(bf16(f32(dy) * f32(w)))             mul's grad_other (``grad * self``), then `_to_copy` back to f32
    ga = RN(g * r)                               mul's grad_self
    gr = Σ RN(g * f32(x))                        mul's grad_other summed to r's shape: `Reduce.cuh` over the row (below)
    gv = RN(RN(gr * -0.5) * RN(RN(r * r) * r))   rsqrt's ``-0.5 * grad * result.pow(3)``; pow 3 is ``base * base * base``
    gp = RN(gv * f32(1.0 / H))                   mean's ``grad.expand(...) / n``: a CPU-scalar divide is a multiply by
                                                 ``(float)(1.0 / n)``, the reciprocal in double (`div_true_kernel_cuda`)
    gb = RN(gp * RN(f32(x) * 2))                 pow 2's ``grad * (2 * self.pow(1))``; pow 1 is a copy
    dx = bf16(RN(ga + gb))                       the two grads of ``x.float()``, then `_to_copy` (cvt.rn.bf16.f32)

``gr`` is `Reduce.cuh`'s sum along the fastest dimension, vectorized along input: LANES threads of a block row read float4s,
lane l the vectors l, l + LANES, ... below H / 4, into four running sums from 0 (one per component), combined
``((s0 + s1) + s2) + s3``, then the warp's ``shfl_down`` tree at offsets LANES / 2 .. 1 (decreasing, 2.14's order).
`reduce_lanes` is LANES for a launch: 32 for H = 576 at 16 or more tokens. A launch whose config splits the row across warps or
blocks, or doesn't vectorize, is refused.

`RmsNormTorchWeightGrad_v1` is one column of the weight's gradient, MulBackward0's ``grad * hb`` (bf16, f32 opmath) summed over
the M tokens to the weight's shape (`sum_to_size`): `Reduce.cuh` over the slow dimension, vectorized along output (four columns
a thread, a 32 x 4 block), each column summed by the block's four warps and, from 1,024 tokens, by `column_ctas` CTAs, whose
partials the last CTA adds (`column_sum`). That count is ceil(M / 64) (64 at M = 4,096) while it is below the device's grid
over CTAs a column; past it, sm_120's 188 multiprocessors set it, and the Definition refuses. On sm_120 the profiler shows
grids (5, 1) at 256 tokens and (5, 64) at 4,096 for H = 576.

`SiluTorch_v1` is `silu_kernel`'s ``x / (1 + ::exp(-x))`` in f32, then bf16; `SiluTorchBackward_v1` is `silu_backward_kernel`'s
``s = 1 / (1 + exp(-x))``, ``dy * s * (1 + x * (1 - s))`` (the kernel autograd runs when the backward doesn't build a graph).
``exp`` is CUDA's ``expf`` (libdevice ``__nv_expf``, `boolean_moe`'s `NvExpfIn_v1` and `NvExpfOut_v1` around its
``ex2.approx.ftz.f32``), and EX2 is a static: the Definition of MUFU.EX2, on sm_120 `mufu.MufuEx2Ftz_v2`, which equals that
unit on every input kernels' sm_120 table read. The divides are ``div.rn.f32``. CONTRACT is the one
contraction the source leaves to nvcc: 1 reads ``1 + x * (1 - s)`` as ``fma(x, RN(1 - s), 1)``, 0 as ``RN(1 + RN(x * RN(1 -
s)))``; the sm_120 SASS of ``silu_backward_kernel`` in torch 2.14's libtorch_cuda.so is the FFMA, CONTRACT = 1.
Every NaN result is 0x7FFF.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, FunctionDefinition, SpecializedDefinition, bind, composite
from verity.ml.circuits.ops.boolean.elementwise import specialized
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry.boolean_moe import nv_expf_in, nv_expf_out

__all__ = ["RmsNormTorch", "RmsNormTorchBackward", "RmsNormTorchIn", "RmsNormTorchOut", "RmsNormTorchWeightGrad", "SiluTorch",
           "SiluTorchBackward", "SiluTorchBackwardOut", "SiluTorchIn", "SiluTorchOut", "Transpose", "column_ctas", "column_sum",
           "mean_factor", "reduce_lanes", "rmsnorm_backward", "rmsnorm_out", "rmsnorm_variance", "row_sum", "silu_backward_out",
           "silu_out"]

U16 = Array(16, BL.BIT)
BF16 = U16
F32 = Array(32, BL.BIT)
L8 = Array(8, BL.BIT)
NAN = fp.F32_NAN_PTX
BF16_NAN = 0x7FFF
ONE = 0x3F800000
#: `mnt_wrapper<float>::MAX_NUM_THREADS` and the warp
MAX_THREADS, WARP = 512, 32
#: the RTX PRO 6000's multiprocessors and threads a multiprocessor (sm_120), which size `setReduceConfig`'s target grid
SM120_SMS, SM120_THREADS = 188, 1536


def _count(name: str, v: Any, least: int = 1) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < least:
        raise ValueError(f"{name} is a count, at least {least}, not {v!r}")
    return v


def _c(value: float) -> list:
    return fp.f32_const(int(np.float32(value).view(np.uint32)))


def _flag(S: Any, k: str) -> bool:
    v = getattr(S, k)
    if v not in (0, 1) or isinstance(v, bool):
        raise ValueError(f"{k} is 0 or 1, not {v!r}")
    return bool(v)


# --- Transpose -------------------------------------------------------------------------------------------------------------

def _transpose_signature(S: Any):
    R, C = _count("R", S.R), _count("C", S.C)
    return (("x", Array(R, Array(C, U16))),), Array(C, Array(R, U16))


Transpose = specialized(
    "Transpose", 1, ("R", "C"),
    lambda S: lambda C, *e: [e[r * S.C + c] for c in range(S.C) for r in range(S.R)], _transpose_signature,
    doc="a tile's transpose on bits, wiring with no gate: out[c][r] = x[r][c] for an R x C tile of 16-bit words")


# --- RMSNorm backward ------------------------------------------------------------------------------------------------------

def _last_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def reduce_lanes(H: int, rows: int) -> int:
    """LANES of `Reduce.cuh`'s f32 sum over the contiguous last dimension of ``rows`` rows of H, `setReduceConfig`'s
    block_width; ValueError for a launch this module doesn't build (unvectorized, or split across warps or blocks)."""
    H, rows = _count("H", H), _count("rows", rows)
    if H < 128 or H % 4:
        raise ValueError(f"H = {H}: Reduce.cuh vectorizes along input only from 128 inputs in whole float4s")
    dim0, cap = H // 4, MAX_THREADS
    d0 = _last_pow2(dim0) if dim0 < cap else cap
    d1 = _last_pow2(rows) if rows < cap else cap
    width = min(d0, WARP)
    height = min(d1, cap // width)
    width = min(d0, cap // height)
    if width > WARP:
        raise ValueError(f"H = {H} at {rows} rows: a block row of {width} threads reduces through shared memory first")
    if -(-H // width) >= min(height * 16, 256):
        raise ValueError(f"H = {H} at {rows} rows: {-(-H // width)} values a thread splits the row across warps")
    return width


def _lanes(S: Any) -> int:
    H, L = _count("H", S.H), _count("LANES", S.LANES)
    if H < 128 or H % 4:
        raise ValueError(f"H = {H}: Reduce.cuh vectorizes along input only from 128 inputs in whole float4s")
    if L > WARP or L & (L - 1):
        raise ValueError(f"LANES is a power of two up to {WARP}, not {L}")
    return L


def row_sum(C, p, lanes: int):
    """`Reduce.cuh`'s vectorized sum of the f32 words p (len a multiple of 4) at ``lanes`` threads, ``+0`` its identity."""
    nv = len(p) // 4
    zero = fp.f32_const(0)
    sums = []
    for lane in range(lanes):
        acc = [zero] * 4
        for v in range(lane, nv, lanes):
            acc = [fp.f32_add(C, a, p[4 * v + i], nan_word=NAN) for i, a in enumerate(acc)]
        s = acc[0]
        for a in acc[1:]:
            s = fp.f32_add(C, s, a, nan_word=NAN)
        sums.append(s)
    off = lanes // 2
    while off:
        sums = [fp.f32_add(C, sums[i], sums[i + off], nan_word=NAN) for i in range(off)]
        off //= 2
    return sums[0]


def rmsnorm_backward(C, dy, x, w, r, lanes: int):
    """One token's dx of `RmsNormTorchBackward_v1` from bf16 dy, x, w (lists of 16-bit words) and the f32 r."""
    H = len(x)
    x32 = [fp.bf16_to_f32(v) for v in x]
    g = [fp.bf16_to_f32(fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b), nan_word=NAN), BF16_NAN))
         for a, b in zip(dy, w, strict=True)]
    ga = [fp.f32_mul(C, gi, r, nan_word=NAN) for gi in g]
    gr = row_sum(C, [fp.f32_mul(C, gi, xi, nan_word=NAN) for gi, xi in zip(g, x32)], lanes)
    r3 = fp.f32_mul(C, fp.f32_mul(C, r, r, nan_word=NAN), r, nan_word=NAN)
    gv = fp.f32_mul(C, fp.f32_mul(C, gr, _c(-0.5), nan_word=NAN), r3, nan_word=NAN)
    gp = fp.f32_mul(C, gv, _c(1.0 / H), nan_word=NAN)
    gb = [fp.f32_mul(C, gp, fp.f32_mul(C, xi, _c(2.0), nan_word=NAN), nan_word=NAN) for xi in x32]
    return [fp.f32_to_bf16(C, fp.f32_add(C, a, b, nan_word=NAN), BF16_NAN) for a, b in zip(ga, gb)]


def _rms_signature(S: Any):
    _lanes(S)
    row = Array(S.H, BF16)
    return (("dy", row), ("x", row), ("w", row), ("r", F32)), row


def _rms_build(S: Any):
    H, lanes = S.H, _lanes(S)
    return lambda C, *e: rmsnorm_backward(C, e[:H], e[H:2 * H], e[2 * H:3 * H], e[3 * H], lanes)


RmsNormTorchBackward = specialized(
    "RmsNormTorchBackward", 1, ("H", "LANES"), _rms_build, _rms_signature,
    doc="torch 2.14 CUDA autograd of HF's RMS norm (bf16) for one token on bits: g = f32(bf16(dy w)); dx = bf16(RN(g r) + "
        "RN(RN(RN(RN(gr -0.5) RN(RN(r r) r)) f32(1/H)) RN(2 f32(x)))), gr = sum RN(g f32(x)) in Reduce.cuh's float4 order at "
        "LANES threads; r the saved rsqrt; NaN 0x7FFF")


# --- RMSNorm forward -------------------------------------------------------------------------------------------------------

def _eps(S: Any) -> int:
    e = S.EPS
    if isinstance(e, bool) or not isinstance(e, int) or not 0 <= e < 1 << 32:
        raise ValueError(f"EPS is an f32 word, not {e!r}")
    return e


def mean_factor(H: int) -> int:
    """`MeanOps`' factor over a row of H, ``(float)num_out / numel``: the f32 word RN(1 / H)."""
    return int((np.float32(1) / np.float32(_count("H", H))).view(np.uint32))


def rmsnorm_variance(C, x, lanes: int, eps: int):
    """``RN(RN(Σ RN(f32(x)²) * RN(1 / H)) + EPS)`` of one token's bf16 x, the sum in `row_sum`'s order."""
    x32 = [fp.bf16_to_f32(v) for v in x]
    s = row_sum(C, [fp.f32_mul(C, a, a, nan_word=NAN) for a in x32], lanes)
    mean = fp.f32_mul(C, s, fp.f32_const(mean_factor(len(x))), nan_word=NAN)
    return fp.f32_add(C, mean, fp.f32_const(eps), nan_word=NAN)


def rmsnorm_out(C, x, w, r):
    """(y, hb) of one token from bf16 x and w and the f32 rsqrt r: hb = bf16(RN(f32(x) r)), y = bf16(RN(f32(w) f32(hb)))."""
    hb = [fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(v), r, nan_word=NAN), BF16_NAN) for v in x]
    y = [fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b), nan_word=NAN), BF16_NAN)
         for a, b in zip(w, hb, strict=True)]
    return y + hb


def _rms_in_signature(S: Any):
    _lanes(S)
    _eps(S)
    return (("x", Array(S.H, BF16)),), F32


RmsNormTorchIn = specialized(
    "RmsNormTorchIn", 1, ("H", "LANES", "EPS"),
    lambda S: lambda C, *x: [rmsnorm_variance(C, x, _lanes(S), _eps(S))], _rms_in_signature,
    doc="HF's RMS norm (torch 2.14 CUDA, bf16) up to its rsqrt for one token on bits: RN(RN(sum RN(f32(x)^2) RN(1/H)) + "
        "EPS), the sum in Reduce.cuh's float4 order at LANES threads; NaN 0x7FFFFFFF")


def _rms_out_signature(S: Any):
    row = Array(_count("H", S.H), BF16)
    return (("x", row), ("w", row), ("r", F32)), Tuple(row, row)


RmsNormTorchOut = specialized(
    "RmsNormTorchOut", 1, ("H",), lambda S: lambda C, *e: rmsnorm_out(C, e[:S.H], e[S.H:2 * S.H], e[2 * S.H]),
    _rms_out_signature,
    doc="HF's RMS norm (torch 2.14 CUDA, bf16) after its rsqrt r for one token on bits: hb = bf16(RN(f32(x) r)), "
        "y = bf16(RN(f32(w) f32(hb))); NaN 0x7FFF")


def _rsqrt(S: Any) -> FunctionDefinition:
    f = bind(S.RSQRT) if isinstance(S.RSQRT, CompositeDefinition) else S.RSQRT
    if not isinstance(f, FunctionDefinition) or isinstance(f, SpecializedDefinition) and not f.complete:
        raise ValueError(f"RSQRT is a complete Definition of rsqrt.approx.f32, not {S.RSQRT!r}")
    params, ret = f.signature()
    if [t for _, t in params] != [F32] or ret != F32:
        raise ValueError(f"RSQRT = {f.id} is not (F32) -> F32 on bits")
    return f


def _refuse_rms(b: dict) -> str | None:
    try:
        if "RSQRT" in b:
            _rsqrt(type("S", (), {"RSQRT": b["RSQRT"]}))
        if "EPS" in b:
            _eps(type("S", (), {"EPS": b["EPS"]}))
    except ValueError as e:
        return str(e)
    return None


def _rms_fwd_signature(S: Any):
    _lanes(S)
    _eps(S)
    _rsqrt(S)
    row = Array(S.H, BF16)
    return (("x", row), ("w", row)), Tuple(row, row, F32)


@composite("RmsNormTorch", 1, ["H", "LANES", "EPS", "RSQRT"], _rms_fwd_signature, conformance="exact-model-tested",
           refuse=_refuse_rms)
def RmsNormTorch(B, S, x, w):
    """torch 2.14's eager HF RMS norm for one bf16 token on bits: `RmsNormTorchIn_v1`, RSQRT, then `RmsNormTorchOut_v1`;
    (y, hb, r)."""
    r = B.call(_rsqrt(S), B.call(bind(RmsNormTorchIn, H=S.H, LANES=S.LANES, EPS=S.EPS), x))
    o = B.call(bind(RmsNormTorchOut, H=S.H), x, w, r)
    return tuple_of(o[0], o[1], r)


# --- RMSNorm weight gradient -----------------------------------------------------------------------------------------------

def column_ctas(M: int, H: int) -> int:
    """CTAs a column of `Reduce.cuh`'s sum over the M rows of an (M, H) bf16 tensor (output-vectorized, a 32 x 4 block), the
    rows' stride 4 * CTAs; ValueError for a launch this module doesn't build (unvectorized, a warp a column, or a CTA count the
    device's grid sets)."""
    M, H = _count("M", M), _count("H", H)
    if H < 128 or H % 4:
        raise ValueError(f"H = {H}: Reduce.cuh vectorizes along output with a 32-thread row only from 128 columns in fours")
    if M < 64:
        raise ValueError(f"M = {M}: below 64 rows each warp sums its own columns")
    vpt = -(-M // 4)
    if vpt < 256:
        return 1
    by_grid = -(-(SM120_SMS * (SM120_THREADS // 128)) // -(-(H // 4) // WARP))
    by_rows = -(-vpt // 16)
    if by_grid < by_rows:
        raise ValueError(f"M = {M}, H = {H}: the CTA count is the device's grid ({by_grid}), not the rows' ({by_rows})")
    return max(by_rows, -(-vpt // 256))


def column_sum(C, p, ctas: int):
    """`Reduce.cuh`'s sum of one column's f32 words p at ``ctas`` CTAs, ``+0`` its identity: thread y of CTA c (y < 4) reads rows
    y + 4c, then every 4 * ctas, into accumulator i of four in turn, combined ((a0 + a1) + a2) + a3; `block_y_reduce` adds y2 to
    y0 and y3 to y1, then y1 to y0. With CTAs, the last block's thread y adds partials y, y + 4, .. to +0 and reduces again."""
    M, stride = len(p), 4 * ctas
    zero = fp.f32_const(0)

    def add(a, b):
        return fp.f32_add(C, a, b, nan_word=NAN)

    def block_y(ys):
        return add(add(ys[0], ys[2]), add(ys[1], ys[3]))

    parts = []
    for c in range(ctas):
        ys = []
        for y in range(4):
            idx, acc = y + 4 * c, [zero] * 4
            while idx + 3 * stride < M:
                acc = [add(a, p[idx + i * stride]) for i, a in enumerate(acc)]
                idx += 4 * stride
            for i in range(4):
                if idx >= M:
                    break
                acc[i] = add(acc[i], p[idx])
                idx += stride
            ys.append(add(add(add(acc[0], acc[1]), acc[2]), acc[3]))
        parts.append(block_y(ys))
    if ctas == 1:
        return parts[0]
    ys = []
    for y in range(4):
        v = zero
        for j in range(y, ctas, 4):
            v = add(v, parts[j])
        ys.append(v)
    return block_y(ys)


def _wgrad_signature(S: Any):
    column_ctas(S.M, S.H)
    col = Array(S.M, BF16)
    return (("dy", col), ("hb", col)), BF16


def _wgrad_build(S: Any):
    M, ctas = S.M, column_ctas(S.M, S.H)

    def build(C, *e):
        p = [fp.bf16_to_f32(fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b), nan_word=NAN), BF16_NAN))
             for a, b in zip(e[:M], e[M:], strict=True)]
        return fp.f32_to_bf16(C, column_sum(C, p, ctas), BF16_NAN)
    return build


RmsNormTorchWeightGrad = specialized(
    "RmsNormTorchWeightGrad", 1, ("M", "H"), _wgrad_build, _wgrad_signature,
    doc="torch 2.14 CUDA autograd of HF's RMS norm weight (bf16) for one of its H columns on bits: bf16(sum f32(bf16(dy "
        "hb))) over M tokens, the sum Reduce.cuh's column order at `column_ctas` CTAs on cc 12.0's 188 SMs; NaN 0x7FFF")


# --- SiLU ------------------------------------------------------------------------------------------------------------------

def _ex2(S: Any) -> FunctionDefinition:
    f = bind(S.EX2) if isinstance(S.EX2, CompositeDefinition) else S.EX2
    if not isinstance(f, FunctionDefinition) or isinstance(f, SpecializedDefinition) and not f.complete:
        raise ValueError(f"EX2 is a complete Definition of MUFU.EX2, not {S.EX2!r}")
    params, ret = f.signature()
    if [t for _, t in params] != [F32] or ret != F32:
        raise ValueError(f"EX2 = {f.id} is not (F32) -> F32 on bits")
    return f


def _neg_x(x):
    return fp.f32_neg(fp.bf16_to_f32(x))


def silu_out(C, x, e, L):
    """``bf16(RN(f32(x) / RN(1 + expf(-x))))``, expf's 2^f e and its scale byte L given."""
    d = fp.f32_add(C, fp.f32_const(ONE), nv_expf_out(C, e, L), nan_word=NAN)
    return fp.f32_to_bf16(C, fp.f32_div(C, fp.bf16_to_f32(x), d, nan_word=NAN), BF16_NAN)


def silu_backward_out(C, dy, x, e, L, contract: bool):
    """``bf16(RN(RN(f32(dy) * s) * t))``, ``s = RN(1 / RN(1 + expf(-x)))`` and ``t = 1 + x * RN(1 - s)``, one FFMA when
    ``contract``."""
    one = fp.f32_const(ONE)
    s = fp.f32_div(C, one, fp.f32_add(C, one, nv_expf_out(C, e, L), nan_word=NAN), nan_word=NAN)
    x32 = fp.bf16_to_f32(x)
    oms = fp.f32_sub(C, one, s, nan_word=NAN)
    t = (fp.f32_fma(C, x32, oms, one, nan_word=NAN) if contract
         else fp.f32_add(C, one, fp.f32_mul(C, x32, oms, nan_word=NAN), nan_word=NAN))
    y = fp.f32_mul(C, fp.f32_mul(C, fp.bf16_to_f32(dy), s, nan_word=NAN), t, nan_word=NAN)
    return fp.f32_to_bf16(C, y, BF16_NAN)


def _in_build(S: Any):
    def build(C, *x):
        fl = [nv_expf_in(C, _neg_x(v)) for v in x]
        return [f for f, _ in fl] + [L for _, L in fl]
    return build


def _in_signature(S: Any):
    n = _count("N", S.N)
    return (("x", Array(n, BF16)),), Tuple(Array(n, F32), Array(n, L8))


SiluTorchIn = specialized(
    "SiluTorchIn", 1, ("N",), _in_build, _in_signature,
    doc="SiLU's expf(-x) up to its MUFU.EX2 for N bf16 x on bits: NvExpfIn_v1 of -f32(x), the ex2 operands and the scale bytes")


def _out_signature(S: Any):
    n = _count("N", S.N)
    return (("x", Array(n, BF16)), ("e", Array(n, F32)), ("L", Array(n, L8))), Array(n, BF16)


SiluTorchOut = specialized(
    "SiluTorchOut", 1, ("N",),
    lambda S: lambda C, *a: [silu_out(C, a[i], a[S.N + i], a[2 * S.N + i]) for i in range(S.N)], _out_signature,
    doc="silu_kernel after its MUFU.EX2 for N elements on bits: bf16(RN(f32(x) / RN(1 + e 2^(L - 127)))), NaN 0x7FFF")


def _bwd_out_signature(S: Any):
    n = _count("N", S.N)
    _flag(S, "CONTRACT")
    row = Array(n, BF16)
    return (("dy", row), ("x", row), ("e", Array(n, F32)), ("L", Array(n, L8))), row


def _bwd_out_build(S: Any):
    n, contract = S.N, _flag(S, "CONTRACT")
    return lambda C, *a: [silu_backward_out(C, a[i], a[n + i], a[2 * n + i], a[3 * n + i], contract) for i in range(n)]


SiluTorchBackwardOut = specialized(
    "SiluTorchBackwardOut", 1, ("N", "CONTRACT"), _bwd_out_build, _bwd_out_signature,
    doc="silu_backward_kernel after its MUFU.EX2 for N elements on bits: s = RN(1 / RN(1 + e 2^(L - 127))), bf16(RN(RN(dy s) "
        "(1 + x RN(1 - s)))), the last one FFMA at CONTRACT = 1; NaN 0x7FFF")


def _refuse_silu(b: dict) -> str | None:
    try:
        if "EX2" in b:
            _ex2(type("S", (), {"EX2": b["EX2"]}))
    except ValueError as e:
        return str(e)
    return None


def _silu_signature(S: Any):
    _ex2(S)
    row = Array(_count("N", S.N), BF16)
    return (("x", row),), row


@composite("SiluTorch", 1, ["N", "EX2"], _silu_signature, conformance="exact-model-tested", refuse=_refuse_silu)
def SiluTorch(B, S, x):
    """torch 2.14's `silu_kernel` for N bf16 elements on bits: `SiluTorchIn_v1`, a batch of N EX2, then `SiluTorchOut_v1`."""
    fl = B.call(bind(SiluTorchIn, N=S.N), x)
    return B.call(bind(SiluTorchOut, N=S.N), x, B.batch(_ex2(S), fl[0], axes=(0,)), fl[1])


def _silu_bwd_signature(S: Any):
    _ex2(S)
    _flag(S, "CONTRACT")
    row = Array(_count("N", S.N), BF16)
    return (("dy", row), ("x", row)), row


@composite("SiluTorchBackward", 1, ["N", "EX2", "CONTRACT"], _silu_bwd_signature, conformance="exact-model-tested",
           refuse=_refuse_silu)
def SiluTorchBackward(B, S, dy, x):
    """torch 2.14's `silu_backward_kernel` for N bf16 elements on bits: `SiluTorchIn_v1` of x, a batch of N EX2, then
    `SiluTorchBackwardOut_v1`."""
    fl = B.call(bind(SiluTorchIn, N=S.N), x)
    e = B.batch(_ex2(S), fl[0], axes=(0,))
    return B.call(bind(SiluTorchBackwardOut, N=S.N, CONTRACT=S.CONTRACT), dy, x, e, fl[1])
