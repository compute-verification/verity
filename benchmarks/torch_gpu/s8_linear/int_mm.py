"""The int_mm backend of the exact W8A8 linear (run.py's `--kernel int_mm`; the plain GEMM
p2_stage's full.py and j10r_sm90's bench call): every coordinate of x W^T is `ScaledMmS8Coordinate_v1{K, DOT}`
(`verity.ml.circuits.ops.gemm`),

    acc[i, j] = sum_k x[i, k] w[j, k]                                  s32, exact in any order for K < 2^17
    y[i, j]   = cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * sx[i]) * sw[j])  two f32 mul.rn, the token's scale first

for x s8 [M, K] with one f32 scale per token sx [M], and w s8 [N, K] (one row per output channel) with one f32 scale per
channel sw [N].

`int_mm` is `torch._int_mm`, cuBLASLt's s8 x s8 -> s32 GEMM (the int8 profile's, `verity_vllm.engine.int8_exact`). It
refuses M <= 16, so a smaller x gets zero rows appended up to `MIN_ROWS`, and the first M rows are kept: a row of the
product reads only its own row of x. The epilogue is one of `EPILOGUES`:

* `fused_scale` (`linear`'s default, CUDA only): one Triton kernel that reads each s32 once and writes each bf16 once. A
  coordinate is `FUSED_PTX`, the Definition's primitives as PTX: `cvt.rn.f32.s32` (`S32ToF32Rn_v1`), `mul.rn.f32` by sx
  then by sw (`F32MulV2_v1`, two roundings, no FTZ), `cvt.rn.bf16.f32` (`F2fpBf16_v1`, every NaN 0x7FFF). Inline PTX,
  so no compiler pass can contract, reorder or flush them; `fused_ptx` returns what a launch compiled and `ptx_audit`
  holds it to that.
* `scale` (`epilogue="eager"`, any device): four eager torch kernels with one rounding each, the s32 -> f32 conversion,
  the two in-place multiplies, then the bf16 cast. Run it eagerly: under `torch.compile` the four would fuse into one
  generated kernel, which run.py doesn't check. On CUDA the cast is `cvt.rn.bf16.f32`; on the CPU torch's cast
  writes another NaN word (0x7FC0 or 0xFFFF), and every other word is the same.
"""
from __future__ import annotations

import re

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

#: the least K whose s8 x s8 sum can leave s32 (`gemm.S8_K_LIMIT`), and the Definition's k step: it binds only multiples
K_LIMIT, K_STEP = 1 << 17, 32
#: the fewest rows `torch._int_mm` takes on CUDA
MIN_ROWS = 17
EPILOGUES = ("fused", "eager")
#: one coordinate of the fused epilogue: $0 the bf16 word, $1 the s32 sum, $2 sx, $3 sw
FUSED_PTX = ("{ .reg .f32 t; cvt.rn.f32.s32 t, $1; mul.rn.f32 t, t, $2; mul.rn.f32 t, t, $3; "
             "cvt.rn.bf16.f32 $0, t; }")
#: the fused kernel's tile (rows, channels) and warps
FUSED_TILE, FUSED_WARPS = (16, 256), 4


def int_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x s8 [M, K] times w s8 [N, K] transposed: the exact s32 sums [M, N]."""
    m = x.shape[0]
    if m < MIN_ROWS:
        x = torch.cat([x, x.new_zeros(MIN_ROWS - m, x.shape[1])])
    return torch._int_mm(x, w.t())[:m]


def scale(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """bf16((f32(acc) * sx[i]) * sw[j]) of s32 acc [M, N]: one rounding an operation, the token's scale first."""
    y = acc.to(torch.float32)
    y.mul_(sx[:, None])
    y.mul_(sw[None, :])
    return y.to(torch.bfloat16) if out is None else out.copy_(y)


if triton is not None:
    @triton.jit
    def _fused_kernel(acc, sx, sw, y, m, n, stride, PTX: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        i = tl.program_id(0) * BM + tl.arange(0, BM)
        j = tl.program_id(1) * BN + tl.arange(0, BN)
        inside = (i < m)[:, None] & (j < n)[None, :]
        a = tl.load(acc + i.to(tl.int64)[:, None] * stride + j[None, :], mask=inside, other=0)
        s = tl.broadcast_to(tl.load(sx + i, mask=i < m, other=0.0)[:, None], (BM, BN))
        t = tl.broadcast_to(tl.load(sw + j, mask=j < n, other=0.0)[None, :], (BM, BN))
        word = tl.inline_asm_elementwise(PTX, "=h,r,f,f", [a, s, t], dtype=tl.int16, is_pure=True, pack=1)
        tl.store(y + i.to(tl.int64)[:, None] * n + j[None, :], word, mask=inside)


def _fused(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor, out: torch.Tensor | None = None):
    m, n = acc.shape
    acc = acc if acc.stride(1) == 1 else acc.contiguous()
    y = torch.empty((m, n), dtype=torch.bfloat16, device=acc.device) if out is None else out
    grid = (triton.cdiv(m, FUSED_TILE[0]), triton.cdiv(n, FUSED_TILE[1]))
    kernel = _fused_kernel[grid](acc, sx.contiguous(), sw.contiguous(), y.view(torch.int16), m, n, acc.stride(0),
                                 PTX=FUSED_PTX, BM=FUSED_TILE[0], BN=FUSED_TILE[1], num_warps=FUSED_WARPS)
    return y, kernel


def fused_scale(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """`scale` as one kernel on CUDA tensors, each s32 read once and each bf16 written once (module docstring)."""
    return _fused(acc, sx, sw, out)[0]


def fused_ptx(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor) -> str:
    """The PTX `fused_scale` runs on these operands: Triton compiles a variant per specialization of m, n and the stride."""
    return _fused(acc, sx, sw)[1].asm["ptx"]


_F32_ARITH = re.compile(r"\b(?:add|sub|mul|fma|mad|div|rcp|sqrt|neg|abs|min|max|ex2|lg2|cvt)(?:\.[a-z0-9]+)*?\.f32\b")


def ptx_audit(ptx: str) -> dict:
    """The fused kernel's PTX holds `FUSED_PTX` once a coordinate and no other f32 arithmetic: as many `mul.rn.f32` as
    twice the `cvt.rn.f32.s32`, as many `cvt.rn.bf16.f32`, and no fma, no `.ftz` and no other f32 instruction."""
    n = {op: len(re.findall(r"\b" + re.escape(op) + r"\b", ptx)) for op in ("cvt.rn.f32.s32", "mul.rn.f32", "cvt.rn.bf16.f32")}
    n |= {"fma": len(re.findall(r"\bfma\.", ptx)), "ftz": len(re.findall(r"\.ftz\b", ptx)),
          "blocks": ptx.count("mul.rn.f32 t, t,") // 2}
    n["other_f32"] = len(_F32_ARITH.findall(ptx)) - n["cvt.rn.f32.s32"] - n["mul.rn.f32"] - n["cvt.rn.bf16.f32"]
    c = n["cvt.rn.f32.s32"]
    n["ok"] = bool(c > 0 and n["mul.rn.f32"] == 2 * c and n["cvt.rn.bf16.f32"] == c == n["blocks"]
                   and n["fma"] == n["ftz"] == n["other_f32"] == 0)
    return n


def linear(x_s8: torch.Tensor, sx_f32: torch.Tensor, w_s8: torch.Tensor, sw_f32: torch.Tensor,
           epilogue: str = "fused", out: torch.Tensor | None = None) -> torch.Tensor:
    """The W8A8 linear: bf16 [M, N] of x s8 [M, K], sx f32 [M], w s8 [N, K] and sw f32 [N], with the fused epilogue or
    the eager one (module docstring), into ``out`` (contiguous bf16 [M, N] on x's device) when given."""
    refusal = refuse(x_s8, sx_f32, w_s8, sw_f32, epilogue, out)
    if refusal:
        raise ValueError(f"int_mm.linear: {refusal}")
    return (fused_scale if epilogue == "fused" else scale)(int_mm(x_s8, w_s8), sx_f32, sw_f32, out)


def refuse(x: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor, epilogue: str = "fused",
           out: torch.Tensor | None = None) -> str | None:
    """Why `linear` can't compute the Definition on these operands, or None."""
    if epilogue not in EPILOGUES:
        return f"epilogue {epilogue!r}: pass one of {EPILOGUES}"
    got = (x.dtype, sx.dtype, w.dtype, sw.dtype)
    if got != (torch.int8, torch.float32, torch.int8, torch.float32):
        return f"dtypes {got}: pass x and w as int8 codes and sx and sw as float32 scales"
    if x.dim() != 2 or w.dim() != 2 or x.shape[1] != w.shape[1]:
        return f"x {tuple(x.shape)} and w {tuple(w.shape)} are not [M, K] and [N, K]"
    (m, k), n = x.shape, w.shape[0]
    if tuple(sx.shape) != (m,) or tuple(sw.shape) != (n,):
        return f"sx {tuple(sx.shape)} and sw {tuple(sw.shape)}: one scale a token ({m}) and one a channel ({n})"
    if k % K_STEP or not 0 < k < K_LIMIT:
        return (f"K = {k}: the Definition binds a positive multiple of {K_STEP} below {K_LIMIT}; pad x and w with zero codes "
                f"to the next multiple, or split K into parts below {K_LIMIT}")
    if n % 8:
        return f"N = {n}: torch._int_mm takes a multiple of 8 channels; pad w with zero rows and drop their outputs"
    if out is not None and (out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n) or not out.is_contiguous()
                            or out.device != x.device):
        return f"out {out.dtype} {tuple(out.shape)} on {out.device}: pass a contiguous bfloat16 [{m}, {n}] on {x.device}"
    if epilogue == "fused" and (triton is None or x.device.type != "cuda"):
        return (f"the fused epilogue is a Triton kernel on CUDA ({x.device.type} tensors"
                f"{', and no Triton here' if triton is None else ''}): pass epilogue='eager'")
    return None
