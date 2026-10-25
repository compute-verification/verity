"""The simplest exact W8A8 linear on a CUDA GPU, the GEMM the H200 demo's walking skeleton runs: every coordinate of
x W^T is `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}` (`verity.ml.circuits.ops.gemm`),

    acc[i, j] = sum_k x[i, k] w[j, k]                                  s32, exact in any order for K < 2^17
    y[i, j]   = cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * sx[i]) * sw[j])  two f32 mul.rn, the token's scale first

for x s8 [M, K] with one f32 scale per token sx [M], and w s8 [N, K] (one row per output channel) with one f32 scale per
channel sw [N].

`int_mm` is `torch._int_mm`, cuBLASLt's s8 x s8 -> s32 GEMM (the int8 profile's, `verity_vllm.engine.int8_exact`). It
refuses M <= 16, so a smaller x gets zero rows appended up to `MIN_ROWS`, and the first M rows are kept: a row of the
product reads only its own row of x. `scale` is the epilogue as four eager kernels with one rounding each: the s32 -> f32
conversion, the two in-place multiplies, then the bf16 cast. Run it eagerly: under `torch.compile` the four would fuse
into one generated kernel, which `check.py` did not check. On CUDA the cast is `cvt.rn.bf16.f32` (every NaN 0x7FFF, the
Definition's `F2fpBf16_v1`); on the CPU torch's cast writes another NaN word (0x7FC0 or 0xFFFF), and every other word is
the same.
"""
from __future__ import annotations

import torch

#: the least K whose s8 x s8 sum can leave s32 (`gemm.S8_K_LIMIT`), and the Definition's k step: it binds only multiples
K_LIMIT, K_STEP = 1 << 17, 32
#: the fewest rows `torch._int_mm` takes on CUDA
MIN_ROWS = 17


def int_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x s8 [M, K] times w s8 [N, K] transposed: the exact s32 sums [M, N]."""
    m = x.shape[0]
    if m < MIN_ROWS:
        x = torch.cat([x, x.new_zeros(MIN_ROWS - m, x.shape[1])])
    return torch._int_mm(x, w.t())[:m]


def scale(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """bf16((f32(acc) * sx[i]) * sw[j]) of s32 acc [M, N]: one rounding an operation, the token's scale first."""
    y = acc.to(torch.float32)
    y.mul_(sx[:, None])
    y.mul_(sw[None, :])
    return y.to(torch.bfloat16)


def linear(x_s8: torch.Tensor, sx_f32: torch.Tensor, w_s8: torch.Tensor, sw_f32: torch.Tensor) -> torch.Tensor:
    """The W8A8 linear: bf16 [M, N] of x s8 [M, K], sx f32 [M], w s8 [N, K] and sw f32 [N] (module docstring)."""
    refusal = refuse(x_s8, sx_f32, w_s8, sw_f32)
    if refusal:
        raise ValueError(f"s8_simple.linear: {refusal}")
    return scale(int_mm(x_s8, w_s8), sx_f32, sw_f32)


def refuse(x: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> str | None:
    """Why `linear` can't compute the Definition on these operands, or None."""
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
    return None
