"""The step's row ops and the W8A8 linear in torch, on whatever device their tensors are on, each the function circuits'
plan states for its Definitions (CMP-92, comment af6de1f1, sections 1 and 1b): every f32 op one IEEE round to nearest
(torch's elementwise CPU and CUDA kernels, no fast math, no FMA across separate ops, no FTZ), every reduction sequential in
ascending index from +0 (a Python loop over the reduced index, vectorized over the rest; never torch's blocked sums), every
bf16 rounding `cvt.rn.bf16.f32`. NaN words are not modelled: no op here produces one from finite inputs.

    rmsnorm    RmsSumSqPart{C=128} per chunk, F32SumPart{G=64} per group of chunk sums, the group sums from +0,
               RmsNormRecip (mean = RN(acc / N), rs = RN(1 / RN(sqrt(RN(mean + eps))))), RmsNormOut
               (bf16(RN(w * bf16(RN(x * rs)))))
    quantize   RowAbsMax{128[, RELU]} per chunk then over the chunks (exact), QuantS8Scale (RN(amax / 127), 1 when amax is 0),
               QuantS8Pair (clamp(rne(RN(x / scale)), -128, 127)); RELU applies max(x, 0) first
    s8_linear  ScaledMmS8Coordinate{K, GEMM_SM90_S8}: the exact s32 sum, then bf16((f32(acc) * sx) * sw), the token's scale first
    add        Bf16Add: bf16(RN(a + b))
    argmax     the first index of the greatest logit (ArgmaxChunkBest / ChunkIdx / Combine)

The s32 sum is `torch._int_mm` on CUDA (exact for K < 2^17 in any order) and an int64 matmul on the CPU: the stand-in for
kernels' KP2-S8 GEMM until it is handed over.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .shape import AMAX_CHUNK, RMS_CHUNK, RMS_GROUP


def f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def div_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RN(a / b) of f32s, through f64 (one f64 op then one rounding to f32 is the correctly rounded f32 result, as 53 >= 2·24 + 2):
    torch's f32 division by a scalar may multiply by its reciprocal."""
    return (a.to(torch.float64) / b.to(torch.float64)).to(torch.float32)


def sqrt_rn(a: torch.Tensor) -> torch.Tensor:
    """RN(sqrt(a)) of an f32, through f64: torch's f32 sqrt on the CPU (AVX-512) is not correctly rounded."""
    return torch.sqrt(a.to(torch.float64)).to(torch.float32)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return bf16(f32(a) + f32(b))


def seq_sum(x: torch.Tensor) -> torch.Tensor:
    """Σ over the last axis, in ascending index from +0."""
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)
    for i in range(x.shape[-1]):
        acc = acc + x[..., i]
    return acc


@dataclass
class Rms:
    out: torch.Tensor      # (B, N) bf16
    chunks: torch.Tensor   # (B, N / 128) f32: each chunk's sum of squares
    groups: torch.Tensor   # (B, ceil(chunks / 64)) f32
    rs: torch.Tensor       # (B,) f32


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> Rms:
    b, n = x.shape
    xf = f32(x)
    chunks = seq_sum((xf * xf).view(b, n // RMS_CHUNK, RMS_CHUNK))
    nc = chunks.shape[1]
    ng = -(-nc // RMS_GROUP)
    padded = torch.zeros(b, ng * RMS_GROUP, dtype=torch.float32, device=x.device)
    padded[:, :nc] = chunks
    groups = seq_sum(padded.view(b, ng, RMS_GROUP))
    total = seq_sum(groups)
    mean = div_rn(total, torch.full_like(total, float(n)))
    root = sqrt_rn(mean + torch.tensor(eps, dtype=torch.float32, device=x.device))
    rs = div_rn(torch.ones_like(root), root)
    h = bf16(xf * rs[:, None])
    return Rms(bf16(f32(w)[None, :] * f32(h)), chunks, groups, rs)


@dataclass
class Quant:
    codes: torch.Tensor    # (B, K) s8
    scale: torch.Tensor    # (B,) f32
    amax: torch.Tensor     # (B,) bf16
    parts: torch.Tensor    # (B, K / 128) bf16: each chunk's absmax


def quantize(x: torch.Tensor, relu: bool = False) -> Quant:
    b, k = x.shape
    xf = f32(x)
    if relu:
        xf = torch.clamp_min(xf, 0.0)
    parts = xf.abs().view(b, k // AMAX_CHUNK, AMAX_CHUNK).amax(-1)
    amax = parts.amax(-1)
    scale = torch.where(amax > 0, div_rn(amax, torch.full_like(amax, 127.0)), torch.ones_like(amax))
    codes = torch.round(div_rn(xf, scale[:, None])).clamp(-128, 127).to(torch.int8)
    return Quant(codes, scale, bf16(amax), bf16(parts))


def s32_mm(xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """(B, K) s8 x (N, K) s8 -> (B, N) s32, exact."""
    if xq.is_cuda:
        b = xq.shape[0]
        m = max(32, -(-b // 8) * 8)
        if m != b:
            xq = torch.cat([xq, torch.zeros(m - b, xq.shape[1], dtype=torch.int8, device=xq.device)])
        return torch._int_mm(xq, wq.t())[:b]
    return (xq.to(torch.int64) @ wq.to(torch.int64).t()).to(torch.int32)


def s8_linear(q: Quant, wq: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    acc = s32_mm(q.codes, wq)
    return bf16((acc.to(torch.float32) * q.scale[:, None]) * sw[None, :])


def argmax(logits: torch.Tensor) -> torch.Tensor:
    """The first index of the greatest value of each row, (B,) int64."""
    return torch.argmax(f32(logits), dim=-1)
