"""The step's row ops and the W8A8 linear in torch, on whatever device their tensors are on, each the function of its units
in circuits' FAMILIES map (`plan`; the semantics are circuits' plan's): every f32 op one IEEE
round to nearest (torch's elementwise kernels, no fast math, no FMA across separate ops), division and sqrt correctly
rounded through f64 (`div_rn`, `sqrt_rn`), every reduction sequential in ascending index from +0 (`seq_sum`, never torch's
blocked sums: a Triton kernel on CUDA, numpy on the host), every bf16 rounding RNE. NaN words are not modelled (no op here
makes one from finite inputs), except that `argmax` and `quantize` take a NaN input as their Definitions do.

    rmsnorm    RmsSumSqChunks_v1 and RmsNormOutSums_v1{C, S, G, N, EPS}: chunk c's sum s_c = sum_i RN(x_i^2), group g's
               sum of its chunk sums, s = the groups' sum, each from +0; mean = RN(s / N), rs = RN(1 / RN(sqrt(RN(mean +
               eps)))), out_j = bf16(RN(w_j * bf16(RN(x_j * rs))))
    quantize   RowAbsMax_v1{N, RELU} over parts and QuantChunkS8_v1{N, RELU}: amax = max |x| (after x < 0 -> +0 under
               RELU), scale = RN(amax / 127) (1 when amax is not > 0), codes = clamp(rne(RN(x / scale)), -128, 127), a NaN 0
    s8_linear  ScaledMmS8CoordinateRows_v1{K, R, GEMM_SM90_S8}: the exact s32 sum (`s32_mm`), then
               bf16(RN(RN(f32(acc) * sx) * sw)) (`scale`)
    add        Bf16AddRow_v2{N}: bf16(RN(a + b))
    embed      EmbeddingCoordinate_v1{V}: bf16(RN(tok[:, token] + pos[:, t]))
    argmax     Argmax_v2{V}: TokenSelect, the first index of the greatest logit, a NaN never taken

The s32 sum is `torch._int_mm` on CUDA (exact for K < 2^17 in any order) and a float64 matmul in row chunks on the CPU (exact: integers below 2^53): the stand-in for
kernels' s8 GEMM until it is handed over.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

#: f64 elements of one weight chunk in the CPU's s32 sum (1 GiB)
F64_CHUNK = 2 ** 27


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


@functools.lru_cache(maxsize=1)
def _seq_sum_kernel() -> Any:
    """A Triton kernel summing each row in ascending index from +0, one lane a row, each step one ``add.rn.f32`` (no FTZ, no
    contraction), or None without Triton."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def kernel(x, out, rows, n, stride, BLOCK: tl.constexpr):
        r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = r < rows
        base = x + r.to(tl.int64) * stride
        acc = tl.zeros((BLOCK,), tl.float32)
        for i in range(n):
            v = tl.load(base + i, mask=live, other=0.0)
            acc = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [acc, v], dtype=tl.float32, is_pure=True,
                                            pack=1)
        tl.store(out + r, acc, mask=live)
    return kernel


def seq_sum(x: torch.Tensor) -> torch.Tensor:
    """(B, N) f32 -> (B,): the sum in ascending index from +0 (sequential, unlike every torch and numpy sum): on CUDA a
    Triton kernel (`_seq_sum_kernel`), else numpy's ``add.accumulate`` on the host."""
    k = _seq_sum_kernel() if x.is_cuda else None
    if k is not None:
        x = x.contiguous().to(torch.float32)
        out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        block = 32
        k[(-(-x.shape[0] // block),)](x, out, x.shape[0], x.shape[1], x.stride(0), BLOCK=block)
        return out
    a = x.detach().to("cpu", torch.float32).numpy()
    s = np.add.accumulate(a, axis=1, dtype=np.float32)[:, -1] + np.float32(0)
    return torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32)).to(x.device)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return bf16(f32(a) + f32(b))


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: torch.Tensor, chunk: int | None = None,
            groups: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """x (B, N) bf16, w (N,) bf16, eps a () f32 -> (B, N) bf16 and the chunk sums (B, N / chunk) f32: each chunk's sum of
    RN(x_i^2), then each of ``groups`` groups' sum of its chunk sums, then the groups' sum, each sequential from +0."""
    b, n = x.shape
    c = chunk or n
    xf = f32(x)
    sums = seq_sum((xf * xf).reshape(b * (n // c), c)).reshape(b, n // c)
    total = seq_sum(seq_sum(sums.reshape(b * groups, -1)).reshape(b, groups))
    mean = div_rn(total, torch.full((b,), float(n), dtype=torch.float32, device=x.device))
    rs = div_rn(torch.ones_like(mean), sqrt_rn(mean + eps))
    h = bf16(xf * rs[:, None])
    return bf16(f32(w)[None, :] * f32(h)), sums


@dataclass
class Quant:
    codes: torch.Tensor    # (B, K) s8
    scale: torch.Tensor    # (B,) f32
    amax: torch.Tensor     # (B, P) bf16 words as s16: the first P - 1 parts' max |x|, then the row's


def quantize(x: torch.Tensor, relu: bool = False, part: int | None = None, every: bool = False) -> Quant:
    """``part``: the words of a `RowAbsMax_v1` part (the whole row by default); the amax words are the parts' but the last
    (every part's with ``every``), then the row's. max |x| is the greatest word with its sign cleared, so a NaN is above +inf
    and the scale of a row with a NaN is 1."""
    xf = f32(x)
    if relu:
        xf = torch.where(xf < 0, torch.zeros_like(xf), xf)
    b, n = x.shape
    p = part or n
    mag = x.view(torch.int16).to(torch.int32) & 0x7FFF
    if relu:
        mag = torch.where(xf == 0, 0, mag)
    mag = mag.reshape(b, n // p, p).amax(-1)
    amax = torch.cat([mag if every else mag[:, :-1], mag.amax(-1, keepdim=True)], 1).to(torch.int16)
    am = (mag.amax(-1) << 16).view(torch.float32)
    scale = torch.where(am > 0, div_rn(am, torch.full_like(am, 127.0)), torch.ones_like(am))
    codes = torch.nan_to_num(torch.round(div_rn(xf, scale[:, None])), nan=0.0).clamp(-128, 127).to(torch.int8)
    return Quant(codes, scale, amax)


def s32_mm(xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """(B, K) s8 x (N, K) s8 -> (B, N) s32, exact."""
    if xq.is_cuda:
        b = xq.shape[0]
        m = max(32, -(-b // 8) * 8)
        if m != b:
            xq = torch.cat([xq, torch.zeros(m - b, xq.shape[1], dtype=torch.int8, device=xq.device)])
        return torch._int_mm(xq, wq.t())[:b]
    # f64 is exact in any summation order: every partial sum is an integer of magnitude <= K * 2^14 < 2^53
    if xq.shape[1] > 2 ** 38:
        raise ValueError(f"K = {xq.shape[1]} is past f64's exact range")
    x, n = xq.to(torch.float64), wq.shape[0]
    rows = max(1, F64_CHUNK // xq.shape[1])
    out = torch.empty(xq.shape[0], n, dtype=torch.int32)
    for lo in range(0, n, rows):
        out[:, lo:lo + rows] = (x @ wq[lo:lo + rows].to(torch.float64).t()).to(torch.int32)
    return out


def scale(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """The linear's epilogue on its s32 sums: bf16(RN(RN(f32(acc) * sx) * sw))."""
    return bf16((acc.to(torch.float32) * sx[:, None]) * sw[None, :])


def s8_linear(q: Quant, wq: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    return scale(s32_mm(q.codes, wq), q.scale, sw)


def argmax(logits: torch.Tensor) -> torch.Tensor:
    """`TokenSelect` over each row, (B,) int64: the first index of the greatest value (+0 = -0), a NaN never taken, and 0
    when the row's first value is NaN or none is above -inf (torch's argmax takes a NaN as the greatest)."""
    xf = f32(logits)
    nan = torch.isnan(xf)
    j = torch.argmax(torch.where(nan, float("-inf"), xf), dim=-1)
    return torch.where(nan[:, 0], torch.zeros_like(j), j)
