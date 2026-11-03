"""NVFP4 as Blackwell's block-scaled FP4 GEMMs read it: E2M1 codes, two a byte (the even element in the low nibble), an
E4M3 scale per 16 values and an fp32 scale per tensor (weights) or per row (activations).

`pack` codes a tensor; `unpack` decodes it to bf16, where every value is exact (an E2M1 value times an E4M3 scale has
at most six significant bits); `linear` is a W4A4 GEMM on the decoded operands, bf16 with an fp32 accumulator, the two
fp32 scales applied after it, its activation coded and decoded in one pass (`qdq`). The emulated GEMM is a bf16 GEMM
to a counter; the codes are what the forward reads. Rounding and decoding are arithmetic, so no op indexes a table by
a value (the counter reads an index as token data).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BLOCK = 16
E2M1_MAX, E4M3_MAX = 6.0, 448.0


def _scales(xf: torch.Tensor, rows: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(blocks (..., K/16, 16), their E4M3 scales, g): x ~ g * scale * E2M1, g over the tensor or each row."""
    amax = xf.abs().amax(-1, keepdim=True) if rows else xf.abs().amax()
    g = amax.clamp(min=1e-30) / (E2M1_MAX * E4M3_MAX)
    blocks = xf.unflatten(-1, (-1, BLOCK))
    s = (blocks.abs().amax(-1) / (E2M1_MAX * g)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    return blocks, s, g


def _round(y: torch.Tensor) -> torch.Tensor:
    """The E2M1 value nearest each of `y`, saturating at ±6: steps of 0.5 below 2, 1 below 4, 2 up to 6."""
    y = y.clamp(-E2M1_MAX, E2M1_MAX)
    q = torch.exp2(torch.floor(torch.log2(y.abs())).clamp(min=0)) / 2
    return torch.round(y / q) * q


def _unit(blocks: torch.Tensor, s: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(blocks in units of their scale, rounded to E2M1; the scale as fp32 (..., K/16, 1))."""
    sf = s.float().unsqueeze(-1)
    step = sf * (g.unsqueeze(-1) if g.dim() else g)
    y = torch.where(step > 0, blocks / step.clamp(min=1e-30), torch.zeros_like(blocks))
    return _round(y), sf


def pack(x: torch.Tensor, rows: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(codes uint8 (..., K/2), scales e4m3 (..., K/16), g fp32): x ~ g * scale * E2M1 along its last axis, g over the
    whole tensor, or over each row with `rows`. K must be a multiple of 16."""
    blocks, s, g = _scales(x.float(), rows)
    v, _ = _unit(blocks, s, g)
    a = v.abs()
    mag = torch.where(a <= 2, a * 2, torch.where(a <= 4, a + 2, a + 1))
    code = (mag.to(torch.uint8) | ((v < 0).to(torch.uint8) << 3)).flatten(-2)
    return code[..., 0::2] | (code[..., 1::2] << 4), s, g


def unpack(codes: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """bf16 values scale * E2M1 (exact), without the fp32 scale g."""
    code = torch.stack((codes & 15, codes >> 4), -1).flatten(-2)
    c = (code & 7).to(torch.bfloat16)
    mag = torch.where(c <= 4, c / 2, torch.where(c <= 6, c - 2, c - 1))
    v = (mag * (1 - 2 * (code >> 3).to(torch.bfloat16))).unflatten(-1, (-1, BLOCK))
    return (v * s.to(torch.bfloat16).unsqueeze(-1)).flatten(-2)


def qdq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`unpack(pack(x, rows=True))` and its rows' g, without forming the codes."""
    blocks, s, g = _scales(x.float(), True)
    v, sf = _unit(blocks, s, g)
    return (v * sf).flatten(-2).to(torch.bfloat16), g


def linear(x: torch.Tensor, w: torch.Tensor, gw: torch.Tensor) -> torch.Tensor:
    """x @ (gw * w).T with x coded to NVFP4 by row first; `w` is a weight's `unpack`, `gw` its g."""
    xq, g = qdq(x)
    return (F.linear(xq, w).float() * (g * gw)).to(torch.bfloat16)


def bits(codes: torch.Tensor, s: torch.Tensor) -> int:
    """A coded tensor's bits: its codes, its scales and its fp32 g."""
    return 8 * (codes.numel() + s.numel()) + 32
