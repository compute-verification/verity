"""NVFP4 as Blackwell's block-scaled FP4 GEMMs read it: E2M1 codes, two a byte (the even element in the low nibble), an
E4M3 scale per 16 values and an fp32 scale per tensor (weights) or per row (activations).

`pack` codes a tensor; `unpack` decodes it to bf16, where every value is exact (an E2M1 value times an E4M3 scale has
at most six significant bits); `linear` is a W4A4 GEMM on the decoded operands, bf16 with an fp32 accumulator, the two
fp32 scales applied after it. The emulated GEMM is a bf16 GEMM to a counter; the codes are what the forward reads.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BLOCK = 16
E2M1_MAX, E4M3_MAX = 6.0, 448.0
#: the E2M1 magnitudes (code & 7 indexes them; bit 3 is the sign) and the midpoints where round-to-nearest switches
VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def pack(x: torch.Tensor, rows: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(codes uint8 (..., K/2), scales e4m3 (..., K/16), g fp32): x ~ g * scale * E2M1 along its last axis, g over the
    whole tensor, or over each row with `rows`. K must be a multiple of 16."""
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True) if rows else xf.abs().amax()
    g = amax.clamp(min=1e-30) / (E2M1_MAX * E4M3_MAX)
    blocks = xf.unflatten(-1, (-1, BLOCK))
    s = (blocks.abs().amax(-1) / (E2M1_MAX * g)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    step = (s.float() * g).unsqueeze(-1)
    y = torch.where(step > 0, blocks / step.clamp(min=1e-30), torch.zeros_like(blocks))
    mid = torch.tensor(MIDPOINTS, device=x.device)
    mag = torch.bucketize(y.abs().clamp(max=E2M1_MAX), mid).to(torch.uint8)
    code = (mag | ((y < 0).to(torch.uint8) << 3)).flatten(-2)
    return code[..., 0::2] | (code[..., 1::2] << 4), s, g


def unpack(codes: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """bf16 values scale * E2M1 (exact), without the fp32 scale g."""
    lut = torch.tensor(VALUES, device=codes.device)
    lut = torch.cat((lut, -lut)).to(torch.bfloat16)
    code = torch.stack((codes & 15, codes >> 4), -1).flatten(-2)
    v = lut[code.long()].unflatten(-1, (-1, BLOCK))
    return (v * s.to(torch.bfloat16).unsqueeze(-1)).flatten(-2)


def linear(x: torch.Tensor, w: torch.Tensor, gw: torch.Tensor) -> torch.Tensor:
    """x @ (gw * w).T with x coded to NVFP4 by row first; `w` is a weight's `unpack`, `gw` its g."""
    q, s, g = pack(x, rows=True)
    return (F.linear(unpack(q, s), w).float() * (g * gw)).to(torch.bfloat16)


def bits(codes: torch.Tensor, s: torch.Tensor) -> int:
    """A coded tensor's bits: its codes, its scales and its fp32 g."""
    return 8 * (codes.numel() + s.numel()) + 32
