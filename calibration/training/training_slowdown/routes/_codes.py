"""Codes a route's step passes a tensor through, each on tensors and nothing of a route: a matrix's rows rounded to
nearest at a few bits under one bf16 scale a row and packed into bytes (`row_codes`, decoded by `row_values`), and
float32 rounded stochastically to bf16 (`round_bf16`) or to int8 codes at one bf16 scale a row (`round_int8`). Not a
route: a route imports it as a sibling (`from ._codes import row_codes`).

The stochastic roundings take their random values from torch's global generator, which a stopped job keeps and its continuation takes back
(`routes/__init__.py`), and nothing else: a generator of a route's own would start again at `build`.
"""
from __future__ import annotations

import torch


def row_codes(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Matrix `x` rounded to nearest at `bits` bits (2, 4 or 8) under one bf16 scale a row, its largest magnitude
    over 2^(bits-1) - 1: the codes, offset to unsigned and packed 8/bits a byte along the row, and the scales."""
    qmax, per = 2 ** (bits - 1) - 1, 8 // bits
    if x.shape[1] % per:
        raise ValueError(f"row_codes: {bits}-bit codes fill whole bytes of a row only, and a row of {x.shape[1]} "
                         f"doesn't")
    scales = (x.abs().amax(1, keepdim=True) / qmax).bfloat16()
    s = scales.float()
    k = (torch.clamp(torch.round(x / torch.where(s > 0, s, 1.0)), -qmax, qmax) + qmax).to(torch.uint8)
    lanes = k.view(k.shape[0], -1, per)
    packed = lanes[..., 0]
    for i in range(1, per):
        packed = packed | (lanes[..., i] << (bits * i))
    return packed, scales


def row_values(packed: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    """The float32 matrix `row_codes` codes as (`packed`, `scales`)."""
    qmax, per = 2 ** (bits - 1) - 1, 8 // bits
    codes = torch.stack([(packed >> (bits * i)) & (2 ** bits - 1) for i in range(per)], -1)
    return (codes.view(packed.shape[0], -1).float() - qmax) * scales.float()


def round_bf16(x: torch.Tensor) -> torch.Tensor:
    """float32 `x` rounded to bf16 stochastically: 16 random low bits added to each word, then cut, so the rounding is
    unbiased and a move under half a bf16 step still lands in expectation."""
    noise = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32)
    return ((x.contiguous().view(torch.int32) + noise) & -65536).view(torch.float32).to(torch.bfloat16)


def round_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 matrix `x` as int8 codes in -127..127 at one bf16 scale a row (its largest magnitude over 127), rounded
    stochastically, so the codes' value is unbiased: the codes and the scales."""
    scales = (x.abs().amax(1, keepdim=True) / 127).bfloat16()
    y = x / torch.where(scales > 0, scales.float(), 1.0)
    return torch.clamp(torch.floor(y + torch.rand(y.shape, device=y.device)), -127, 127).to(torch.int8), scales
