"""The codes a step computes with: a gradient, a momentum or a stored weight coded inside a step, as functions on
tensors. `codecs` is E's alone. Every random draw is from torch's global generator, which the harness seeds with the
job's seed and keeps across a stop.

A grouped code rounds each `group` consecutive entries of a tensor, stochastically and so unbiased (E[value] = x),
onto a uniform grid of levels times an f16 scale a group, at least the group's largest magnitude:

- `grid(bits)`: the levels a code takes, ±1 at one bit, else (c − L)/L for c in 0..2L, L = 2^(bits−1) − 1;
- `encode`, `encode_at`: x's codes at the group's own scale (its largest magnitude, times 1 + 2^-9) or at given
  scales, and `values`, the tensor they stand for;
- `pack`, `unpack`: x as it waits between two of a step's blocks, its codes as `bits` bool planes (one bit an entry)
  and its scales, coded in a rotated basis when a `rotation` (a Hadamard matrix with random signs on its rows, so a
  few large entries no longer set a whole group's scale) is given;
- `sticky_add`: a stored tensor's codes and scales after a step is added to the value they stand for, each group
  keeping its scale until a value outgrows it, then `HEADROOM` times its largest magnitude;
- `vote_pack`, `vote_unpack`: one bit an entry and no scale, the sign, a zero drawn ±1 at random (a majority vote's
  tie).
"""
from __future__ import annotations

import math

import torch

#: a sticky scale's growth over its group's largest magnitude when a value outgrows it
HEADROOM = 1.125


def codebook(bits: int) -> torch.Tensor:
    """The f16 grid `grid` takes its levels from, padded with zeros to 2^bits entries (E's codebook)."""
    if bits == 1:
        return torch.tensor([-1.0, 1.0], dtype=torch.float16)
    half = 2 ** (bits - 1) - 1
    cb = torch.zeros(2 ** bits, dtype=torch.float16)
    cb[:2 * half + 1] = (torch.arange(2 * half + 1, dtype=torch.float64) - half) / half
    return cb


def levels(bits: int) -> int:
    return 2 if bits == 1 else 2 ** bits - 1


def grid(bits: int) -> torch.Tensor:
    return codebook(bits)[:levels(bits)].float()


def zero(bits: int) -> int:
    """The code of the value 0 (the grid's middle level; at one bit, which has none, the first)."""
    return 0 if bits == 1 else levels(bits) // 2


def rotation(group: int) -> torch.Tensor:
    """An orthonormal group × group basis H·D: the Sylvester–Hadamard matrix H with random signs D on its rows."""
    if group & (group - 1):
        raise ValueError(f"a rotated code's group ({group}) is a power of two")
    h = torch.ones(1, 1)
    while h.shape[0] < group:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    signs = torch.randint(0, 2, (group,)).float() * 2 - 1
    return signs[:, None] * (h / math.sqrt(group))


def encode(x: torch.Tensor, group: int, levels_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = (x.float().reshape(-1, group).abs().amax(1) * (1 + 2 ** -9)).to(torch.float16)
    return encode_at(x, s, group, levels_)


def encode_at(x: torch.Tensor, s: torch.Tensor, group: int, levels_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x's codes (int64, one row a group) at the f16 scales `s`, each at least its group's largest magnitude."""
    flat = x.float().reshape(-1, group)
    div = torch.where(s > 0, s.float(), torch.ones_like(s, dtype=torch.float32))
    v = (flat / div[:, None]).clamp(-1, 1)
    top = len(levels_) - 1
    lo = ((v + 1) * (top / 2)).floor().clamp(0, top - 1).long()
    g_lo, g_hi = levels_[lo], levels_[lo + 1]
    p = ((v - g_lo) / (g_hi - g_lo)).clamp(0, 1)
    return lo + (torch.rand(p.shape, device=p.device) < p).long(), s


def values(codes: torch.Tensor, s: torch.Tensor, levels_: torch.Tensor, shape) -> torch.Tensor:
    return (levels_[codes.long()] * s.float()[:, None]).reshape(shape)


def pack(x: torch.Tensor, bits: int, group: int, levels_: torch.Tensor,
         rot: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """x's codes as `bits` bool planes stacked (bits × groups × group) and its f16 scales."""
    y = x.float().reshape(-1, group) @ rot if rot is not None else x
    codes, s = encode(y, group, levels_)
    return torch.stack([((codes >> b) & 1).bool() for b in range(bits)]), s


def unpack(planes: torch.Tensor, s: torch.Tensor, shape, levels_: torch.Tensor,
           rot: torch.Tensor | None = None) -> torch.Tensor:
    codes = sum(p.long() << b for b, p in enumerate(planes))
    y = values(codes, s, levels_, codes.shape)
    return (y @ rot.T).reshape(shape) if rot is not None else y.reshape(shape)


def sticky_add(codes: torch.Tensor, scales: torch.Tensor, step: torch.Tensor, group: int,
               levels_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The uint8 codes and f16 scales of value(`codes`, `scales`) + `step`."""
    new = values(codes, scales, levels_, step.shape) + step
    need = new.float().reshape(-1, group).abs().amax(1)
    s = torch.where(need > scales.float(), (need * HEADROOM).to(torch.float16), scales)
    out, s = encode_at(new, s, group, levels_)
    return out.to(torch.uint8), s


def vote_pack(x: torch.Tensor) -> torch.Tensor:
    coin = torch.randint(0, 2, x.shape, device=x.device, dtype=torch.int8).bool()
    return torch.where(x == 0, coin, x > 0)


def vote_unpack(b: torch.Tensor) -> torch.Tensor:
    return torch.where(b, 1.0, -1.0)
