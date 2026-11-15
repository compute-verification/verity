"""The `lf` model, the starts' architecture: a dense Llama-style decoder (RMSNorm, SwiGLU, RoPE, no biases), input and
output embeddings tied. Its parameter names are θ's tensor names (`embed`, `blocks.{i}.{attn_norm, mlp_norm, wq, wk,
wv, wo, w_gate, w_up, w_down}`, `norm`), so `LM.of(shape).load_state_dict(θ)` loads a start.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

CTX = 1024
ROPE_THETA = 10000.0


@dataclass(frozen=True)
class Size:
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, std: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.normal_(self.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).type_as(x)


class Block(nn.Module):
    def __init__(self, size: Size, std: float, out_std: float):
        super().__init__()
        d = size.d_model
        self.n_head = size.n_head
        self.attn_norm = nn.Parameter(torch.ones(d))
        self.mlp_norm = nn.Parameter(torch.ones(d))
        self.wq, self.wk, self.wv = (Linear(d, d, std) for _ in range(3))
        self.wo = Linear(d, d, out_std)
        self.w_gate, self.w_up = Linear(d, size.d_ff, std), Linear(d, size.d_ff, std)
        self.w_down = Linear(size.d_ff, d, out_std)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = F.rms_norm(x, (d,), self.attn_norm)
        q, k, v = (w(h).view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        a = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, is_causal=True)
        x = x + self.wo(a.transpose(1, 2).reshape(b, t, d))
        h = F.rms_norm(x, (d,), self.mlp_norm)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class LM(nn.Module):
    def __init__(self, size: Size, vocab: int, ctx: int = CTX):
        super().__init__()
        self.size, self.vocab, self.ctx = size, vocab, ctx
        std, out_std = 0.02, 0.02 / math.sqrt(2 * size.n_layer)
        self.embed = nn.Parameter(torch.empty(vocab, size.d_model))
        nn.init.normal_(self.embed, std=std)
        self.blocks = nn.ModuleList(Block(size, std, out_std) for _ in range(size.n_layer))
        self.norm = nn.Parameter(torch.ones(size.d_model))
        hd = size.d_model // size.n_head
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        angles = torch.outer(torch.arange(ctx, dtype=torch.float32), freqs)
        # One position's row at a time, so each in the calling thread. torch cuts a larger cos or sin across its
        # threads, and the first one a process computes can come back with a thread's share at low accuracy: the
        # model would then be another function in one process than in the next.
        self.register_buffer("cos", torch.stack([row.cos() for row in angles]), persistent=False)
        self.register_buffer("sin", torch.stack([row.sin() for row in angles]), persistent=False)

    @classmethod
    def of(cls, shape: dict) -> LM:
        """The model of a start's `shape` (`starts.toml`)."""
        if shape.get("family", "lf") != "lf" or not shape.get("tied", True):
            raise ValueError(f"model: no architecture for {shape.get('family')!r} (tied {shape.get('tied')}): lf, tied")
        return cls(Size(shape["d_model"], shape["n_layer"], shape["n_head"], shape["d_ff"]), shape["vocab"],
                   shape.get("ctx", CTX))

    def hidden(self, idx: torch.Tensor, upto: int | None = None) -> torch.Tensor:
        """The residual stream after the first `upto` blocks (all of them by default)."""
        x = F.embedding(idx, self.embed)
        t = idx.shape[1]
        for b in self.blocks[:upto]:
            x = b(x, self.cos[:t], self.sin[:t])
        return x

    def top(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """The logits from the residual stream entering block `start`."""
        t = x.shape[1]
        for b in self.blocks[start:]:
            x = b(x, self.cos[:t], self.sin[:t])
        return F.linear(F.rms_norm(x, (x.shape[-1],), self.norm), self.embed)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.top(self.hidden(idx, 0), 0)
