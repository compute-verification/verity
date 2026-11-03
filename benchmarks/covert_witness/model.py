"""The `lf` model, learning_frontier's dense decoder: Llama-style (RMSNorm, SwiGLU, RoPE, no biases), context 1024, the GPT-2
vocabulary padded to 50,304 (data v3's starts: bpe32k's 32,000, `Size.config(vocab=32000)`), input and output embeddings
tied. Its parameter names are the start registry's tensor names.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

VOCAB = 50304
CTX = 1024
ROPE_THETA = 10000.0


@dataclass(frozen=True)
class Size:
    name: str
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int

    def config(self, vocab: int = VOCAB) -> dict:
        return {"family": "lf", **asdict(self), "vocab": vocab, "ctx": CTX, "tied": True}


SIZES = {s.name: s for s in (
    Size("s", 512, 6, 8, 1408),
    Size("m", 768, 8, 12, 2048),
    Size("l", 1024, 12, 16, 2816),
    Size("tiny", 64, 2, 2, 192),
)}


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, std: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.normal_(self.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class Block(nn.Module):
    def __init__(self, size: Size, std: float, out_std: float):
        super().__init__()
        d = size.d_model
        self.n_head = size.n_head
        self.attn_norm = nn.Parameter(torch.ones(d))
        self.mlp_norm = nn.Parameter(torch.ones(d))
        self.wq, self.wk, self.wv = Linear(d, d, std), Linear(d, d, std), Linear(d, d, std)
        self.wo = Linear(d, d, out_std)
        self.w_gate, self.w_up = Linear(d, size.d_ff, std), Linear(d, size.d_ff, std)
        self.w_down = Linear(size.d_ff, d, out_std)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = F.rms_norm(x, (d,), self.attn_norm)
        q, k, v = (w(h).view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.wo(a)
        h = F.rms_norm(x, (d,), self.mlp_norm)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).type_as(x)


class LM(nn.Module):
    def __init__(self, size: Size, ctx: int = CTX, vocab: int = VOCAB):
        super().__init__()
        self.size, self.vocab = size, vocab
        std, out_std = 0.02, 0.02 / math.sqrt(2 * size.n_layer)
        self.embed = nn.Parameter(torch.empty(vocab, size.d_model))
        nn.init.normal_(self.embed, std=std)
        self.blocks = nn.ModuleList(Block(size, std, out_std) for _ in range(size.n_layer))
        self.norm = nn.Parameter(torch.ones(size.d_model))
        hd = size.d_model // size.n_head
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        angles = torch.outer(torch.arange(ctx, dtype=torch.float32), freqs)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def hidden(self, idx: torch.Tensor, upto: int | None = None) -> torch.Tensor:
        """The residual stream after the first `upto` blocks (all of them by default)."""
        t = idx.shape[1]
        x = F.embedding(idx, self.embed)
        cos, sin = self.cos[:t], self.sin[:t]
        for b in self.blocks[:upto]:
            x = b(x, cos, sin)
        return x

    def top(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """Logits from the residual stream entering block `start`."""
        t = x.shape[1]
        cos, sin = self.cos[:t], self.sin[:t]
        for b in self.blocks[start:]:
            x = b(x, cos, sin)
        x = F.rms_norm(x, (x.shape[-1],), self.norm)
        return F.linear(x, self.embed)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.top(self.hidden(idx, 0), 0)
