"""Toy models for the capture tests: a 2-layer MLP, a tiny decoder (pre-norm, causal SDPA, GELU MLP, untied head), LoRA
adapters on its q and v projections, and a small conv net."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def mlp(F_in: int = 8, H: int = 16, O: int = 4) -> nn.Module:
    return nn.Sequential(nn.Linear(F_in, H), nn.ReLU(), nn.Linear(H, O))


class Block(nn.Module):
    def __init__(self, H: int, heads: int, I: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(H), nn.LayerNorm(H)
        self.q, self.k, self.v, self.o = (nn.Linear(H, H, bias=False) for _ in range(4))
        self.up, self.down = nn.Linear(H, I, bias=False), nn.Linear(I, H, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        h = self.ln1(x)
        q, k, v = (f(h).view(B, S, self.heads, H // self.heads).transpose(1, 2) for f in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, H))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self, V: int = 32, H: int = 16, heads: int = 2, I: int = 32, L: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.blocks = nn.ModuleList(Block(H, heads, I) for _ in range(L))
        self.ln_f = nn.LayerNorm(H)
        self.head = nn.Linear(H, V, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def lm_loss(model: nn.Module, batch) -> torch.Tensor:
    ids, labels = batch
    return F.cross_entropy(model(ids).flatten(0, 1), labels.flatten())


class LoRA(nn.Module):
    """``base(x) + s · B(A(x))`` with the base frozen."""

    def __init__(self, base: nn.Linear, r: int) -> None:
        super().__init__()
        self.base = base
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        self.s = 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * self.s


def lora(model: TinyLM, r: int) -> TinyLM:
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.blocks:
        b.q, b.v = LoRA(b.q, r), LoRA(b.v, r)
    return model


def conv_net(C: int = 3, K: int = 8, side: int = 8, O: int = 4) -> nn.Module:
    return nn.Sequential(nn.Conv2d(C, K, 3), nn.ReLU(), nn.Flatten(), nn.Linear(K * (side - 2) ** 2, O))
