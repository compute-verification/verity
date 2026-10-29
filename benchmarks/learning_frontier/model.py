"""The learning-frontier model: a Llama-style decoder (RMSNorm, SwiGLU, RoPE, no biases) with tied input and output
embeddings, and optional per-window rank-r factor pairs on every matrix.

With `rank > 0`, each matrix W (d_out x d_in) is used as W + s * B @ A with A (r x d_in) and B (d_out x r) the window's trainable
factors and W frozen; the tied embedding E (V x d) is one more such matrix, its factor pair serving both the lookup and the
output head. `merge()` folds s * B @ A into W and draws the next window's factors: A ~ N(0, 1/d_in) from a generator seeded by
(seed, window), B = 0, so every window starts from the merged model unchanged. The scale is s = sqrt(d_in / r): at a window's
first step, when B = 0, an AdamW step moves each entry of W by about the learning rate, as a dense step does, so one
learning-rate grid serves both families.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

VOCAB = 50304  # GPT-2's 50257 tokens, padded to a multiple of 64
CTX = 1024
ROPE_THETA = 10000.0


@dataclass(frozen=True)
class Size:
    name: str
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int


SIZES = {s.name: s for s in (
    Size("s", 512, 6, 8, 1408),
    Size("m", 768, 8, 12, 2048),
    Size("l", 1024, 12, 16, 2816),
    Size("tiny", 64, 2, 2, 192),
)}


class Linear(nn.Module):
    """y = x W^T, plus s (x A^T) B^T while a rank-r factor pair is attached."""

    def __init__(self, d_in: int, d_out: int, std: float):
        super().__init__()
        self.d_in, self.d_out, self.rank, self.scale = d_in, d_out, 0, 0.0
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.normal_(self.weight, std=std)

    def attach(self, rank: int) -> None:
        self.rank, self.scale = rank, math.sqrt(self.d_in / rank)
        self.lora_a = nn.Parameter(torch.zeros(rank, self.d_in, device=self.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(self.d_out, rank, device=self.weight.device))
        self.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        if self.rank:
            y = y + self.scale * F.linear(F.linear(x, self.lora_a), self.lora_b)
        return y


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
        self.size, self.vocab, self.rank, self.emb_scale = size, vocab, 0, 0.0
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

    def linears(self) -> list[Linear]:
        return [m for m in self.modules() if isinstance(m, Linear)]

    def norms(self) -> list[nn.Parameter]:
        return [self.norm, *(p for b in self.blocks for p in (b.attn_norm, b.mlp_norm))]

    def attach(self, rank: int, seed: int) -> None:
        """Freeze every matrix and give it a rank-`rank` factor pair; draw window 0's factors."""
        self.rank, self.emb_scale = rank, math.sqrt(self.size.d_model / rank)
        for m in self.linears():
            m.attach(rank)
        self.emb_a = nn.Parameter(torch.zeros(rank, self.size.d_model, device=self.embed.device))
        self.emb_b = nn.Parameter(torch.zeros(self.vocab, rank, device=self.embed.device))
        self.embed.requires_grad_(False)
        self.draw(seed, 0)

    def factor_pairs(self) -> list[tuple[torch.Tensor, nn.Parameter, nn.Parameter, float]]:
        """(frozen matrix, A, B, scale) for every factor pair, the embedding's last."""
        return [(m.weight, m.lora_a, m.lora_b, m.scale) for m in self.linears()] + [
            (self.embed, self.emb_a, self.emb_b, self.emb_scale)]

    @torch.no_grad()
    def draw(self, seed: int, window: int) -> None:
        g = torch.Generator(device=self.embed.device).manual_seed(seed * 1_000_003 + window)
        for _, a, b, _ in self.factor_pairs():
            a.copy_(torch.randn(a.shape, generator=g, device=a.device) / math.sqrt(a.shape[1]))
            b.zero_()

    @torch.no_grad()
    def merge(self, seed: int, next_window: int) -> None:
        for w, a, b, s in self.factor_pairs():
            w.add_(s * (b @ a))
        self.draw(seed, next_window)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        x = F.embedding(idx, self.embed)
        if self.rank:
            x = x + self.emb_scale * (F.embedding(idx, self.emb_b) @ self.emb_a)
        cos, sin = self.cos[:t], self.sin[:t]
        for b in self.blocks:
            x = b(x, cos, sin)
        x = F.rms_norm(x, (x.shape[-1],), self.norm)
        logits = F.linear(x, self.embed)
        if self.rank:
            logits = logits + self.emb_scale * F.linear(F.linear(x, self.emb_a), self.emb_b)
        return logits


def matrix_shapes(size: Size, vocab: int = VOCAB) -> list[tuple[int, int]]:
    """Every matrix's (d_in, d_out): each layer's attention q, k, v, o and MLP gate, up, down, then the tied embedding (V x d)."""
    d = size.d_model
    layer = [(d, d)] * 4 + [(d, size.d_ff)] * 2 + [(size.d_ff, d)]
    return layer * size.n_layer + [(vocab, d)]


def counts(size: Size, rank: int = 0, ctx: int = CTX, vocab: int = VOCAB) -> dict:
    """Parameter counts and the analytic training FLOPs per token.

    Matrices count d_in * d_out each; the tied head is a V x d matmul. Dense: 6 FLOPs per matrix entry per token (forward,
    input gradient, weight gradient). Per-window low rank: the frozen matrices skip their weight gradient (4 per entry) and each
    factor pair costs 6 per factor entry, the embedding's counted on the head side (its lookup is a gather). Causal attention,
    QK^T and AV over half the context on average: 2 * ctx * d per layer forward, 6 * n_layer * ctx * d with the backward.
    """
    d, L = size.d_model, size.n_layer
    shapes = matrix_shapes(size, vocab)
    n_mat = sum(i * o for i, o in shapes[:-1])
    n_norm = (2 * L + 1) * d
    n_embed = vocab * d
    attn = 6 * L * ctx * d
    out = {"non_embedding": n_mat + n_norm, "matrices": n_mat, "norms": n_norm, "embedding": n_embed,
           "total": n_mat + n_norm + n_embed, "attention_flops_per_token": attn,
           "flops_per_token_6n": 6 * (n_mat + n_norm),
           "dense_flops_per_token": 6 * (n_mat + n_embed) + attn}
    if rank:
        n_fac = sum(rank * (i + o) for i, o in shapes)
        out |= {"rank": rank, "factors": n_fac, "trained_per_window": n_fac + n_norm,
                "lowrank_flops_per_token": 4 * (n_mat + n_embed) + 6 * n_fac + attn}
    return out
