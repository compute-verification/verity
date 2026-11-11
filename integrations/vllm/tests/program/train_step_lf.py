"""The `lf` model of the training-efficiency ladder (the `model.py` of training's slowdown code), for
the training-step frontend's tests and the grader's (`benchmarks/training_slowdown/grader`): a Llama-style decoder (RMSNorm,
SwiGLU, RoPE, no biases, tied input and output embeddings) over the GPT-2 vocabulary padded to 50,304, its loss the next-token
cross-entropy, and the plain AdamW the step is derived with (the ladder trains its matrices with Muon instead).

The ladder runs the forward under CUDA bf16 autocast, which for this model casts each linear's two operands to bf16 and leaves
every other op as it is (attention's operands are bf16 already). `LM(..., matmul_dtype=torch.bfloat16)` writes those casts out;
a trace of the step under CPU autocast fails in scaled-dot-product attention's backward. Without it the model is f32."""
from __future__ import annotations

import math
from dataclasses import dataclass

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


SIZES = {s.name: s for s in (Size("s", 512, 6, 8, 1408), Size("tiny", 64, 2, 2, 192))}


def linear(x: torch.Tensor, w: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """`F.linear(x, w)`, both operands cast to `dtype` first when one is given."""
    return F.linear(x, w) if dtype is None else F.linear(x.to(dtype), w.to(dtype))


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, std: float, matmul_dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.normal_(self.weight, std=std)
        self.matmul_dtype = matmul_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.matmul_dtype)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).type_as(x)


class Block(nn.Module):
    def __init__(self, size: Size, std: float, out_std: float, matmul_dtype: torch.dtype | None = None):
        super().__init__()
        d, dt = size.d_model, matmul_dtype
        self.n_head = size.n_head
        self.attn_norm = nn.Parameter(torch.ones(d))
        self.mlp_norm = nn.Parameter(torch.ones(d))
        self.wq, self.wk, self.wv = Linear(d, d, std, dt), Linear(d, d, std, dt), Linear(d, d, std, dt)
        self.wo = Linear(d, d, out_std, dt)
        self.w_gate, self.w_up = Linear(d, size.d_ff, std, dt), Linear(d, size.d_ff, std, dt)
        self.w_down = Linear(size.d_ff, d, out_std, dt)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = F.rms_norm(x, (d,), self.attn_norm)
        q, k, v = (w(h).view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.wo(a)
        h = F.rms_norm(x, (d,), self.mlp_norm)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class LM(nn.Module):
    def __init__(self, size: Size, ctx: int = CTX, vocab: int = VOCAB, matmul_dtype: torch.dtype | None = None):
        super().__init__()
        std, out_std = 0.02, 0.02 / math.sqrt(2 * size.n_layer)
        self.matmul_dtype = matmul_dtype
        self.embed = nn.Parameter(torch.empty(vocab, size.d_model))
        nn.init.normal_(self.embed, std=std)
        self.blocks = nn.ModuleList(Block(size, std, out_std, matmul_dtype) for _ in range(size.n_layer))
        self.norm = nn.Parameter(torch.ones(size.d_model))
        hd = size.d_model // size.n_head
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        angles = torch.outer(torch.arange(ctx, dtype=torch.float32), freqs)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        x = F.embedding(idx, self.embed)
        cos, sin = self.cos[:t], self.sin[:t]
        for b in self.blocks:
            x = b(x, cos, sin)
        return linear(F.rms_norm(x, (x.shape[-1],), self.norm), self.embed, self.matmul_dtype)


def loss_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy of a batch of token rows: the model reads each row's first T tokens and predicts the next."""
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))


def square_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The sum of the logits' squares: the tests' loss of one element-wise op and one sum, with no cross-entropy."""
    return model(x[:, :-1]).float().square().sum()


#: the ladder's AdamW hyperparameters (its embedding group's betas and eps), here over every parameter
ADAMW = {"lr": 6e-3, "betas": (0.9, 0.95), "eps": 1e-10, "weight_decay": 0.01}


def meta_step(size: str = "s", batch: int = 2, ctx: int = CTX, vocab: int = VOCAB,
              matmul_dtype: torch.dtype | None = None) -> tuple[LM, torch.Tensor]:
    """The model at `size` and a batch of `batch` rows of `ctx + 1` tokens, on the meta device (shapes only)."""
    with torch.device("meta"):
        model = LM(SIZES[size], ctx=ctx, vocab=vocab, matmul_dtype=matmul_dtype)
    return model, torch.empty(batch, ctx + 1, dtype=torch.long, device="meta")
