"""Model configurations for rho_attack: the parameterized maps of one layer and what follows from them.

A map is a weight matrix ``d_in -> d_out``. ``count`` copies are held per layer (routed experts: all of them) and a token
passes through ``active`` of them. ``resid_in`` marks a map whose input is a normed residual stream (so its fresh part is
the residual's fresh correction ``F``); ``resid_out`` marks one that writes into the residual (so its output gradient is the
residual's gradient correction ``G``). The rest are interior: their input or output gradient lives only inside the layer.

Published configs: Qwen2.5 (HF ``config.json``), Kimi K2 (HF ``config.json``; every layer taken as MoE, the one dense
layer ignored). DeepSeek-V4-Pro from the finding note (arXiv 2606.19348: 61 layers, d = 7168, 384 routed experts with 6
active plus 1 shared, expert width 3072, 49B active of 1.6T, residual widened to 4d by hyper-connections). Its attention
is not MLA and its shapes aren't in the note, so it is four ``d x w`` maps whose width ``w`` makes the active count 49B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Map:
    name: str
    d_in: int
    d_out: int
    resid_in: bool
    resid_out: bool
    count: int = 1
    active: float = 1.0
    expert: bool = False
    mlp: bool = False  # in the layer's MLP (or MoE) half rather than its attention half
    softmax: bool = False  # the LM head: its output gradient is softmax - onehot, rebuilt from a per-token logsumexp

    @property
    def params(self) -> int:
        return self.d_in * self.d_out * self.count


@dataclass(frozen=True)
class Config:
    name: str
    layers: int
    d: int
    vocab: int
    maps: tuple[Map, ...]
    attn_width: int  # n_heads * head_dim of the score/value products, for attention FLOPs
    hc: int = 1  # residual streams (hyper-connections): the crossing is hc * d wide
    tied: bool = False
    n_experts: int = 0
    n_active: int = 0
    ctx_eff: int = 2048  # attended positions per token (causal average of a 4096 context, or a sparse top-k)
    extra: dict = field(default_factory=dict)

    @property
    def head(self) -> Map:
        return Map("head", self.d, self.vocab, True, False, softmax=True)

    @property
    def d_cross(self) -> int:
        return self.hc * self.d

    @property
    def h(self) -> float:
        """Raw bits of one token id."""
        return math.log2(self.vocab)

    @property
    def layer_params(self) -> int:
        return sum(m.params for m in self.maps)

    @property
    def params(self) -> int:
        emb = self.vocab * self.d * (1 if self.tied else 2)
        return self.layers * self.layer_params + emb

    @property
    def active_params(self) -> float:
        emb = self.vocab * self.d * (1 if self.tied else 2)
        return self.layers * sum(m.d_in * m.d_out * m.active for m in self.maps) + emb

    @property
    def f_layer(self) -> float:
        """Forward FLOPs per token in one layer: the maps plus the score and value products."""
        return 2 * sum(m.d_in * m.d_out * m.active for m in self.maps) + 4 * self.attn_width * self.ctx_eff

    @property
    def f_head(self) -> float:
        return 2 * self.d * self.vocab

    @property
    def f(self) -> float:
        """Forward FLOPs per token."""
        return self.layers * self.f_layer + self.f_head

    @property
    def values_per_token(self) -> float:
        """Counted values per token: outputs of parameterized maps, min(d_in, d_out) per token per map (finding §4)."""
        return self.layers * sum(min(m.d_in, m.d_out) * m.active for m in self.maps) + min(self.d, self.vocab)

    @property
    def d_eff(self) -> float:
        """Training FLOPs per counted value over 6 (so 6 d_eff = 3 f / values)."""
        return 3 * self.f / (6 * self.values_per_token)

    def phi(self, traffic: float = 1.0) -> float:
        """Honest FLOPs per unregistered input bit: a forward per raw token id (prefill), times a traffic factor."""
        return traffic * self.f / self.h


def dense(name: str, layers: int, d: int, ffn: int, heads: int, kv_heads: int, head_dim: int, vocab: int,
          tied: bool) -> Config:
    q, kv = heads * head_dim, kv_heads * head_dim
    maps = (
        Map("q", d, q, True, False), Map("k", d, kv, True, False), Map("v", d, kv, True, False),
        Map("o", q, d, False, True),
        Map("gate", d, ffn, True, False, mlp=True), Map("up", d, ffn, True, False, mlp=True),
        Map("down", ffn, d, False, True, mlp=True),
    )
    return Config(name, layers, d, vocab, maps, attn_width=q, tied=tied)


def moe(name: str, layers: int, d: int, attn: tuple[Map, ...], attn_width: int, n_experts: int, n_active: int,
        n_shared: int, e_ffn: int, vocab: int, hc: int = 1, ctx_eff: int = 2048) -> Config:
    expert = dict(count=n_experts, active=n_active, expert=True, mlp=True)
    shared = dict(count=n_shared, active=n_shared, mlp=True)
    maps = attn + (
        Map("router", d, n_experts, True, False, mlp=True),
        Map("s_gate", d, e_ffn, True, False, **shared), Map("s_up", d, e_ffn, True, False, **shared),
        Map("s_down", e_ffn, d, False, True, **shared),
        Map("e_gate", d, e_ffn, True, False, **expert), Map("e_up", d, e_ffn, True, False, **expert),
        Map("e_down", e_ffn, d, False, True, **expert),
    )
    return Config(name, layers, d, vocab, maps, attn_width=attn_width, hc=hc, n_experts=n_experts, n_active=n_active,
                  ctx_eff=ctx_eff)


def deepseek_v4_pro() -> Config:
    layers, d, vocab, e_ffn, active_total = 61, 7168, 129280, 3072, 49e9
    experts_active = layers * 7 * 3 * d * e_ffn
    attn_params = (active_total - 2 * vocab * d - experts_active) / layers
    w = int(round(attn_params / (4 * d)))
    attn = (Map("q", d, w, True, False), Map("k", d, w, True, False), Map("v", d, w, True, False),
            Map("o", w, d, False, True))
    # compressed sparse attention: a top-k of 2048 attended positions
    return moe("deepseek-v4-pro", layers, d, attn, attn_width=w, n_experts=384, n_active=6, n_shared=1, e_ffn=e_ffn,
               vocab=vocab, hc=4, ctx_eff=2048)


def kimi_k2() -> Config:
    d, heads = 7168, 64
    attn = (Map("q_a", d, 1536, True, False), Map("q_b", 1536, heads * 192, False, False),
            Map("kv_a", d, 576, True, False), Map("kv_b", 512, heads * 256, False, False),
            Map("o", heads * 128, d, False, True))
    return moe("kimi-k2", 61, d, attn, attn_width=heads * 192, n_experts=384, n_active=8, n_shared=1, e_ffn=2048,
               vocab=163840)


CONFIGS = {
    "qwen2.5-1.5b": lambda: dense("qwen2.5-1.5b", 28, 1536, 8960, 12, 2, 128, 151936, tied=True),
    "qwen2.5-7b": lambda: dense("qwen2.5-7b", 28, 3584, 18944, 28, 4, 128, 152064, tied=False),
    "qwen2.5-72b": lambda: dense("qwen2.5-72b", 80, 8192, 29568, 64, 8, 128, 152064, tied=False),
    "deepseek-v4-pro": deepseek_v4_pro,
    "kimi-k2": kimi_k2,
}


def get(name: str) -> Config:
    return CONFIGS[name]()
