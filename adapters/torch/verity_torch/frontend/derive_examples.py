"""Source modules for the derivation tests that are plain PyTorch: ordinary modules with NO Verity annotations.

* S1 (reference mode): `nn.Linear`, `MLP2` (Linear -> ReLU -> Linear), f32.
* HF-eager-style classes (`HFDecoderLayer`, `HFTinyLM`, `HFGreedy`) for the honest "different circuit" negative (three q/k/v
  Linears, decomposed RMSNorm / SDPA), and GPT-2-style classes (`Conv1D`, `GPT2Block`, `GPT2Greedy`).
* Negatives on plain PyTorch: an op with no rule, a hidden constant, an in-place op, a custom op with no binding, a returned
  weight, data-dependent control flow, and a Mixtral-style MoE gathered by its routing index.

`DERIVE_SOURCE` names each class by its old place in the vLLM adapter, so moving it here changed no derived program's id.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------
# S1: plain f32 modules
# ---------------------------------------------------------------------------------------------------------


class MLP2(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h, bias=bias)
        self.fc2 = nn.Linear(d_h, d_out, bias=bias)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TiedMLP(nn.Module):
    """fc2 shares fc1's weight (square): one tensor, two uses — the derived program must have ONE Input."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.fc2.weight = self.fc1.weight

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------------------------------------
# HF-eager-style layer: a DIFFERENT circuit (three Linears, decomposed norm, SDPA)
# ---------------------------------------------------------------------------------------------------------


class HFRMSNorm(nn.Module):
    def __init__(self, H: int, eps: float, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return self.weight * xf.to(x.dtype)


class HFRotaryEmbedding(nn.Module):
    """transformers `LlamaRotaryEmbedding` (default rope type) at the op interface: `inv_freq` non-persistent buffer,
    `forward(x, position_ids) -> (cos, sin)` as `[B, T, D]` tables in x's dtype (`cat(freqs, freqs)`, `attention_scaling`)."""
    inv_freq: torch.Tensor

    def __init__(self, D: int, base: float = 10000.0, attention_scaling: float = 1.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.int64).float() / D))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = attention_scaling

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def hf_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (hf_rotate_half(q) * sin)
    k_embed = (k * cos) + (hf_rotate_half(k) * sin)
    return q_embed, k_embed


class HFDecoderLayer(nn.Module):
    """HF Llama-style layer at the op interface (three projections, `LlamaRMSNorm`, `repeat_kv`, SDPA; with `rope=True` also
    `LlamaRotaryEmbedding` + `apply_rotary_pos_emb` on the `[1, NH, T, D]` layout HF uses). Input `[T, H]`."""

    def __init__(self, C: dict, dtype=torch.bfloat16, rope: bool = False) -> None:
        super().__init__()
        H, NH, KVH, D, INTER = C["H"], C["NH"], C["KVH"], C["D"], C["I"]
        self.NH, self.KVH, self.D = NH, KVH, D
        self.rotary_emb = HFRotaryEmbedding(D, C.get("ROPE_THETA", 10000.0)) if rope else None
        self.q_proj = nn.Linear(H, NH * D, bias=C.get("BIAS", True), dtype=dtype)
        self.k_proj = nn.Linear(H, KVH * D, bias=C.get("BIAS", True), dtype=dtype)
        self.v_proj = nn.Linear(H, KVH * D, bias=C.get("BIAS", True), dtype=dtype)
        self.o_proj = nn.Linear(NH * D, H, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(H, INTER, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(H, INTER, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(INTER, H, bias=False, dtype=dtype)
        self.input_layernorm = HFRMSNorm(H, C["EPS"], dtype)
        self.post_attention_layernorm = HFRMSNorm(H, C["EPS"], dtype)

    def forward(self, hidden_states, position_embeddings=None):
        T = hidden_states.shape[0]
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.rotary_emb is not None or position_embeddings is not None:
            q = self.q_proj(h).view(1, T, self.NH, self.D).transpose(1, 2)
            k = self.k_proj(h).view(1, T, self.KVH, self.D).transpose(1, 2)
            v = self.v_proj(h).view(1, T, self.KVH, self.D).transpose(1, 2)
            if position_embeddings is None:  # layer-local rotary (a single-layer example); HF computes it once at model level
                assert self.rotary_emb is not None
                position_ids = torch.arange(T).unsqueeze(0)
                position_embeddings = self.rotary_emb(h, position_ids)
            cos, sin = position_embeddings
            q, k = hf_apply_rotary_pos_emb(q, k, cos, sin)
            k = k.repeat_interleave(self.NH // self.KVH, dim=1)
            v = v.repeat_interleave(self.NH // self.KVH, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(T, self.NH * self.D)
        else:
            q = self.q_proj(h).view(T, self.NH, self.D).transpose(0, 1)
            k = self.k_proj(h).view(T, self.KVH, self.D).transpose(0, 1)
            v = self.v_proj(h).view(T, self.KVH, self.D).transpose(0, 1)
            k = k.repeat_interleave(self.NH // self.KVH, dim=0)
            v = v.repeat_interleave(self.NH // self.KVH, dim=0)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(T, self.NH * self.D)
        h = residual + self.o_proj(a)
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + h


class NegMoE(nn.Module):
    """Mixtral-style sparse MoE block: router -> softmax -> top-k -> the chosen experts' weights gathered by the routing INDEX.
    The expert selection is data-dependent (boundary 2: a fixed computation never derives from routing); refused."""

    def __init__(self, H: int = 16, INTER: int = 32, E: int = 4, k: int = 2) -> None:
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)
        self.w1 = nn.Parameter(torch.randn(E, INTER, H))
        self.w2 = nn.Parameter(torch.randn(E, H, INTER))
        self.k = k

    def forward(self, x):  # [T, H]
        w, idx = torch.topk(F.softmax(self.gate(x), dim=-1), self.k, dim=-1)
        out = torch.zeros_like(x)
        for j in range(self.k):
            e = idx[:, j]
            h = F.silu(torch.einsum("th,tih->ti", x, self.w1[e]))
            out = out + w[:, j:j + 1] * torch.einsum("ti,thi->th", h, self.w2[e])
        return out


class HFTinyLM(nn.Module):
    """HF `LlamaForCausalLM` at the op interface: `embed_tokens` (int64 ids), ONE model-level `rotary_emb` whose (cos, sin) feed
    every layer (HF's sharing), `NL` layers, final `norm`, `lm_head` tied to the embedding (`tie_word_embeddings`), greedy token
    of the last row. Input `ids[T]` int64 -> `i32[1]`."""

    def __init__(self, C: dict, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(C["V"], C["H"], dtype=dtype)
        self.rotary_emb = HFRotaryEmbedding(C["D"], C.get("ROPE_THETA", 10000.0))
        self.layers = nn.ModuleList([HFDecoderLayer(C, dtype, rope=False) for _ in range(C["NL"])])
        self.norm = HFRMSNorm(C["H"], C["EPS"], dtype)
        self.lm_head = nn.Linear(C["H"], C["V"], bias=False, dtype=dtype)
        self.lm_head.weight = self.embed_tokens.weight  # tied

    def forward(self, ids):
        T = ids.shape[0]
        h = self.embed_tokens(ids)
        position_embeddings = self.rotary_emb(h, torch.arange(T).unsqueeze(0))
        for layer in self.layers:
            h = layer(h, position_embeddings)
        logits = self.lm_head(self.norm(h[T - 1:T]))
        return torch.argmax(logits, dim=-1).to(torch.int32)


class HFGreedy(nn.Module):
    """Bounded HF request: LP int64 prompt tokens, STEPS greedy steps with the prefix recomputed each step (HF `generate` without
    the DynamicCache: the KV dependency is the dataflow itself; each step's rotary table covers its own T). Returns the STEPS+1
    selected tokens (i32)."""

    def __init__(self, C: dict, STEPS: int = 2, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.STEPS = STEPS
        self.lm = HFTinyLM(C, dtype)

    def forward(self, prompt):
        toks = prompt
        out = []
        for _ in range(self.STEPS + 1):
            nxt = self.lm(toks)  # i32[1]
            out.append(nxt)
            toks = torch.cat([toks, nxt])
        return torch.cat(out)


# ---------------------------------------------------------------------------------------------------------
# GPT-2 style (HF transformers.models.gpt2 classes at the op interface): LayerNorm, Conv1D (addmm), gelu_new
# ---------------------------------------------------------------------------------------------------------


class Conv1D(nn.Module):
    """transformers.pytorch_utils.Conv1D: `x @ W[K, N] + b` via `torch.addmm` (the weight is stored transposed)."""

    def __init__(self, nf: int, nx: int) -> None:
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class GPT2MLPBlock(nn.Module):
    """`x + mlp(ln_2(x))` of a GPT-2 block (the attention half is a decomposed pattern: see HF negatives)."""

    def __init__(self, H: int, INTER: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln_2 = nn.LayerNorm(H, eps=eps)
        self.c_fc = Conv1D(INTER, H)
        self.c_proj = Conv1D(H, INTER)

    def forward(self, x):
        h = self.ln_2(x)
        h = self.c_fc(h)
        h = F.gelu(h, approximate="tanh")  # HF NewGELUActivation == gelu tanh form
        h = self.c_proj(h)
        return x + h


class GPT2Attention(nn.Module):
    """GPT-2 `_attn` at the op interface: merged `c_attn` Conv1D, `q @ k^T / sqrt(d)`, causal `masked_fill(-inf)`,
    softmax in f32, `p @ v`, `c_proj`."""

    def __init__(self, H: int, NH: int) -> None:
        super().__init__()
        self.nh, self.d = NH, H // NH
        self.c_attn = Conv1D(3 * H, H)
        self.c_proj = Conv1D(H, H)

    def forward(self, x):  # [T, H]
        T = x.shape[0]
        q, k, v = self.c_attn(x).split(x.shape[-1], dim=-1)
        q = q.view(T, self.nh, self.d).transpose(0, 1)
        k = k.view(T, self.nh, self.d).transpose(0, 1)
        v = v.view(T, self.nh, self.d).transpose(0, 1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s.float(), dim=-1).to(x.dtype)
        y = (p @ v).transpose(0, 1).reshape(T, self.nh * self.d)
        return self.c_proj(y)


class GPT2Block(nn.Module):
    def __init__(self, H: int, NH: int, INTER: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(H, eps=eps)
        self.attn = GPT2Attention(H, NH)
        self.mlp = GPT2MLPBlock(H, INTER, eps)  # ln_2 + mlp + residual

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return self.mlp(x)


class GPT2TinyLM(nn.Module):
    """wte + wpe[arange(T)] -> NL blocks -> ln_f -> tied lm_head (logits [T, V]); int64 token ids as HF gives them."""

    def __init__(self, V: int = 32, H: int = 16, NH: int = 2, INTER: int = 32, NL: int = 2, MAXPOS: int = 8, eps: float = 1e-5) -> None:
        super().__init__()
        self.wte = nn.Embedding(V, H)
        self.wpe = nn.Embedding(MAXPOS, H)
        self.h = nn.ModuleList([GPT2Block(H, NH, INTER, eps) for _ in range(NL)])
        self.ln_f = nn.LayerNorm(H, eps=eps)
        self.lm_head = nn.Linear(H, V, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, tokens):  # [T] int64
        pos = torch.arange(tokens.shape[0], device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)
        for blk in self.h:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


class GPT2Greedy(nn.Module):
    """Bounded GPT-2 request: LP prompt tokens, STEPS greedy steps, the prefix recomputed each step (no host cache: the
    KV dependency is the dataflow itself). Returns the STEPS+1 selected tokens."""

    def __init__(self, STEPS: int = 2, **kw) -> None:
        super().__init__()
        self.STEPS = STEPS
        self.lm = GPT2TinyLM(**kw)

    def forward(self, prompt):
        toks = prompt
        out = []
        for _ in range(self.STEPS + 1):
            logits = self.lm(toks)
            nxt = torch.argmax(logits[-1], dim=-1)
            out.append(nxt)
            toks = torch.cat([toks, nxt.view(1)])
        return torch.stack(out)


# ---------------------------------------------------------------------------------------------------------
# negatives
# ---------------------------------------------------------------------------------------------------------


class NegUnknownOp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class NegHiddenConstant(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        scale = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # lifted constant: hidden state
        return self.fc(x) * scale


class NegInPlace(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        y = self.fc(x)
        y.add_(1.0)  # aten.add_: in-place without a rule
        return y


@functools.cache
def _mystery_op() -> torch.library.Library:
    """`someone_else::mystery`, defined at its first use and kept for the process: a custom op with a fake impl."""
    lib = torch.library.Library("someone_else", "FRAGMENT")
    lib.define("mystery(Tensor a) -> Tensor")
    torch.library.register_fake("someone_else::mystery", lambda a: torch.empty_like(a))
    return lib


class NegCustomOp(nn.Module):
    """A custom op with a fake impl but no Verity binding (`someone_else::mystery`)."""

    def __init__(self) -> None:
        super().__init__()
        _mystery_op()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return torch.ops.someone_else.mystery(self.fc(x))


class NegReturnsWeight(nn.Module):
    """Returns a parameter alongside the result (a returned Input: representable, but the program must say so)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(x), self.fc.bias


class NegCond(nn.Module):
    """Data-dependent control flow (`torch.cond` on a value)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        y = self.fc(x)
        return torch.cond(y.sum() > 0, lambda t: t + 1.0, lambda t: t - 1.0, (y,))


for _cls in (MLP2, TiedMLP, HFRMSNorm, HFRotaryEmbedding, HFDecoderLayer, NegMoE, HFTinyLM, HFGreedy, Conv1D, GPT2MLPBlock, GPT2Attention, GPT2Block, GPT2TinyLM, GPT2Greedy, NegUnknownOp, NegHiddenConstant, NegInPlace, NegCustomOp, NegReturnsWeight, NegCond):
    _cls.DERIVE_SOURCE = f"verity_vllm.program.frontend.derive_examples.{_cls.__qualname__}"
del _cls

__all__ = ["MLP2", "TiedMLP", "HFRMSNorm", "HFRotaryEmbedding", "hf_rotate_half", "hf_apply_rotary_pos_emb", "HFDecoderLayer", "NegMoE", "HFTinyLM", "HFGreedy", "Conv1D", "GPT2MLPBlock", "GPT2Attention", "GPT2Block", "GPT2TinyLM", "GPT2Greedy", "NegUnknownOp", "NegHiddenConstant", "NegInPlace", "NegCustomOp", "NegReturnsWeight", "NegCond"]
