"""Source modules for the derivation tests: ordinary PyTorch modules with NO Verity annotations.

* S1 (reference mode): `nn.Linear`, `MLP2` (Linear -> ReLU -> Linear), f32.
* S2/S3 (profile mode): vLLM-style classes — merged QKV / merged gate_up linears, in-place kernel interface
  (`rules/vllm_iface.py`), `unified_attention` with an implicit per-layer KV cache — `DecoderLayer`, `Model`,
  `GreedyServe` (prefill + STEPS greedy decode steps, an ordinary Python loop).
* HF-eager-style classes (`HFDecoderLayer`) for the honest "different circuit" negative (three q/k/v Linears,
  decomposed RMSNorm / SDPA).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from verity_vllm.program.frontend.rules.vllm_iface import iface  # registers the stand-in ops


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
# vLLM-style building blocks (the classes vllm/model_executor/layers/* expose, at the kernel interface)
# ---------------------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """vllm.model_executor.layers.layernorm.RMSNorm (forward_cuda): fused add+norm when a residual is given."""

    def __init__(self, hidden_size: int, eps: float, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        if residual is not None:
            torch.ops.vllm_iface.fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            return x, residual
        out = torch.empty_like(x)
        torch.ops.vllm_iface.rms_norm(out, x, self.weight, self.variance_epsilon)
        return out


class LinearNoTP(nn.Module):
    """A (Q)KV/MergedColumn/RowParallelLinear at TP=1: `F.linear(x, weight, bias)`; returns (out, None) like vLLM."""

    def __init__(self, in_f: int, out_f: int, bias: bool, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_f, dtype=dtype)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias), None


class RotaryEmbedding(nn.Module):
    """vllm.model_executor.layers.rotary_embedding.RotaryEmbedding (NeoX): bf16 cos/sin cache buffer computed at init
    (`_compute_cos_sin_cache`), applied in place by `_C.rotary_embedding`."""

    def __init__(self, head_size: int, max_position: int, base: float, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.head_size = head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        torch.ops.vllm_iface.rotary_embedding(positions, query, key, self.head_size, self.cos_sin_cache, True)
        return query, key


_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}


def get_rope(head_size: int, max_position: int, base: float, dtype=torch.bfloat16) -> RotaryEmbedding:
    """vllm.model_executor.layers.rotary_embedding.get_rope: ONE RotaryEmbedding instance (one cos/sin table) is shared
    by every layer with the same parameters — the table is tied state, not per-layer state."""
    key = (head_size, max_position, float(base), dtype, str(torch.get_default_device()) if hasattr(torch, "get_default_device") else "")
    r = _ROPE_DICT.get(key)
    if r is None:
        r = RotaryEmbedding(head_size, max_position, base, dtype)
        _ROPE_DICT[key] = r
    return r


class SiluAndMul(nn.Module):
    def forward(self, x):
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        torch.ops.vllm_iface.silu_and_mul(out, x)
        return out


class Attention(nn.Module):
    """vllm.attention.layer.Attention (V1): `unified_attention(q, k, v, layer_name)`; the KV cache is forward-context state."""

    def __init__(self, layer_name: str) -> None:
        super().__init__()
        self.layer_name = layer_name

    def forward(self, q, k, v, pad=None):
        out = torch.ops.vllm_iface.unified_attention(q, k, v, self.layer_name)
        if pad is not None:
            # padded engine (serve3-spec §6): after the step's attention (which read the raw row), the appended cache
            # row is replaced by `live ? row : 0` so a finished request's steps append the zero row
            live, zero, slot = pad
            torch.ops.vllm_iface.unified_cache_write(torch.where(live, k, zero), torch.where(live, v, zero), self.layer_name, slot)
        return out


class DecoderLayer(nn.Module):
    """vLLM LlamaDecoderLayer / Qwen2DecoderLayer shape: merged qkv_proj, o_proj, merged gate_up_proj, down_proj."""

    def __init__(self, C: dict, layer_idx: int, dtype=torch.bfloat16) -> None:
        super().__init__()
        H, NH, KVH, D, I = C["H"], C["NH"], C["KVH"], C["D"], C["I"]
        self.NH, self.KVH, self.D = NH, KVH, D
        self.qkv_proj = LinearNoTP(H, (NH + 2 * KVH) * D, bias=C.get("BIAS", True), dtype=dtype)
        self.o_proj = LinearNoTP(NH * D, H, bias=False, dtype=dtype)
        self.rotary_emb = get_rope(D, C["MAX_POS"], C["ROPE_THETA"], dtype=dtype)
        self.attn = Attention(f"model.layers.{layer_idx}.self_attn.attn")
        self.gate_up_proj = LinearNoTP(H, 2 * I, bias=False, dtype=dtype)
        self.act_fn = SiluAndMul()
        self.down_proj = LinearNoTP(I, H, bias=False, dtype=dtype)
        self.input_layernorm = RMSNorm(H, C["EPS"], dtype=dtype)
        self.post_attention_layernorm = RMSNorm(H, C["EPS"], dtype=dtype)

    def forward(self, positions, hidden_states, residual, pad=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.NH * self.D, self.KVH * self.D, self.KVH * self.D], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, pad)
        hidden_states, _ = self.o_proj(attn_output)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        gate_up, _ = self.gate_up_proj(hidden_states)
        x = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(x)
        return hidden_states, residual


class SingleLayer(nn.Module):
    """One decoder layer as a standalone module (S2): positions T0..T0+T-1 are static (an attribute)."""

    def __init__(self, C: dict, layer_idx: int, pos0: int = 0, first: bool = True) -> None:
        super().__init__()
        self.layer = DecoderLayer(C, layer_idx)
        self.pos0 = pos0
        self.first = first

    def forward(self, hidden_states, residual=None):
        positions = torch.arange(self.pos0, self.pos0 + hidden_states.shape[0])
        return self.layer(positions, hidden_states, None if self.first else residual)


class Model(nn.Module):
    def __init__(self, C: dict, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(C["V"], C["H"], dtype=dtype)
        self.layers = nn.ModuleList([DecoderLayer(C, i, dtype) for i in range(C["NL"])])
        self.norm = RMSNorm(C["H"], C["EPS"], dtype=dtype)

    def forward(self, input_ids, positions, last_only: bool = True, pad=None):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, pad)
        if last_only:
            hidden_states, residual = hidden_states[-1:], residual[-1:]
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CausalLM(nn.Module):
    def __init__(self, C: dict, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.model = Model(C, dtype)
        self.lm_head = nn.Linear(C["H"], C["V"], bias=False, dtype=dtype)
        if C.get("tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions, last_only: bool = True, sampled_rows_only: bool = False, pad=None):
        h = self.model(input_ids, positions, last_only, pad)
        if sampled_rows_only and not last_only:
            h = h[-1:]  # vLLM's LogitsProcessor: logits for the sampled (last) row only
        return self.lm_head(h)


class GreedyServe(nn.Module):
    """An ordinary serving loop: prefill LP prompt tokens, then STEPS greedy decode steps (argmax), each decode step
    feeding the sampled token back. Token ids are int32 (the certified program's I32 tokens). `final_norm_all_rows`
    reproduces vLLM's model forward (final norm over every prefill row; only the last row is sampled) — extra dead
    work the certified program does not contain."""

    def __init__(self, C: dict, steps: int, final_norm_all_rows: bool = False, vllm_rows: bool = False) -> None:
        super().__init__()
        self.lm = CausalLM(C)
        self.steps = steps
        # vllm_rows: exactly vLLM's row policy — final norm over EVERY prefill row (model forward), LM head over the
        # sampled row only (LogitsProcessor gathers selected_token_indices first). CAPCORR's 01:15 finding: the capture
        # commits the norm outputs of the 31 non-final prefill rows, which Serve_v1 does not contain.
        self.last_only = not (final_norm_all_rows or vllm_rows)
        self.sampled_rows_only = vllm_rows

    def forward(self, prompt):
        LP = prompt.shape[0]
        positions = torch.arange(0, LP)
        logits = self.lm(prompt, positions, self.last_only, self.sampled_rows_only)
        tok = torch.argmax(logits[-1], dim=-1)
        toks = [tok]
        for s in range(self.steps):
            positions = torch.arange(LP + s, LP + s + 1)
            logits = self.lm(tok.view(1), positions, self.last_only, self.sampled_rows_only)
            tok = torch.argmax(logits[-1], dim=-1)
            toks.append(tok)
        return torch.stack(toks)


class StochasticServe(nn.Module):
    """`GreedyServe` with the seeded temperature + top-p select of vLLM's sampler as the stand-in op
    `vllm_iface.gumbel_topp_token_select(logits, seed, temperature, top_p, pos, splits)` — the shape of `cb_a.vllm_meta.StochasticRequest`
    over the example CausalLM.  `seed` (int64 [1]) and `splits` (int32 [steps+1]: the derived split count
    per sampling event, S_t = SplitsFor_v1(|live(t)|, num_SMs)) are forward INPUTS; temperature / top_p are constructor constants;
    `pos` is the logits row's position (LP-1+k for the k-th sample).  `literal_seed` (negative): the seed created inside forward —
    hidden state, refused.  `literal_splits` (negative): S as a tensor literal / a Python int inside forward — refused likewise (S is
    the workload's derived value, never the wrapper's).  `temperature=0` (negative): the rule refuses (greedy is TokenSelect_v1).
    `constant_S`: the registered split count, passed to every select as `S` (no `splits` input)."""

    def __init__(self, C: dict, steps: int, temperature: float = 0.8, top_p: float = 0.95, literal_seed: bool = False,
                 literal_splits: int | None = None, constant_S: int | None = None) -> None:
        super().__init__()
        self.lm = CausalLM(C)
        self.steps = steps
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.literal_seed = literal_seed
        self.literal_splits = literal_splits
        self.constant_S = constant_S

    def _select(self, logits, seed, pos: int, entry):
        return iface.gumbel_topp_token_select(logits[-1:], seed, self.temperature, self.top_p, pos, entry, S=self.constant_S)

    def forward(self, prompt, seed, splits=None):
        if self.literal_seed:
            seed = torch.tensor([1234], dtype=torch.int64)
        if self.literal_splits is not None:
            splits = torch.full((self.steps + 1,), int(self.literal_splits), dtype=torch.int32)
        entry = (lambda s: None) if self.constant_S is not None else (lambda s: splits[s:s + 1])
        LP = prompt.shape[0]
        logits = self.lm(prompt, torch.arange(0, LP))
        tok = self._select(logits, seed, LP - 1, entry(0))
        toks = [tok]
        for s in range(self.steps):
            logits = self.lm(tok, torch.arange(LP + s, LP + s + 1))
            tok = self._select(logits, seed, LP + s, entry(s + 1))
            toks.append(tok)
        return torch.cat(toks)


class PaddedGreedyServe(nn.Module):
    """A padded, early-terminating greedy serving loop as an ordinary module (serve3-spec §3–§8): prefill LP tokens,
    then a FIXED number `d_max` of decode steps whatever the tokens are. The realized length is a *value*: one live bit
    per sampling event, `live_0 = 1`, `live_{g+1} = live_g & ~(y_g == EOS)` on the raw sample; the emitted slot is
    `live_g ? y_g : PAD(0)`; the next step consumes the muxed slot; every decode step's appended K/V row is
    `live ? row : 0` against one zero per step (the cache clear of a finished request). Returns (slots, live flags),
    `D_MAX+1 = d_max+1` each; only the slots are designated outputs (GUIDANCE §32.6).

    `guard_inside` (negative): mux the hidden state of the step body too — a program the spec forbids (§6: nothing
    inside the step is guarded); it derives (honestly) as its own flat program and is NOT `Serve_v3`."""

    def __init__(self, C: dict, d_max: int, eos_id: int, guard_inside: bool = False) -> None:
        super().__init__()
        self.lm = CausalLM(C)
        self.d_max = d_max
        self.eos_id = eos_id
        self.guard_inside = guard_inside

    def _event(self, logits, live):
        y = torch.argmax(logits[-1], dim=-1).view(1)  # the raw sample (i32)
        tok = torch.where(live, y, 0)  # PAD_TOKEN_ID = 0; keyed on the event's liveness
        live = live & ~(y == self.eos_id)  # absorbing
        return tok, live

    def forward(self, prompt):
        LP = prompt.shape[0]
        logits = self.lm(prompt, torch.arange(0, LP))
        live = torch.ones(1, dtype=torch.bool)  # live_0: a request always samples y_0
        tok, live = self._event(logits, live)
        toks, lives = [tok], [live]
        for s in range(1, self.d_max + 1):  # decode step s at position LP+s-1
            slot = LP + s - 1
            zero = torch.zeros((), dtype=torch.bfloat16)  # the step's one zero (broadcast by every mux)
            logits = self.lm(tok, torch.arange(slot, slot + 1), pad=(live, zero, slot))
            if self.guard_inside:
                logits = torch.where(live, logits, zero)
            tok, live = self._event(logits, live)
            toks.append(tok)
            lives.append(live)
        return torch.cat(toks), torch.cat(lives)


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
        H, NH, KVH, D, I = C["H"], C["NH"], C["KVH"], C["D"], C["I"]
        self.NH, self.KVH, self.D = NH, KVH, D
        self.rotary_emb = HFRotaryEmbedding(D, C.get("ROPE_THETA", 10000.0)) if rope else None
        self.q_proj = nn.Linear(H, NH * D, bias=C.get("BIAS", True), dtype=dtype)
        self.k_proj = nn.Linear(H, KVH * D, bias=C.get("BIAS", True), dtype=dtype)
        self.v_proj = nn.Linear(H, KVH * D, bias=C.get("BIAS", True), dtype=dtype)
        self.o_proj = nn.Linear(NH * D, H, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(H, I, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(H, I, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(I, H, bias=False, dtype=dtype)
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

    def __init__(self, H: int = 16, I: int = 32, E: int = 4, k: int = 2) -> None:
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)
        self.w1 = nn.Parameter(torch.randn(E, I, H))
        self.w2 = nn.Parameter(torch.randn(E, H, I))
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

    def __init__(self, H: int, I: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln_2 = nn.LayerNorm(H, eps=eps)
        self.c_fc = Conv1D(I, H)
        self.c_proj = Conv1D(H, I)

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
    def __init__(self, H: int, NH: int, I: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(H, eps=eps)
        self.attn = GPT2Attention(H, NH)
        self.mlp = GPT2MLPBlock(H, I, eps)  # ln_2 + mlp + residual

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return self.mlp(x)


class GPT2TinyLM(nn.Module):
    """wte + wpe[arange(T)] -> NL blocks -> ln_f -> tied lm_head (logits [T, V]); int64 token ids as HF gives them."""

    def __init__(self, V: int = 32, H: int = 16, NH: int = 2, I: int = 32, NL: int = 2, MAXPOS: int = 8, eps: float = 1e-5) -> None:
        super().__init__()
        self.wte = nn.Embedding(V, H)
        self.wpe = nn.Embedding(MAXPOS, H)
        self.h = nn.ModuleList([GPT2Block(H, NH, I, eps) for _ in range(NL)])
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


class NegDataDependent(nn.Module):
    """A computed token (TokenSelect) used as an index into a *value*: a data-dependent gather (the row read
    depends on the data). Embedding rows are the only table a computed token may select (Embedding kind)."""

    def __init__(self, V: int = 8, H: int = 8) -> None:
        super().__init__()
        self.fc = LinearNoTP(H, V, bias=False)

    def forward(self, x):
        logits, _ = self.fc(x)
        tok = torch.argmax(logits[-1], dim=-1).view(1)
        return logits[tok]


class NegMutatesWeight(nn.Module):
    """`rms_norm` writes its `out` operand — here a registered buffer (host state written by the forward)."""

    def __init__(self, T: int = 2, H: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.register_buffer("scratch", torch.zeros(T, H, dtype=torch.bfloat16))

    def forward(self, x):
        torch.ops.vllm_iface.rms_norm(self.scratch, x, self.weight, 1e-5)
        return self.scratch


class NegStaleAlias(nn.Module):
    """Reads `qkv` (the whole tensor) after rotary_embedding rotated q and k in place through views."""

    def __init__(self, C: dict) -> None:
        super().__init__()
        H, NH, KVH, D = C["H"], C["NH"], C["KVH"], C["D"]
        self.NH, self.KVH, self.D = NH, KVH, D
        self.qkv_proj = LinearNoTP(H, (NH + 2 * KVH) * D, bias=False)
        self.rotary_emb = RotaryEmbedding(D, C["MAX_POS"], C["ROPE_THETA"])

    def forward(self, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.NH * self.D, self.KVH * self.D, self.KVH * self.D], dim=-1)
        positions = torch.arange(0, hidden_states.shape[0])
        self.rotary_emb(positions, q, k)
        return qkv  # stale: the source reads the pre-mutation alias


class NegCustomOp(nn.Module):
    """A custom op with a fake impl but no Verity binding (`examples.py`'s `someone_else::mystery`)."""

    def __init__(self) -> None:
        super().__init__()
        from verity_vllm.program.frontend import examples  # noqa: F401  (defines someone_else::mystery)
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


__all__ = ["MLP2", "TiedMLP", "RMSNorm", "LinearNoTP", "RotaryEmbedding", "SiluAndMul", "Attention", "DecoderLayer", "SingleLayer", "Model",
           "CausalLM", "GreedyServe", "HFRMSNorm", "HFDecoderLayer", "NegUnknownOp", "NegHiddenConstant", "NegInPlace", "NegDataDependent",
           "NegCustomOp", "NegReturnsWeight", "NegCond", "Conv1D", "GPT2MLPBlock", "GPT2Attention", "GPT2Block", "GPT2TinyLM", "GPT2Greedy",
           "NegMutatesWeight", "NegStaleAlias"]
