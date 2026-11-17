"""Source modules for the derivation tests that are vLLM-style, and the plain-PyTorch ones of
`verity_torch.frontend.derive_examples`, which this module passes on so its importers read every example from one module,
but for its sparse MoE (`NegMoE`), whose name is a vLLM layer class's and so no literal here (P8).

* S2/S3 (profile mode): vLLM-style classes — merged QKV / merged gate_up linears, in-place kernel interface
  (`rules/vllm_iface.py`), `unified_attention` with an implicit per-layer KV cache — `DecoderLayer`, `Model`,
  `GreedyServe` (prefill + STEPS greedy decode steps, an ordinary Python loop), `StochasticServe` and `PaddedGreedyServe`.
* Negatives on the kernel interface: a data-dependent gather of a computed token, a kernel writing a buffer, a stale alias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from verity_torch.frontend.derive_examples import (
    MLP2,
    Conv1D,
    GPT2Attention,
    GPT2Block,
    GPT2Greedy,
    GPT2MLPBlock,
    GPT2TinyLM,
    HFDecoderLayer,
    HFGreedy,
    HFRMSNorm,
    HFRotaryEmbedding,
    HFTinyLM,
    NegCond,
    NegCustomOp,
    NegHiddenConstant,
    NegInPlace,
    NegReturnsWeight,
    NegUnknownOp,
    TiedMLP,
    hf_apply_rotary_pos_emb,
    hf_rotate_half,
)

from verity_vllm.program.frontend.rules.vllm_iface import (
    iface,  # registers the stand-in ops
)

#: a padded engine's attention operand (`Attention.forward`): (live, the zero row, the cache slot), or None unpadded.
_Pad = tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None

# ---------------------------------------------------------------------------------------------------------
# vLLM-style building blocks (the classes vllm/model_executor/layers/* expose, at the kernel interface)
# ---------------------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """vllm.model_executor.layers.layernorm.RMSNorm (forward_cuda): fused add+norm when a residual is given."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            torch.ops.vllm_iface.fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            return x, residual
        out = torch.empty_like(x)
        torch.ops.vllm_iface.rms_norm(out, x, self.weight, self.variance_epsilon)
        return out


class LinearNoTP(nn.Module):
    """A (Q)KV/MergedColumn/RowParallelLinear at TP=1: `F.linear(x, weight, bias)`; returns (out, None) like vLLM."""

    def __init__(self, in_f: int, out_f: int, bias: bool, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_f, dtype=dtype)) if bias else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight, self.bias), None


class RotaryEmbedding(nn.Module):
    """vllm.model_executor.layers.rotary_embedding.RotaryEmbedding (NeoX): bf16 cos/sin cache buffer computed at init
    (`_compute_cos_sin_cache`), applied in place by `_C.rotary_embedding`."""

    def __init__(self, head_size: int, max_position: int, base: float, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.head_size = head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        torch.ops.vllm_iface.silu_and_mul(out, x)
        return out


class Attention(nn.Module):
    """vllm.attention.layer.Attention (V1): `unified_attention(q, k, v, layer_name)`; the KV cache is forward-context state."""

    def __init__(self, layer_name: str) -> None:
        super().__init__()
        self.layer_name = layer_name

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad: _Pad = None) -> torch.Tensor:
        out = torch.ops.vllm_iface.unified_attention(q, k, v, self.layer_name)
        if pad is not None:
            # padded engine (serve3-spec §6): after the step's attention (which read the raw row), the appended cache
            # row is replaced by `live ? row : 0` so a finished request's steps append the zero row
            live, zero, slot = pad
            torch.ops.vllm_iface.unified_cache_write(torch.where(live, k, zero), torch.where(live, v, zero), self.layer_name, slot)
        return out


class DecoderLayer(nn.Module):
    """vLLM LlamaDecoderLayer / Qwen2DecoderLayer shape: merged qkv_proj, o_proj, merged gate_up_proj, down_proj."""

    def __init__(self, C: dict, layer_idx: int, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        H, NH, KVH, D, INTER = C["H"], C["NH"], C["KVH"], C["D"], C["I"]
        self.NH, self.KVH, self.D = NH, KVH, D
        self.qkv_proj = LinearNoTP(H, (NH + 2 * KVH) * D, bias=C.get("BIAS", True), dtype=dtype)
        self.o_proj = LinearNoTP(NH * D, H, bias=False, dtype=dtype)
        self.rotary_emb = get_rope(D, C["MAX_POS"], C["ROPE_THETA"], dtype=dtype)
        self.attn = Attention(f"model.layers.{layer_idx}.self_attn.attn")
        self.gate_up_proj = LinearNoTP(H, 2 * INTER, bias=False, dtype=dtype)
        self.act_fn = SiluAndMul()
        self.down_proj = LinearNoTP(INTER, H, bias=False, dtype=dtype)
        self.input_layernorm = RMSNorm(H, C["EPS"], dtype=dtype)
        self.post_attention_layernorm = RMSNorm(H, C["EPS"], dtype=dtype)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None, pad: _Pad = None) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(self.pos0, self.pos0 + hidden_states.shape[0])
        return self.layer(positions, hidden_states, None if self.first else residual)


class Model(nn.Module):
    def __init__(self, C: dict, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(C["V"], C["H"], dtype=dtype)
        self.layers = nn.ModuleList([DecoderLayer(C, i, dtype) for i in range(C["NL"])])
        self.norm = RMSNorm(C["H"], C["EPS"], dtype=dtype)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, last_only: bool = True, pad: _Pad = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, pad)
        if last_only:
            assert residual is not None
            hidden_states, residual = hidden_states[-1:], residual[-1:]
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CausalLM(nn.Module):
    def __init__(self, C: dict, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.model = Model(C, dtype)
        self.lm_head = nn.Linear(C["H"], C["V"], bias=False, dtype=dtype)
        if C.get("tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, last_only: bool = True, sampled_rows_only: bool = False, pad: _Pad = None) -> torch.Tensor:
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

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
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
    `vllm_iface.gumbel_topp_token_select(logits, seed, temperature, top_p, pos, S)` — the shape of `cb_a.vllm_meta.StochasticRequest`
    over the example CausalLM.  `seed` (int64 [1]) is a forward INPUT; temperature / top_p and the split count `S` (the registered
    one) are constructor constants; `pos` is the logits row's position (LP-1+k for the k-th sample).  `literal_seed` (negative): the
    seed created inside forward — hidden state, refused.  `temperature=0` (negative): the rule refuses (greedy is TokenSelect_v1)."""

    def __init__(self, C: dict, steps: int, temperature: float = 0.8, top_p: float = 0.95, literal_seed: bool = False,
                 S: int = 32) -> None:
        super().__init__()
        self.lm = CausalLM(C)
        self.steps = steps
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.literal_seed = literal_seed
        self.S = int(S)

    def _select(self, logits, seed, pos: int):
        return iface.gumbel_topp_token_select(logits[-1:], seed, self.temperature, self.top_p, pos, self.S)

    def forward(self, prompt, seed):
        if self.literal_seed:
            seed = torch.tensor([1234], dtype=torch.int64)
        LP = prompt.shape[0]
        logits = self.lm(prompt, torch.arange(0, LP))
        tok = self._select(logits, seed, LP - 1)
        toks = [tok]
        for s in range(self.steps):
            logits = self.lm(tok, torch.arange(LP + s, LP + s + 1))
            tok = self._select(logits, seed, LP + s)
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
# negatives
# ---------------------------------------------------------------------------------------------------------


class NegDataDependent(nn.Module):
    """A computed token (TokenSelect) used as an index into a *value*: a data-dependent gather (the row read
    depends on the data). Embedding rows are the only table a computed token may select (Embedding kind)."""

    def __init__(self, V: int = 8, H: int = 8) -> None:
        super().__init__()
        self.fc = LinearNoTP(H, V, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.fc(x)
        tok = torch.argmax(logits[-1], dim=-1).view(1)
        return logits[tok]


class NegMutatesWeight(nn.Module):
    """`rms_norm` writes its `out` operand — here a registered buffer (host state written by the forward)."""

    def __init__(self, T: int = 2, H: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.register_buffer("scratch", torch.zeros(T, H, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.NH * self.D, self.KVH * self.D, self.KVH * self.D], dim=-1)
        positions = torch.arange(0, hidden_states.shape[0])
        self.rotary_emb(positions, q, k)
        return qkv  # stale: the source reads the pre-mutation alias


__all__ = ["MLP2", "TiedMLP", "RMSNorm", "LinearNoTP", "RotaryEmbedding", "SiluAndMul", "Attention", "DecoderLayer", "SingleLayer", "Model",
           "CausalLM", "GreedyServe", "HFRMSNorm", "HFDecoderLayer", "NegUnknownOp", "NegHiddenConstant", "NegInPlace", "NegDataDependent",
           "NegCustomOp", "NegReturnsWeight", "NegCond", "Conv1D", "GPT2MLPBlock", "GPT2Attention", "GPT2Block", "GPT2TinyLM", "GPT2Greedy",
           "NegMutatesWeight", "NegStaleAlias", "HFGreedy", "HFRotaryEmbedding", "HFTinyLM", "hf_apply_rotary_pos_emb",
           "hf_rotate_half"]
