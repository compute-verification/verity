"""The vLLM kernel interface the profile rules bind: `torch.ops._C.*` (vLLM's custom ops) when vLLM is importable,
and otherwise (laptop / pod without vLLM) an interface-identical stand-in library `vllm_iface` with the SAME
schemas — mutable operands, argument order, defaults — and fake (meta) implementations only.

The stand-ins have NO numerical implementation: they exist so that ordinary vLLM-style module classes
(`derive_examples.py`) export to graphs whose nodes are the kernel calls the profile's registered composites
model. Whether a stand-in and the real `_C` op agree is a question about the *module classes*, not about the
derivation; the rules target both by schema.

Schemas (from vllm/csrc/torch_bindings.cpp, `d9105ea80`):
    rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()
    fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()
    rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, int head_size, Tensor cos_sin_cache, bool is_neox) -> ()
    silu_and_mul(Tensor! result, Tensor input) -> ()
    gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()
    silu_and_mul_per_block_quant(Tensor! out, Tensor input, Tensor! scales, int group_size, Tensor? scale_ub, bool is_scale_transposed) -> ()
and of `_moe_C` (vllm/csrc/moe/torch_bindings.cpp; the call `_custom_ops.grouped_topk` makes):
    grouped_topk(Tensor scores, int n_group, int topk_group, int topk, bool renormalize, float routed_scaling_factor, Tensor bias,
                 int scoring_func) -> (Tensor, Tensor)
and vLLM V1's model-level attention entry point (vllm/attention/layer.py):
    unified_attention(Tensor query, Tensor key, Tensor value, str layer_name) -> Tensor
"""

from __future__ import annotations

import torch

_LIB_NAME = "vllm_iface"
_lib = torch.library.Library(_LIB_NAME, "FRAGMENT")  # noqa: TOR901

_DEFINED = getattr(torch.ops, _LIB_NAME, None)


def _define_once(schema: str, name: str, fake) -> None:
    try:
        _lib.define(schema)
    except RuntimeError:  # already defined in this process
        return
    torch.library.register_fake(f"{_LIB_NAME}::{name}")(fake)


_define_once("rms_norm(Tensor(a!) result, Tensor input, Tensor weight, float epsilon) -> ()", "rms_norm",
             lambda result, input, weight, epsilon: None)
_define_once("fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float epsilon) -> ()", "fused_add_rms_norm",
             lambda input, residual, weight, epsilon: None)
_define_once("rotary_embedding(Tensor positions, Tensor(a!) query, Tensor(b!)? key, int head_size, Tensor cos_sin_cache, bool is_neox) -> ()",
             "rotary_embedding", lambda positions, query, key, head_size, cos_sin_cache, is_neox: None)
_define_once("silu_and_mul(Tensor(a!) result, Tensor input) -> ()", "silu_and_mul", lambda result, input: None)
_define_once("gelu_tanh_and_mul(Tensor(a!) out, Tensor input) -> ()", "gelu_tanh_and_mul", lambda out, input: None)
_define_once("silu_and_mul_per_block_quant(Tensor(a!) out, Tensor input, Tensor(b!) scales, int group_size, Tensor? scale_ub, "
             "bool is_scale_transposed) -> ()", "silu_and_mul_per_block_quant",
             lambda out, input, scales, group_size, scale_ub, is_scale_transposed: None)
_define_once("grouped_topk(Tensor scores, int n_group, int topk_group, int topk, bool renormalize, float routed_scaling_factor, "
             "Tensor bias, int scoring_func=0) -> (Tensor, Tensor)", "grouped_topk",
             lambda scores, n_group, topk_group, topk, renormalize, routed_scaling_factor, bias, scoring_func=0:
                 (scores.new_empty((scores.shape[0], topk), dtype=torch.float32), scores.new_empty((scores.shape[0], topk), dtype=torch.int32)))
_define_once("unified_attention(Tensor query, Tensor key, Tensor value, str layer_name) -> Tensor", "unified_attention",
             lambda query, key, value, layer_name: torch.empty_like(query))
# The implicit-cache counterpart of `_C_cache_ops.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...)`:
# (re)write the K/V row(s) of `layer_name`'s cache at static `slot` — what a padded engine does after a step whose request has
# finished (the appended row is replaced by the pad row). Not a vLLM op today; a stand-in with an explicit schema so that a padded
# serving module can be an ordinary module (serve3-spec §6: the row muxes sit at the step's output boundary, never inside the body).
_define_once("unified_cache_write(Tensor key, Tensor value, str layer_name, int slot) -> ()", "unified_cache_write",
             lambda key, value, layer_name, slot: None)
# The padded engine's step-boundary cache clear for a model whose KV cache is implicit engine state (the real vLLM model:
# rows are written by `unified_kv_cache_update` inside the attention layer, so a wrapper cannot mux them itself): for EVERY
# layer's cache, row `slot` (the row this step appended) := live ? row : zero (one broadcast zero per step). Model-agnostic
# (no layer names, no layer count). Stand-in with an explicit schema, like `unified_cache_write`.
_define_once("unified_cache_pad(Tensor live, Tensor zero, int slot) -> ()", "unified_cache_pad", lambda live, zero, slot: None)
# [R15 stoch] The stochastic token select of vLLM d9105ea80's sampler for ONE request with temperature + top-p (no top-k / penalties /
# logit bias / min_p), as one op the wrapper authors per sampled step: `Sampler.sample` = f32 widening -> `_temperature_kernel` ->
# `apply_top_k_top_p_triton` (`_apply_topp_split`, `splits` splits) -> `_gumbel_sample_kernel` (Gumbel-max with the stream key
# `tl.randint(seed, pos)`) -> block argmax.  `logits` bf16 [.., V]; `seed` int64 [1] (the request's `SamplingParams.seed`, a RUNTIME
# input); `temperature` / `top_p` the request's declared policy (STRUCTURAL: constants of the Program); `pos` the position of the
# logits row (`input_batch.positions[logits_indices]`: the last input token of the step); `splits` int32 [1] the pipeline's split count
# for this event -- [R16 stoch, coord M-0169] a DERIVED input of the workload (S_t = SplitsFor_v1(|live(t)|, num_SMs)) the request wrapper
# reads from its request-owned `splits` Input, not a static.  [single-request workloads] A single-request workload passes `splits=None`
# and `S` = SplitsFor_v1(1, num_SMs), the constant split count of its every event.  Not a vLLM op: a stand-in with an explicit schema like
# `unified_cache_pad`; the frontend rule `GumbelTopPTokenSelect[vllm]` binds it to `GumbelTopPTokenSelect_v1{V}` (verity_vllm.program.registry.sampling;
# S its 6th operand).
_define_once("gumbel_topp_token_select(Tensor logits, Tensor seed, float temperature, float top_p, int pos, Tensor? splits, int? S=None) -> Tensor",
             "gumbel_topp_token_select",
             lambda logits, seed, temperature, top_p, pos, splits, S=None: logits.new_empty(tuple(logits.shape[:-1]), dtype=torch.int32))

# [GLM-5.3] DeepSeek sparse attention (vllm/models/deepseek_v32/attention.py) as three ops, the leaves `export_compat.dsa_ops_leaf`
# puts in place of `fused_norm_rope`, `fused_q` and `_sparse_indexer_and_attn` while exporting.  The served kernels write the MLA and
# indexer caches through a slot mapping and read them back through the forward context; here the cache rows are the first op's outputs
# and the third op's operands, and each layer's history is the derive's state keyed by `layer_name` (as `unified_attention_with_output`'s).
#   dsa_norm_rope -> (q_c normed, index K e4m3 [T, Dk], index K scale f32 [T]); kv_c_out / k_pe_out written (the MLA cache row)
#   dsa_q         -> (q_pe roped bf16, index q e4m3 [T, H, Dk], index weights f32 [T, H])
#   dsa_indexer_attn -> the attention rows [T, NH, kv_lora] bf16 (before the W_UV bmm)
# On a layer without an indexer the index outputs are empty ([T, 0], [T, 0, 0]) and the third op's index operands are None.
def _dsa_norm_rope_fake(positions, q_c, q_w, q_eps, kv_c, kv_w, kv_eps, k_pe, cos_sin, index_k, index_k_w, index_k_b, index_k_eps,
                        index_cos_sin, index_rope_interleave, kv_c_out, k_pe_out):
    T, Dk = q_c.shape[0], (index_k.shape[-1] if index_k is not None else 0)
    return (q_c.new_empty(q_c.shape), q_c.new_empty((T, Dk), dtype=torch.float8_e4m3fn),
            q_c.new_empty((T,) if index_k is not None else (T, 0), dtype=torch.float32))


def _dsa_q_fake(positions, q_pe, cos_sin, index_q, index_cos_sin, index_weights, softmax_scale, head_scale, index_rope_interleave):
    T = q_pe.shape[0]
    iq = index_q.shape if index_q is not None else (T, 0, 0)
    return (q_pe.new_empty(q_pe.shape), q_pe.new_empty(iq, dtype=torch.float8_e4m3fn), q_pe.new_empty(iq[:2], dtype=torch.float32))


_define_once("dsa_norm_rope(Tensor positions, Tensor q_c, Tensor q_w, float q_eps, Tensor kv_c, Tensor kv_w, float kv_eps, Tensor k_pe, "
             "Tensor cos_sin, Tensor? index_k, Tensor? index_k_w, Tensor? index_k_b, float index_k_eps, Tensor? index_cos_sin, "
             "bool index_rope_interleave, Tensor(a!) kv_c_out, Tensor(b!) k_pe_out) -> (Tensor, Tensor, Tensor)",
             "dsa_norm_rope", _dsa_norm_rope_fake)
_define_once("dsa_q(Tensor positions, Tensor q_pe, Tensor cos_sin, Tensor? index_q, Tensor? index_cos_sin, Tensor? index_weights, "
             "float softmax_scale, float head_scale, bool index_rope_interleave) -> (Tensor, Tensor, Tensor)", "dsa_q", _dsa_q_fake)
_define_once("dsa_indexer_attn(Tensor positions, Tensor ql_nope, Tensor q_pe, Tensor kv_c, Tensor k_pe, Tensor? index_q, "
             "Tensor? index_weights, Tensor? index_k, Tensor? index_k_scale, str layer_name, int topk, float scale) -> Tensor",
             "dsa_indexer_attn", lambda positions, ql_nope, q_pe, kv_c, k_pe, index_q, index_weights, index_k, index_k_scale, layer_name,
             topk, scale: ql_nope.new_empty(ql_nope.shape))
# [GLM-5.3] vLLM's cuteDSL router GEMM `ll_bf16_gemm(x [M, K] bf16, w [N, K] bf16) -> [M, N] f32` (`GateLinear` tier 1, M <= 16 on
# sm_90+), the leaf `export_compat.moe_gate_leaf` puts in its place while exporting: a CUTLASS DSL kernel that is no dispatcher op
_define_once("ll_bf16_gemm(Tensor x, Tensor w) -> Tensor", "ll_bf16_gemm",
             lambda x, w: x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32))

iface = torch.ops.vllm_iface


def real_C():
    """vLLM's `_C` ops if importable (never on the laptop)."""
    try:
        import vllm  # noqa: F401
        return torch.ops._C
    except Exception:  # noqa: BLE001
        return None


def targets_for(name: str) -> set:
    """Both the stand-in and (if present) the real op, for one kernel name."""
    out = {getattr(iface, name).default}
    C = real_C()
    if C is not None and hasattr(C, name):
        out.add(getattr(C, name).default)
    if name == "unified_attention":
        try:
            out.add(torch.ops.vllm.unified_attention.default)
        except Exception:  # noqa: BLE001
            pass
    return out


__all__ = ["iface", "targets_for", "real_C"]
