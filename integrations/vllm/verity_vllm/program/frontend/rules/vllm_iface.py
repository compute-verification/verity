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
    silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit, float alpha=1., float beta=0.) -> ()
    gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()
    silu_and_mul_per_block_quant(Tensor! out, Tensor input, Tensor! scales, int group_size, Tensor? scale_ub, bool is_scale_transposed) -> ()
    per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! output_s, int group_size, float eps, float fp8_min,
                              float fp8_max, bool scale_ue8m0, bool column_major_scales, bool tma_aligned) -> ()
and of `_moe_C` (vllm/csrc/moe/torch_bindings.cpp; the call `_custom_ops.grouped_topk` makes):
    grouped_topk(Tensor scores, int n_group, int topk_group, int topk, bool renormalize, float routed_scaling_factor, Tensor bias,
                 int scoring_func) -> (Tensor, Tensor)
    topk_softplus_sqrt(Tensor! topk_weights, Tensor! topk_indices, Tensor! token_expert_indices, Tensor gating_output, bool renormalize,
                       float routed_scaling_factor, Tensor? bias, Tensor? input_ids, Tensor? tid2eid, Tensor? is_padding,
                       Tensor? bias_vl=None, int image_sentinel_lo=0) -> ()
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
_define_once("silu_and_mul_with_clamp(Tensor(a!) result, Tensor input, float limit, float alpha=1., float beta=0.) -> ()",
             "silu_and_mul_with_clamp", lambda result, input, limit, alpha=1.0, beta=0.0: None)
_define_once("gelu_tanh_and_mul(Tensor(a!) out, Tensor input) -> ()", "gelu_tanh_and_mul", lambda out, input: None)
_define_once("silu_and_mul_per_block_quant(Tensor(a!) out, Tensor input, Tensor(b!) scales, int group_size, Tensor? scale_ub, "
             "bool is_scale_transposed) -> ()", "silu_and_mul_per_block_quant",
             lambda out, input, scales, group_size, scale_ub, is_scale_transposed: None)
# `_C.per_token_group_fp8_quant` (vllm/csrc/torch_bindings.cpp): the per-group activation quant of the block-FP8 linears and the
# block-FP8 Triton experts' input (`moe_kernel_quantize_input` -> `per_token_group_quant_fp8`, which passes `use_ue8m0 =
# is_deep_gemm_e8m0_used()`: True at vLLM's sm_90 defaults).
_define_once("per_token_group_fp8_quant(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s, int group_size, float eps, "
             "float fp8_min, float fp8_max, bool scale_ue8m0, bool column_major_scales, bool tma_aligned) -> ()",
             "per_token_group_fp8_quant",
             lambda input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0, column_major_scales, tma_aligned: None)
_define_once("grouped_topk(Tensor scores, int n_group, int topk_group, int topk, bool renormalize, float routed_scaling_factor, "
             "Tensor bias, int scoring_func=0) -> (Tensor, Tensor)", "grouped_topk",
             lambda scores, n_group, topk_group, topk, renormalize, routed_scaling_factor, bias, scoring_func=0:
                 (scores.new_empty((scores.shape[0], topk), dtype=torch.float32), scores.new_empty((scores.shape[0], topk), dtype=torch.int32)))
_define_once("topk_softplus_sqrt(Tensor(a!) topk_weights, Tensor(b!) topk_indices, Tensor(c!) token_expert_indices, Tensor gating_output, "
             "bool renormalize, float routed_scaling_factor, Tensor? bias, Tensor? input_ids, Tensor? tid2eid, Tensor? is_padding, "
             "Tensor? bias_vl=None, int image_sentinel_lo=0) -> ()", "topk_softplus_sqrt",
             lambda topk_weights, topk_indices, token_expert_indices, gating_output, renormalize, routed_scaling_factor, bias, input_ids,
             tid2eid, is_padding, bias_vl=None, image_sentinel_lo=0: None)
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
# A request's cached prefix (vLLM's prefix cache): rows 0..C-1 of every attention layer's key/value cache, which the scheduler serves
# from blocks an earlier request computed instead of recomputing them.  `prefix` bf16 [L, 2, C, KVH*D] (layer in forward order, key |
# value, row, leaf): the request wrapper's input, called before its prefill; each layer's history opens with its C rows.  Not a vLLM
# op: a stand-in with an explicit schema like `unified_cache_pad` (rule `KvPrefix[vllm]`).
_define_once("kv_prefix(Tensor prefix) -> ()", "kv_prefix", lambda prefix: None)
# The stochastic token select of vLLM d9105ea80's sampler for ONE request with temperature + top-p (no top-k / penalties /
# logit bias / min_p), as one op the wrapper authors per sampled step: `Sampler.sample` = f32 widening -> `_temperature_kernel` ->
# `apply_top_k_top_p_triton` (`_apply_topp_split`, `splits` splits) -> `_gumbel_sample_kernel` (Gumbel-max with the stream key
# `tl.randint(seed, pos)`) -> block argmax.  `logits` bf16 [.., V]; `seed` int64 [1] (the request's `SamplingParams.seed`, a RUNTIME
# input); `temperature` / `top_p` the request's declared policy (STRUCTURAL: constants of the Program); `pos` the position of the
# logits row (`input_batch.positions[logits_indices]`: the last input token of the step); `splits` int32 [1] the pipeline's split count
# for this event -- a DERIVED input of the workload (S_t = SplitsFor_v1(|live(t)|, num_SMs)) the request wrapper
# reads from its request-owned `splits` Input, not a static.  [registered S] The Build passes `splits=None` and `S` = the registered
# split count SplitsFor_v1(1, num_SMs) (`topp_geometry.registered_splits`), at which every served event runs (`engine.topp_pin`).
# Not a vLLM op: a stand-in with an explicit schema like `unified_cache_pad`; the frontend rule `GumbelTopPTokenSelect[vllm]` binds it
# to `GumbelTopPTokenSelect_v1{V}` (verity.ml.definitions.registry.sampling; S its 6th operand).
_define_once("gumbel_topp_token_select(Tensor logits, Tensor seed, float temperature, float top_p, int pos, Tensor? splits, int? S=None) -> Tensor",
             "gumbel_topp_token_select",
             lambda logits, seed, temperature, top_p, pos, splits, S=None: logits.new_empty(tuple(logits.shape[:-1]), dtype=torch.int32))

# vLLM's default Hopper block-FP8 linear.  The real op is registered in
# `kernels/linear/scaled_mm/flashinfer.py`; this stand-in lets CPU-only derive
# tests author the same exported node.
_define_once("dynamic_flashinfer_deepgemm_blockscale_gemm(Tensor input, Tensor weight, Tensor weight_scale, int group_size, "
             "bool use_deep_gemm_e8m0) -> Tensor", "dynamic_flashinfer_deepgemm_blockscale_gemm",
             lambda input, weight, weight_scale, group_size, use_deep_gemm_e8m0:
                 input.new_empty((*input.shape[:-1], weight.shape[0]), dtype=torch.bfloat16))

# [GLM-5.3] DeepSeek sparse attention (vllm/models/deepseek_v32/attention.py) as three ops, the leaves `export_compat.dsa_ops_leaf`
# puts in place of `fused_norm_rope`, `fused_q` and `_sparse_indexer_and_attn` while exporting.  The served kernels write the MLA and
# indexer caches through a slot mapping and read them back through the forward context; here the cache rows are the first op's outputs
# and the third op's operands, and each layer's history is the derive's state keyed by `layer_name` (as `unified_attention_with_output`'s).
#   dsa_norm_rope -> (q_c normed, index K e4m3 [T, Dk], index K scale f32 [T]); kv_c_out / k_pe_out written (the MLA cache row)
#   dsa_q         -> (q_pe roped bf16, index q e4m3 [T, H, Dk], index weights f32 [T, H])
#   dsa_indexer_attn -> the attention rows [T, NH, kv_lora] bf16 (before the W_UV bmm)
#   dsa_dense_mha    -> in its place on a prefill the layer serves by dense MHA (`SparseMLACommonImpl.forward_mha`: the prefill
#                       backend's causal varlen attention over the call's own rows), the attention rows [T, NH, v_head] bf16 of
#                       q = [q_nope | q_pe roped], k = [k_nope | k_pe] and v (k_nope and v kv_b_proj's up projection, outside the
#                       op); q and k go in as their parts, since k_pe broadcast over the heads is no view of a T-row prefill's rows;
#                       kv_c, k_pe and the index K rows are the call's cache rows, as the third op's
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
_define_once("dsa_dense_mha(Tensor positions, Tensor q_nope, Tensor q_pe, Tensor k_nope, Tensor v, Tensor kv_c, Tensor k_pe, "
             "Tensor? index_k, Tensor? index_k_scale, str layer_name, int topk, float scale) -> Tensor",
             "dsa_dense_mha", lambda positions, q_nope, q_pe, k_nope, v, kv_c, k_pe, index_k, index_k_scale, layer_name, topk, scale:
                 v.new_empty((q_nope.shape[0], q_nope.shape[1], v.shape[-1])))
# [GLM-5.3] vLLM's cuteDSL router GEMM `ll_bf16_gemm(x [M, K] bf16, w [N, K] bf16) -> [M, N] f32` (`GateLinear` tier 1, M <= 16 on
# sm_90+), the leaf `export_compat.moe_gate_leaf` puts in its place while exporting: a CUTLASS DSL kernel that is no dispatcher op
_define_once("ll_bf16_gemm(Tensor x, Tensor w) -> Tensor", "ll_bf16_gemm",
             lambda x, w: x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32))


# [DSV4] DeepSeek-V4's manifold-constrained hyper-connections (vllm/model_executor/kernels/mhc/tilelang.py): the three ops vLLM
# registers from the functions DeepSeek-V4's model calls (`vllm::<name>`, the schemas here are theirs), and layer 0's broadcast
# pre, which vLLM does not register (`export_compat.mhc_pre_broadcast_leaf` puts this op, of the schema `infer_schema` gives the
# function, in its place).  residual [T, HC, H] bf16; post mix [T, HC, 1], comb mix [T, HC, HC] f32; the layer input [T, H] bf16.
def _mhc_mixes(t, HC):
    T = t.shape[0]
    return t.new_empty((T, HC, 1), dtype=torch.float32), t.new_empty((T, HC, HC), dtype=torch.float32)


def _mhc_fused_post_pre_fake(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                             hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None, norm_eps=1e-6):
    T, HC, H = residual.shape
    return (residual.new_empty((T, HC, H)), *_mhc_mixes(residual, HC), residual.new_empty((T, H)))


def _mhc_pre_broadcast_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                            n_splits=1, norm_weight=None, norm_eps=1e-6, fn_broadcast=None):
    T, H = residual.shape
    HC = fn.shape[1] // H
    return (residual.new_empty((T, HC, H)), *_mhc_mixes(residual, HC), residual.new_empty((T, H)))


_MHC_PRE = ("Tensor fn, Tensor hc_scale, Tensor hc_base, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, "
            "SymInt sinkhorn_repeat, SymInt n_splits=1")
_define_once("mhc_fused_post_pre_tilelang(Tensor x, Tensor residual, Tensor post_layer_mix, Tensor comb_res_mix, " + _MHC_PRE +
             ", SymInt tile_n=1, Tensor? norm_weight=None, float norm_eps=9.9999999999999995e-07) -> (Tensor, Tensor, Tensor, Tensor)",
             "mhc_fused_post_pre_tilelang", _mhc_fused_post_pre_fake)
_define_once("mhc_pre_broadcast_tilelang(Tensor residual, " + _MHC_PRE + ", Tensor? norm_weight=None, float norm_eps=1e-06, "
             "Tensor? fn_broadcast=None) -> (Tensor, Tensor, Tensor, Tensor)", "mhc_pre_broadcast_tilelang", _mhc_pre_broadcast_fake)
_define_once("mhc_post_tilelang(Tensor x, Tensor residual, Tensor post_layer_mix, Tensor comb_res_mix) -> Tensor", "mhc_post_tilelang",
             lambda x, residual, post_layer_mix, comb_res_mix: residual.new_empty(residual.shape))
_define_once("hc_head_fused_kernel_tilelang(Tensor hs_flat, Tensor fn, Tensor hc_scale, Tensor hc_base, float rms_eps, float hc_eps) -> Tensor",
             "hc_head_fused_kernel_tilelang", lambda hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps:
                 hs_flat.new_empty((hs_flat.shape[0], hs_flat.shape[-1]), dtype=torch.bfloat16))


# [DSV4] DeepSeek-V4's sparse MLA (vllm/models/deepseek_v4/attention.py, nvidia/flashmla.py) as three ops, the leaves
# `dsv4_leaves.dsv4_attention_leaves` puts in place of the kernels that write and read its caches through the forward context:
#   dsv4_qnorm_rope_kv_insert -> (q with its heads padded to NHP [T, NHP, D] bf16, the fp8_ds_mla window row: NoPE [T, D - DR] e4m3,
#                                 its scale bytes [T, (D - DR) / QB + 1] u8, RoPE [T, DR] bf16); DR the rotary width of cos_sin
#   dsv4_compressor_state     -> the compressor's state row (`save_partial_states`): kv [T, W] f32, score [T, W] f32, W = ape's width
#   dsv4_sparse_attn          -> the rank's live heads [T, heads, D] bf16
# The window rows, the compressor's state rows and the compressed rows are each layer's history, keyed by `layer_name`.
def _dsv4_insert_fake(q, kv, positions, cos_sin, padded_heads, eps, quant_block):
    T, D, DR = q.shape[0], q.shape[-1], cos_sin.shape[-1]
    return (q.new_empty((T, padded_heads, D)), q.new_empty((T, D - DR), dtype=torch.float8_e4m3fn),
            q.new_empty((T, (D - DR) // quant_block + 1), dtype=torch.uint8), q.new_empty((T, DR)))


_define_once("dsv4_qnorm_rope_kv_insert(Tensor q, Tensor kv, Tensor positions, Tensor cos_sin, SymInt padded_heads, float eps, "
             "SymInt quant_block) -> (Tensor, Tensor, Tensor, Tensor)", "dsv4_qnorm_rope_kv_insert", _dsv4_insert_fake)
_define_once("dsv4_compressor_state(Tensor kv_score, Tensor positions, Tensor ape) -> (Tensor, Tensor)", "dsv4_compressor_state",
             lambda kv_score, positions, ape: (kv_score.new_empty((kv_score.shape[0], ape.shape[1])),
                                               kv_score.new_empty((kv_score.shape[0], ape.shape[1]))))
_define_once("dsv4_sparse_attn(Tensor q, Tensor nope, Tensor scale, Tensor rope, Tensor positions, Tensor cos_sin, Tensor sink, "
             "Tensor? state_kv, Tensor? state_score, Tensor? norm_weight, str layer_name, SymInt heads, SymInt ratio, SymInt window, "
             "SymInt topk, float softmax_scale, float eps) -> Tensor", "dsv4_sparse_attn",
             lambda q, nope, scale, rope, positions, cos_sin, sink, state_kv, state_score, norm_weight, layer_name, heads, ratio, window,
                  topk, softmax_scale, eps: q.new_empty((q.shape[0], heads, q.shape[-1])))


# `vllm::fused_inv_rope_fp8_quant_kernel` (vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py), the o-projection's
# input: (bytes [G, T, d] e4m3, scales [G, T, scale_inner] f32, or int32 packed UE8M0 when TMA-aligned, over a buffer of
# tma_aligned_T rows a group)
def _inv_rope_quant_fake(o, positions, cos_sin_cache, heads_per_group, quant_group_size, chunks_per_head, rope_start, half_rope,
                         tma_aligned_scales, fp8_max, tma_aligned_T, num_tokens, n_groups, d, scale_inner):
    scale = o.new_empty(n_groups * scale_inner * tma_aligned_T, dtype=torch.int32 if tma_aligned_scales else torch.float32)
    return (o.new_empty((n_groups, num_tokens, d), dtype=torch.float8_e4m3fn),
            scale.as_strided((n_groups, num_tokens, scale_inner), (scale_inner * tma_aligned_T, 1, tma_aligned_T)))


_define_once("fused_inv_rope_fp8_quant_kernel(Tensor o, Tensor positions, Tensor cos_sin_cache, SymInt heads_per_group, "
             "SymInt quant_group_size, SymInt chunks_per_head, SymInt rope_start, SymInt half_rope, bool tma_aligned_scales, "
             "float fp8_max, SymInt tma_aligned_T, SymInt num_tokens, SymInt n_groups, SymInt d, SymInt scale_inner) -> (Tensor, Tensor)",
             "fused_inv_rope_fp8_quant_kernel", _inv_rope_quant_fake)
# DeepGEMM's `fp8_einsum(expr, (a, a_scale), (b, b_scale), d, c=None, recipe)` (csrc/apis/einsum.hpp), which writes `d`
# (`export_compat.deep_gemm_fp8_einsum_leaf`)
_define_once("deep_gemm_fp8_einsum(str expr, Tensor a, Tensor a_scale, Tensor b, Tensor b_scale, Tensor(a!) d, int[] recipe) -> ()",
             "deep_gemm_fp8_einsum", lambda expr, a, a_scale, b, b_scale, d, recipe: None)
# `fused_marlin_moe`'s `moe_align_block_size` as its launches read it, from the router ids and a key per pair (`marlin_moe_leaf`):
# (each pair's moe block [pairs] i32, num_tokens_post_padded / moe_block_size [1] i32)
_define_once("marlin_moe_align(Tensor topk_ids, Tensor key, int num_experts, int moe_block_size) -> (Tensor, Tensor)", "marlin_moe_align",
             lambda topk_ids, key, num_experts, moe_block_size: (topk_ids.new_empty((topk_ids.numel(),)), topk_ids.new_empty((1,))))
# one `moe_wna16_marlin_gemm` launch of `fused_marlin_moe` on W4A16 float4_e2m1f experts, over the logical weights w [E, N, K/2] and
# their e8m0 scales [E, N, K/32], each pair at its moe block `blk` of `par` (`marlin_moe_align`): rows [x rows * top_k, N] bf16
_define_once("marlin_moe_mxfp4_gemm(Tensor x, Tensor w, Tensor w_scale, Tensor topk_ids, Tensor blk, Tensor par, Tensor topk_weights, "
             "int moe_block_size, int top_k, bool mul_topk_weights) -> Tensor",
             "marlin_moe_mxfp4_gemm", lambda x, w, w_scale, topk_ids, blk, par, topk_weights, moe_block_size, top_k, mul_topk_weights:
             x.new_empty((x.shape[0] * top_k, w.shape[1])))

iface = torch.ops.vllm_iface


def real_C():
    """vLLM's `_C` ops if importable (never on the laptop)."""
    try:
        import vllm  # noqa: F401
        return torch.ops._C
    except Exception:  # noqa: BLE001
        return None


def targets_for(name: str) -> set:
    """Both the stand-in and (if present) the real op, `torch.ops._C`'s or `torch.ops.vllm`'s, for one kernel name."""
    out = {getattr(iface, name).default}
    C = real_C()
    if C is not None and hasattr(C, name):
        out.add(getattr(C, name).default)
    try:
        out.add(getattr(torch.ops.vllm, name).default)
    except (AttributeError, RuntimeError):
        pass
    return out


__all__ = ["iface", "targets_for", "real_C"]
