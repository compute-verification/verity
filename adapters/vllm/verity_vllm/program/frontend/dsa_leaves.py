"""4. DeepSeek sparse attention (vLLM `models/deepseek_v32/attention.py`: GLM-5.3, DeepSeek-V3.2) as leaf ops (`vllm_iface`).

Its two Triton kernels run outside the MLA attention op and write the MLA and indexer caches through the forward context's slot
mapping, which the meta export has no cache for (`fused_norm_rope` asserts `slot_mapping is not None`); its attention reads those
caches back through the same context.  `dsa_ops_leaf` patches them for a model's layers, `dsa_leaves` at one site.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import torch

from verity_vllm.engine import hooks

__models__ = ("zai-org/GLM-5.3",)


def _dsa_layers(model) -> list:
    return [m for m in (model.modules() if model is not None else ())
            if hasattr(m, "_sparse_indexer_and_attn") and hasattr(m, "fused_qkv_a_proj")]


@contextmanager
def dsa_ops_leaf(model=None):
    """4. DeepSeek sparse attention (`DeepseekV32Attention.forward`, GLM-5.3, DeepSeek-V3.2) as three leaf ops.  Its two Triton kernels
    run outside the MLA attention op and write the MLA and indexer caches through the forward context's slot mapping, which the meta
    export has no cache for (`fused_norm_rope` asserts `slot_mapping is not None`); its attention reads those caches back through the
    same context.  The kernels are patched in the module the layers' `forward` is defined in, the globals it reads them from.  See
    `dsa_leaves`.  A model without a sparse attention layer is unchanged."""
    sites = {sys.modules[type(m).forward.__module__] for m in _dsa_layers(model)}
    if len(sites) > 1:
        raise NotImplementedError(f"dsa_ops_leaf: sparse attention layers whose forward reads its kernels from {len(sites)} modules")
    if not sites:
        yield
        return
    with dsa_leaves(sites.pop(), model):
        yield


@contextmanager
def dsa_leaves(site, model=None):
    """For the duration, ``site`` (the module `DeepseekV32Attention.forward` reads its kernels from) calls `vllm_iface.dsa_norm_rope`
    for `fused_norm_rope` and `vllm_iface.dsa_q` for `fused_q`, and every module of ``model`` with a `_sparse_indexer_and_attn` gets
    one that calls `vllm_iface.dsa_indexer_attn` and then the absorbed value projection its body ends with, `bmm(attn, W_UV)` into
    `output`.  A call vLLM serves by dense MHA (`vllm_dsa_dense.takes_dense_mha`) runs `forward_mha`'s body: `kv_b_proj` up projects
    the latent to k_nope and v, then `vllm_iface.dsa_dense_mha` over q's and k's parts (`_concat_k_nope_k_pe`'s copies, which its rule
    assembles), its rows flattened into `output`.  The cache rows the kernels write are the first op's outputs (`kv_c_out`,
    `k_pe_out`, the index K and its scale) and the attention op's operands, so each layer's cache is the derive's state
    (`rules/vllm_dsa.py`).  Over an `fp8_ds_mla` KV cache the first op is `vllm_iface.dsa_norm_rope_ds_mla`, whose two further
    outputs are the cache row's NoPE bytes and scales (`DS_MLA_TILE` values a scale), and the attention ops take them in kv_c's
    place: `vllm_iface.dsa_flashmla_attn`, and on the dense-MHA route `vllm_iface.dsa_dense_mha_ds_mla` (`rules/vllm_dsa_fp8.py`).
    What the shim does not model is refused at export: prefill context parallelism, decode context parallelism, an fp8 query, a
    quantized KV cache in another layout, a caller-supplied `q_c_out` or `index_k_out`.  `W_UK_T` and `W_UV` are Parameters the
    engine registers after loading; a meta model gets them from `vllm_meta.mla_absorbed_weights_as_the_loader_does`."""
    from verity_vllm.program.frontend.rules import vllm_dsa_dense, vllm_iface  # noqa: F401  (vllm_iface registers the ops)
    from verity_vllm.program.frontend.rules.vllm_dsa_fp8 import DS_MLA, DS_MLA_TILE
    iface = torch.ops.vllm_iface
    pending: dict[int, tuple] = {}
    seen: dict[str, int] = {}
    h = hooks.Hooks()

    def fused_norm_rope(positions, q_c, q_w, q_eps, kv_c, kv_w, kv_eps, k_pe, k_cs, index_k, index_k_w, index_k_b, index_k_eps, index_cs,
                        topk_indices_buffer, slot_mapping=None, indexer_k_cache=None, mla_kv_cache=None, mla_kv_cache_dtype="auto",
                        mla_k_scale=None, has_indexer=True, index_rope_interleave=False, q_c_out=None, kv_c_out=None, k_pe_out=None,
                        index_k_out=None):
        if q_c_out is not None or index_k_out is not None or kv_c_out is None or k_pe_out is None:
            raise NotImplementedError("dsa_leaves: fused_norm_rope with q_c_out / index_k_out, or without kv_c_out / k_pe_out")
        if mla_kv_cache_dtype not in ("auto", "bfloat16", DS_MLA):
            raise NotImplementedError(f"dsa_leaves: MLA KV cache dtype {mla_kv_cache_dtype!r} (a bf16 cache and {DS_MLA}'s rows are "
                                      f"modelled)")
        ix = (index_k, index_k_w, index_k_b, index_cs) if has_indexer else (None, None, None, None)
        args = (positions, q_c, q_w, float(q_eps), kv_c, kv_w, float(kv_eps), k_pe, k_cs, ix[0], ix[1], ix[2], float(index_k_eps), ix[3],
                bool(index_rope_interleave))
        row = None
        if mla_kv_cache_dtype == DS_MLA:
            q, ik, iks, *row = iface.dsa_norm_rope_ds_mla(*args, DS_MLA_TILE, kv_c_out, k_pe_out)
        else:
            q, ik, iks = iface.dsa_norm_rope(*args, kv_c_out, k_pe_out)
        pending[id(kv_c_out)] = ((ik, iks) if has_indexer else (None, None), row)
        return q

    def fused_q(positions, q_pe, q_pe_cs, index_q, index_q_cs, ql_nope, q_scale, index_weights, softmax_scale, head_scale,
                has_indexer=True, index_rope_interleave=False, quantize_mqa=True):
        if quantize_mqa:
            raise NotImplementedError("dsa_leaves: fused_q with quantize_mqa (an fp8 MQA query)")
        ix = (index_q, index_q_cs, index_weights) if has_indexer else (None, None, None)
        q_pe_out, iq, iw = iface.dsa_q(positions, q_pe, q_pe_cs, ix[0], ix[1], ix[2], float(softmax_scale), float(head_scale),
                                       bool(index_rope_interleave))
        return iq, iw, q_pe_out

    def indexer_and_attn(layer):
        def run(positions, q_c, q_nope, q_pe, index_q_fp8, index_k, index_weights_out, kv_c, k_pe, ql_nope, mqa_q, output):
            if layer.use_pcp or layer.impl.dcp_world_size > 1 or layer._fp8_query or layer._fp8_kv_needs_view:
                raise NotImplementedError(f"dsa_leaves: {layer.layer_name} under context parallelism or an fp8 KV cache")
            (ik, iks), row = pending.pop(id(kv_c))
            if (row is not None) != (layer.kv_cache_dtype == DS_MLA):
                raise NotImplementedError(f"dsa_leaves: {layer.layer_name}'s KV cache dtype {layer.kv_cache_dtype!r} is not the one its "
                                          f"cache rows were written in")
            has = ik is not None
            n, topk = output.shape[0], int(layer.topk_indices_buffer.shape[-1])
            rows = seen[layer.layer_name] = seen.get(layer.layer_name, 0) + n
            # the call's cache rows: bf16 [kv_c | k_pe], or fp8_ds_mla's NoPE bytes, scales and k_pe
            cache = (kv_c, k_pe) if row is None else (*row, k_pe)
            if vllm_dsa_dense.takes_dense_mha(layer, n, rows, topk):
                nh, dn, dv = layer.num_heads, layer.qk_nope_head_dim, layer.v_head_dim
                k_nope, v = layer.kv_b_proj(kv_c)[0].view(-1, nh, dn + dv).split([dn, dv], dim=-1)
                dense = iface.dsa_dense_mha if row is None else iface.dsa_dense_mha_ds_mla
                o = dense(positions, q_nope, mqa_q, k_nope, v, *cache, ik, iks, layer.layer_name, topk, float(layer.scale))
                output.copy_(o.flatten(start_dim=-2))
                return
            attn = iface.dsa_indexer_attn if row is None else iface.dsa_flashmla_attn
            a = attn(positions, ql_nope, mqa_q, *cache, index_q_fp8 if has else None, index_weights_out if has else None, ik, iks,
                     layer.layer_name, topk, float(layer.scale))
            nh, lat, v = layer.num_heads, layer.kv_lora_rank, layer.v_head_dim
            y = torch.bmm(a.view(n, nh, lat).transpose(0, 1), layer.W_UV)
            output.copy_(y.transpose(0, 1).reshape(n, nh * v))
        return run

    h.patch(site, "fused_norm_rope", fused_norm_rope)
    h.patch(site, "fused_q", fused_q)
    for m in _dsa_layers(model):
        h.patch(m, "_sparse_indexer_and_attn", indexer_and_attn(m))
    with h:
        yield


__all__ = ["dsa_leaves", "dsa_ops_leaf"]
