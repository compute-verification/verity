"""10. DeepSeek-V4's sparse MLA (vLLM `models/deepseek_v4/attention.py`, `nvidia/flashmla.py`) as three leaf ops (`vllm_iface`).

Its kernels write the window cache, the compressor's state cache and the compressed cache through the forward context's slot
mappings, and FlashMLA reads them back through block tables and index buffers: a meta export has neither the caches nor the
metadata (`_fused_qnorm_rope_kv_insert` asserts its window metadata).  For the duration of `dsv4_attention_leaves`, each
DeepSeek-V4 attention layer of the model calls

* `dsv4_qnorm_rope_kv_insert` for `_fused_qnorm_rope_kv_insert`: the padded, normed and roped query, and the token's fp8_ds_mla
  window row (`torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`; `registry.dsv4_qnorm_insert`);
* `dsv4_compressor_state` for its compressor: the state row `save_partial_states` writes (`registry.dsv4_indexer`'s
  `Dsv4CompressorScore`); the compression the compressor's second kernel runs at a window's end is the attention rule's, which
  owns the history the window is read from;
* `dsv4_sparse_attn` for `forward_mqa`: the layer's live heads, over its window rows and its compressed rows by position
  (`flash_mla_with_kvcache` / `flash_mla_sparse_fwd`), written into the padded output's live heads, the only heads the layer reads.

A ratio-4 layer's indexer is the short path, every compressed row selected, whenever the model's length leaves fewer compressed
rows than its top-k (`DeepseekV4Indexer.forward`): its compressor writes an indexer cache only the long path reads, and its
selection is the attention rule's.  A layer the leaves do not model refuses at export: another cache layout than fp8_ds_mla, or
an indexer whose long path the model's length can reach.
"""

from __future__ import annotations

from contextlib import contextmanager

from verity_vllm.engine import hooks

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

#: fp8_ds_mla's NoPE quantisation block (`DeepseekCompressor._quant_block` at head 512; one UE8M0 byte per 64 NoPE elements)
FP8_DS_MLA_QB = 64


def dsv4_layers(model) -> list:
    """The modules of ``model`` that are DeepSeek-V4 attention layers."""
    return [m for m in (model.modules() if model is not None else ())
            if hasattr(m, "_fused_qnorm_rope_kv_insert") and hasattr(m, "forward_mqa") and hasattr(m, "swa_cache_layer")]


@contextmanager
def dsv4_attention_leaves(model=None):
    """See the module docstring.  A model without a DeepSeek-V4 attention layer is unchanged."""
    layers = dsv4_layers(model)
    if not layers:
        yield
        return
    from verity_vllm.program.frontend.rules.vllm_iface import iface
    rows: dict[int, tuple] = {}
    states: dict[int, tuple] = {}

    def insert(layer):
        def run(q, kv, positions, attn_metadata):
            if layer.kv_cache_dtype != "fp8_ds_mla":
                raise NotImplementedError(f"dsv4_attention_leaves: {layer.prefix}'s kv cache {layer.kv_cache_dtype!r} (only fp8_ds_mla "
                                          "is modelled)")
            qp, nope, scale, rope = iface.dsv4_qnorm_rope_kv_insert(q, kv, positions, layer.rotary_emb.cos_sin_cache,
                                                                    int(layer.padded_heads), float(layer.eps), FP8_DS_MLA_QB)
            rows[id(kv)] = (nope, scale, rope)
            return qp
        return run

    def compressor(layer):
        def run(kv_score, positions, rotary_emb):
            if rotary_emb is not layer.rotary_emb:
                raise NotImplementedError(f"dsv4_attention_leaves: {layer.prefix}'s compressor with another rotary table than its layer's")
            states[id(layer)] = iface.dsv4_compressor_state(kv_score, positions, layer.compressor.ape)
        return run

    def indexer(layer):
        def run(*args, **kwargs):
            ix = layer.indexer
            if ix.max_model_len > ix.topk_tokens:
                raise NotImplementedError(f"dsv4_attention_leaves: {ix.prefix} at {ix.max_model_len} compressed rows takes the indexer's "
                                          f"long path (top-{ix.topk_tokens}), which is not modelled")
            return None, None, None
        return run

    def forward_mqa(layer):
        def run(q, kv, positions, output):
            nope, scale, rope = rows.pop(id(kv))
            comp, ix = layer.compressor, layer.indexer
            state_kv, state_score = states.pop(id(layer)) if comp is not None else (None, None)
            o = iface.dsv4_sparse_attn(q, nope, scale, rope, positions, layer.rotary_emb.cos_sin_cache, layer.attn_sink, state_kv,
                                       state_score, comp.norm.weight if comp is not None else None, layer.prefix,
                                       int(layer.n_local_heads), int(layer.compress_ratio), int(layer.window_size),
                                       int(ix.topk_tokens) if ix is not None else 0, float(layer.scale),
                                       float(comp.rms_norm_eps) if comp is not None else 0.0)
            output[:, :layer.n_local_heads].copy_(o)
        return run

    with hooks.Hooks() as h:
        for m in layers:
            h.patch(m, "_fused_qnorm_rope_kv_insert", insert(m))
            h.patch(m, "forward_mqa", forward_mqa(m))
            if m.compressor is not None:
                h.patch(m.compressor, "forward", compressor(m))
            if m.indexer is not None:
                h.patch(m.indexer, "forward", indexer(m))
        yield


__all__ = ["FP8_DS_MLA_QB", "dsv4_attention_leaves", "dsv4_layers"]
