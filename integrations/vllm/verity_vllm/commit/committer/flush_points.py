"""The gather collector's flush points by module leaf (`native_collect`).  Torch-free: the acquisition plan (`acquire.plan`) reads the
same tables in a process where torch does not import."""

# With a reference held from the hook to the flush, only in-place overwrites bound the windows.  On Llama the three zero-window
# overwriters are RoPE after qkv_proj, the post-attention fused norm after o_proj and the next layer's input fused norm after down_proj ->
# flush (one gather launch) at exactly those hooks + step end.  Two more exist on the other architectures, where the C2 oracle sees them as
# committed bytes != the Program's value: (1) QK-norm models (OLMoE, Qwen3-*): `ops.rotary_embedding` rotates q/k IN PLACE right after
# `q_norm`/`k_norm` -> post-hook flush at the two norms; (2) fused-MoE blocks have NO down_proj hook, so the residual
# `post_attention_layernorm/1` (and, at TP, the block's received all-reduce output `<L>.mlp.experts/out`) would stay staged until the next
# layer's qkv_proj flush -- after `ops.fused_add_rms_norm(x, residual)` overwrites BOTH in place at that layer's `input_layernorm` -> flush
# BEFORE the fused-add consumer runs (pre-forward flush at `input_layernorm` and the final `norm`).  Flushing with nothing staged is a no-op
# (native flush returns early), so dense workloads pay nothing; QK-norm workloads pay two more gather launches per layer.
# declared lifetime facts (with PRE_FLUSH_LEAVES): the correspondence annotations carry no alias / in-place facts to derive them from
GATHER_FLUSH_LEAVES = frozenset({"qkv_proj", "o_proj", "down_proj", "q_norm", "k_norm"})
#: pre-forward flush points: the in-place consumers of the previous block's staged outputs (fused add-RMSNorm of residual + block output)
PRE_FLUSH_LEAVES = frozenset({"input_layernorm", "norm"})
