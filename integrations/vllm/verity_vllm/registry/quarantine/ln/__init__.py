"""verity_vllm.registry.quarantine.ln — lane `cov-ln` (coverage campaign 2026-09-17, EleutherAI/pythia-160m, `GPTNeoXForCausalLM`):
provisional exact definitions for what the LayerNorm / erf-GELU family adds over the Llama/Qwen kernel set under vLLM
d9105ea80, eager, `VLLM_BATCH_INVARIANT=1`, sm_89 — one module per definition (one `SEMANTIC_ID` each), a `*_difftest.py`
adapter next to each.

  nn.LayerNorm (weight + bias)   aten.layer_norm -> vectorized_layer_norm_kernel<bf16, float>  ->  LayerNormAten{N, EPS}
  nn.GELU() (erf)                aten.gelu(approximate='none') -> vectorized_elementwise_kernel  ->  GeluErf{N} over GeluErfBf16
                                 (GeluErfBf16 = the kernel's exhaustive measured bf16 table, `gelu_erf_table.py`)

Not here (no new Definition needed): the partial rotary (`rotary_pct = 0.25`, rot_dim = 16 of head dim 64) is the registered
`RoPE_v1{NHEADS, D=16}` on the `[M, NH, 16]` sub-view of each head (pattern `rope.rotary_embedding_partial`); the parallel
residual `x + attn + mlp` is two `ResidualAdd_v1{H}` (b1); the biases of every dense layer are `BiasAdd_v1{N}` (b1).
"""
from __future__ import annotations

from verity_vllm.registry.quarantine.ln import gelu_erf, gelu_erf_bf16, layer_norm_aten  # noqa: F401

ALL_DEFINITIONS = [layer_norm_aten.LayerNormAten, gelu_erf.GeluErf]
ALL_PRIMITIVES = [gelu_erf_bf16.GeluErfBf16]
