"""Per-op bindings for the graph `torch.export(strict=False)` produces from the ACTUAL pinned vLLM model classes under the
eager batch-invariant profile (CB-A, R9 2026-09-18). Every rule here names an *operator* (a vLLM custom op, a Triton kernel
launched directly by vLLM's batch-invariant code, an ATen cast); none names a model class, a layer, or an op order.

What the export of a real vLLM class contains that the vLLM-*shaped* examples (`derive_examples.py`) did not:

* `verity_cba::triton_launch` nodes (`triton_iface.py`): under `VLLM_BATCH_INVARIANT=1` the GEMMs
  (`aten::linear` -> `linear_batch_invariant` -> `matmul_kernel_persistent`, `batch_invariant.py:1039-1106`) and the
  un-fused RMSNorm (`rms_norm_batch_invariant` -> `_rms_norm_kernel`, `:977-1036`) are Python-level Triton launches, not
  torch ops. The launch node carries the kernel name and its constexprs, which the rules check against the profile's
  kernel pins (the same pins the eager pattern matchers use) before binding `Gemm{K,N}` / `RMSNormTriton{N,EPS}`.
* `vllm::unified_kv_cache_update(k, v, layer_name) -> Tensor` + `vllm::unified_attention_with_output(q, k, v, output!,
  layer_name, output_scale?, output_block_scale?, kv_cache_dummy_dep?) -> ()` (`layers/attention/attention.py:706-784`):
  the KV write and the attention read are two ops tied by a dummy data dependency. `KvCacheUpdate[vllm]` records the
  pending write; `UnifiedAttentionWithOutput[vllm]` owns the per-layer KV history (positions from the static `positions`
  argument's row count, exactly as `UnifiedAttention[vllm]`), checks the pending write is this call's own rows, emits one
  `Attention{T=P}` per row and rebinds `output`.
* `aten.to.dtype(int32 ids, int64)` (`VocabParallelEmbedding.forward`): the I32 token id widened for `aten.embedding`;
  value-preserving for ids in [0, V) -> a reinterpretation, no gate.

Profile pins (declared record, `verity_vllm/engine/profiles/vllm_d9105ea80_sm89_eager.py:60-80`; a launch that differs is
refused, never adapted to).
"""

from __future__ import annotations

from verity_vllm.program.frontend.rules.profile import FusedAddRmsNormRule as _FusedAddRmsNormRule
from verity_vllm.program.frontend.rules.triton_iface import triton_launch
from verity_vllm.program.frontend.rules.vllm_bindings.attention import KvCacheUpdateRule, UnifiedAttentionWithOutputRule
from verity_vllm.program.frontend.rules.vllm_bindings.bf16_gemm import CublasLinearRule, CublasRouterRule, LlBf16GemmRule
from verity_vllm.program.frontend.rules.vllm_bindings.cache_pad import CachePadRule
from verity_vllm.program.frontend.rules.vllm_bindings.collectives import (AllGather2Rule, AllGatherRule, AllReduce2Rule, AllReduceRule,
                                                                          EmbeddingShardRule, FusedAllReduceRmsNormRule)
from verity_vllm.program.frontend.rules.vllm_bindings.cuda_norm import CudaFusedAddRmsNormRule, CudaRmsNormRule
from verity_vllm.program.frontend.rules.vllm_bindings.elementwise import Bf16ScalarChainRule, GeluTanhAndMulRule, ScalarTensorMulRule
from verity_vllm.program.frontend.rules.vllm_bindings.fp8 import CutlassScaledMmFp8BlockRule, PerTokenGroupFp8QuantRule, SameDtypeCastRule
from verity_vllm.program.frontend.rules.vllm_bindings.fused_norm import FusedAddRmsNormBIRule
from verity_vllm.program.frontend.rules.vllm_bindings.layer_norm import LayerNormAtenRule
from verity_vllm.program.frontend.rules.vllm_bindings.norm_chain import (AddScalarRule, AddWidenedRule, MeanDimBIRule, MulF32Rule,
                                                                         NarrowF32ToBf16Rule, RsqrtRule, SquareRule, WidenBf16ToF32Rule)
from verity_vllm.program.frontend.rules.vllm_bindings.observations import (_OBSERVED, active_allreduce_settings, active_target_profile, clear_observations,
                                                                           declared_derived_accounting, observe_attention_impls, observe_export_facts,
                                                                           observe_allreduce_settings, observe_launch_context, observe_target_profile)
from verity_vllm.program.frontend.rules.vllm_bindings.pins import (GEMM_PINS, MEAN_PINS, RMS_PINS, _pins_ok, gemm_pins_for, lookup_key,
                                                                   tuned_gemm_table_sha256)
from verity_vllm.program.frontend.rules.vllm_bindings.triton_launches import TritonGemmRule, TritonRmsNormRule
from verity_vllm.program.frontend.rules.vllm_bindings.views import AllocViewRule, IntIdWidenRule


VLLM_BINDING_RULES = [TritonRmsNormRule(), TritonGemmRule(), CublasLinearRule(), LlBf16GemmRule(), CublasRouterRule(),
                      KvCacheUpdateRule(), UnifiedAttentionWithOutputRule(), AllocViewRule(), IntIdWidenRule(),
                      CachePadRule(), CudaRmsNormRule(), CudaFusedAddRmsNormRule(), FusedAddRmsNormBIRule(), ScalarTensorMulRule(),
                      GeluTanhAndMulRule(), Bf16ScalarChainRule(),
                      # [R15 dense] the unfused norm chain (forward_native norms as ATen on an f32 copy)
                      WidenBf16ToF32Rule(), NarrowF32ToBf16Rule(), SquareRule(), MeanDimBIRule(), RsqrtRule(), AddScalarRule(), AddWidenedRule(),
                      MulF32Rule(),
                      LayerNormAtenRule(),   # GPT-NeoX's nn.LayerNorm (torch's vectorized CUDA kernel)
                      AllReduce2Rule(), AllGather2Rule(), AllReduceRule(), FusedAllReduceRmsNormRule(), AllGatherRule(), EmbeddingShardRule(),
                      # [lane vllm-tp-n] WORLD-ary rules after the world-2 ones
                      # [R15 fp8] the block-scaled FP8 linear (Fp8LinearMethod -> CutlassFp8BlockScaledMMKernel) + the same-dtype cast it ends with
                      PerTokenGroupFp8QuantRule(), CutlassScaledMmFp8BlockRule(), SameDtypeCastRule()]


def vllm_ruleset():
    """`DEFAULT_RULES` + the bindings above (the default rules already bind `_C.rotary_embedding`, `_C.silu_and_mul`, `aten.embedding`,
    `aten.argmax`, views/allocs). The default `FusedAddRmsNorm[vllm]` is REPLACED by `FusedAddRmsNorm[vllm-bi]`, which derives the
    kernel dispatch from the export instead of assuming it."""
    from verity_vllm.program.frontend.rules import DEFAULT_RULES
    from verity_vllm.program.frontend.rules.base import RuleSet
    base = [r for r in DEFAULT_RULES.rules if type(r) is not _FusedAddRmsNormRule]
    from verity_vllm.program.frontend.rules.vllm_moe import VLLM_MOE_RULES  # [R13 MOE-01] the fused-MoE block's leaf kernels
    from verity_vllm.program.frontend.rules.vllm_moe_fp8 import VLLM_MOE_FP8_RULES  # the block-FP8 fused_moe_kernel launches
    from verity_vllm.program.frontend.rules.vllm_moe_router import VLLM_MOE_ROUTER_RULES  # the one-group sigmoid router (grouped_topk)
    from verity_vllm.program.frontend.rules.vllm_dsa import (VLLM_DSA_PATTERN_RULES,  # the large q-nope/value absorb chain, before generic views
                                                             VLLM_DSA_RULES)  # DeepSeek sparse attention's leaves (export_compat.dsa_leaves)
    from verity_vllm.program.frontend.rules.vllm_sampling import VLLM_SAMPLING_RULES  # [R15 stoch] the seeded temperature + top-p token select (GumbelTopPTokenSelect_v1{V,S})
    return RuleSet("default+vllm-bindings", VLLM_DSA_PATTERN_RULES + base + VLLM_BINDING_RULES + VLLM_MOE_RULES + VLLM_MOE_FP8_RULES + VLLM_MOE_ROUTER_RULES
                   + VLLM_DSA_RULES + list(VLLM_SAMPLING_RULES), list(DEFAULT_RULES.families))


__all__ = ["AllocViewRule", "GEMM_PINS", "LayerNormAtenRule", "MEAN_PINS", "RMS_PINS", "TritonGemmRule", "VLLM_BINDING_RULES", "_OBSERVED", "_pins_ok",
           "active_allreduce_settings", "active_target_profile", "clear_observations", "declared_derived_accounting", "gemm_pins_for", "lookup_key",
           "observe_allreduce_settings", "observe_attention_impls", "observe_export_facts", "observe_launch_context", "observe_target_profile",
           "triton_launch", "tuned_gemm_table_sha256", "vllm_ruleset"]
