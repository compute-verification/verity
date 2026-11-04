"""`AttentionSoftcap_v1{T, NH, KVH, D, BN, CAP}` — FA2 attention WITH logit soft-capping (`flash_fwd_splitkv_kernel<..., Is_softcap=true>`,
vLLM + Gemma-2 `attn_logit_softcapping = 50`), STRUCTURE ONLY: the body of `Attention_v3` ->
`AttentionHead_v3` -> `AttnBlock_v3` with the one stage FA2 adds, in the place FA2 adds it.

FA2 (`flash_api.cpp::set_params_fprop`, `flash_fwd_kernel.h::compute_attn_1rowblock`, `softmax.h::apply_softcap`):

    params.softcap           = softmax_scale / softcap            (float / float -> RN32)
    params.scale_softmax_log2 = softcap * M_LOG2E                  (float * double -> double, narrowed to float)
    acc_s = Q K^T                                                  (mma, f32 accumulators)          -- stage A  (unchanged)
    acc_s(i) = tanh.approx.f32(acc_s(i) * params.softcap)          (`cutlass::fast_tanh`, one MUFU.TANH) -- stage A'  (NEW)
    mask(acc_s); softmax_rescale_o(acc_s, acc_o, params.scale_softmax_log2)                         -- stages B1..B5, C (unchanged)

so relative to `AttnBlock_v3` (b1.py): every score `s[c] = DotBf16{K=D}(q, k[c])` becomes `s'[c] = MufuTanh(F32MulFtz(s[c], SOFTCAP_MUL))`
BEFORE the row max (B1), the rescale factor (B2) and the exp2 stage (B3), and the exp2 stage's constant is `CAP * log2e` instead of
`D^-0.5 * log2e`; everything else (lane sums, P -> bf16, O accumulation, epilogue `Fa2InvSum`) is word for word `AttnBlock_v3`.
The mask is applied after the softcap in the kernel, but masked columns contribute P = 0 exactly and are skipped here as in v3.

Conformance: NOT established.  `MufuTanh` (prims.py) is a quarantine primitive whose table is a measured-shard pilot; until the full
sm_89 table lands, evaluating this composite raises `MufuTanhUnmeasured` on any score word outside the measured shards.  The tests in
`softcap_tests.py` check the STRUCTURE (gate population = v3 + NVIS x {MufuTanh, F32MulFtz} + the two constants, the constants'
bit patterns, static wiring); `softcap_difftest.py` is the adapter for the pod difftest vs `_vllm_fa2_C.varlen_fwd(softcap=CAP)`.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.AttentionSoftcap`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import AttentionSoftcap, AttentionHeadSoftcap, AttnBlockSoftcap, softcap_mul_bits, softcap_scale_log2_bits, M_LOG2E

SEMANTIC_ID = "dense/AttentionSoftcap@v1"

__all__ = ['AttentionSoftcap', 'AttentionHeadSoftcap', 'AttnBlockSoftcap', 'softcap_mul_bits', 'softcap_scale_log2_bits', 'M_LOG2E', 'SEMANTIC_ID']
