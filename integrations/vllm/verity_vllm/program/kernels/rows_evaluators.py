"""The replay's row evaluators as data: family -> the registry reference that evaluates its relation (`EVALUATORS`, recorded in
every report) and, for a fitted twin, what pinned it (`EVALUATOR_PROVENANCE`).  `rows` re-exports both and registers the row kernels."""
from __future__ import annotations


#: spec family -> the registry reference that evaluates its relation (recorded in every report)
EVALUATORS = {
    "Gemm_v1": "verity_vllm.program.kernels.derived_rows.dot_bf16 (DotBf16_v1: ascending k16 AmpereBF16TcDot16 chain from 0) + f2fp_bf16 (GemmCoordinate_v1); "
               # [revb M-0461] the k16 step's measured total family is named on the record, not only by the tree commit
               # (literals, held equal to registry.tc_total's constants by test_sampled_replay)
               "k16 step = registry.tc_total ampere_bf16_m16n8k16 total family v3",
    # [R17 dense, F-r16-20b / ruling R17-3] the statics-lifted GEMM every non-(Ampere, FA2) target binds (targets.gemm_spec -> b1.GemmV2):
    # `Gemm_v2{K,N,DOT}` -- the SAME VU as Gemm_v1 (one token row: x[K] @ W[N,K]^T -> [N], the instance row Q(P) selects; the registry's
    # GemmCoordinate is its interior), with the k16 accumulator step read from the DOT static of THIS row's spec: HopperBF16WgmmaDot16_v1
    # (wgmma m64nNk16: one group of 16 + acc, 26-bit adder, floor -133, the H100 measurement of record) -> derived_rows.dot_bf16_hopper;
    # AmpereBF16TcDot16_v2 -> dot_bf16.  A DOT with no registered twin is refused BY NAME (`Declined`), never another adder.
    # The ascending-from-0 chain is the cuBLASLt-without-split-K GEMM of a VLLM_BATCH_INVARIANT=1 step on cc 9.0 (vLLM keeps cuBLASLt there,
    # split-K off by workspace config), pinned word for word on the H100 at every Qwen2.5-1.5B shape x M (0 / 27.2M words; derived_rows).
    "Gemm_v2": "verity_vllm.program.kernels.derived_rows.dot_bf16_for(DOT) (DotBf16_v2{K,DOT}: ascending k16 chain of the DOT static's step from 0 -- "
               "HopperBF16WgmmaDot16_v1 = tc_dot16_hopper (16:26:-133, silicon_total.HOPPER_BF16_WGMMA_K16; cuBLASLt split-K off = the "
               "batch-invariant Linear of record), AmpereBF16TcDot16_v2 = tc_dot16) + f2fp_bf16 (GemmCoordinate_v2); "
               "k16 steps = registry.tc_total hopper_bf16_wgmma_k16 family v1 / ampere_bf16_m16n8k16 total family v3",
    # [R17 dense, ruling R17-9] the NeoX rotation of one token row's q / k heads with ONE bf16 cos/sin cache row (a `weights` field slice
    # the Program's args name by position): b1.RoPE{NHEADS,D} = NHEADS x RoPEHead{D} = prims.RopeOut / RopeOutAdd per coordinate.
    "RoPE_v1": "verity_vllm.program.kernels.derived_rows.rope_rows (RoPE_v1{NHEADS,D}: out[i] = bf16_rn(RN32(x[i]*c[i] - x[i+R]*s[i])), "
               "out[i+R] = bf16_rn(RN32(x[i+R]*c[i] + x[i]*s[i])), R = D/2, cs = cos_sin_cache[pos] -- prims.RopeOut / RopeOutAdd); "
               # [revb F-r17b-14 / M-0122] the qualification that travels with every evaluated RoPE_v1 row
               "verified against the SERVED cos_sin table, pinned by the MAN-09c analytic bound (<= 1-2 bf16 ulp on ~1-2.5 % of words), "
               "not by digest -- see weights_pin.check_cos_sin n_diff / max_ulp",
    # [R17 dense, ruling R17-11 (coord M-0136, from R17-10: the query of record is Q_module_v1, the attention ROW is the selected unit)]
    # one attention row from the committed q + the T committed K/V members the Program names by ref (every earlier step's committed word,
    # read by the resolver at the producer's step -- 'state-linked' was never a resolver gap) to the committed output: the registered
    # Definitions b1.AttentionV3 (FA2) / b1.AttentionV2 (FA3 on sm_90, DOT / INV statics resolved by name) evaluated gate for gate.
    "Attention_v3": "verity_vllm.program.kernels.derived_rows.attention_rows (Attention_v3{T,NH,KVH,D,BN}: FA2 -- GQA over AttentionHead_v3, one query "
                    "at position T-1 over the T committed K/V rows the Program names, ceil(T/BN) blocks last -> first, tc_dot16 step, Fa2InvSum); "
                    "k16 step = registry.tc_total ampere_bf16_m16n8k16 total family v3",
    "Attention_v2": "verity_vllm.program.kernels.derived_rows.attention_rows (Attention_v2{T,NH,KVH,D,BN,DOT,INV}: the DOT / INV statics resolved by name "
                    "-- HopperBF16WgmmaDot16_v1 + Fa3InvSum_v1 = FA3 on sm_90 (== verity_vllm.program.kernels.fa3_model.fa3_rows); "
                    "AmpereBF16TcDot16_v2 + Fa2InvSum_v1 = FA2 under the lifted name); "
                    # [revb M-0148 (5) / rev] the silicon envelope of the FA3 relation, stated with the evaluator
                    "FA3 envelope MEASURED exact on the H100 production capture at hdim 128 / kBlockN 128, T = 1..527 (39 leaves, "
                    "hopper-profile.md s1b); rows beyond it (1024/128, hdim 64 / kBlockN 192) are measured BY THE COMMIT -- a chain "
                    "mismatch FAILs by name, never a silent word; "
                    "k16 steps = registry.tc_total hopper_bf16_wgmma_k16 family v1 / ampere_bf16_m16n8k16 total family v3",
    "Attention_v4": "verity_vllm.program.kernels.derived_rows.attention_rows (Attention_v4{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM}: Attention_v2's evaluator "
                    "with block b guarding a running max of -inf only when b >= MASKED_FROM -- FA3's per-iteration Check_inf, "
                    "registry.fa3_check_inf; TargetProfile.fa3_construction)",
    "Attention_v5": "verity_vllm.program.kernels.derived_rows.attention_rows (Attention_v5{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM}: Attention_v2's evaluator "
                    "with block b guarding the rescale's running max of -inf only when b >= MASKED_FROM, max * scale in every block -- FA2's "
                    "per-iteration Check_inf, registry.fa2_check_inf; targets.fa2_check_inf)",
    "RMSNormTriton_v1": "verity_vllm.program.kernels.derived_rows.rmsnorm_triton_rows",
    "RMSNormFusedCuda_v2": "verity_vllm.program.kernels.derived_rows.rmsnorm_fused_v2_rows",
    "SiluMul_v1": "verity_vllm.program.kernels.derived_rows.silu_mul_rows",
    "Embedding_v1": "row gather of the registered embedding field (Embedding_v1)",
    "BiasAdd_v1": "Bf16Add on the composed interior Gemm output + the registered bias slice (one f32 add, RNE; biased linears, MAN-07)",
    "GemmBias_v1": "Gemm_v1's row (dot_bf16 + f2fp_bf16), then BiasAdd_v1's Bf16Add with the bias slice, per coordinate (GemmBiasCoordinate_v1)",
    "GemmBiasF32Epilogue_v1": "Gemm_v2's f32 chain (the DOT static's k16 twin), the bias slice widened and added in f32, then f2fp_bf16, per coordinate",
    "GemvBiasF32_v1": "verity_vllm.program.kernels.gemv_rows.gemv_bias_f32 (cuBLAS gemvx at M = 1: T strided FFMA chains, a halving tree, f32 bias)",
    "TokenSelect_v1": "first-max argmax over the bf16 logits (TokenSelect_v1: Bf16GtStrict scan, lowest index on ties)",
    # [R16 stoch, rev F-r16-20(a)] the stochastic selector IS replayed: the registered composite GumbelTopPTokenSelect_v1{V} (TemperatureScale
    # -> TopPMask_v1 with the committed `splits` S as its operand -> GumbelSelectF32: Philox draws from the committed seed + the Const64 pos)
    # evaluated by the core reference (`verity.evaluation.evaluate` on `sampling_topp_difftest.spec_registered`) on the committed bf16
    # logits row + the Program's Const32 top_p / temperature + the committed runner.sampler/seed and /splits words of the (step, request)
    "GumbelTopPTokenSelect_v1": "verity_vllm.program.registry.sampling.GumbelTopPTokenSelect{V} via verity.evaluation.evaluate (TemperatureScale -> "
                                "TopPMask_v1 (S = the committed `splits` operand) -> GumbelSelectF32 on the committed seed + Const64 pos)",
    "GumbelTopPTokenSelectSharedGreedy_v1": "GumbelTopPTokenSelect_v1's row (the same function, registry.sampling.SHARED_GREEDY; "
                                            "TargetProfile.sampler_construction = shared-greedy)",
    "GumbelTopPTokenSelect_v2": "GumbelTopPTokenSelect_v1's row at the split count S (registry.topp_words, a single-request Program's select): "
                                "`splits` = S as v1 reads it, any other value as an unlaunched count, where v1 keeps no lane either",
    # [R16 fp8] the block-scaled FP8 linear (Qwen3-4B-FP8, CutlassFp8BlockScaledMMKernel): the quant is INTERIOR (members [], composed
    # from the committed bf16 row), the GEMM row is the VU -- registered e4m3 weight + f32 `weight_scale_inv` slices (8- and 32-bit fields)
    "Fp8GroupQuant_v1": "verity_vllm.program.kernels.derived_rows.fp8_group_quant_row (per-token-group(128) e4m3 quant: exact fmaxf fold from eps, div.rn, cvt.rn.satfinite.e4m3)",
    "ScaledMmFp8Block_v1": "verity_vllm.program.kernels.derived_rows.scaled_mm_fp8_block_row (ascending k: G/32 HopperE4m3QgmmaDot32 steps from +0 per 128-block, "
                           "FFMA(temp, FMUL(sx, sw), acc), cvt.rn.bf16.f32) on the composed Fp8GroupQuant output + the registered e4m3 / f32 weight slices",
    "ScaledMmFp8BlockSharedScale_v1": "verity_vllm.program.kernels.derived_rows.scaled_mm_fp8_block_row (ScaledMmFp8Block_v1's function and signature: "
                                      "each weight block's scale products once, registry.fp8.ScaledMmFp8BlockSharedScale)",
    # [MAN-06f v3 unit 2, moe R16] the fused-MoE block relations (verity_vllm.program.registry.moe bodies; R13 moe_kernel_xcheck = the reference against
    # the real kernels): every plane word is PRODUCED by an evaluated relation over committed plane words + registered expert weights
    "MoeRouterTopK_v1": "verity_vllm.program.registry.moe.MoeRouterTopK evaluated per token with the registry's scalar primitives (f32 softmax in the kernel's "
                        "E/VPT-thread geometry + xor butterflies, NvExpf_v1 exp, F32Div rcp, non-finite -> 0, TOPK rounds of strict first-max argmax "
                        "masked to -10000.f) -> router_w f32 + router_ids i32",
    "MoeRouterTopKNorm_v1": "MoeRouterTopK then the renormalising epilogue (selected sum ascending from 0.f, denom = sum > 0 ? sum : 1, one F32Div, "
                            "one F32Mul per slot; MOE-04) -> router_w f32 + router_ids i32",
    "MoeRouterTopKOrdered_v1": "the MoeRouterTopK_v1 row (the same function in selection order: registry.moe.MoeRouterTopKOrdered) -> router_w + router_ids",
    "MoeRouterTopKOrderedNorm_v1": "the MoeRouterTopKNorm_v1 row (registry.moe.MoeRouterTopKOrderedNorm) -> router_w + router_ids",
    "MoeExpertGemm_v1": "verity_vllm.program.kernels.derived_rows.dot_bf16 (DotBf16_v1 k16 chain from 0) over the expert row W13[e] gathered by the committed "
                        "router_ids word + f2fp_bf16 (MoeExpertCoordinate_v1: the w13 launch, MUL_ROUTED_WEIGHT=False) -> up",
    "MoeExpertGemmW_v1": "dot_bf16 over W2[e] gathered by the committed router_ids word, the committed routed weight multiplied into the f32 "
                         "accumulator (one F32Mul) before the single f2fp_bf16 (MoeExpertCoordinateW_v1: the w2 launch, MUL_ROUTED_WEIGHT=True) -> down",
    # [reqval 287783ec6 / 9f7a75f6b; moe MOE-PAD-SPEC route A (3)] the PADDED fixed-slot block, one row per token (verity_vllm/program/registry/moe_pad.py
    # `_fixed_core`, node for node): MoeRouterTopK on the row's logits input (MoeFfnFixed_v1: on f2fp(dot_bf16(x, wg)) first); per slot e the
    # routed weight g_e = alpha_j where ids[j] == e else 0.0f (SelectF32 chain); EVERY slot body on its own fixed weights -- up = f2fp(dot_bf16(x,
    # W13[e])) (Gemm), act = silu_mul (SiluMul), down = f2fp(mul_rn(dot_bf16(act, W2[e]), g_e)) (MoeSlotDown = MoeExpertCoordinateW with the row
    # fixed); out = MoeSum of the selected slots' down words in id order.  ONE evaluation yields every member the query may name: `out`
    # (Q_module_v1) and the slot units + router members (Q_module_pad_v1) -- memoised per (statics, x, logits) so the picks of one token row
    # share the 64-body evaluation
    "MoeFfnFixedL_v1": "padded_moe_block: MoeRouterTopK (moe_router_topk) on the committed logits input; per slot dot_bf16 + f2fp_bf16 (up), "
                       "silu_mul_rows (act), dot_bf16 * g_e (mul_rn) + f2fp_bf16 (down); MoeSum in id order -> out + slot<e>/{up,act,down,g} + router_ids/w",
    "MoeFfnFixed_v1": "padded_moe_block with the router Gemm inside (logits = f2fp(dot_bf16(x, wg))), otherwise MoeFfnFixedL_v1",
    "MoeSum_v1": "per coordinate: f32 sum of the TOPK committed down words ascending from 0.f (F32Add) then one f2fp_bf16 (MoeSumCoordinate_v1) -> sum",
    # [R19 int-20, F-r19-int-20] `aten.mul.Tensor(x bf16 [N], s bf16 0-d TENSOR)` -- b1.Bf16MulScalarTensor{N}: ATen's bf16 mul_kernel in opmath
    # float, f32(x_j) * f32(s) (exact widenings, ONE RN multiply) stored by c10::BFloat16(float) = cvt.rn.bf16.f32 (NaN -> 0x7FFF).  The scalar is
    # a graph VALUE the row wires (arg 1: a registered buffer / input slice), never a constant of the evaluator.  Evaluated with the registry's
    # own word helpers (derived_rows: bf16_to_f32_words = Bf16ToF32, mul_rn = F32Mul, f2fp_bf16 = F2fpBf16), node for node the Definition's body.
    # When the Program's dataflow reads its output from another module it is a boundary Value of the v2 query (`model/out`) and the replay
    # evaluates it as a VU against the committed member.
    "Bf16MulScalarTensor_v1": "verity_vllm.program.kernels.derived_rows bf16_to_f32_words (Bf16ToF32) on x[N] and on the one-word scalar s, mul_rn (F32Mul: "
                              "one RN multiply), f2fp_bf16 (F2fpBf16 = cvt.rn.bf16.f32, NaN -> 0x7FFF) -- b1.Bf16MulScalarTensor{N} node for node -> out",
}

#: [R16 fp8, rev M-0100] WHAT PINNED an evaluator that is a fitted numerics twin rather than the Definition itself: the record carries it
#: beside `evaluators` so the finalizer's Commit basis can name 'FP8 GEMM strata: twin pinned to H100 words (fixture <sha>)'.  The npz
#: holds complete operand sets recorded on the H100 by the R15 fp8 lane's `out/gen/sweep/evidence/fp8/mini_probe.py`: "p" (M=1, K=256,
#: N=256; the swap_ab Pingpong CUTLASS variant) and "c" (M=4, K=384, N=256; the Cooperative variant), each = bf16 x, the kernel's xq / sx,
#: the e4m3 w and f32 ws, the kernel's bf16 y.  The twins reproduce every word (tests/program/test_derived_rows_fp8.py) and agree with
#: the gate-level Definitions gate by gate (`verity_vllm/program/registry/fp8.py`; the k32 step vs `veritor.core.silicon_total_fp8.tc_dot_total_e4m3`).
_FP8_TWIN_PROVENANCE = {
    "fixture": "tests/program/data/fp8_block_operands.npz",
    "fixture_sha256": "80e176c0319f804ed574acb3ad63aa431ba03e3f55762fe07ae63477f2bf9a81",
    "captured_from": ["_C.per_token_group_fp8_quant (GroupShape(1, 128), eps 1e-10, [-448, 448])", "_C.cutlass_scaled_mm blockwise (CutlassFp8BlockScaledMMKernel, 128x128 weight_scale_inv)"],
    "device": "cp-h100, NVIDIA H100 80GB HBM3 (cc 9.0), vLLM 0.28.1rc1.dev472+gd9105ea80, torch 2.13.0+cu129",
    "captured_utc": "2026-09-19",
    "shapes": {"p": "M=1 K=256 N=256 (swap_ab Pingpong)", "c": "M=4 K=384 N=256 (Cooperative)"},
    "tests": ["tests/program/test_derived_rows_fp8.py::test_scaled_mm_fp8_block_row_matches_the_kernel_words[p|c]",
              "tests/program/test_derived_rows_fp8.py::test_fp8_group_quant_row_matches_the_kernel_words[p|c]",
              "tests/program/test_derived_rows_fp8.py::test_twins_match_the_definitions_gate_by_gate",
              "tests/program/test_derived_rows_fp8.py::test_tc_dot32_e4m3_matches_the_total_model_on_random_operands_and_accumulators",
              "tests/program/test_fp8.py::test_fp8_block_gemm_matches_kernel[p|c] (the Definitions vs the same words)"],
    "definitions": "verity_vllm/program/registry/fp8.py: Fp8GroupQuant_v1{K,G}, ScaledMmFp8Block_v1{K,N,G}, HopperE4m3QgmmaDot32_v1 (the reference; the twins are checked against them, never the other way round)",
}
#: family -> provenance of its evaluator when it is a fitted twin (a family absent here is evaluated by the registry's own derived rows)
EVALUATOR_PROVENANCE: dict[str, dict] = {"Fp8GroupQuant_v1": _FP8_TWIN_PROVENANCE, "ScaledMmFp8Block_v1": _FP8_TWIN_PROVENANCE,
                                          "ScaledMmFp8BlockSharedScale_v1": _FP8_TWIN_PROVENANCE}

__all__ = ["EVALUATORS", "EVALUATOR_PROVENANCE"]
