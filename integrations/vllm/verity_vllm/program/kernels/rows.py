"""Row kernels: the numpy evaluators of the Program rows the sampled replay re-evaluates, one per Definition family.

A row kernel is `fn(statics, operands) -> {member: words}`: `statics` as the spec text renders them (or as a binding record
holds them), `operands[g]` the word array of the row's argument g.  It returns every member a query may name on the row: the
Definition's return value (`RowKernel.returns`, in return order) and, for the padded MoE block, its slot units and router
members too.  It raises `Declined` with the reason when the instance is outside what it reproduces (an unregistered DOT / INV
step, a wiring of another arity, a non-finite lane of a vectorised twin) and ValueError on operands the Definition does not
take.

Every row kernel is registered with core `verity.ml.circuits.evaluation` under the kernel name "rows" against its Definition id
(`kernel_registry.register`).  The core kernel returns the `returns` members flattened, and declines the instances the row
kernel declines or raises ValueError on, and those with a bf16 operand or result word that is not finite: the domain
`self_check` covers.  The replay calls the row kernel on whatever words were committed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.circuits.evaluation.batch import type_dtype, type_shape
from verity_vllm.program.kernels.kernel_registry import Declined, register
from verity_vllm.program.kernels.moe_router_rows import moe_router_topk
from verity_vllm.program.kernels import value_rows as VR

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
    **VR.EVALUATORS,
    "SiluMul_v1": "verity_vllm.program.kernels.derived_rows.silu_mul_rows",
    "Embedding_v1": "row gather of the registered embedding field (Embedding_v1)",
    "BiasAdd_v1": "Bf16Add on the composed interior Gemm output + the registered bias slice (one f32 add, RNE; biased linears, MAN-07)",
    "GemmBias_v1": "Gemm_v1's row (dot_bf16 + f2fp_bf16), then BiasAdd_v1's Bf16Add with the bias slice, per coordinate (GemmBiasCoordinate_v1)",
    "GemmBias_v2": "Gemm_v2's row (the DOT static's k16 chain + f2fp_bf16), then BiasAdd_v1's Bf16Add with the bias slice, per coordinate "
                   "(GemmBiasCoordinate_v2)",
    "TokenSelect_v1": "first-max argmax over the bf16 logits (TokenSelect_v1: Bf16GtStrict scan, lowest index on ties)",
    # [R16 stoch, rev F-r16-20(a)] the stochastic selector IS replayed: the registered composite GumbelTopPTokenSelect_v1{V} (TemperatureScale
    # -> TopPMask_v1 with the committed `splits` S as its operand -> GumbelSelectF32: Philox draws from the committed seed + the Const64 pos)
    # evaluated by the core reference (`verity.ml.circuits.evaluation.evaluate` on `sampling_topp_difftest.spec_registered`) on the committed bf16
    # logits row + the Program's Const32 top_p / temperature + the committed runner.sampler/seed and /splits words of the (step, request)
    "GumbelTopPTokenSelect_v1": "verity_vllm.program.registry.sampling.GumbelTopPTokenSelect{V} via verity.ml.circuits.evaluation.evaluate (TemperatureScale -> "
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
    # the same rows with the k16 step read from the DOT static (registry.moe MoeExpertGemm_v2 / MoeExpertGemmW_v2: a target whose
    # fused_moe_kernel tl.dot is not the Ampere/Ada step, frontend.targets.moe_expert_dot); an unregistered DOT is declined by name
    "MoeExpertGemm_v2": "derived_rows.dot_bf16_for(DOT) (DotBf16_v2{K,DOT}) over W13[e] gathered by the committed router_ids word + f2fp_bf16 "
                        "(MoeExpertCoordinate_v2) -> up",
    "MoeExpertGemmW_v2": "derived_rows.dot_bf16_for(DOT) over W2[e], the committed routed weight into the f32 accumulator (one F32Mul), then "
                         "f2fp_bf16 (MoeExpertCoordinateW_v2) -> down",
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

GEMM_CHUNK = 4096                       # Gemm coordinates evaluated per vectorised k16 chain (bounds the int64 temporaries to ~0.5 GB)
SAMPLED_MEMBER = "sampled_token_ids"

_DOT_STATIC_FN = re.compile(r'^\{"fn":"([A-Za-z0-9_]+)"\}$')


def prim_static(statics: dict[str, Any], key: str, family: str) -> str:
    """[R17 dense, ruling R17-3] A function-valued static (`DOT` of `Gemm_v2{K,N,DOT}` / `Attention_v2{..., DOT, INV}`, `INV` of the latter)
    -> the registered primitive id.  The spec text renders it as `DOT={"fn":"HopperBF16WgmmaDot16_v1"}` (verity.ml.circuits.codec._spec_id); a bare
    `<Name>_v<k>` id is accepted too.  Anything else is unresolved BY NAME: the evaluator never guesses the architecture from the target."""
    v = statics.get(key, "")
    raw = str(getattr(v, "id", v) or "")
    m = _DOT_STATIC_FN.match(raw)
    if m:
        return m.group(1)
    if re.match(r"^[A-Za-z0-9]+_v\d+$", raw):
        return raw
    raise Declined(f"{family} {key} static {raw!r} does not name a registered primitive")


def gemm_dot_prim(statics: dict[str, Any]) -> str:
    """The DOT static of a `Gemm_v2{K,N,DOT}` row -> the registered k16 primitive id (see `prim_static`)."""
    return prim_static(statics, "DOT", "Gemm_v2")


def _eps(statics: dict[str, Any]) -> float:
    """`EPS={"f64":"0x1.4f8b588e368f1p-17"}` (the spec's f64 hex literal, kept as a string by spec_statics) -> float."""
    e = statics.get("EPS")
    if isinstance(e, str) and e.lstrip().startswith("{"):
        e = json.loads(e)
    if isinstance(e, dict) and "f64" in e:
        return float.fromhex(str(e["f64"]))
    if isinstance(e, str):
        return float.fromhex(e) if e.lstrip("-").startswith("0x") else float(e)
    return float(e)


def _bf16_to_f32(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)


def token_select_first_max(logits: np.ndarray) -> int:
    """`TokenSelect_v1{V}`: scan with Bf16GtStrict -- the lowest index of the maximal bf16 logit (NaN never wins: NaN > x is false)."""
    f = _bf16_to_f32(logits).astype(np.float64)
    f = np.where(np.isnan(f), -np.inf, f)
    return int(np.argmax(f))          # np.argmax returns the FIRST maximal index


#: the registered stochastic selector's operands, in signature order (verity_vllm.program.registry.sampling.GumbelTopPTokenSelect): word width in bits
_GUMBEL_TOPP_OPERANDS = (("logits", 16), ("top_p", 32), ("seed", 64), ("pos", 64), ("temp", 32), ("splits", 32))


def gumbel_topp_token_select(statics: dict[str, Any], operands: list[np.ndarray]) -> int:
    """[R16 stoch, rev F-r16-20(a)] `GumbelTopPTokenSelect_v1{V}` on its six operand word arrays -- (bf16 logits[V], f32 top_p, i64 seed,
    i64 pos, f32 temp, i32 splits) -- evaluated by the REGISTRY (the composite of record: TemperatureScale -> TopPMask_v1{V} with `splits`
    as the S operand -> GumbelSelectF32, Philox draws from (seed, pos)); returns the sampled id.  No re-implementation: the core
    reference (`verity.ml.circuits.evaluation.evaluate`, the one-call walk the difftest (G4) and the Match-time stoch recompute run).  The words
    arrive as the committed / Const encodings; every operand is checked against the signature's leaf count before the walk (a slice
    that is not exactly one f32 word is a fault of the Program's wiring, named here, never a silent truncation)."""
    from verity.ml.circuits.evaluation import evaluate
    from verity_vllm.program.registry import sampling_topp_difftest as DT
    V = int(statics["V"])
    if len(operands) != len(_GUMBEL_TOPP_OPERANDS):
        raise ValueError(f"GumbelTopPTokenSelect_v1 takes {len(_GUMBEL_TOPP_OPERANDS)} operands, the row wires {len(operands)}")
    words: list[np.ndarray] = []
    for (pname, bits), w in zip(_GUMBEL_TOPP_OPERANDS, operands):
        a = np.asarray(w).reshape(-1)
        n = V if pname == "logits" else 1
        if a.size != n:
            raise ValueError(f"GumbelTopPTokenSelect_v1 operand {pname}: {a.size} words, the signature has {n}")
        if a.dtype.itemsize * 8 > bits and bits < 64 and int(a.max(initial=0)) >= (1 << bits):
            raise ValueError(f"GumbelTopPTokenSelect_v1 operand {pname}: a word exceeds Value<{bits}>")
        words.append(a.astype(np.uint64))
    out = evaluate(DT.spec_registered({"V": V}), *(w.tolist() for w in words))
    return int(out[0]) & 0xFFFFFFFF


def _gemm(dot: Callable, statics: dict[str, Any], ops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """f2fp_bf16(dot(x, W[n, :])) for every coordinate n of one token row: (out [N] bf16 words, non-finite lanes)."""
    from verity_vllm.program.kernels import derived_rows as D
    K, N = int(statics["K"]), int(statics["N"])
    x = np.asarray(ops[0], dtype=np.uint16).reshape(K)
    w = np.asarray(ops[1], dtype=np.uint16).reshape(N, K)
    out = np.empty(N, dtype=np.uint16)
    nonfin = np.zeros(N, dtype=bool)
    for n0 in range(0, N, GEMM_CHUNK):                                        # memory-bounded: the k16 chain is vectorised over <= GEMM_CHUNK coordinates
        acc, nf = dot(x[None, :], w[n0:n0 + GEMM_CHUNK])
        out[n0:n0 + GEMM_CHUNK] = D.f2fp_bf16(acc).astype(np.uint16)
        nonfin[n0:n0 + GEMM_CHUNK] = nf
    return out, nonfin


def gemm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Gemm_v1{K,N}`: the ascending k16 AmpereBF16TcDot16 chain from 0 per coordinate, then f2fp_bf16 (the float64 chain of
    `dense_rows` for a row past `FAST_GEMM_WORDS` weight words, word for word the same)."""
    from verity_vllm.program.kernels import derived_rows as D
    K, N = int(statics["K"]), int(statics["N"])
    if K * N >= dense_rows.FAST_GEMM_WORDS:
        out, _nonfin = dense_rows.gemm_ampere_row(np.asarray(ops[0], dtype=np.uint16).reshape(K), np.asarray(ops[1], dtype=np.uint16).reshape(N, K))
        return {"0": out, "out": out}
    out, _nonfin = _gemm(D.dot_bf16, statics, ops)
    return {"0": out, "out": out}


def gemm_dot(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Gemm_v2{K,N,DOT}`: the same row as `Gemm_v1` with the k16 step read from the DOT static, resolved to its vectorised twin by
    name.  An unregistered DOT, and a coordinate with a non-finite operand or accumulator, is declined by name."""
    from verity_vllm.program.kernels import derived_rows as D
    # Gemm_v1 is the accepted Ampere chain.  An unregistered DOT is unresolved BY NAME (the row stays 'not evaluated' with the id in
    # its why), never evaluated under another architecture's adder.
    dot_id = gemm_dot_prim(statics)
    try:
        dot = D.dot_bf16_for(dot_id)
    except KeyError:
        raise Declined(f"no registered evaluator for Gemm_v2 with DOT={dot_id} (k16 step twins: {sorted(D.TC_DOT16_BY_PRIM)})") from None
    K, N = int(statics["K"]), int(statics["N"])
    if K * N >= dense_rows.FAST_GEMM_WORDS and dot_id in dense_rows.STEPS:
        out, nonfin = dense_rows.gemm_row(np.asarray(ops[0], dtype=np.uint16).reshape(K), np.asarray(ops[1], dtype=np.uint16).reshape(N, K), dot_id)
    else:
        out, nonfin = _gemm(dot, statics, ops)
    if nonfin.any():
        # the vectorised twins flag a non-finite operand / accumulator lane instead of modelling the total semantics (tc_dot16 docstring):
        # such a coordinate is outside the twin's domain -> unresolved by name (the ScaledMmFp8Block convention), never a silent word
        raise Declined(f"Gemm_v2: {int(nonfin.sum())} output coordinates have a non-finite bf16 operand or accumulator "
                       "(outside the vectorised k16 twin's domain)")
    return {"0": out, "out": out}


def rope(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`RoPE_v1{NHEADS,D}`: arg 0 = the q (or k) slice of the biased qkv row (NHEADS*D bf16 words), arg 1 = ONE row of the registered
    `weights` field `...rotary_emb.cos_sin_cache` (D bf16 words: [cos[0:D/2] | sin[0:D/2]] of THIS row's position -- the Program's
    slice names the position, nothing here infers it).  The row's one member is `0` (query) or `1` (key); the relation is the same,
    the manifest names the member, so both names are returned."""
    from verity_vllm.program.kernels import derived_rows as D
    NH, Dh = int(statics["NHEADS"]), int(statics["D"])
    x = np.asarray(ops[0], dtype=np.uint16).reshape(NH, Dh)
    cs = np.asarray(ops[1], dtype=np.uint16).reshape(Dh)
    out = D.rope_rows(x, cs).reshape(NH * Dh)
    return {"0": out, "1": out, "out": out}


def attention(family: str, statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """One attention ROW of the Program (the Q_module_v1 unit): arg 0 = the RoPE query member (NH*D words); arg 1 = the T committed KEY
    members (rotary_emb member `1`, KVH*D words each) of every token row this query attends to, in Program order = key position;
    arg 2 = the T committed V slices of the same rows' qkv members.  Every K/V value is a named committed word of the request (any
    earlier step: the resolver reads it at the producer's step) -- nothing is inferred from a cache.  Attention_v3 is FA2
    (Ampere/Ada: tc_dot16, Fa2InvSum); Attention_v2 binds the step and the epilogue as statics (Hopper FA3: wgmma step, Fa3InvSum)
    -- resolved BY NAME to their twins, any other id is unresolved by name."""
    from verity_vllm.program.kernels import derived_rows as D
    # The AttnBlock INTERIORS inside the row (Q_fine_v1 units) live in the M1 export stream, which the replay does not read -- the
    # report names that mechanism with the row count (ATTENTION_INTERIOR_MECHANISM).
    T, NH, KVH, Dh, BN = (int(statics[k]) for k in ("T", "NH", "KVH", "D", "BN"))
    arity = 4 if family in VR.SCALE_INPUT else 3
    if len(ops) != arity:                                                    # [R17-14] the Definition's arity (q, kc, vc[, scale_log2]): a row wiring another is by name
        raise Declined(f"{family}: the row wires {len(ops)} args, the registered Definition takes {arity} (q, K history, V history{', scale_log2' * (arity == 4)})")
    if family in ("Attention_v2", "Attention_v4", "Attention_v5", "Attention_v11"):
        dot_id, inv_id = prim_static(statics, "DOT", family), prim_static(statics, "INV", family)
    else:
        dot_id, inv_id = "AmpereBF16TcDot16_v2", "Fa2InvSum_v1"
    masked_from = int(statics["MASKED_FROM"]) if family in ("Attention_v4", "Attention_v5", "Attention_v11") else None
    scale_bits = VR.word32(ops[3], family, "scale_log2") if arity == 4 else None
    if dot_id not in D.TC_DOT16_BY_PRIM or inv_id not in D.INV_SUM_BY_PRIM:
        raise Declined(f"no registered evaluator for {family} with DOT={dot_id}, INV={inv_id} "
                       f"(twins: {sorted(D.TC_DOT16_BY_PRIM)} x {sorted(D.INV_SUM_BY_PRIM)})")
    q = np.asarray(ops[0], dtype=np.uint16).reshape(NH, Dh)
    kc = np.asarray(ops[1], dtype=np.uint16)
    vc = np.asarray(ops[2], dtype=np.uint16)
    if kc.size != T * KVH * Dh or vc.size != T * KVH * Dh:
        raise Declined(f"{family}: K history {kc.size} / V history {vc.size} words != T*KVH*D = {T}*{KVH}*{Dh}")
    try:
        out, nonfin = D.attention_rows(q, kc.reshape(T, KVH, Dh), vc.reshape(T, KVH, Dh), BN, dot_prim=dot_id, inv_prim=inv_id,
                                       masked_from=masked_from, rescale_only=family in ("Attention_v5", "Attention_v11"), scale_bits=scale_bits)
    except ValueError as e:                                                  # a geometry outside the registered relation: by name
        raise Declined(f"{family}: {e}") from None
    if nonfin.any():
        raise Declined(f"{family}: {int(nonfin.sum())} heads have a non-finite operand / accumulator (outside the vectorised twins' domain)")
    return {"out": np.asarray(out, dtype=np.uint16).reshape(NH * Dh)}


def rmsnorm_triton(statics: dict[str, Any], ops: Sequence[np.ndarray], family: str = "RMSNormTriton_v1") -> dict[str, np.ndarray]:
    from verity_vllm.program.kernels import derived_rows as D
    N = int(statics["N"])
    r = D.rmsnorm_triton_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, N), np.asarray(ops[1], dtype=np.uint16).reshape(N), _eps(statics) if "EPS" in statics else VR.eps(ops, 2, family))
    return {"out": np.asarray(r["out"], dtype=np.uint16).reshape(N)}


def rmsnorm_fused(statics: dict[str, Any], ops: Sequence[np.ndarray], family: str = "RMSNormFusedCuda_v2") -> dict[str, np.ndarray]:
    from verity_vllm.program.kernels import derived_rows as D
    N = int(statics["N"])
    r = D.rmsnorm_fused_v2_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, N), np.asarray(ops[1], dtype=np.uint16).reshape(1, N),
                                np.asarray(ops[2], dtype=np.uint16).reshape(1, N), _eps(statics) if "EPS" in statics else VR.eps(ops, 3, family))
    return {"0": np.asarray(r["normed"], dtype=np.uint16).reshape(N), "1": np.asarray(r["resid"], dtype=np.uint16).reshape(N)}


def silu_mul(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    from verity_vllm.program.kernels import derived_rows as D
    I = int(statics["I"])
    r = D.silu_mul_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, 2 * I))
    return {"out": np.asarray(r["out"], dtype=np.uint16).reshape(I)}


def embedding(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    H = int(statics["H"])
    ids = np.asarray(ops[0]).reshape(-1)
    tid = int(ids[0])
    table = np.asarray(ops[1], dtype=np.uint16).reshape(-1, H)
    if not 0 <= tid < table.shape[0]:
        raise ValueError(f"token id {tid} outside the embedding table ({table.shape[0]} rows)")
    return {"out": table[tid].copy()}


def bias_add(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    from verity_vllm.program.kernels import derived_rows as D
    N = int(statics["N"])
    y = D.as_f32(D.bf16_to_f32_words(np.asarray(ops[0], dtype=np.uint16).reshape(N)))
    b = D.as_f32(D.bf16_to_f32_words(np.asarray(ops[1], dtype=np.uint16).reshape(N)))
    return {"0": D.f2fp_bf16(D.as_u32((y + b).astype(np.float32)))}         # Bf16Add: one f32 add, RNE to bf16


def gemm_bias(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`GemmBias_v1{K,N}(x, W, b)`: `Gemm_v1`'s row, then `BiasAdd_v1`'s add of the bias slice, per coordinate."""
    if len(ops) != 3:
        raise ValueError(f"GemmBias_v1 takes 3 operands (x, W, b), the row wires {len(ops)}")
    y = gemm(statics, (ops[0], ops[1]))["0"]                                  # the replay's operands index, they don't slice
    return bias_add({"N": statics["N"]}, (y, ops[2]))


def gemm_bias_dot(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`GemmBias_v2{K,N,DOT}(x, W, b)`: `Gemm_v2`'s row, then `BiasAdd_v1`'s add of the bias slice, per coordinate."""
    if len(ops) != 3:
        raise ValueError(f"GemmBias_v2 takes 3 operands (x, W, b), the row wires {len(ops)}")
    y = gemm_dot(statics, (ops[0], ops[1]))["0"]
    return bias_add({"N": statics["N"]}, (y, ops[2]))


def mul_scalar_tensor(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """b1.Bf16MulScalarTensor{N}: out_j = F2fpBf16(F32Mul(Bf16ToF32(x_j), Bf16ToF32(s))) -- arg 0 the N bf16 words of the row, arg 1
    the ONE bf16 word of the scalar tensor (the Definition's `s: Array(1, BF16)`); another arity / width is by name."""
    from verity_vllm.program.kernels import derived_rows as D
    family = "Bf16MulScalarTensor_v1"
    N = int(statics["N"])
    if len(ops) != 2:
        raise Declined(f"{family}: the row wires {len(ops)} args, the registered Definition takes 2 (x, s)")
    x = np.asarray(ops[0], dtype=np.uint16).reshape(-1)
    s = np.asarray(ops[1], dtype=np.uint16).reshape(-1)
    if x.size != N:
        raise Declined(f"{family}: x has {x.size} words, the signature has N = {N}")
    if s.size != 1:
        raise Declined(f"{family}: the scalar operand has {s.size} words, the signature has 1 (Array(1, BF16))")
    xf = D.bf16_to_f32_words(x)
    sf = np.broadcast_to(D.bf16_to_f32_words(s), xf.shape)
    return {"out": D.f2fp_bf16(D.mul_rn(xf, sf)).astype(np.uint16)}


def token_select(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    V = int(statics["V"])
    return {SAMPLED_MEMBER: np.asarray([token_select_first_max(np.asarray(ops[0], dtype=np.uint16).reshape(V))], dtype=np.uint64)}


def gumbel_topp(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    return {SAMPLED_MEMBER: np.asarray([gumbel_topp_token_select(statics, [ops[g] for g in range(len(ops))])], dtype=np.uint64)}


def gumbel_topp_split(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`GumbelTopPTokenSelect_v2{V,S}`: v1 wherever `splits` is S; any other value keeps no lane, as v1 at the unlaunched count 0 does."""
    ops = list(ops)
    s = np.asarray(ops[5])
    ops[5] = np.where(s == int(statics["S"]), s, 0).astype(s.dtype)
    return gumbel_topp(statics, ops)


def fp8_group_quant(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Interior of the FP8 linear: ports 0 (K e4m3 words) and 1 (K/G f32 group scales), in the Definition's return order -- the
    consumer's `['n', row, lo, hi]` slices index the concatenation."""
    from verity_vllm.program.kernels import derived_rows as D
    K, G = int(statics["K"]), int(statics["G"])
    xq, sx = D.fp8_group_quant_row(np.asarray(ops[0], dtype=np.uint16).reshape(K), G=G)
    return {"0": np.asarray(xq, dtype=np.uint8), "1": np.asarray(sx, dtype=np.uint32)}


def scaled_mm_fp8_block(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`_C.cutlass_scaled_mm` blockwise on one token row: xq / sx composed from the interior quant, w (N x K e4m3) and ws (cdiv(N, G) x
    K/G f32, `weight_scale_inv`) the registered checkpoint words.  A lane with a NaN e4m3 byte or a non-finite accumulator is outside
    the twin's domain -> unresolved by name (the tc_dot16 convention), never a silent word."""
    from verity_vllm.program.kernels import derived_rows as D
    K, N, G = int(statics["K"]), int(statics["N"]), int(statics["G"])
    xq = np.asarray(ops[0], dtype=np.uint8).reshape(K)
    sx = np.asarray(ops[1], dtype=np.uint32).reshape(K // G)
    w = np.asarray(ops[2], dtype=np.uint8).reshape(N, K)
    ws = np.asarray(ops[3], dtype=np.uint32).reshape(-(-N // G), K // G)
    out, nonfin = D.scaled_mm_fp8_block_row(xq, sx, w, ws, G=G, chunk=GEMM_CHUNK)
    if nonfin.any():
        raise Declined(f"ScaledMmFp8Block: {int(nonfin.sum())} output coordinates have a NaN e4m3 operand or a non-finite accumulator "
                       "(outside the vectorised twin's domain)")
    return {"0": np.asarray(out, dtype=np.uint16)}


def moe_router(statics: dict[str, Any], ops: Sequence[np.ndarray], *, renormalize: bool) -> dict[str, np.ndarray]:
    E, TOPK, VPT = int(statics["E"]), int(statics["TOPK"]), int(statics.get("VPT", 8))
    w, ids = moe_router_topk(np.asarray(ops[0], dtype=np.uint16).reshape(E), E=E, TOPK=TOPK, VPT=VPT, renormalize=renormalize)
    return {"router_w": w, "router_ids": ids}


def moe_expert_gemm(statics: dict[str, Any], ops: Sequence[np.ndarray], *, routed: bool, dotted: bool = False) -> dict[str, np.ndarray]:
    """`MoeExpertGemm_v1(x, e, W13)` (up) and, `routed`, `MoeExpertGemmW_v1(x, e, g, W2)` (down): dot_bf16 over the expert row gathered
    by the committed router_ids word, the routed weight multiplied into the f32 accumulator before the one bf16 conversion.  `dotted`:
    the `_v2{E,K,N,DOT}` rows, the chain of the DOT static's step; an unregistered DOT and a non-finite lane are declined by name."""
    from verity_vllm.program.kernels import derived_rows as D
    dot = D.dot_bf16
    if dotted:
        fam = "MoeExpertGemmW_v2" if routed else "MoeExpertGemm_v2"
        dot_id = prim_static(statics, "DOT", fam)
        try:
            dot = D.dot_bf16_for(dot_id)
        except KeyError:
            raise Declined(f"no registered evaluator for {fam} with DOT={dot_id} (k16 step twins: {sorted(D.TC_DOT16_BY_PRIM)})") from None
    E, K, N = int(statics["E"]), int(statics["K"]), int(statics["N"])
    x = np.asarray(ops[0], dtype=np.uint16).reshape(K)
    e = int(np.asarray(ops[1]).reshape(-1)[0])
    if not 0 <= e < E:
        raise ValueError(f"routed expert id {e} outside the expert axis (E = {E})")
    wi = 3 if routed else 2
    wall = np.asarray(ops[wi], dtype=np.uint16)
    if wall.size != E * N * K:
        raise ValueError(f"expert weight slab has {wall.size} words, not E*N*K = {E}*{N}*{K}")
    w = wall.reshape(E, N, K)[e]                                              # the gather by the VALUE operand e (MoeExpertRow): one expert's [N, K]
    gw = np.asarray(ops[2]).reshape(-1)[0] if routed else None
    out = np.empty(N, dtype=np.uint16)
    for n0 in range(0, N, GEMM_CHUNK):
        acc, nf = dot(x[None, :], w[n0:n0 + GEMM_CHUNK])
        if dotted and nf.any():
            raise Declined(f"{fam}: {int(nf.sum())} output coordinates have a non-finite bf16 operand or accumulator "
                           "(outside the vectorised k16 twin's domain)")
        if gw is not None:                                                    # routed weight into the f32 accumulator BEFORE the one bf16 conversion
            acc = D.mul_rn(acc, np.full(acc.shape, int(gw) & 0xFFFFFFFF, dtype=np.uint32))
        out[n0:n0 + GEMM_CHUNK] = D.f2fp_bf16(acc).astype(np.uint16)
    return {"down" if gw is not None else "up": out}


def moe_ffn_fixed(statics: dict[str, Any], ops: Sequence[np.ndarray], *, router_gemm: bool) -> dict[str, np.ndarray]:
    return padded_moe_block(statics, [ops[g] for g in range(len(ops))], router_gemm=router_gemm)


def moe_sum(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    from verity_vllm.program.kernels import derived_rows as D
    TOPK, H = int(statics["TOPK"]), int(statics["H"])
    y = np.asarray(ops[0], dtype=np.uint16).reshape(TOPK, H)
    acc = np.zeros(H, dtype=np.uint32)                                        # Const32[0]; ascending slot order, one RNE at the end
    for k in range(TOPK):
        acc = D.add_rn(acc, D.bf16_to_f32_words(y[k]))
    return {"sum": D.f2fp_bf16(acc).astype(np.uint16)}


_PADDED_BLOCK_MEMO: "collections.OrderedDict[tuple, dict[str, np.ndarray]]" = collections.OrderedDict()
_PADDED_BLOCK_MEMO_MAX = 4


def padded_moe_block(statics: dict[str, Any], args: list[np.ndarray], *, router_gemm: bool) -> dict[str, np.ndarray]:
    """`MoeFfnFixedL_v1{E,TOPK,VPT,H,I}(x, logits, w13, w2)` (and `MoeFfnFixed_v1(x, wg, w13, w2)` with `router_gemm`) on one token row --
    the vectorised twin of `verity_vllm.program.registry.moe_pad._fixed_core`, statement for statement, built from the twins the MOE-01 evaluators
    already use (dot_bf16 / f2fp_bf16 / silu_mul_rows / mul_rn / add_rn / moe_router_topk).  Returns EVERY member a query may name on the
    row: `out` [H] bf16, `router_ids` [TOPK] i32, `router_w` [TOPK] f32, and per slot e `slot<e>/up` [2I] bf16, `slot<e>/act` [I] bf16,
    `slot<e>/down` [H] bf16, `slot<e>/g` [1] f32 (0.0f where the router did not pick e; the body is evaluated for EVERY slot, as the
    construction computes it).  Memoised on (statics, x, logits/wg) so the picks of one token row share one 64-body evaluation."""
    from verity_vllm.program.kernels import derived_rows as D

    E, TOPK, VPT, H, I = int(statics["E"]), int(statics["TOPK"]), int(statics.get("VPT", 8)), int(statics["H"]), int(statics["I"])
    if len(args) != 4:
        raise ValueError(f"padded MoE block takes 4 operands (x, {'wg' if router_gemm else 'logits'}, w13, w2), the row wires {len(args)}")
    x = np.asarray(args[0], dtype=np.uint16).reshape(H)
    a1 = np.asarray(args[1], dtype=np.uint16).reshape(-1)
    w13 = np.asarray(args[2], dtype=np.uint16)
    w2 = np.asarray(args[3], dtype=np.uint16)
    if w13.size != E * 2 * I * H:
        raise ValueError(f"w13 slab has {w13.size} words, not E*2I*H = {E}*{2 * I}*{H}")
    if w2.size != E * H * I:
        raise ValueError(f"w2 slab has {w2.size} words, not E*H*I = {E}*{H}*{I}")
    # [revb F-r17b-52 minor (b)] the key covers the four operands: x, logits/wg AND the two weight slabs (a flipped W13 bit must recompute)
    key = (E, TOPK, VPT, H, I, bool(router_gemm), hashlib.sha256(x.tobytes()).digest(), hashlib.sha256(a1.tobytes()).digest(),
           hashlib.sha256(np.ascontiguousarray(w13).tobytes()).digest(), hashlib.sha256(np.ascontiguousarray(w2).tobytes()).digest())
    hit = _PADDED_BLOCK_MEMO.get(key)
    if hit is not None:
        _PADDED_BLOCK_MEMO.move_to_end(key)
        return hit

    def gemm(v: np.ndarray, w: np.ndarray, g: int | None = None) -> np.ndarray:
        """f2fp(dot_bf16(v, w[n, :])) per coordinate, the routed weight multiplied into the f32 accumulator first when `g` is given."""
        N = w.shape[0]
        out = np.empty(N, dtype=np.uint16)
        for n0 in range(0, N, GEMM_CHUNK):
            acc, _nf = D.dot_bf16(v[None, :], w[n0:n0 + GEMM_CHUNK])
            if g is not None:
                acc = D.mul_rn(acc, np.full(acc.shape, int(g) & 0xFFFFFFFF, dtype=np.uint32))
            out[n0:n0 + GEMM_CHUNK] = D.f2fp_bf16(acc).astype(np.uint16)
        return out

    if router_gemm:
        if a1.size != E * H:
            raise ValueError(f"wg has {a1.size} words, not E*H = {E}*{H}")
        logits = gemm(x, a1.reshape(E, H))
    else:
        if a1.size != E:
            raise ValueError(f"logits input has {a1.size} words, not E = {E}")
        logits = a1
    alpha, ids = moe_router_topk(logits, E=E, TOPK=TOPK, VPT=VPT, renormalize=False)
    alpha = np.asarray(alpha, dtype=np.uint32).reshape(TOPK)
    ids = np.asarray(ids, dtype=np.uint32).reshape(TOPK)
    w13 = w13.reshape(E, 2 * I, H)
    w2 = w2.reshape(E, H, I)
    res: dict[str, np.ndarray] = {"router_ids": ids.copy(), "router_w": alpha.copy()}
    downs = np.empty((E, H), dtype=np.uint16)
    for e in range(E):
        g = 0                                                                     # ZERO32; SelectF32 chain in reversed j order -> the lowest j that hits e
        for j in reversed(range(TOPK)):
            if int(ids[j]) == e:
                g = int(alpha[j])
        up = gemm(x, w13[e])
        act = np.asarray(D.silu_mul_rows(up.reshape(1, 2 * I))["out"], dtype=np.uint16).reshape(I)
        down = gemm(act, w2[e], g)
        downs[e] = down
        res[f"slot{e}/up"], res[f"slot{e}/act"], res[f"slot{e}/down"] = up, act, down
        res[f"slot{e}/g"] = np.asarray([g], dtype=np.uint32)
    acc = np.zeros(H, dtype=np.uint32)                                            # MoeSum: Const32[0]; the selected slots in id order, one RNE
    for j in range(TOPK):
        e = int(ids[j])
        if not 0 <= e < E:
            raise ValueError(f"router id {e} outside the expert axis (E = {E})")
        acc = D.add_rn(acc, D.bf16_to_f32_words(downs[e]))
    res["out"] = D.f2fp_bf16(acc).astype(np.uint16)
    _PADDED_BLOCK_MEMO[key] = res
    while len(_PADDED_BLOCK_MEMO) > _PADDED_BLOCK_MEMO_MAX:
        _PADDED_BLOCK_MEMO.popitem(last=False)
    return res


@dataclass(frozen=True)
class RowKernel:
    family: str
    fn: Callable[[dict[str, Any], Sequence[np.ndarray]], dict[str, np.ndarray]]
    returns: tuple[str, ...]


KERNEL = "rows"
DOMAIN = ("instances whose bf16 operand and result words are all finite, outside the row kernel's own declines (an unregistered "
          "DOT / INV step, a non-finite lane of a vectorised twin, an out-of-range index)")
ROWS: dict[str, RowKernel] = {}


_FINITE_CHUNK = 1 << 22                                                        # words per scan: a weight table is never masked whole


def _finite16(w: np.ndarray) -> bool:
    w = np.asarray(w).reshape(-1)
    return not any(bool(((w[i:i + _FINITE_CHUNK] & 0x7F80) == 0x7F80).any()) for i in range(0, w.size, _FINITE_CHUNK))


def _instance(k: RowKernel) -> Callable[..., np.ndarray]:
    def fn(*args: np.ndarray, **statics: Any) -> np.ndarray:
        if not all(_finite16(a) for a in args if a.dtype == np.uint16):
            raise Declined("a bf16 operand word is not finite")
        got = k.fn(statics, list(args))
        words = [np.asarray(got[m]).reshape(-1) for m in k.returns]
        if not all(_finite16(w) for w in words if w.dtype == np.uint16):
            raise Declined("a bf16 result word is not finite")
        return np.concatenate([w.astype(np.uint64) for w in words])
    return fn


def _words(rng: np.random.Generator, shape: tuple[int, ...], width: int) -> np.ndarray:
    """Finite random words of one leaf width: bf16 at a random scale per instance with a few zeros, subnormals and +-1, f32 normal,
    e4m3 without its two NaN bytes, 64-bit uniform."""
    if width == 16:
        scale = rng.choice([0.02, 1.0, 40.0], size=(shape[0],) + (1,) * (len(shape) - 1))
        w = ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        edge = rng.choice(np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x3F80, 0xBF80], dtype=np.uint16), size=shape)
        return np.where(rng.random(shape) < 1 / 32, edge, w)
    if width == 32:
        return rng.standard_normal(shape).astype(np.float32).view(np.uint32)
    if width == 8:
        return (rng.integers(0, 0x7F, shape) | (rng.integers(0, 2, shape) << 7)).astype(np.uint8)
    if width == 64:
        return rng.integers(0, np.iinfo(np.int64).max, shape).astype(np.uint64)
    raise TypeError(f"no sampler for Value<{width}>")


def _sample(index: dict[int, str] | None = None) -> Callable[..., list[np.ndarray]]:
    """A self-check sampler: `_words` per parameter; `index` maps a parameter position to the static bounding it (a token or expert
    id), drawn in range but for one in sixteen."""
    def sample(rng: np.random.Generator, n: int, params: Sequence[Any], **statics: Any) -> list[np.ndarray]:
        out = []
        for j, pt in enumerate(params):
            shape = (n, *type_shape(pt))
            if index and j in index:
                bound = int(statics[index[j]])
                w = np.where(rng.random(shape) < 1 / 16, bound + rng.integers(0, 4, shape), rng.integers(0, bound, shape))
            else:
                widths = set(pt.leaf_widths())
                if len(widths) != 1:
                    raise TypeError(f"no sampler for a parameter of mixed widths {sorted(widths)}")
                w = _words(rng, shape, widths.pop())
            out.append(np.asarray(w).astype(type_dtype(pt)))
        return out
    return sample


def _sample_gumbel(rng: np.random.Generator, n: int, params: Sequence[Any], **statics: Any) -> list[np.ndarray]:
    """logits, top_p in [0.05, 1], seed, pos, temperature in [0.3, 2], and splits S in {1, 2, 4, 8}."""
    f32 = lambda a: np.asarray(a, dtype=np.float32).view(np.uint32)  # noqa: E731
    return [_words(rng, (n, int(statics["V"])), 16), f32(rng.uniform(0.05, 1.0, n)), _words(rng, (n,), 64),
            rng.integers(0, 1 << 20, n).astype(np.uint64), f32(rng.uniform(0.3, 2.0, n)), rng.choice([1, 2, 4, 8], n).astype(np.uint32)]


def _sample_gumbel_split(rng: np.random.Generator, n: int, params: Sequence[Any], **statics: Any) -> list[np.ndarray]:
    """`_sample_gumbel` with splits = S on three rows in four (a Program's `Const32[S]`), another count on the rest."""
    w = _sample_gumbel(rng, n, params, **statics)
    w[5] = np.where(rng.random(n) < 0.75, int(statics["S"]), w[5]).astype(np.uint32)
    return w


def _check_targets() -> dict[str, list]:
    """Family -> the small specialisations its self-check binds (the served statics are too wide for the gate reference)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import b1, fa2_check_inf, fa3_check_inf, fp8, gemm_bias, hopper, moe, moe_pad, sampling, topp_words
    P = b1.P
    fa2 = {"DOT": P.AmpereBF16TcDot16, "INV": P.Fa2InvSum}
    fa3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
    attn = {"T": 20, "NH": 2, "KVH": 1, "D": 16, "BN": 16}
    router = {"E": 8, "TOPK": 2, "VPT": 8}
    return {
        "Gemm_v1": [bind(b1.Gemm, K=32, N=3)],
        "Gemm_v2": [bind(b1.GemmV2, K=32, N=3, DOT=P.AmpereBF16TcDot16), bind(b1.GemmV2, K=32, N=3, DOT=hopper.HopperBF16WgmmaDot16)],
        "RoPE_v1": [bind(b1.RoPE, NHEADS=2, D=8)],
        "Attention_v3": [bind(b1.AttentionV3, **attn)],
        "Attention_v2": [bind(b1.AttentionV2, **attn, **fa2), bind(b1.AttentionV2, **attn, **fa3)],
        # T = 40, BN = 16: three blocks; MASKED_FROM 2 leaves blocks 1 and 0 unguarded, 0 guards every block (= Attention_v2)
        "Attention_v4": [bind(fa3_check_inf.AttentionV4, **dict(attn, T=40), **fa3, MASKED_FROM=m) for m in (2, 0)],
        "Attention_v5": [bind(fa2_check_inf.AttentionV5, **dict(attn, T=40), DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum, MASKED_FROM=m)
                         for m in (2, 0)],
        "RMSNormTriton_v1": [bind(b1.RMSNormTriton, N=16, EPS=1e-6)],
        "RMSNormFusedCuda_v2": [bind(b1.RMSNormFusedCuda, N=16, EPS=1e-6)],
        **VR.check_targets(),
        "SiluMul_v1": [bind(b1.SiluMul, I=8)],
        "Embedding_v1": [bind(b1.Embedding, V=5, H=4)],
        "BiasAdd_v1": [bind(b1.BiasAdd, N=8)],
        "GemmBias_v1": [bind(gemm_bias.GemmBias, K=32, N=3)],
        "GemmBias_v2": [bind(gemm_bias.GemmBiasV2, K=32, N=3, DOT=P.AmpereBF16TcDot16),
                        bind(gemm_bias.GemmBiasV2, K=32, N=3, DOT=hopper.HopperBF16WgmmaDot16)],
        "Bf16MulScalarTensor_v1": [bind(b1.Bf16MulScalarTensor, N=8)],
        "TokenSelect_v1": [bind(b1.TokenSelect, V=16)],
        "GumbelTopPTokenSelect_v1": [bind(sampling.GumbelTopPTokenSelect, V=16)],
        "GumbelTopPTokenSelectSharedGreedy_v1": [bind(sampling.GumbelTopPTokenSelectSharedGreedy, V=16)],
        "GumbelTopPTokenSelect_v2": [bind(topp_words.GumbelTopPTokenSelectV2, V=16, S=4)],
        "Fp8GroupQuant_v1": [bind(fp8.Fp8GroupQuant, K=256, G=128)],
        "ScaledMmFp8Block_v1": [bind(fp8.ScaledMmFp8Block, K=128, N=n, G=128) for n in (128, 192)],
        "ScaledMmFp8BlockSharedScale_v1": [bind(fp8.ScaledMmFp8BlockSharedScale, K=128, N=n, G=128) for n in (128, 192)],
        "MoeRouterTopK_v1": [bind(moe.MoeRouterTopK, **router)],
        "MoeRouterTopKNorm_v1": [bind(moe.MoeRouterTopKNorm, **router)],
        "MoeRouterTopKOrdered_v1": [bind(moe.MoeRouterTopKOrdered, **router)],
        "MoeRouterTopKOrderedNorm_v1": [bind(moe.MoeRouterTopKOrderedNorm, **router)],
        "MoeExpertGemm_v1": [bind(moe.MoeExpertGemm, E=2, K=32, N=3)],
        "MoeExpertGemmW_v1": [bind(moe.MoeExpertGemmW, E=2, K=32, N=3)],
        "MoeExpertGemm_v2": [bind(moe.MoeExpertGemmV2, E=2, K=32, N=3, DOT=d) for d in (P.AmpereBF16TcDot16, hopper.HopperBF16WgmmaDot16)],
        "MoeExpertGemmW_v2": [bind(moe.MoeExpertGemmWV2, E=2, K=32, N=3, DOT=d) for d in (P.AmpereBF16TcDot16, hopper.HopperBF16WgmmaDot16)],
        "MoeFfnFixedL_v1": [bind(moe_pad.MoeFfnFixedL, **router, H=16, I=16)],
        "MoeFfnFixed_v1": [bind(moe_pad.MoeFfnFixed, **router, H=16, I=16)],
        "MoeSum_v1": [bind(moe.MoeSum, TOPK=2, H=8)],
    }


def _row(family: str, fn: Callable, returns: tuple[str, ...], sample: Callable | None = None, check_targets: Callable | None = None, batch: Callable | None = None) -> None:
    k = ROWS[family] = RowKernel(family, fn, returns)
    register(family, _instance(k), kernel=KERNEL, domain=DOMAIN, sample=sample or _sample(),
             check_targets=check_targets or (lambda rng: _check_targets()[family]), batch=batch)


_row("Gemm_v1", gemm, ("0",))
_row("Gemm_v2", gemm_dot, ("0",), batch=lambda x, w, **statics: dense_rows.gemm_dot_rows(gemm_dot_prim, x, w, **statics))
_row("RoPE_v1", rope, ("0",))
_row("Attention_v3", partial(attention, "Attention_v3"), ("out",))
_row("Attention_v2", partial(attention, "Attention_v2"), ("out",))
_row("Attention_v4", partial(attention, "Attention_v4"), ("out",))
_row("Attention_v5", partial(attention, "Attention_v5"), ("out",))
_row("RMSNormTriton_v1", rmsnorm_triton, ("out",))
_row("RMSNormFusedCuda_v2", rmsnorm_fused, ("0", "1"))
_row("Attention_v10", partial(attention, "Attention_v10"), ("out",), VR.sample_value(_sample(), VR.scale_words))
_row("Attention_v11", partial(attention, "Attention_v11"), ("out",), VR.sample_value(_sample(), VR.scale_words))
_row("RMSNormTriton_v3", partial(rmsnorm_triton, family="RMSNormTriton_v3"), ("out",), VR.sample_value(_sample(), VR.eps_words))
_row("RMSNormFusedCuda_v4", partial(rmsnorm_fused, family="RMSNormFusedCuda_v4"), ("0", "1"), VR.sample_value(_sample(), VR.eps_words))
_row("SiluMul_v1", silu_mul, ("out",))
_row("Embedding_v1", embedding, ("out",), _sample({0: "V"}))
_row("BiasAdd_v1", bias_add, ("0",))
_row("GemmBias_v1", gemm_bias, ("0",))
_row("GemmBias_v2", gemm_bias_dot, ("0",))
_row("Bf16MulScalarTensor_v1", mul_scalar_tensor, ("out",))
_row("TokenSelect_v1", token_select, (SAMPLED_MEMBER,))
_row("GumbelTopPTokenSelect_v1", gumbel_topp, (SAMPLED_MEMBER,), _sample_gumbel)
_row("GumbelTopPTokenSelectSharedGreedy_v1", gumbel_topp, (SAMPLED_MEMBER,), _sample_gumbel)
_row("GumbelTopPTokenSelect_v2", gumbel_topp_split, (SAMPLED_MEMBER,), _sample_gumbel_split)
_row("Fp8GroupQuant_v1", fp8_group_quant, ("0", "1"))
_row("ScaledMmFp8Block_v1", scaled_mm_fp8_block, ("0",))
_row("ScaledMmFp8BlockSharedScale_v1", scaled_mm_fp8_block, ("0",))
_row("MoeRouterTopK_v1", partial(moe_router, renormalize=False), ("router_w", "router_ids"))
_row("MoeRouterTopKNorm_v1", partial(moe_router, renormalize=True), ("router_w", "router_ids"))
_row("MoeRouterTopKOrdered_v1", partial(moe_router, renormalize=False), ("router_w", "router_ids"))
_row("MoeRouterTopKOrderedNorm_v1", partial(moe_router, renormalize=True), ("router_w", "router_ids"))
_row("MoeExpertGemm_v1", partial(moe_expert_gemm, routed=False), ("up",), _sample({1: "E"}))
_row("MoeExpertGemmW_v1", partial(moe_expert_gemm, routed=True), ("down",), _sample({1: "E"}))
_row("MoeExpertGemm_v2", partial(moe_expert_gemm, routed=False, dotted=True), ("up",), _sample({1: "E"}))
_row("MoeExpertGemmW_v2", partial(moe_expert_gemm, routed=True, dotted=True), ("down",), _sample({1: "E"}))
_row("MoeFfnFixedL_v1", partial(moe_ffn_fixed, router_gemm=False), ("out",))
_row("MoeFfnFixed_v1", partial(moe_ffn_fixed, router_gemm=True), ("out",))
_row("MoeSum_v1", moe_sum, ("sum",))

from verity_vllm.program.kernels import dense_rows  # noqa: E402  -- the dense chain's kernels (registers them) and the float64 GEMM chain
from verity_vllm.program.kernels import act_rows, softcap_rows  # noqa: E402  -- SiluMul_v2's row, AttentionSoftcap_v2's head twin


def attention_softcap(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    from verity_vllm.program.kernels.ir_rows import f64_statics
    f = softcap_rows.FAMILY
    return softcap_rows.attention_softcap(f64_statics(statics), ops, prim_static(statics, "DOT", f), prim_static(statics, "INV", f))


EVALUATORS |= {m.FAMILY: m.EVALUATOR for m in (softcap_rows, act_rows)}
_row(softcap_rows.FAMILY, attention_softcap, ("out",), check_targets=softcap_rows.check_targets)
_row(act_rows.FAMILY, act_rows.silu_mul_expf, ("out",), check_targets=act_rows.check_targets)

from verity_vllm.program.kernels import ir_rows  # noqa: E402

for _f in ir_rows.FAMILIES:
    EVALUATORS[_f] = ir_rows.EVALUATOR
    _k = ROWS[_f] = RowKernel(_f, partial(ir_rows.ir_row, _f), ("out",))
    register(_f, _instance(_k), kernel=KERNEL, domain=DOMAIN, sample=_sample(), check_targets=lambda rng, f=_f: ir_rows.check_targets(f))
