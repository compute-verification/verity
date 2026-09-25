"""Sampled exact VU replay -- the local replay check of record (serialized as `C2`; DECISIONS-2 §4 C2; board M-0356 (1), M-0394).

For a sampled VU `V` (one Call of the Program) the check is

    F_V( tau(in(V)) ) == tau(out(V))      bit-for-bit,

where `tau` reads the COMMITTED bytes of the Values the Program's dataflow names (the Call's `args`), and `F_V` is the
registered relation of the Call's Definition evaluated by the registry's own reference (`verity_vllm.program.registry.derived_rows`,
the same numpy models the Match stage's fold checks use).  Nothing here reads the collector's emissions to decide WHAT is
required or WHERE an input lives: the population is the Program (its Calls, the instances.json.gz rows), the addresses are the required-value
manifest's Value rule -- the v2 query's population the manifest of record was built from (`required.Population.members_of`) -- and the
binding map fixed at finalize
(occurrence resolution: a Call's Value -> its occurrence key -> committed bytes).

    row.args = [[["n", j, lo, hi]], [["p", 1, lo, hi]]]
        "n" j lo hi -> producer Call j's OUTPUT element slice [lo, hi) -> its occurrence key (op_path, member, request-local row) -> committed bytes
        "p" 1 lo hi -> a slice of the registered `weights` parameter (bound by reference: read from the model's parameters)
        "p" 0 k k+1 -> the declared prompt id k -> committed `input_token_ids` row k of the request's arrival step
        a TokenSelect producer -> committed `sampled_token_ids` of THAT Call's step (the previous step's designated output)

Strata (recorded): (spec incl. shape, request, phase in {prefill, first-decode, mid, last}; `last` = the last EXECUTED step, F-r17b-39) x `per_stratum`, seeded -- distinct
numerical mechanisms x requests x lifetime boundaries.  Not-evaluable Calls are counted by reason (never silently skipped); see NOT_YET.
Attention_v2/v3 Calls ARE evaluated (R17-11): the query at position T-1 over the T committed K/V Values the Program names by ref -- the
resolver reads every producer at its own step, so the request's history is committed words, never a cache.

Ordered boundary linkage (the `linkage` check; exhaustive, NOT sampled): every request's committed `input_token_ids` at its arrival step == the DECLARED
prompt (workload manifest), and at every later step == that request's committed `sampled_token_ids` of the previous step.  This is
the prescribed-input / designated-output linkage of §4 and replaces the Match-oracle occurrence-resolution gate at B>=2 (F-cb-23).

PROTOCOL SUMMARY (Commit stage: local replay (`C2`) of record = sampled exact VU replay + exhaustive boundary linkage (`linkage`))
  Purpose: from the COMMITTED bytes alone (binding map + committed reader), re-evaluate a stratified sample of Calls with the
    registry's own relations and require bit equality; link every request's committed input ids to its declared prompt and to its
    own previous-step sampled ids over the executed prefix.  Needs no Match record.
  Inputs: the request Programs' instances docs (ProgramIndex), the binding map, `committed(step, op_path, member, rank)`, the weights
    reader, arrivals, the required-value manifest (Q(P) of record), `executed_prefix` facts per request, the seed (+ an optional verifier challenge).
  Outputs: the replay record {result True|False|None, grade, sample{seed, seed_source, challenge}, strata, population{query_total,
    not_executed_by_rule, ...}, uncovered_by_name, query_checks, catch_probability}; the linkage record {result, why, pairs, linked, unlinked}.
  Invariants: result True only if EVERY sampled VU replays bit-exactly AND every non-empty stratum was evaluated (None = partial,
    never PASS); the seed is never prover-choosable in advance (run root after commitment, or sha256(challenge|root)[:8]); Calls at
    request-relative steps >= n_exec leave the pool into a NAMED class before any draw; a request linked over fewer steps than its
    cap must end on a declared EOS id at steps[-1 - lag] (undeclared EOS / unattested lag = unlinked by name); weights_check.equal
    False fails the replay by name.
  Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import os
import re
import random
import time
from collections import Counter
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from verity_vllm.query.manifest import format as RM
from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.check.oracle_compare import entry_read_name as _entry_read_name, same_name_invocations as _same_name_invocations
from verity_vllm.check.replay.replay_codes import NOT_YET, WHY_CLASSES_BY_NAME, WHY_CLASSES_FAULT, why_class  # noqa: F401  (this module's record vocabulary)
from verity_vllm.check.replay.replay_dump import dump_row_words
from verity_vllm.commit.opened import OpeningNotVerified, UnverifiedUnlinked, opened_check, opened_in_children

SCHEMA = "verity-sweep/sampled-replay/v1"
LINKAGE_SCHEMA = "verity-sweep/boundary-linkage/v1"

#: the binding map's `encoding.dtype` is the collector's torch dtype name; the manifest's short names are accepted too
_ITEMSIZE = {"bf16": 2, "f16": 2, "u16": 2, "i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8,
             "bfloat16": 2, "float16": 2, "int16": 2, "uint16": 2, "float32": 4, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8, "float64": 8,
             # [revb F-r17b-52 (iii)] the LIFTED words of the padded MoE construction (manifest_format.LIFTED_DTYPES): Value<17> -> u32, Value<33> -> u64
             "l17": 4, "l33": 8}
_NPDTYPE = {2: np.uint16, 4: np.uint32, 8: np.uint64}

#: spec family -> the registry reference that evaluates its relation (recorded in every report)
EVALUATORS = {
    "Gemm_v1": "verity_vllm.program.registry.derived_rows.dot_bf16 (DotBf16_v1: ascending k16 AmpereBF16TcDot16 chain from 0) + f2fp_bf16 (GemmCoordinate_v1); "
               # [revb M-0461] the k16 step's measured total family is named on the record, not only by the tree commit
               # (literals, held equal to registry.tc_total's constants by test_sampled_replay)
               "k16 step = registry.tc_total ampere_bf16_m16n8k16 total family v3",
    # [R17 dense, F-r16-20b / ruling R17-3] the statics-lifted GEMM every non-(Ampere, FA2) target binds (targets.gemm_spec -> b1.GemmV2):
    # `Gemm_v2{K,N,DOT}` -- the SAME VU as Gemm_v1 (one token row: x[K] @ W[N,K]^T -> [N], the instance row Q(P) selects; the registry's
    # GemmCoordinate is its interior), with the k16 accumulator step read from the DOT static of THIS row's spec: HopperBF16WgmmaDot16_v1
    # (wgmma m64nNk16: one group of 16 + acc, 26-bit adder, floor -133, the H100 measurement of record) -> derived_rows.dot_bf16_hopper;
    # AmpereBF16TcDot16_v1 -> dot_bf16.  A DOT with no registered twin is refused BY NAME (`_Unresolved`), never another adder.
    # The ascending-from-0 chain is the cuBLASLt-without-split-K GEMM of a VLLM_BATCH_INVARIANT=1 step on cc 9.0 (vLLM keeps cuBLASLt there,
    # split-K off by workspace config), pinned word for word on the H100 at every Qwen2.5-1.5B shape x M (0 / 27.2M words; derived_rows).
    "Gemm_v2": "verity_vllm.program.registry.derived_rows.dot_bf16_for(DOT) (DotBf16_v2{K,DOT}: ascending k16 chain of the DOT static's step from 0 -- "
               "HopperBF16WgmmaDot16_v1 = tc_dot16_hopper (16:26:-133, silicon_total.HOPPER_BF16_WGMMA_K16; cuBLASLt split-K off = the "
               "batch-invariant Linear of record), AmpereBF16TcDot16_v1 = tc_dot16) + f2fp_bf16 (GemmCoordinate_v2); "
               "k16 steps = registry.tc_total hopper_bf16_wgmma_k16 family v1 / ampere_bf16_m16n8k16 total family v3",
    # [R17 dense, ruling R17-9] the NeoX rotation of one token row's q / k heads with ONE bf16 cos/sin cache row (a `weights` field slice
    # the Program's args name by position): b1.RoPE{NHEADS,D} = NHEADS x RoPEHead{D} = prims.RopeOut / RopeOutAdd per coordinate.
    "RoPE_v1": "verity_vllm.program.registry.derived_rows.rope_rows (RoPE_v1{NHEADS,D}: out[i] = bf16_rn(RN32(x[i]*c[i] - x[i+R]*s[i])), "
               "out[i+R] = bf16_rn(RN32(x[i+R]*c[i] + x[i]*s[i])), R = D/2, cs = cos_sin_cache[pos] -- prims.RopeOut / RopeOutAdd); "
               # [revb F-r17b-14 / M-0122] the qualification that travels with every evaluated RoPE_v1 row
               "verified against the SERVED cos_sin table, pinned by the MAN-09c analytic bound (<= 1-2 bf16 ulp on ~1-2.5 % of words), "
               "not by digest -- see weights_pin.check_cos_sin n_diff / max_ulp",
    # [R17 dense, ruling R17-11 (coord M-0136, from R17-10: the query of record is Q_module_v1, the attention ROW is the selected unit)]
    # one attention row from the committed q + the T committed K/V members the Program names by ref (every earlier step's committed word,
    # read by the resolver at the producer's step -- 'state-linked' was never a resolver gap) to the committed output: the registered
    # Definitions b1.AttentionV3 (FA2) / b1.AttentionV2 (FA3 on sm_90, DOT / INV statics resolved by name) evaluated gate for gate.
    "Attention_v3": "verity_vllm.program.registry.derived_rows.attention_rows (Attention_v3{T,NH,KVH,D,BN}: FA2 -- GQA over AttentionHead_v3, one query "
                    "at position T-1 over the T committed K/V rows the Program names, ceil(T/BN) blocks last -> first, tc_dot16 step, Fa2InvSum); "
                    "k16 step = registry.tc_total ampere_bf16_m16n8k16 total family v3",
    "Attention_v2": "verity_vllm.program.registry.derived_rows.attention_rows (Attention_v2{T,NH,KVH,D,BN,DOT,INV}: the DOT / INV statics resolved by name "
                    "-- HopperBF16WgmmaDot16_v1 + Fa3InvSum_v1 = FA3 on sm_90 (== verity_vllm.program.kernels.fa3_model.fa3_rows); "
                    "AmpereBF16TcDot16_v1 + Fa2InvSum_v1 = FA2 under the lifted name); "
                    # [revb M-0148 (5) / rev] the silicon envelope of the FA3 relation, stated with the evaluator
                    "FA3 envelope MEASURED exact on the H100 production capture at hdim 128 / kBlockN 128, T = 1..527 (39 leaves, "
                    "hopper-profile.md s1b); rows beyond it (1024/128, hdim 64 / kBlockN 192) are measured BY THE COMMIT -- a chain "
                    "mismatch FAILs by name, never a silent word; "
                    "k16 steps = registry.tc_total hopper_bf16_wgmma_k16 family v1 / ampere_bf16_m16n8k16 total family v3",
    "RMSNormTriton_v1": "verity_vllm.program.registry.derived_rows.rmsnorm_triton_rows",
    "RMSNormFusedCuda_v2": "verity_vllm.program.registry.derived_rows.rmsnorm_fused_v2_rows",
    "SiluMul_v1": "verity_vllm.program.registry.derived_rows.silu_mul_rows",
    "Embedding_v1": "row gather of the registered embedding field (Embedding_v1)",
    "BiasAdd_v1": "Bf16Add on the composed interior Gemm output + the registered bias slice (one f32 add, RNE; biased linears, MAN-07)",
    "TokenSelect_v1": "first-max argmax over the bf16 logits (TokenSelect_v1: Bf16GtStrict scan, lowest index on ties)",
    # [R16 stoch, rev F-r16-20(a)] the stochastic selector IS replayed: the registered composite GumbelTopPTokenSelect_v1{V} (TemperatureScale
    # -> TopPMask_v1 with the committed `splits` S as its operand -> GumbelSelectF32: Philox draws from the committed seed + the Const64 pos)
    # evaluated by the registry (`verity_vllm.check.difftest.evaluate_spec` on `sampling_topp_difftest.spec_registered`) on the committed bf16
    # logits row + the Program's Const32 top_p / temperature + the committed runner.sampler/seed and /splits words of the (step, request)
    "GumbelTopPTokenSelect_v1": "verity_vllm.program.registry.sampling.GumbelTopPTokenSelect{V} via verity_vllm.check.difftest.evaluate_spec (TemperatureScale -> "
                                "TopPMask_v1 (S = the committed `splits` operand) -> GumbelSelectF32 on the committed seed + Const64 pos)",
    # [R16 fp8] the block-scaled FP8 linear (Qwen3-4B-FP8, CutlassFp8BlockScaledMMKernel): the quant is INTERIOR (members [], composed
    # from the committed bf16 row), the GEMM row is the VU -- registered e4m3 weight + f32 `weight_scale_inv` slices (8- and 32-bit fields)
    "Fp8GroupQuant_v1": "verity_vllm.program.registry.derived_rows.fp8_group_quant_row (per-token-group(128) e4m3 quant: exact fmaxf fold from eps, div.rn, cvt.rn.satfinite.e4m3)",
    "ScaledMmFp8Block_v1": "verity_vllm.program.registry.derived_rows.scaled_mm_fp8_block_row (ascending k: G/32 HopperE4m3QgmmaDot32 steps from +0 per 128-block, "
                           "FFMA(temp, FMUL(sx, sw), acc), cvt.rn.bf16.f32) on the composed Fp8GroupQuant output + the registered e4m3 / f32 weight slices",
    # [MAN-06f v3 unit 2, moe R16] the fused-MoE block relations (verity_vllm.program.registry.moe bodies; R13 moe_kernel_xcheck = the reference against
    # the real kernels): every plane word is PRODUCED by an evaluated relation over committed plane words + registered expert weights
    "MoeRouterTopK_v1": "verity_vllm.program.registry.moe.MoeRouterTopK evaluated per token with the registry's scalar primitives (f32 softmax in the kernel's "
                        "E/VPT-thread geometry + xor butterflies, NvExpf_v1 exp, F32Div rcp, non-finite -> 0, TOPK rounds of strict first-max argmax "
                        "masked to -10000.f) -> router_w f32 + router_ids i32",
    "MoeRouterTopKNorm_v1": "MoeRouterTopK then the renormalising epilogue (selected sum ascending from 0.f, denom = sum > 0 ? sum : 1, one F32Div, "
                            "one F32Mul per slot; MOE-04) -> router_w f32 + router_ids i32",
    "MoeExpertGemm_v1": "verity_vllm.program.registry.derived_rows.dot_bf16 (DotBf16_v1 k16 chain from 0) over the expert row W13[e] gathered by the committed "
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
    "Bf16MulScalarTensor_v1": "verity_vllm.program.registry.derived_rows bf16_to_f32_words (Bf16ToF32) on x[N] and on the one-word scalar s, mul_rn (F32Mul: "
                              "one RN multiply), f2fp_bf16 (F2fpBf16 = cvt.rn.bf16.f32, NaN -> 0x7FFF) -- b1.Bf16MulScalarTensor{N} node for node -> out",
}
#: [R16 fp8] a `weights` field's leaf width (its `Value<b>`) -> the word dtype the registered slice is read as (bf16 -> u16, e4m3 -> u8, f32 -> u32)
_NP_BY_BITS = {8: np.uint8, 16: np.uint16, 32: np.uint32}

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
EVALUATOR_PROVENANCE: dict[str, dict] = {"Fp8GroupQuant_v1": _FP8_TWIN_PROVENANCE, "ScaledMmFp8Block_v1": _FP8_TWIN_PROVENANCE}
#: [R17 dense, ruling R17-11] the attention families whose ROW the replay evaluates; their block interiors are Q_fine_v1 units (R17-10 (3))
ATTENTION_ROW_FAMILIES = ("Attention_v3", "Attention_v2")
#: named in population() whenever a Program carries attention rows: what the row relation does not read (a Q_fine_v1 research item, not a
#: gap under the query of record)
ATTENTION_INTERIOR_MECHANISM = ("AttnBlock interiors (Q_fine_v1 units: s[c] dots, PV k-step chains, max chain, lane sums, epilogue) live in the "
                                "fa2/fa3 M1 export stream; the sampled replay evaluates the attention ROW (the Q_module_v1 unit) from its "
                                "committed q + K/V history to its committed output and does not read those interior words (R17-10 (3) / R17-11). "
                                "[R17-14] The row evaluator is driven by the row's DECLARED signature, not by an attention-specific hand list: "
                                "its inputs are exactly the args the Program row wires (the Definition's arity: q, K history, V history -- each "
                                "ref resolved to committed words at its producer's step; a row wiring another arity is unresolved by name), its "
                                "statics come from the spec text (T, NH, KVH, D, BN, and for Attention_v2 DOT / INV by id), and every member "
                                "the manifest requires of the row is compared (a member the evaluator does not produce = not evaluated, by name).")

GEMM_CHUNK = 4096                       # Gemm coordinates evaluated per vectorised k16 chain (bounds the int64 temporaries to ~0.5 GB)
SAMPLER_OP = "runner.sampler"
SAMPLED_MEMBER = "sampled_token_ids"
IDS_MEMBER = "in"
SEED_PARAM = "seed"                     # the stochastic Program's seed input (Param name; committed as runner.sampler/seed per (step, request))
#: the controller / guard families of a request Program's root body (the manifest header's `guard_rows` counts them)
GUARD_PREFIXES = ("LStepLive", "LSelect", "LBot", "LIsEOS", "LEmitNext")
_CONST_ROW = re.compile(r"^Const(32|64)\[0x([0-9a-fA-F]+)\]_v\d+$")


_DOT_STATIC_FN = re.compile(r'^\{"fn":"([A-Za-z0-9_]+)"\}$')


def prim_static(statics: dict[str, Any], key: str, family: str) -> str:
    """[R17 dense, ruling R17-3] A function-valued static (`DOT` of `Gemm_v2{K,N,DOT}` / `Attention_v2{..., DOT, INV}`, `INV` of the latter)
    -> the registered primitive id.  The spec text renders it as `DOT={"fn":"HopperBF16WgmmaDot16_v1"}` (verity.ir.codec._spec_id); a bare
    `<Name>_v<k>` id is accepted too.  Anything else is unresolved BY NAME: the evaluator never guesses the architecture from the target."""
    raw = str(statics.get(key, "") or "")
    m = _DOT_STATIC_FN.match(raw)
    if m:
        return m.group(1)
    if re.match(r"^[A-Za-z0-9]+_v\d+$", raw):
        return raw
    raise _Unresolved(f"{family} {key} static {raw!r} does not name a registered primitive")


def gemm_dot_prim(statics: dict[str, Any]) -> str:
    """The DOT static of a `Gemm_v2{K,N,DOT}` row -> the registered k16 primitive id (see `prim_static`)."""
    return prim_static(statics, "DOT", "Gemm_v2")


def const_word(row: dict | None) -> np.ndarray | None:
    """A `Const32[0x..]_v1` / `Const64[0x..]_v1` Program row -> its one word (u32 / u64), else None.  The constant IS the Program's
    text (top_p, temperature, pos of the stochastic selector): read from the spec, no committed identity to look up."""
    m = _CONST_ROW.match(str((row or {}).get("spec") or ""))
    if m is None:
        return None
    bits, hexv = int(m.group(1)), int(m.group(2), 16)
    if hexv >= (1 << bits):
        return None
    return np.asarray([hexv], dtype=np.uint32 if bits == 32 else np.uint64)


# ------------------------------------------------------------------------------------------------------------------------ weights

def parse_struct(type_str: str) -> list[tuple[str, int, list[int], int]]:
    """`{a: Array<3,Array<2,Value<16>>>, b: Array<4,Value<16>>}` -> [(name, element count, shape, value bits)] in declaration order."""
    s = type_str.strip()
    if not (s.startswith("{") and s.endswith("}")):
        raise ValueError("weights parameter is not a struct type")
    body, fields, depth, buf = s[1:-1], [], 0, ""
    for ch in body:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            fields.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        fields.append(buf)
    out = []
    for f in fields:
        name, _, ty = f.strip().partition(":")
        ty = ty.strip()
        shape, rest, bits = [], ty, 0
        while rest.startswith("Array<"):
            n, _, rest = rest[len("Array<"):].partition(",")
            shape.append(int(n))
            rest = rest.strip()
            if rest.endswith(">"):
                rest = rest[:-1]
        if rest.startswith("Value<"):
            bits = int(rest[len("Value<"):].rstrip(">").rstrip(">"))
        count = 1
        for n in shape:
            count *= n
        out.append((name.strip(), count, shape, bits))
    return out


#: the query engine whose population addresses a Call's committed output (`addresses.rule` on the record)
V2_QUERY = "v2-query"


class ProgramIndex:
    """The Program's rows with their manifest addresses: (op_path, member, request-local step, request-local row) per output slice.

    A row's members come from the v2 query's population of the SAME Program (`verity_vllm.query.required.Population.members_of`: the
    required Values a Call produces under `Q_module_body_v1` + the policy, the rows the manifest counts), read with the correspondence of
    its Build dir (`program_dir`; without one, the rows' recorded names)."""

    def __init__(self, inst: dict[str, Any], *, prefix: str | None = None, attention_paths: list[str] | None = None, tp: dict[str, Any] | None = None,
                 program_dir: str | None = None):
        self.rows: list[dict] = list(inst["rows"])
        # [tp TP-08d] a RANK Program (derive_step --tp W --rank r; result.json `tp` = {"world": W, "rank": r}): its population carries the
        # TP-04 policy (the rank-local partials `part_rank<r>` / `shard_rank<r>` beside every collective output), the one the rank's
        # manifest was built with.  `tp` = {"rank": r, "world": W}.
        self.tp: dict[str, Any] | None = (dict(rank=int(tp.get("rank", 0)), world=int(tp.get("world", 1))) if tp and int(tp.get("world", 1) or 1) > 1 else None)
        # [v2 flip, P2] Call id -> [(member, width, dtype)] of the required Values it produces, and the Program path they are spelt under
        # (`Population.program_path`: its owning module, a collective site's site path)
        self._v2: dict[int, list[tuple[str, int, str]]] = {}
        self._v2_module: dict[int, str] = {}
        self._v2_fault: dict[int, str] = {}
        from verity_vllm.correspondence.reader_for_query import Correspondence
        from verity_vllm.query.program_view import from_instances
        from verity_vllm.query.required import population
        P = from_instances(inst)
        corr = Correspondence.of(P, program_dir, implementation_paths=attention_paths or ())
        pop = population(P, corr, tp=self.tp)
        for cid in pop.producers():
            mem = pop.members_of(cid)
            if mem:
                self._v2[cid] = [(m.member, m.width, m.dtype) for m in mem]
                self._v2_module[cid] = pop.program_path(cid)
        self.correspondence: dict[str, Any] | None = corr.provenance()
        del P, corr, pop
        self.params: list[list] = [list(p) for p in inst.get("params", [])]
        self.prefix = RM.wrapper_prefix_of(self.rows, attention_paths or []) if prefix is None else prefix
        self.segs = RM.segment_steps(self.rows)
        self._seg_lo = [lo for lo, _hi in self.segs]
        self.by_i = {int(r["i"]): r for r in self.rows}
        self.pnames = [p[0] for p in self.params]
        self.fields: list[tuple[str, int, list[int], int]] = []
        self.foff: list[int] = []
        wt = next((p[1] for p in self.params if p[0] == "weights"), None)
        if wt:
            self.fields = parse_struct(wt)
            off = 0
            for _n, c, _s, _b in self.fields:
                self.foff.append(off)
                off += c
        self._desc: dict[int, dict | None] = {}
        # [MAN-06f v3, moe R16] the fused-MoE block rows of each step: per MoE module its geometry (M routed tokens, TOPK, E, H, I, N_up --
        # `manifest_format.moe_geometry`, the same numbers the manifest sizes the `moe_block_stream` planes with) and, per row, its
        # PAIR / TOKEN ordinal among the rows of its plane kind in Program order (router / sum rows: token t; up / act / down rows: pair
        # p = t*TOPK + k).  `describe` addresses these rows at their PLANE (op_path `moe/L<k>`, member = the plane).
        self._moe_geo: list[dict[str, dict[str, Any]]] = [RM.moe_geometry(self.rows[lo: hi + 1]) for lo, hi in self.segs]
        self._moe_ord: dict[int, int] = {}
        mcounts: dict[tuple, int] = {}
        for r in self.rows:
            mod0 = RM.split_name(r.get("name") or "")[0]
            kind = moe_plane_kind(RM.spec_family(r["spec"]), mod0)
            if kind is not None:
                key = (mod0, kind, self.step_of(int(r["i"])))
                self._moe_ord[int(r["i"])] = mcounts.get(key, 0)
                mcounts[key] = self._moe_ord[int(r["i"])] + 1
        self._ordinal: dict[int, int] = {}
        counts: dict[tuple, int] = {}
        for r in self.rows:
            i = int(r["i"])
            s = self.step_of(i)
            # the request-local ROW of a Call = its ordinal among the Calls producing the same (module, member) in its step, Program order --
            # exactly the rows `required.request_manifest` counts per identity; a Call whose members would sit at different rows is not one
            # addressable row (named, never read at a guessed row)
            mem2 = self._v2.get(i)
            if mem2:
                ords = []
                for m in mem2:
                    k2 = (self._v2_module[i], m[0], s)
                    ords.append(counts.get(k2, 0))
                    counts[k2] = ords[-1] + 1
                if len(set(ords)) == 1:
                    self._ordinal[i] = ords[0]
                else:
                    self._v2_fault[i] = f"members {[m[0] for m in mem2]} sit at request-local rows {ords}: not one row"

    @property
    def n_steps(self) -> int:
        return len(self.segs)

    def ids_params(self) -> set[int]:
        """The Program parameter(s) carrying the declared prompt ids: every parameter that is not the registered `weights` record and not a
        sampler input (`seed`, `splits`) -- the same reading `_gather_inputs` resolves a `['p', a, k, k+1]` slice by (the committed
        `input_token_ids` row k of the arrival step)."""
        return {a for a, n in enumerate(self.pnames) if n not in ("weights", SEED_PARAM, RM.SPLITS_PARAM)}

    def embed_op_path(self) -> str | None:
        """The op_path of the Program's Embedding rows (the `input_token_ids` identity lives under it as member `in`): the first row whose
        arguments READ the prompt-ids parameter (`ids_params`) -- the Program's own dataflow names the consumer of the declared ids; the
        row's family name never enters (Draft 3.1 §12.4)."""
        if not hasattr(self, "_embed_op"):
            self._embed_op = None
            ids = self.ids_params()
            for r in self.rows:
                if any(sl and sl[0] == "p" and int(sl[1]) in ids for g in (r.get("args") or []) for sl in g):
                    d = self.describe(r)
                    self._embed_op = d["op_path"] if d else None
                    break
        return self._embed_op

    def step_of(self, i: int) -> int:
        return max(0, bisect.bisect_right(self._seg_lo, int(i)) - 1)

    def describe(self, r: dict) -> dict | None:
        """Family, statics, module -> op_path, members (name, width, dtype) and request-local row of a Program row; None if unmodelled."""
        i = int(r["i"])
        if i in self._desc:
            return self._desc[i]
        fam = RM.spec_family(r["spec"])
        st = RM.spec_statics(r["spec"])
        mod, node, hint, row = RM.split_name(r.get("name") or "")
        d: dict | None
        kind = moe_plane_kind(fam, mod)
        # [F-r19-int-24] FusedMoE.forward returns the MoeSum: a boundary Value (`<experts>/out`) beside its plane copy (MOE-TP-02 below)
        tp_sum = kind == "sum" and bool(self._v2.get(i))
        if tp_sum:
            kind = None
        if RM.is_token_select(fam):
            d = {"family": fam, "statics": st, "module": "", "op_path": SAMPLER_OP, "hint": hint, "row": 0,
                 "members": [(SAMPLED_MEMBER, 1, "i64")], "step": self.step_of(i)}
        elif kind is not None:
            d = self._describe_plane(i, fam, st, mod, hint, row, kind)
        else:
            fault = self._v2_fault.get(i)
            mod2 = self._v2_module.get(i, mod)
            d = None if fault else {"family": fam, "statics": st, "module": mod2, "op_path": RM.collector_path(mod2, self.prefix), "hint": hint,
                                    "row": int(self._ordinal.get(i, 0)), "name_row": row, "members": list(self._v2.get(i, ())), "step": self.step_of(i)}
        if tp_sum and d is not None and d["members"]:
            # [MOE-TP-02 / moe, tp M-1528] on a TP rank the MoeSum row's ONE output is committed under TWO required names: the module's
            # `part_rank<r>` (tp_rank_partials, TP-04/TP-06: the all-reduce input) AND the MOE-01 plane `moe/L<k>/sum` (moe_block_stream,
            # staged by MoeBlockSource on the rank exactly as at TP1).  The manifest requires both (TP-06); the
            # replay addressed the row at the partial only, so F-r17b-38's reconciliation read every `moe/L<k>/sum` identity as
            # `identities_without_rows` (OLMoE TP2 B8 int-6: 6,944 per rank, COMMIT FAIL).  `copies` = the plane address of the same
            # output: recorded in the population under the plane identity AND read back and compared against the same evaluated words
            # (two committed copies, one relation, both checked).  The plane mapping is the checked one (_describe_plane: geometry,
            # ordinal, dataflow); a mapping failure is NAMED here and the plane identity stays without rows -- fail closed, never aliased.
            # The same holds at TP1: the MoeSum is the module's returned tensor `<experts>/out` (F-r19-int-24).
            pd = self._describe_plane(i, fam, st, mod, hint, row, "sum")
            if pd.get("plane_error") or not pd["members"]:
                d["copy_error"] = pd.get("plane_error") or "sum plane has no member"
            else:
                pm = pd["members"][0]
                why = "F-r19-int-24: the MoeSum output = the MoE module's returned tensor (a boundary Value) = its moe_block_stream sum plane"
                d["copies"] = [{"op_path": pd["op_path"], "member": pm[0], "width": int(pm[1]), "dtype": str(pm[2]), "row": int(pd["row"]),
                                "of": d["members"][0][0], "why": why}]
        self._desc[i] = d
        return d

    def _describe_plane(self, i: int, fam: str, st: dict[str, Any], mod: str, hint: str, name_row: int | None, kind: str) -> dict:
        """[MAN-06f v3] a fused-MoE block row -> its `moe_block_stream` PLANE address: op_path `moe/L<layer>`, members = the plane(s) the row
        produces (router: `router_w` f32[TOPK] + `router_ids` i32[TOPK] per token; up: `up` bf16[N_up] per pair; act (the SiluMul under the
        MoE runner): `act` bf16[I] per pair; down: `down` bf16[H] per pair; sum: `sum` bf16[H] per token), row = the row's token / pair
        ordinal in Program order (pair-major p = t*TOPK + k: MOE-03 source order, `moe_geometry` docstring).  The plane's request-local
        element range is the binding's per-request split of the batched plane (binding.py: rows == tokens -> the request's token rows, x
        TOPK for `act`), so `store.words(rid, step, "moe/L<k>", plane, row*width + lo, ...)` reads exactly this row's words.  The mapping
        is CHECKED against the Program, never guessed: the module's geometry must be consistent (expert-GEMM rows == M*TOPK, the rule
        build_manifest sizes the planes with), the ordinal must lie inside the plane (M tokens / M*TOPK pairs) and must equal the row's own
        `[k]` name index when the Program names one; a violation yields members [] + `plane_error` -> counted by NAME as a mapping failure
        (population), never a silent read of another pair's words."""
        s = self.step_of(i)
        g = self._moe_geo[s].get(mod) or {}
        layer = RM.layer_of(mod)
        M, TOPK = int(g.get("M", 0) or 0), int(g.get("TOPK", 0) or 0)
        ord_ = int(self._moe_ord.get(i, -1))
        err: str | None = None
        if layer < 0:
            err = f"MoE module {mod!r} has no `.layers.<k>` index (plane name moe/L<k> undefined)"
        elif M <= 0 or TOPK <= 0:
            err = f"MoE module {mod!r} step {s}: no router rows / TOPK in the Program (M={M}, TOPK={TOPK})"
        elif int(g.get("expert_gemm_rows", 0)) != M * TOPK:
            err = f"MoE module {mod!r} step {s}: {g.get('expert_gemm_rows')} expert-GEMM rows != M*TOPK = {M}*{TOPK}"
        if kind == "router":
            mem = [("router_w", TOPK, "f32"), ("router_ids", TOPK, "i32")]
            n_rows = M
        elif kind == "up":
            mem = [("up", int(st.get("N", g.get("N_up", 0))), "bf16")]
            n_rows = M * TOPK
        elif kind == "act":
            mem = [("act", int(st.get("I", g.get("I", 0))), "bf16")]
            n_rows = M * TOPK
        elif kind == "down":
            mem = [("down", int(st.get("N", g.get("H", 0))), "bf16")]
            n_rows = M * TOPK
        else:                                                                    # "sum"
            mem = [("sum", int(st.get("H", g.get("H", 0))), "bf16")]
            n_rows = M
        if err is None and not (0 <= ord_ < n_rows):
            err = f"{fam} row ordinal {ord_} outside the {kind} plane of {mod!r} step {s} ({n_rows} rows = {'M' if n_rows == M else 'M*TOPK'})"
        if err is None and name_row is not None and int(name_row) != ord_:
            err = f"{fam} row ordinal {ord_} != its Program name index [{name_row}] ({mod!r} step {s}, {kind} plane)"
        if err is None and any(int(m[1]) <= 0 for m in mem):
            err = f"{fam} {kind} plane width 0 ({mod!r} step {s}: statics {st}, geometry {g})"
        if err is None:
            err = self._plane_dataflow_error(i, kind, ord_, TOPK, mod)
        # `plane` = the plane the row PRODUCES: the first member's name (the router row produces two: router_w + router_ids)
        d = {"family": fam, "statics": st, "module": mod, "op_path": f"moe/L{layer}", "hint": hint, "row": ord_, "name_row": name_row,
             "members": ([] if err else mem), "step": s, "plane": mem[0][0], "plane_kind": kind, "moe_module": mod}
        if err:
            d["plane_error"] = err
        return d

    def _plane_dataflow_error(self, i: int, kind: str, p: int, TOPK: int, mod: str) -> str | None:
        """The pair-major reading of a plane row's ordinal (p = t*TOPK + k) cross-checked against the Program's OWN dataflow: the up row p
        reads router row t's id slot k (`['n', router, TOPK + k, TOPK + k + 1]`), the act row p reads up row p, the down row p reads act
        row p + router row t's id slot k + weight slot k, the sum row t reads down rows t*TOPK .. t*TOPK + TOPK - 1 in slot order.  A Program
        whose MoE rows are not in that order (or not of that shape) is named here -- the replay never reads a plane word of another pair.
        (Measured on the OLMoE B1 256/32 request Program: 119,392 MoE rows, 0 inconsistencies.)"""
        r = self.by_i[i]
        args = r.get("args") or []

        def prod(gi: int, si: int = 0) -> tuple[str | None, int, int, int] | None:
            if gi >= len(args) or si >= len(args[gi]) or args[gi][si][0] != "n":
                return None
            j, lo, hi = int(args[gi][si][1]), int(args[gi][si][2]), int(args[gi][si][3])
            pr = self.by_i.get(j)
            if pr is None:
                return None
            pm = RM.split_name(pr.get("name") or "")[0]
            pk = moe_plane_kind(RM.spec_family(pr["spec"]), pm) if pm == mod else None
            return pk, int(self._moe_ord.get(j, -1)), lo, hi

        t, k = (p // TOPK, p % TOPK) if TOPK > 0 else (p, 0)
        if kind == "up":
            e = prod(1)
            if e is None or e[0] != "router" or e[1] != t or (e[2], e[3]) != (TOPK + k, TOPK + k + 1):
                return f"up row {p} does not read router row {t} id slot {k} (Program dataflow {e}): not pair-major"
        elif kind == "act":
            u = prod(0)
            if u is None or u[0] != "up" or u[1] != p or u[2] != 0:
                return f"act row {p} does not read up row {p} (Program dataflow {u}): not pair-major"
        elif kind == "down":
            a, e, g = prod(0), prod(1), prod(2)
            if a is None or a[0] != "act" or a[1] != p or a[2] != 0:
                return f"down row {p} does not read act row {p} (Program dataflow {a}): not pair-major"
            if e is None or e[0] != "router" or e[1] != t or (e[2], e[3]) != (TOPK + k, TOPK + k + 1):
                return f"down row {p} does not read router row {t} id slot {k} (Program dataflow {e}): not pair-major"
            if g is None or g[0] != "router" or g[1] != t or (g[2], g[3]) != (k, k + 1):
                return f"down row {p} does not read router row {t} weight slot {k} (Program dataflow {g}): not pair-major"
        elif kind == "sum":
            sl = args[0] if args else []
            if len(sl) != TOPK:
                return f"sum row {p} reads {len(sl)} slices, not TOPK = {TOPK}"
            for kk in range(TOPK):
                y = prod(0, kk)
                if y is None or y[0] != "down" or y[1] != p * TOPK + kk or y[2] != 0:
                    return f"sum row {p} slot {kk} does not read down row {p * TOPK + kk} (Program dataflow {y}): not pair-major"
        return None

    def member_at(self, j: int, lo: int, hi: int) -> tuple[dict, str, int, int, str] | None:
        """Producer row j's output slice [lo, hi) -> (desc, member, member-local lo, hi, dtype); None if it is not inside one member."""
        pr = self.by_i.get(int(j))
        if pr is None:
            return None
        d = self.describe(pr)
        if not d:
            return None
        off = 0
        for m in d["members"]:
            name, width, dtype = m[0], int(m[1]), str(m[2])
            if off <= lo and hi <= off + width:
                return d, name, lo - off, hi - off, dtype
            off += width
        return None

    def weight_slice(self, lo: int, hi: int) -> tuple[str, int, int, list[int]] | None:
        """Element slice of the `weights` struct -> (field name, field-local lo, hi, field shape); None if it spans fields."""
        if not self.fields:
            return None
        k = bisect.bisect_right(self.foff, lo) - 1
        if k < 0:
            return None
        name, count, shape, _bits = self.fields[k]
        base = self.foff[k]
        if lo < base or hi > base + count:
            return None
        return name, lo - base, hi - base, shape

    def weight_bits(self, field: str) -> int:
        """[R16 fp8] the leaf width the Program declares for a `weights` field (`Value<16>` bf16, `Value<8>` e4m3, `Value<32>` f32 block scales)."""
        for name, _count, _shape, bits in self.fields:
            if name == field:
                return int(bits) or 16
        return 16


# ------------------------------------------------------------------------------------------------------------------ committed store

def alias_table(manifest: dict[str, Any] | None) -> dict[str, tuple[str, int]]:
    """[rel M-0367 / reqval] `manifest.op_path_aliases` as ONE table: Program op_path -> (canonical op_path, invocation).  Shared modules
    (vLLM's one RotaryEmbedding instance called by every layer) are addressed by the required manifest at the FIRST layer's path with
    invocation = layer order; the per-layer Program paths are recorded as aliases ('model.layers.7.self_attn.rotary_emb' ->
    ['model.layers.0.self_attn.rotary_emb', 7]).  The population reconciliation (`query_population._key`) and the committed store
    (`CommittedStore.words`) resolve through THIS table and no other -- the two never disagree on which identity a Program path names."""
    return {str(k): (str(v[0]), int(v[1])) for k, v in ((manifest or {}).get("op_path_aliases") or {}).items()} if manifest is not None else {}


def resolve_op_path(aliases: dict[str, tuple[str, int]], op_path: str) -> tuple[str, int]:
    """(canonical op_path, invocation) of a Program op_path: its alias entry, else itself at invocation 0 -- EXACTLY (a path with no alias
    names invocation 0 and nothing else; F-r17b-17: `invocation` is 0 for every non-shared module row)."""
    return aliases.get(op_path, (op_path, 0))


def _invocation(e: dict) -> int:
    return int((e.get("identity") or {}).get("invocation", 0) or 0)


class CommittedStore:
    """Committed words of a bound identity: (request, ENGINE step, op_path, member, request-local element slice) via the binding map.

    [R16 rev F-r16-08 / M-0077; BRIEF §3 C1 'one UNAMBIGUOUS binding'] every entry of a key is KEPT (not 'first wins'): `words` reads the
    ONE entry of the resolved invocation whose element_range covers the slice; two entries of that invocation covering it = an ambiguous
    binding (two buffers under one required identity, the replay could read either) -> (None, 'ambiguous binding: ...'), the VU fails by
    name.

    [rel M-0367 / R17-11] the Program op_path is resolved through the manifest's `op_path_aliases` (`aliases`, the SAME table the
    population reconciliation uses) to (canonical path, invocation) and the read is of THAT invocation's entry EXACTLY: a shared module's
    layer-N call reads invocation N of the canonical path; a path with no alias reads invocation 0; an invocation with no entry is
    'no committed identity' BY NAME.  Invocations are never collapsed to the lowest (the pre-M-0367 store did, so the first evaluator that
    read a shared module's output -- the Attention_v3/v2 row, q = rotary_emb/0 -- missed layers 1..N by name on every record: a wrong or
    missing invocation read fails closed, it never reads a neighbour's words)."""

    def __init__(self, bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None],
                 aliases: dict[str, Any] | None = None):
        self.committed = committed
        # accepts the manifest's raw form (alias -> [canonical, invocation]) or the parsed table (alias -> (canonical, invocation))
        self.aliases: dict[str, tuple[str, int]] = {str(k): (str(v[0]), int(v[1])) for k, v in (aliases or {}).items()}
        self.idx: dict[tuple, list[dict]] = {}
        for e in bmap.get("entries") or []:
            ident = e.get("identity") or {}
            if "step" not in ident:
                continue
            key = (str(ident.get("request_id", "r0")), int(ident["step"]), str(ident.get("op_path")), str(ident.get("output_member")))
            self.idx.setdefault(key, []).append(e)
        # [moe F-moe-r17-06] a shared module's invocations share ONE member name in the collector's layouts: the reader is asked for the
        # entry's EXACT meta (`<name>#ord<tensor_ordinal>`, oracle_compare.entry_read_name) whenever more than one invocation is bound under
        # that name at the step -- by the bare name every layer read layer 0's words (the OLMoE mixedstop record: layers 1..15 unequal)
        self.name_inv: dict[tuple[int, str], int] = _same_name_invocations(bmap)

    @classmethod
    def of_record(cls, bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None], manifest: dict[str, Any] | None) -> "CommittedStore":
        """The store the replay of record uses: the binding map read through the manifest's alias table (None manifest = no aliases: a
        shared module's layer-N path is then missing by name, never silently the lowest invocation)."""
        return cls(bmap, committed, alias_table(manifest))

    def resolve(self, op_path: str) -> tuple[str, int]:
        return resolve_op_path(self.aliases, op_path)

    def entries(self, rid: str, step: int, op_path: str, member: str) -> tuple[list[dict], str, int, list[int]]:
        """The entries of the RESOLVED (canonical path, invocation) -> (entries, canonical, invocation, invocations bound under the key)."""
        canon, inv = self.resolve(op_path)
        es_all = self.idx.get((rid, int(step), canon, member)) or []
        return [x for x in es_all if _invocation(x) == inv], canon, inv, sorted({_invocation(x) for x in es_all})

    def has(self, rid: str, step: int, op_path: str, member: str) -> bool:
        return bool(self.entries(rid, step, op_path, member)[0])

    def words(self, rid: str, step: int, op_path: str, member: str, elo: int, ehi: int) -> tuple[np.ndarray | None, str]:
        es, canon, inv_min, bound = self.entries(rid, step, op_path, member)
        if not es:
            via = f" [alias -> {canon} invocation {inv_min}]" if canon != op_path else (f" [invocation {inv_min}]" if bound else "")
            have = f"; bound invocations of ({canon}/{member}) at that step: {bound}" if bound else ""
            return None, f"no committed identity for ({rid}, step {step}, {op_path}/{member}){via}{have}"

        def _covers(x: dict) -> bool:
            r = (x.get("identity") or {}).get("element_range") or [0, None]
            return r[1] is not None and int(r[0]) <= elo and ehi <= int(r[1])

        cov = [x for x in es if _covers(x)]
        if len(cov) > 1:
            where = ", ".join(f"{x.get('name')}@{x.get('stream_offset')}" for x in cov[:4])
            return None, (f"ambiguous binding: {len(cov)} entries for ({rid}, step {step}, {op_path}/{member}) [invocation {inv_min}] cover "
                          f"[{elo}, {ehi}) -- {where}; C1 requires one unambiguous binding per required identity (F-r16-08)")
        e = cov[0] if cov else es[0]
        ident = e.get("identity") or {}
        rng = ident.get("element_range") or [0, None]
        if not cov:
            return None, f"element slice [{elo}, {ehi}) outside the bound identity range {rng} of ({rid}, step {step}, {op_path}/{member})"
        dtype = str((e.get("encoding") or {}).get("dtype") or "")
        isz = _ITEMSIZE.get(dtype)
        if isz is None:
            return None, f"unknown dtype {dtype!r} of ({rid}, step {step}, {op_path}/{member})"
        eo = int(e.get("element_offset") or 0)
        b = self.committed(int(step), _entry_read_name(e, self.name_inv.get((int(step), str(e.get("name"))), 1)), (eo + elo) * isz, (eo + ehi) * isz)
        if b is None or len(b) != (ehi - elo) * isz:
            return None, f"committed bytes not retained for ({rid}, step {step}, {op_path}/{member}) [{elo}, {ehi})"
        return np.frombuffer(bytes(b), dtype=_NPDTYPE[isz]).copy(), dtype


# ------------------------------------------------------------------------------------------------------------------------ evaluate

def _eps(statics: dict[str, Any]) -> float:
    """`EPS={"f64":"0x1.4f8b588e368f1p-17"}` (the spec's f64 hex literal, kept as a string by spec_statics) -> float."""
    e = statics.get("EPS")
    if isinstance(e, str) and e.lstrip().startswith("{"):
        import json as _json
        e = _json.loads(e)
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
    as the S operand -> GumbelSelectF32, Philox draws from (seed, pos)); returns the sampled id.  No re-implementation: the same
    `evaluate_spec` walk the difftest (G4) and the Match-time stoch recompute run.  The words arrive as the committed / Const encodings;
    every operand is checked against the signature's leaf count before the walk (a slice that is not exactly one f32 word is a fault
    of the Program's wiring, named here, never a silent truncation)."""
    from verity_vllm.check.difftest import evaluate_spec
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
    out = evaluate_spec(DT.spec_registered({"V": V}), words)
    return int(out[0]) & 0xFFFFFFFF


def evaluate(family: str, statics: dict[str, Any], inputs: list[list[np.ndarray]], weights: list[np.ndarray | None]) -> dict[str, np.ndarray]:
    """F_V for the supported families.  `inputs[g]` = the committed words of arg group g (one array per slice, concatenated in order);
    `weights[g]` = the registered slice when arg group g is a `weights` slice.  Returns {member: words}."""
    from verity_vllm.program.registry import derived_rows as D

    def arg(g: int) -> np.ndarray:
        if weights[g] is not None:
            return weights[g]
        return np.concatenate(inputs[g]) if len(inputs[g]) > 1 else inputs[g][0]

    if family in ("Gemm_v1", "Gemm_v2"):
        K, N = int(statics["K"]), int(statics["N"])
        # [R17 dense, ruling R17-3] Gemm_v2{K,N,DOT}: the k16 step is the row's DOT static (`{"fn":"<prim id>"}` in the spec text), resolved
        # to its vectorised twin by name; Gemm_v1 is the accepted Ampere chain.  An unregistered DOT is unresolved BY NAME (the row stays
        # 'not evaluated' with the id in its why), never evaluated under another architecture's adder.
        if family == "Gemm_v1":
            dot = D.dot_bf16
        else:
            dot_id = gemm_dot_prim(statics)
            try:
                dot = D.dot_bf16_for(dot_id)
            except KeyError:
                raise _Unresolved(f"no registered evaluator for Gemm_v2 with DOT={dot_id} (k16 step twins: {sorted(D.TC_DOT16_BY_PRIM)})") from None
        x = np.asarray(arg(0), dtype=np.uint16).reshape(K)
        w = np.asarray(arg(1), dtype=np.uint16).reshape(N, K)
        out = np.empty(N, dtype=np.uint16)
        nonfin = np.zeros(N, dtype=bool)
        for n0 in range(0, N, GEMM_CHUNK):                                        # memory-bounded: the k16 chain is vectorised over <= GEMM_CHUNK coordinates
            acc, nf = dot(x[None, :], w[n0:n0 + GEMM_CHUNK])
            out[n0:n0 + GEMM_CHUNK] = D.f2fp_bf16(acc).astype(np.uint16)
            nonfin[n0:n0 + GEMM_CHUNK] = nf
        if family == "Gemm_v2" and nonfin.any():
            # the vectorised twins flag a non-finite operand / accumulator lane instead of modelling the total semantics (tc_dot16 docstring):
            # such a coordinate is outside the twin's domain -> unresolved by name (the ScaledMmFp8Block convention), never a silent word
            raise _Unresolved(f"Gemm_v2: {int(nonfin.sum())} output coordinates have a non-finite bf16 operand or accumulator "
                              "(outside the vectorised k16 twin's domain)")
        return {"0": out, "out": out}
    if family == "RoPE_v1":
        # [R17 dense, ruling R17-9] RoPE_v1{NHEADS,D}: arg 0 = the q (or k) slice of the biased qkv row (NHEADS*D bf16 words), arg 1 = ONE row
        # of the registered `weights` field `...rotary_emb.cos_sin_cache` (D bf16 words: [cos[0:D/2] | sin[0:D/2]] of THIS row's position --
        # the Program's slice names the position, nothing here infers it).  The row's one member is `0` (query) or `1` (key); the relation
        # is the same, the manifest names the member, so both names are returned.
        NH, Dh = int(statics["NHEADS"]), int(statics["D"])
        x = np.asarray(arg(0), dtype=np.uint16).reshape(NH, Dh)
        cs = np.asarray(arg(1), dtype=np.uint16).reshape(Dh)
        out = D.rope_rows(x, cs).reshape(NH * Dh)
        return {"0": out, "1": out, "out": out}
    if family in ("Attention_v3", "Attention_v2"):
        # [R17 dense, ruling R17-11 / BRIEF §4] one attention ROW of the Program (the Q_module_v1 unit): arg 0 = the RoPE query member
        # (NH*D words); arg 1 = the T committed KEY members (rotary_emb member `1`, KVH*D words each) of every token row this query
        # attends to, in Program order = key position; arg 2 = the T committed V slices of the same rows' qkv members.  Every K/V value
        # is a named committed word of the request (any earlier step: the resolver reads it at the producer's step) -- nothing is
        # inferred from a cache.  Attention_v3 is FA2 (Ampere/Ada: tc_dot16, Fa2InvSum); Attention_v2 binds the step and the epilogue
        # as statics (Hopper FA3: wgmma step, Fa3InvSum) -- resolved BY NAME to their twins, any other id is unresolved by name.  The
        # AttnBlock INTERIORS inside the row (Q_fine_v1 units) live in the M1 export stream, which this replay does not read -- the
        # report names that mechanism with the row count (ATTENTION_INTERIOR_MECHANISM).
        T, NH, KVH, Dh, BN = (int(statics[k]) for k in ("T", "NH", "KVH", "D", "BN"))
        if len(inputs) != 3:                                                     # [R17-14] the Definition's arity (q, kc, vc): a row wiring another is by name
            raise _Unresolved(f"{family}: the row wires {len(inputs)} args, the registered Definition takes 3 (q, K history, V history)")
        if family == "Attention_v2":
            dot_id, inv_id = prim_static(statics, "DOT", family), prim_static(statics, "INV", family)
        else:
            dot_id, inv_id = "AmpereBF16TcDot16_v1", "Fa2InvSum_v1"
        if dot_id not in D.TC_DOT16_BY_PRIM or inv_id not in D.INV_SUM_BY_PRIM:
            raise _Unresolved(f"no registered evaluator for {family} with DOT={dot_id}, INV={inv_id} "
                              f"(twins: {sorted(D.TC_DOT16_BY_PRIM)} x {sorted(D.INV_SUM_BY_PRIM)})")
        q = np.asarray(arg(0), dtype=np.uint16).reshape(NH, Dh)
        kc = np.asarray(arg(1), dtype=np.uint16)
        vc = np.asarray(arg(2), dtype=np.uint16)
        if kc.size != T * KVH * Dh or vc.size != T * KVH * Dh:
            raise _Unresolved(f"{family}: K history {kc.size} / V history {vc.size} words != T*KVH*D = {T}*{KVH}*{Dh}")
        try:
            out, nonfin = D.attention_rows(q, kc.reshape(T, KVH, Dh), vc.reshape(T, KVH, Dh), BN, dot_prim=dot_id, inv_prim=inv_id)
        except ValueError as e:                                                  # a geometry outside the registered relation: by name
            raise _Unresolved(f"{family}: {e}") from None
        if nonfin.any():
            raise _Unresolved(f"{family}: {int(nonfin.sum())} heads have a non-finite operand / accumulator (outside the vectorised twins' domain)")
        out = np.asarray(out, dtype=np.uint16).reshape(NH * Dh)
        return {"out": out}
    if family == "RMSNormTriton_v1":
        N = int(statics["N"])
        r = D.rmsnorm_triton_rows(np.asarray(arg(0), dtype=np.uint16).reshape(1, N), np.asarray(arg(1), dtype=np.uint16).reshape(N), _eps(statics))
        return {"out": np.asarray(r["out"], dtype=np.uint16).reshape(N)}
    if family == "RMSNormFusedCuda_v2":
        N = int(statics["N"])
        r = D.rmsnorm_fused_v2_rows(np.asarray(arg(0), dtype=np.uint16).reshape(1, N), np.asarray(arg(1), dtype=np.uint16).reshape(1, N),
                                    np.asarray(arg(2), dtype=np.uint16).reshape(1, N), _eps(statics))
        return {"0": np.asarray(r["normed"], dtype=np.uint16).reshape(N), "1": np.asarray(r["resid"], dtype=np.uint16).reshape(N)}
    if family == "SiluMul_v1":
        I = int(statics["I"])
        r = D.silu_mul_rows(np.asarray(arg(0), dtype=np.uint16).reshape(1, 2 * I))
        return {"out": np.asarray(r["out"], dtype=np.uint16).reshape(I)}
    if family == "Embedding_v1":
        H = int(statics["H"])
        ids = np.asarray(arg(0)).reshape(-1)
        tid = int(ids[0])
        table = np.asarray(arg(1), dtype=np.uint16).reshape(-1, H)
        if not 0 <= tid < table.shape[0]:
            raise ValueError(f"token id {tid} outside the embedding table ({table.shape[0]} rows)")
        return {"out": table[tid].copy()}
    if family == "BiasAdd_v1":
        N = int(statics["N"])
        y = D.as_f32(D.bf16_to_f32_words(np.asarray(arg(0), dtype=np.uint16).reshape(N)))
        b = D.as_f32(D.bf16_to_f32_words(np.asarray(arg(1), dtype=np.uint16).reshape(N)))
        return {"0": D.f2fp_bf16(D.as_u32((y + b).astype(np.float32)))}         # Bf16Add: one f32 add, RNE to bf16
    if family == "Bf16MulScalarTensor_v1":
        # [R19 int-20] b1.Bf16MulScalarTensor{N}: out_j = F2fpBf16(F32Mul(Bf16ToF32(x_j), Bf16ToF32(s))) -- arg 0 the N bf16 words of the row,
        # arg 1 the ONE bf16 word of the scalar tensor (the Definition's `s: Array(1, BF16)`); another arity / width is by name
        N = int(statics["N"])
        if len(inputs) != 2:
            raise _Unresolved(f"{family}: the row wires {len(inputs)} args, the registered Definition takes 2 (x, s)")
        x = np.asarray(arg(0), dtype=np.uint16).reshape(-1)
        s = np.asarray(arg(1), dtype=np.uint16).reshape(-1)
        if x.size != N:
            raise _Unresolved(f"{family}: x has {x.size} words, the signature has N = {N}")
        if s.size != 1:
            raise _Unresolved(f"{family}: the scalar operand has {s.size} words, the signature has 1 (Array(1, BF16))")
        xf = D.bf16_to_f32_words(x)
        sf = np.broadcast_to(D.bf16_to_f32_words(s), xf.shape)
        return {"out": D.f2fp_bf16(D.mul_rn(xf, sf)).astype(np.uint16)}
    if family == "TokenSelect_v1":
        V = int(statics["V"])
        return {SAMPLED_MEMBER: np.asarray([token_select_first_max(np.asarray(arg(0), dtype=np.uint16).reshape(V))], dtype=np.uint64)}
    if family == "GumbelTopPTokenSelect_v1":
        return {SAMPLED_MEMBER: np.asarray([gumbel_topp_token_select(statics, [arg(g) for g in range(len(inputs))])], dtype=np.uint64)}
    if family == "Fp8GroupQuant_v1":
        # [R16 fp8] interior of the FP8 linear: ports 0 (K e4m3 words) and 1 (K/G f32 group scales), in the Definition's return order --
        # the consumer's `['n', row, lo, hi]` slices index the concatenation (`_interior_slice`)
        K, G = int(statics["K"]), int(statics["G"])
        xq, sx = D.fp8_group_quant_row(np.asarray(arg(0), dtype=np.uint16).reshape(K), G=G)
        return {"0": np.asarray(xq, dtype=np.uint8), "1": np.asarray(sx, dtype=np.uint32)}
    if family == "ScaledMmFp8Block_v1":
        # [R16 fp8] `_C.cutlass_scaled_mm` blockwise on one token row: xq / sx composed from the interior quant, w (N x K e4m3) and
        # ws (N/G x K/G f32, `weight_scale_inv`) the registered checkpoint words.  A lane with a NaN e4m3 byte or a non-finite
        # accumulator is outside the twin's domain -> unresolved by name (the tc_dot16 convention), never a silent word.
        K, N, G = int(statics["K"]), int(statics["N"]), int(statics["G"])
        xq = np.asarray(arg(0), dtype=np.uint8).reshape(K)
        sx = np.asarray(arg(1), dtype=np.uint32).reshape(K // G)
        w = np.asarray(arg(2), dtype=np.uint8).reshape(N, K)
        ws = np.asarray(arg(3), dtype=np.uint32).reshape(N // G, K // G)
        out, nonfin = D.scaled_mm_fp8_block_row(xq, sx, w, ws, G=G, chunk=GEMM_CHUNK)
        if nonfin.any():
            raise _Unresolved(f"ScaledMmFp8Block: {int(nonfin.sum())} output coordinates have a NaN e4m3 operand or a non-finite accumulator "
                              "(outside the vectorised twin's domain)")
        return {"0": np.asarray(out, dtype=np.uint16)}
    if family in ("MoeRouterTopK_v1", "MoeRouterTopKNorm_v1"):
        E, TOPK, VPT = int(statics["E"]), int(statics["TOPK"]), int(statics.get("VPT", 8))
        w, ids = moe_router_topk(np.asarray(arg(0), dtype=np.uint16).reshape(E), E=E, TOPK=TOPK, VPT=VPT, renormalize=family.endswith("Norm_v1"))
        return {"router_w": w, "router_ids": ids}
    if family in ("MoeExpertGemm_v1", "MoeExpertGemmW_v1"):
        E, K, N = int(statics["E"]), int(statics["K"]), int(statics["N"])
        x = np.asarray(arg(0), dtype=np.uint16).reshape(K)
        e = int(np.asarray(arg(1)).reshape(-1)[0])
        if not 0 <= e < E:
            raise ValueError(f"routed expert id {e} outside the expert axis (E = {E})")
        wi = 3 if family == "MoeExpertGemmW_v1" else 2
        wall = np.asarray(arg(wi), dtype=np.uint16)
        if wall.size != E * N * K:
            raise ValueError(f"expert weight slab has {wall.size} words, not E*N*K = {E}*{N}*{K}")
        w = wall.reshape(E, N, K)[e]                                              # the gather by the VALUE operand e (MoeExpertRow): one expert's [N, K]
        gw = np.asarray(arg(2)).reshape(-1)[0] if family == "MoeExpertGemmW_v1" else None
        out = np.empty(N, dtype=np.uint16)
        for n0 in range(0, N, GEMM_CHUNK):
            acc, _nf = D.dot_bf16(x[None, :], w[n0:n0 + GEMM_CHUNK])
            if gw is not None:                                                    # routed weight into the f32 accumulator BEFORE the one bf16 conversion
                acc = D.mul_rn(acc, np.full(acc.shape, int(gw) & 0xFFFFFFFF, dtype=np.uint32))
            out[n0:n0 + GEMM_CHUNK] = D.f2fp_bf16(acc).astype(np.uint16)
        return {"down" if gw is not None else "up": out}
    if family in ("MoeFfnFixedL_v1", "MoeFfnFixed_v1"):
        return padded_moe_block(statics, [arg(g) for g in range(len(inputs))], router_gemm=family == "MoeFfnFixed_v1")
    if family == "MoeSum_v1":
        TOPK, H = int(statics["TOPK"]), int(statics["H"])
        y = np.asarray(arg(0), dtype=np.uint16).reshape(TOPK, H)
        acc = np.zeros(H, dtype=np.uint32)                                        # Const32[0]; ascending slot order, one RNE at the end
        for k in range(TOPK):
            acc = D.add_rn(acc, D.bf16_to_f32_words(y[k]))
        return {"sum": D.f2fp_bf16(acc).astype(np.uint16)}
    raise KeyError(family)


def _tp_member_alias(got: dict[str, Any], member: str) -> Any | None:
    """[TP-08d, MOE-TP-01] The evaluated value a rank Program's TP member (`part_rank<r>` / `shard_rank<r>`) compares against when the
    relation did not name it: the evaluator's `0` / `out`, else the relation's ONE output whatever its name (MoeSum's `sum` = the rank's
    MoE-block partial before the all-reduce).  None for any other member name, or for a multi-output relation (never a guess between two)."""
    if member in got or not member.startswith((RM.TP_PARTIAL_MEMBER.split("{")[0], RM.TP_SHARD_MEMBER.split("{")[0])):
        return None
    if "0" in got:
        return got["0"]
    if "out" in got:
        return got["out"]
    if len(got) == 1:
        return next(iter(got.values()))
    return None


def _interior_slice(got: dict[str, np.ndarray], lo: int, hi: int) -> np.ndarray | None:
    """[R16 fp8] The words [lo, hi) of an INTERIOR producer's output refs = the concatenation of its members in return order (a
    Fp8GroupQuant row returns (xq[K], sx[K/G]); a member-less Gemm returns its one output).  None when the slice leaves the output."""
    parts = [np.asarray(v).reshape(-1) for v in got.values()]
    total = sum(int(p.size) for p in parts)
    if lo < 0 or hi > total:
        return None
    if len(parts) == 1:
        return parts[0][lo:hi]
    off, out = 0, []
    for p in parts:
        a, b = max(lo, off), min(hi, off + p.size)
        if a < b:
            out.append(p[a - off:b - off])
        off += p.size
    if len(out) == 1:
        return out[0]
    return np.concatenate([o.astype(np.uint32) for o in out])       # a slice across members of different widths: widened words (values kept)


_SLOT_MEMBER_RE = re.compile(r"^slot(?P<e>\d+)/(up|act|down|g)$")


def padded_slot_enable(member: str, got: dict[str, np.ndarray], store: "CommittedStore", rid: str, step: int, op_path: str, row: int,
                       topk: int) -> tuple[bool | None, str]:
    """[revb F-r17b-52 (ii)] the enable a_e of a padded slot member of one token row, from the COMMITTED router_ids of that row (the block's
    enable is the step's run bit -> the router members are Active in every executed step; a_e = [e in ids]).  None + why when the committed
    ids are not available (the slot's verdict is then not evaluable by name -- never the twin's ids)."""
    m = _SLOT_MEMBER_RE.match(member)
    if m is None:
        return True, ""                                                          # router_ids / router_w: Active under the block's enable
    e = int(m.group("e"))
    w, why = store.words(rid, step, op_path, "router_ids", row * topk, (row + 1) * topk)
    if w is None:
        return None, f"slot enable of {member}: the row's committed router_ids are not available -- {why}"
    ids = np.asarray(w).reshape(-1)
    bot = RM.lifted_bot(RM.LIFTED_F32)
    if any(int(v) & bot for v in ids):
        return None, f"slot enable of {member}: a committed router id of the row is bot ({[hex(int(v)) for v in ids]}) in an executed step"
    return bool(any(int(v) == e for v in ids)), ""


def lifted_compare(have: np.ndarray, payload: np.ndarray, active: bool, dtype: str) -> tuple[bool, int, int | None, str]:
    """[revb F-r17b-52 (ii)] committed LIFTED words (`have`, the plane's machine words) vs the twin's unlifted payload under the slot enable:
    Active -> every word must be Active(payload) = payload exactly (tag clear); disabled -> every word must be enc(bot) = 1 << w exactly
    (never INV, never a bot with a nonzero payload).  Returns (equal, differing_elements, first_diff, verdict)."""
    wbits, nbytes = RM.LIFTED_DTYPES[dtype]
    bot = 1 << wbits
    dt = np.uint32 if nbytes == 4 else np.uint64
    have = np.asarray(have).reshape(-1).astype(dt, copy=False)
    if active:
        want = np.asarray(payload).reshape(-1).astype(dt, copy=False)
        verdict = "active: word == Active(twin value)"
    else:
        want = np.full(have.shape if have.size else np.asarray(payload).reshape(-1).shape, bot, dtype=dt)
        verdict = "disabled: word == enc(bot) exactly"
    if have.shape != want.shape:
        return False, -1, None, verdict + f" -- width {have.shape} vs {want.shape}"
    diff = have != want
    nd = int(np.count_nonzero(diff))
    if nd == 0:
        return True, 0, None, verdict
    first = int(np.flatnonzero(diff)[0])
    hv = int(have[first])
    kind = ("bot word under an Active enable (pad R17-42 s5)" if (active and hv == bot) else
            "value word under a disabled enable" if (not active and not (hv & bot)) else
            "bot with a nonzero payload / INV" if (not active and (hv & bot) and hv != bot) else
            "INV word (bot bit + nonzero payload) under an Active enable" if (active and (hv & bot)) else "payload differs")   # revb n-52-2
    return False, nd, first, verdict + f" -- FAIL: {kind}"


_PADDED_BLOCK_MEMO: "collections.OrderedDict[tuple, dict[str, np.ndarray]]" = collections.OrderedDict()
_PADDED_BLOCK_MEMO_MAX = 4


def padded_moe_block(statics: dict[str, Any], args: list[np.ndarray], *, router_gemm: bool) -> dict[str, np.ndarray]:
    """`MoeFfnFixedL_v1{E,TOPK,VPT,H,I}(x, logits, w13, w2)` (and `MoeFfnFixed_v1(x, wg, w13, w2)` with `router_gemm`) on one token row --
    the vectorised twin of `verity_vllm.program.registry.moe_pad._fixed_core`, statement for statement, built from the twins the MOE-01 evaluators
    already use (dot_bf16 / f2fp_bf16 / silu_mul_rows / mul_rn / add_rn / moe_router_topk).  Returns EVERY member a query may name on the
    row: `out` [H] bf16, `router_ids` [TOPK] i32, `router_w` [TOPK] f32, and per slot e `slot<e>/up` [2I] bf16, `slot<e>/act` [I] bf16,
    `slot<e>/down` [H] bf16, `slot<e>/g` [1] f32 (0.0f where the router did not pick e; the body is evaluated for EVERY slot, as the
    construction computes it).  Memoised on (statics, x, logits/wg) so the picks of one token row share one 64-body evaluation."""
    from verity_vllm.program.registry import derived_rows as D

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


def moe_router_topk(logits: np.ndarray, *, E: int, TOPK: int, VPT: int, renormalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`MoeRouterTopK_v1{E,TOPK,VPT}` (`_moe_C.topk_softmax`, one token row) evaluated with the registry's scalar primitives, statement for
    statement the body of `verity_vllm.program.registry.moe.MoeRouterTopK`: E bf16 logits -> f32 (Bf16ToF32); thread t of E/VPT holds columns
    t*VPT .. t*VPT+VPT-1: thread max (F32Max, ascending), warp butterfly xor T/2 .. 1 (lane 0); `ex_i = NvExpf(x_i - max)` (F32Sub, the
    CUDA math-library expf sequence, MUFU.EX2 from the measured table); thread sum from 0.f (F32Add, ascending), butterfly; `rcp = 1.f / sum`
    (F32Div, IEEE); `p_i = ex_i * rcp` (F32Mul), non-finite -> +0; TOPK rounds: lowest-index strict argmax (F32GtStrict scan from p[0]),
    the winner masked to -10000.f.  `renormalize` (MoeRouterTopKNorm_v1, MOE-04): selected sum ascending from 0.f, denom = sum > 0 ? sum :
    1.f, scale = 1.f / denom (F32Div), w_k = w_k * scale (F32Mul).  Returns (router_w f32 words [TOPK] u32, router_ids i32 words [TOPK] u32)."""
    from verity_vllm.program.registry import moe as RMOE
    from verity_vllm.program.registry import prims as P
    from verity_vllm.program.registry import ref_prims as R

    if E % VPT:
        raise ValueError(f"E = {E} is not a multiple of VPT = {VPT}")
    T = E // VPT
    x = [P.Bf16ToF32.evaluate(int(v)) for v in np.asarray(logits, dtype=np.uint16).reshape(E)]

    def butterfly(op, vals: list[int]) -> int:
        mask = len(vals) // 2
        while mask >= 1:
            vals = [op(vals[t], vals[t ^ mask]) for t in range(len(vals))]
            mask //= 2
        return vals[0]

    tmax = []
    for t in range(T):
        m = x[t * VPT]
        for j in range(1, VPT):
            m = P.F32Max.evaluate(m, x[t * VPT + j])
        tmax.append(m)
    rmax = butterfly(P.F32Max.evaluate, tmax)
    ex = [RMOE.nv_expf_bits(R.F32Sub.evaluate(x[i], rmax)) for i in range(E)]
    tsum = []
    for t in range(T):
        s = 0
        for j in range(VPT):
            s = P.F32Add.evaluate(s, ex[t * VPT + j])
        tsum.append(s)
    rsum = butterfly(P.F32Add.evaluate, tsum)
    one = 0x3F800000
    rcp = P.F32Div.evaluate(one, rsum)
    p = []
    for i in range(E):
        q = P.F32Mul.evaluate(ex[i], rcp)
        p.append(q if RMOE.F32IsFinite.evaluate(q) else 0)
    neg_big = 0xC61C4000                                                          # -10000.f
    weights, ids = [], []
    for _ in range(TOPK):
        best, best_i = p[0], 0
        for i in range(1, E):
            if R.F32GtStrict.evaluate(p[i], best):
                best, best_i = p[i], i
        weights.append(best)
        ids.append(best_i)
        p = [neg_big if i == best_i else p[i] for i in range(E)]
    if renormalize:
        acc = 0
        for k in range(TOPK):
            acc = P.F32Add.evaluate(acc, weights[k])
        denom = acc if R.F32GtStrict.evaluate(acc, 0) else one
        scale = P.F32Div.evaluate(one, denom)
        weights = [P.F32Mul.evaluate(w, scale) for w in weights]
    return (np.asarray([w & 0xFFFFFFFF for w in weights], dtype=np.uint32), np.asarray([i & 0xFFFFFFFF for i in ids], dtype=np.uint32))


# --------------------------------------------------------------------------------------------------------------------- the sampler

def _phase(step: int, last_step: int) -> str:
    """Phase axis of the stratum key.  `last_step` = the request's LAST EXECUTED step: cap - 1, or -- [revb F-r17b-39] under an executed
    prefix (R17-26 / R17-1) -- n_exec - 1, the lag step of an EOS-stopped request (the R17-1 / F-r17-14 boundary), so that request's
    `last` stratum exists and is sampled deterministically like every other request's; steps past it are 'not-executed' (split out of the
    pool by name, never a 'last' or a 'mid')."""
    if step == 0:
        return "prefill"
    if step == last_step:
        return "last"
    if step > last_step:
        return "not-executed"
    return "first-decode" if step == 1 else "mid"


def phase_anchor(pi_n_steps: int, ep: dict | None) -> dict[str, Any]:
    """[revb F-r17b-39] The step the `last` phase of a request is anchored to, NAMED: n_exec - 1 under a fault-free executed prefix that
    stops before the cap (the lag step), else cap - 1.  On the record per request (`strata_keys.phase_anchor`)."""
    cap_last = int(pi_n_steps) - 1
    if ep and not ep.get("faults") and int(ep["n_exec"]) < int(ep["cap"]):
        n_exec, lag = int(ep["n_exec"]), int(ep.get("lag") or 0)
        # [revb n-39-2] under LAG 0 the last executed step is the EOS step, not a lag step -- the text says which
        last_is = "the lag step" if lag > 0 else "the EOS step (LAG 0: no lag step)"
        return {"last_step": n_exec - 1, "anchor": "n_exec - 1", "cap_last_step": cap_last, "n_exec": n_exec,
                "lag_step": (n_exec - 1 if lag > 0 else None),
                "why": (f"the request stopped at a_r {ep.get('a_r')} under LAG {lag}: steps [{n_exec}, {int(ep['cap'])}) are not executed "
                        f"(R17-1); its last EXECUTED step {n_exec - 1} is {last_is} -- the `last` stratum is anchored there, not at the cap "
                        f"(F-r17b-39)")}
    if ep and not ep.get("faults"):
        # [revb NIT n-39-1] an EOS stop at a_r = cap - LAG under LAG > 0 has n_exec == cap: the cap anchors, AND its last step IS a lag
        # step (F-r17b-35's boundary) -- the flag follows the STOP (a_r < cap and LAG > 0), not n_exec < cap.
        n_exec, a_r, lag = int(ep["n_exec"]), int(ep.get("a_r") if ep.get("a_r") is not None else ep["cap"]), int(ep.get("lag") or 0)
        eos_stop_lag = a_r < int(ep["cap"]) and lag > 0
        return {"last_step": cap_last, "anchor": "cap - 1", "cap_last_step": cap_last, "n_exec": n_exec,
                "lag_step": (n_exec - 1 if eos_stop_lag else None),
                "why": (f"the request stopped at a_r {a_r} under LAG {lag} and its executed prefix reaches the cap (n_exec {n_exec} == cap): the cap "
                        f"anchors and its last step {n_exec - 1} is the lag step (n-39-1)" if eos_stop_lag else "the request ran to its cap")}
    return {"last_step": cap_last, "anchor": "cap - 1", "cap_last_step": cap_last, "n_exec": None,
            "lag_step": None, "why": ("no executed prefix" if not ep else "executed prefix has faults: the cap anchors")}


def _lifetime(engine_step: int, ring_slots: int | None) -> str:
    """Storage-reuse axis: with a windowed collector the step's device window is ring slot (engine_step % ring_slots); from engine step
    ring_slots on, every committed identity occupies storage the run already committed (and released) under an earlier step's identities
    -- the post-overwrite / stale-window reads live there.  Unwindowed collectors: one value, stated."""
    if not ring_slots or int(ring_slots) <= 0:
        return "n/a (unwindowed storage)"
    return "recycled-window" if int(engine_step) >= int(ring_slots) else "fresh-window"


def moe_plane_kind(family: str, module: str) -> str | None:
    """[MAN-06f v3] the fused-MoE block row kinds whose values the collector commits as `moe_block_stream` PLANES (MOE-03: six planes per
    MoE layer per step, pair-major) rather than as module outputs: `router` (MoeRouterTopK / MoeRouterTopKNorm: router_w + router_ids
    per token), `up` (MoeExpertGemm: the w13 launch per pair), `act` (the SiluMul rows under the MoE runner module `...experts`: the
    silu_and_mul per pair), `down` (MoeExpertGemmW: the w2 launch per pair), `sum` (MoeSum per token).  None for every other row.  From the
    Program's spec family + module name alone."""
    if family.startswith("MoeRouterTopK"):
        return "router"
    if family.startswith("MoeExpertGemmW"):
        return "down"
    if family.startswith("MoeExpertGemm"):
        return "up"
    if family.startswith("MoeSum"):
        return "sum"
    if family.startswith("SiluMul") and module.rsplit(".", 1)[-1] == RM.MOE_LEAF:
        return "act"
    return None


#: the `moe_block_stream` address schema (the manifest / `binding.py` write `moe/L<layer>` as the op_path of the six
#: planes): parsed here to label the (request, step, layer) triple -- the family's own address vocabulary, not a module-path rule
_MOE_LAYER_RE = re.compile(r"^moe/L(\d+)$")


def moe_plane_alias_check(store: CommittedStore, *, max_named: int = 12) -> dict[str, Any]:
    """[FLUSH-03 fact of record; rev M-0494 (3)] per (request, step, MoE layer) with BOTH planes committed: the committed `up` bytes differ
    from the committed `down` bytes.  Before FLUSH-03 the gather collector committed the same layer's down plane under the up name (vLLM's
    TritonExperts carves intermediate_cache1 = up and intermediate_cache3 = down out of ONE workspace; with no flush between the block's two
    GEMM launches the staged up view was read after the w2 launch had overwritten it -- moe M-0205/M-0232, 22/22 same-layer pairs identical
    on the R15/int-1 OLMoE records).  The comparison is over the COMMON PREFIX of the two request-local slices (min(len) bytes): when
    2I == H (OLMoE) that is the whole plane; when the widths differ the alias would still show as identical leading bytes (one base pointer,
    two row strides), so a prefix test catches it on every geometry.  Result: True = every compared pair differs; False = an identical pair
    (named: request, step, layer, bytes) -- the up plane is the down plane's alias; None = no MoE layer with both planes committed (dense
    model, or the planes not bound).  A FACT beside the replay (the MAN-06f v3 GEMM relation already fails an aliased plane by name);
    the known-good root re-pin of the OLMoE reference (int-4) stands on this line, not on the run reproducing itself."""
    layers: dict[tuple[str, int, int], dict[str, tuple[int, int]]] = {}
    for (rid, step, op, mem), es in store.idx.items():
        m = _MOE_LAYER_RE.match(str(op))
        if not m or mem not in ("up", "down"):
            continue
        rng = None
        for e in es:                                                            # the bound request-local range (every invocation under the key)
            r = (e.get("identity") or {}).get("element_range") or [0, None]
            if r[1] is not None:
                rng = (int(r[0]), int(r[1])) if rng is None else (min(rng[0], int(r[0])), max(rng[1], int(r[1])))
        if rng is not None:
            layers.setdefault((rid, int(step), int(m.group(1))), {})[mem] = rng
    compared, identical, errors, per_layer = 0, [], [], {}
    widths: dict[str, set[int]] = {"up": set(), "down": set()}
    for (rid, step, layer), mems in sorted(layers.items()):
        if "up" not in mems or "down" not in mems:
            continue
        got: dict[str, bytes] = {}
        for mem in ("up", "down"):
            lo, hi = mems[mem]
            w, why = store.words(rid, step, f"moe/L{layer}", mem, lo, hi)
            if w is None:
                if len(errors) < max_named:
                    errors.append({"request_id": rid, "step": step, "layer": layer, "plane": mem, "why": str(why)[:200]})
                break
            got[mem] = w.tobytes()
            widths[mem].add(hi - lo)
        if len(got) < 2:
            continue
        n = min(len(got["up"]), len(got["down"]))
        compared += 1
        pl = per_layer.setdefault(f"L{layer}", {"compared": 0, "identical": 0})
        pl["compared"] += 1
        if n > 0 and got["up"][:n] == got["down"][:n]:
            pl["identical"] += 1
            if len(identical) < max_named:
                identical.append({"request_id": rid, "step": step, "layer": layer, "bytes_compared": n,
                                  "sha256": hashlib.sha256(got["up"][:n]).hexdigest()[:16]})
    n_ident = sum(v["identical"] for v in per_layer.values())
    result = None if compared == 0 else (n_ident == 0)
    return {"schema": "verity-sweep/moe-plane-alias-check/v1", "result": result, "compared": compared, "identical_n": n_ident,
            "identical": identical, "read_errors_n": len(errors), "read_errors": errors, "per_layer": per_layer,
            "plane_words": {k: sorted(v) for k, v in widths.items()},
            "rule": "for every (request, step, MoE layer) with both planes committed: committed up bytes != committed down bytes over their common "
                    "prefix (min length); identical = the up plane is the down plane's alias (pre-FLUSH-03 defect); None = no MoE layer with both planes",
            "why": (None if result is not False else
                    f"{n_ident} of {compared} (request, step, layer) triples commit IDENTICAL up and down plane bytes -- the up plane is the down plane's "
                    f"alias (FLUSH-03): first {identical[0]['request_id']} step {identical[0]['step']} L{identical[0]['layer']}")}


def _interior_feeders_without_evaluator(pi: ProgramIndex, r: dict, *, depth: int = 0) -> set[str]:
    """[MAN-06g, hopper BLOCKER M-0748 F-hopper-BIAS2 / rev-cb M-0751 (b)] the INTERIOR producers this row's VU would have to compose that
    this replay cannot evaluate: an `['n', producer, lo, hi]` input whose producer row claims no committed member (interior: the pre-bias
    Gemm under a BiasAdd, MAN-07) and whose family has no registered evaluator (Gemm_v2 on Hopper / batch-invariant Ada Programs), or a
    composable interior whose own inputs have that defect (bounded by INTERIOR_DEPTH).  Derived from the Program rows alone (families +
    member claims), never from the collector's emissions.  Empty when every interior input composes (Gemm_v1 under BiasAdd_v1)."""
    out: set[str] = set()
    for group in r.get("args") or []:
        for sl in group:
            if not sl or sl[0] != "n":
                continue
            a, lo, hi = int(sl[1]), int(sl[2]), int(sl[3])
            if pi.member_at(a, lo, hi) is not None:
                continue                                                         # a committed member: read at replay, nothing to compose
            pr = pi.by_i.get(a)
            if const_word(pr) is not None:
                continue                                                         # a Program constant (Const32/64[..]): read from the text, nothing to compose
            pd = pi.describe(pr) if pr is not None else None
            if pd is None or pd["members"]:
                continue                                                         # unmodelled / a member-straddling slice: a replay-time fault, not a name
            pf = pd["family"]
            if pf not in EVALUATORS:
                out.add(f"{pf} ({NOT_YET.get(pf, 'no registered evaluator')})")
            elif depth + 1 >= INTERIOR_DEPTH:
                out.add(f"{pf} (interior chain deeper than {INTERIOR_DEPTH})")
            else:
                out |= _interior_feeders_without_evaluator(pi, pr, depth=depth + 1)
    return out


def population(pi: ProgramIndex, rid: str, *, rank: int = 0, arrive: int = 0, ring_slots: int | None = None,
               addresses: dict[tuple, dict] | None = None, last_step: int | None = None,
               codes: dict[str, dict] | None = None) -> tuple[list[dict], Counter, Counter]:
    """Every Program row of one request is in the population: the evaluable VUs [{i, spec, family, phase, stratum}], the not-evaluable
    count by reason, and the not-evaluable count BY FAMILY (rev-cb M-0411 O4: strata of the record by name, never dropped).
    `ring_slots`: the collector's window ring depth -- an identity committed at ENGINE step >= ring_slots lives in storage the ring
    already committed under another step's identities (window recycle): the `lifetime` stratum axis (rev-cb O3 / coord M-0451 O3).
    `addresses` [R17-3 reqval]: an accumulator keyed by the row's MANIFEST address (request, request-relative step, op_path, member) ->
    {spec, family, class ('vu' | a why class), why, rows}: every row that produces a required identity is recorded at that identity,
    evaluable or not, so `query_population` can reconcile the replay's reach against Q(P) identity by identity (a row with no member --
    structure, interior -- produces no identity and is not recorded).
    `last_step` [revb F-r17b-39]: the request's last EXECUTED step (`phase_anchor(...)["last_step"]`: n_exec - 1 under an executed prefix,
    the lag step of an EOS-stopped request); None = the cap's last step.  The POPULATION is the same either way (every row is enumerated);
    only the phase label of the stratum key moves.
    `codes`: an accumulator, not-evaluable reason -> its structured code (`replay_codes.reason_code`), what the verdict reads."""
    vus, not_ev, not_ev_fam = [], Counter(), Counter()
    last = pi.n_steps - 1 if last_step is None else int(last_step)

    def _not_ev(why: str, cls: str, fam: str, *, domain: bool = False, tags: tuple[str, ...] = ()) -> None:
        not_ev[why] += 1
        if codes is not None:
            codes[why] = RC.reason_code(cls, fam, domain=domain, tags=tags)

    def _addr(d: dict | None, fam: str, spec: str, cls: str, why: str | None) -> None:
        if addresses is None or d is None:
            return
        # [MOE-TP-02] a member committed under a second required name (the TP rank's MoeSum: part_rank<r> AND the moe/L<k>/sum plane) is
        # recorded at BOTH identities -- the same row, the same class
        # `manifest_family`: the manifest family the address belongs to (the label build_manifest assigns), read off the address rule that
        # produced it -- the sampler op, a plane description (`plane_kind`) / a plane copy (MOE-TP-02), else the member's own family
        # (`manifest_format.alias_family`: a rank-local partial vs an instance output) -- never parsed back out of the op_path string
        keys = [(str(d["op_path"]), str(m[0]), "moe_block_stream" if d.get("plane_kind") else None) for m in d["members"]] \
            + [(str(c["op_path"]), str(c["member"]), "moe_block_stream") for c in d.get("copies") or []]
        for op, mem, mfam in keys:
            k = (rid, int(d["step"]), op, mem)
            a = addresses.get(k)
            if a is None:
                mfam = "sampled_token_ids" if op == SAMPLER_OP else (mfam or RM.alias_family(mem))
                addresses[k] = {"spec": spec, "family": fam, "class": cls, "why": why, "rows": 1, "manifest_family": mfam}
            else:
                a["rows"] += 1
                if a["class"] != cls:                                            # one identity, rows of two classes: named, never averaged
                    a["class"], a["why"] = "mixed", f"{a['class']} + {cls}: {a['why']} | {why}"

    for r in pi.rows:
        fam = RM.spec_family(r["spec"])
        if fam in NOT_YET:
            _not_ev(NOT_YET[fam], why_class(NOT_YET[fam]), fam, domain=True)
            not_ev_fam[fam] += 1
            if addresses is not None:
                try:
                    _addr(pi.describe(r), fam, r["spec"], why_class(NOT_YET[fam]), NOT_YET[fam])
                except Exception:  # noqa: BLE001  -- a NOT_YET row the address rule cannot describe is still counted above
                    pass
            continue
        d = pi.describe(r)
        if d is not None and d.get("plane_error"):
            # [MAN-06f v3] a fused-MoE block row whose row -> plane slice mapping the Program does not support (inconsistent geometry, an
            # ordinal outside the plane, a name index that disagrees with Program order): NAMED, class `mapping` -- a fault of the record's
            # Program shape, never a guess at another pair's words and never carried on a PASS.
            _not_ev(f"MoE plane row -> plane slice mapping failed: {d['plane_error']}", "mapping", fam)
            not_ev_fam[f"{fam} (plane mapping failed)"] += 1
            continue
        if fam not in EVALUATORS:
            if d is None:
                _not_ev(f"unmodelled family {fam} (manifest incomplete -- coverage fails closed)", "mapping", fam)
                not_ev_fam[fam] += 1
            elif not d["members"]:
                not_ev_fam[f"{fam} (structure / interior: no identity)"] += 1        # Const*, BiasAdd, ...: counted, no relation to replay
            elif d.get("plane"):
                # [MAN-06f v3 unit 1] a plane PRODUCER (router / expert GEMM / sum) whose relation has no evaluator in this version: its
                # plane IS read -- as the committed input of the rows it feeds (up -> the act SiluMul, sum -> the next fused norm) -- but
                # the plane's own production is not re-evaluated.  Named under the moe_block_stream class so the verdict's (B)-conditioned
                # population-gap rule keeps gating it until the evaluator lands (unit 2), never a silent 'no evaluator' domain exclusion.
                w = (f"no registered evaluator for {fam}: moe_block_stream plane producer ({d['plane']} plane read as its consumers' committed input; "
                     f"evaluator = MAN-06f v3 unit 2)")
                _not_ev(w, "plane-fed", fam, tags=("MAN-06f",))
                not_ev_fam[f"{fam} (moe_block_stream plane producer: no evaluator)"] += 1
                _addr(d, fam, r["spec"], "plane-fed", w)
            else:
                w = f"no registered evaluator for {fam}"
                _not_ev(w, "no-evaluator", fam)
                not_ev_fam[fam] += 1
                _addr(d, fam, r["spec"], "no-evaluator", w)
            continue
        if d is None:
            _not_ev(f"unmodelled family {fam}", "mapping", fam)
            not_ev_fam[fam] += 1
            continue
        if not d["members"]:                                                     # [MAN-07 / R15 dense] an evaluable family whose row has no
            not_ev_fam[f"{fam} (interior: composed into its epilogue's VU / a hinted row of the module is its returned tensor)"] += 1
            continue                                                             # committed member of its own: the pre-bias Gemm under a BiasAdd,
                                                                                 # the pre-cap logits Gemm -- interior, no VU, counted by name
        ife = _interior_feeders_without_evaluator(pi, r)
        if ife:
            # [MAN-06g, hopper M-0748 / exec M-0749 / rev-cb M-0751 (b)] the VU's input is an INTERIOR (uncommitted) value of a fused producer
            # this replay cannot compose (the pre-bias Gemm_v2 under Qwen2.5's biased qkv_proj: BiasAdd_v1 is the producer of record, its
            # input is interior, Gemm_v2 has no evaluator): counted BY NAME as not evaluable -- the same shape as the MoE planes -- never
            # sampled as a stratum that stays 'not evaluated' (partial = FAIL closed on every biased-linear row).  The values ARE required
            # and covered by coverage + the C2 oracle-compare; the composed evaluator F(x, W, b) is the v2 follow-up.
            w = f"fed by interior producer(s) without an evaluator ({', '.join(sorted(ife))}): composition not possible in this replay (MAN-06g, v2)"
            _not_ev(w, "interior-composed-v2", fam, tags=("MAN-06g",))
            not_ev_fam[f"{fam} (fed by interior {', '.join(sorted(x.split(' (')[0] for x in ife))}: composed evaluator, v2)"] += 1
            _addr(d, fam, r["spec"], "interior-composed-v2", w)
            continue
        _addr(d, fam, r["spec"], "vu", None)
        st = d["step"]
        # stratum = (spec incl. shape, request, rank, phase, MODULE op_path): the module axis makes every (module, request, phase) cell
        # a stratum of its own, so a relabel / a wrong-value emission of ANY module at a request's prefill, first-decode or last step is
        # sampled deterministically (the cell holds one row group); only a mid-phase single-identity fault is sampled with probability
        # n / (steps in the phase) -- targets (`--replay-target`) pin those for the negatives of record
        life = _lifetime(arrive + st, ring_slots)
        sspec = _stratum_spec(fam, r["spec"], d.get("statics") or {})
        vus.append({"i": int(r["i"]), "spec": r["spec"], "family": fam, "request_id": rid, "step": st, "phase": _phase(st, last), "rank": int(rank),
                    "op_path": d["op_path"], "lifetime": life, "stratum": f"{sspec}|{rid}|rank{int(rank)}|{_phase(st, last)}|{d['op_path']}|{life}"})
    return vus, not_ev, not_ev_fam


# ------------------------------------------------------------------------------------------------------- Q(P) reconciliation (R17-3)

#: [R17-3 / reqval] the query families the exact replay has NO VU row for, with the mechanism of record that checks their values.  The
#: population itself is the manifest's (Q(P)); this names, per family, what covers it when no evaluator can -- a required family with no
#: entry here is reported `class: unstated` (fail closed on the record; never silently "covered").
QUERY_FAMILY_MECHANISM: dict[str, str] = {
    "weights": "registered inputs: MAN-09 weights pin (served parameters == the checkpoint of record, by digest) + C1 coverage / C3 openings",
    "input_token_ids": "prescribed input: boundary linkage (the committed ids at the arrival step == the declared prompt; sampled id[s-1] -> id[s]) + C1 / C3",
    # [revb n-2.16-1] the sampler families name what CONSUMES them, as input_token_ids' entry does: the committed seed / splits are the
    # inputs of the sampler VU (GumbelTopPTokenSelect_v1) whose exact replay in form (A) reads them by binding, and of stoch_recompute's
    # R17-2 attribution (the served token re-derived from the committed logits + seed + splits) -- not only a linkage check on their values
    "sampler_seed": ("prescribed input: prescribed_input_linkage (committed seed == the declared SamplingParams.seed, ruling 36) + C1 / C3; "
                     "CONSUMED BY the sampler VU GumbelTopPTokenSelect_v1's exact replay, which reads the committed seed as its Philox key "
                     "(seed, pos) -- the committed sampled id is reproducible from the committed logits only with it (form (A) per pick; the "
                     "seed is a bound input of every sampled stratum) -- and by stoch_recompute's R17-2 attribution (served token re-derived "
                     "from committed logits + seed + splits, every row) [stoch M-0804 wording]"),
    "sampler_splits": ("prescribed input: prescribed_input_linkage (committed S == SplitsFor_v1(live geometry), REQ-03 / ruling 36) + C1 / C3; "
                       "CONSUMED BY the same relation (the sampler VU GumbelTopPTokenSelect_v1's exact replay) as its 6th operand: the top-p mask "
                       "kernel's split count S = SplitsFor_v1(|live(s)|, num_SMs), "
                       "derived, never a static (F-stoch-01 = a wrong S; form (A): S is a bound input of every sampled stratum) and by "
                       "stoch_recompute's R17-2 attribution (served token re-derived from committed logits + seed + splits) [stoch M-0804 wording]"),
    "fa2_hidden_m1_stream": ("attention interior (the launch's M1 export stream): no VU row of its own; the attention VU it belongs to is state-linked "
                             "(NOT_YET) -- C1 / C3 + form (B) where present; form (A) over it is an engineering gap, not a domain (R17-3)"),
    "fa3_hidden_m1_stream": ("attention interior (the Hopper launch's M1 export stream): no VU row of its own; the attention VU it belongs to is "
                             "state-linked (NOT_YET) -- C1 / C3 + form (B) where present; form (A) over it is an engineering gap, not a domain (R17-3)"),
}
#: the query families whose identities ARE produced by Program rows (a VU or a named non-evaluable row must address each of them)
QUERY_FAMILIES_WITH_ROWS = ("instance_outputs", "sampled_token_ids", "moe_block_stream", "tp_rank_partials")

#: [COORD R17-27 / Daniel, verbatim] the scope of the class -- on the rule, on every record that carries it, in the C2 schema doc
NOT_EXECUTED_SCOPE = ("NOT-EXECUTED-BY-RULE is an active-reference implementation classification only. In the lifted fixed Program the "
                      "corresponding positions are REQUIRED logical values assigned the canonical inactive representation tau(e) = bot; "
                      "there is no not-required-because-not-executed category.")

#: [COORD R17-26 (1), R17-27 (a)(b)(c); rev M-0308 (1)-(6) / reqval] the R17-1 executed prefix of a request, as the C2 population
#: CLASSIFIES it (never as it shrinks by it).
EXECUTED_PREFIX_RULE = ("R17-1: a request that served a_r < cap tokens (its served token a_r-1 a declared EOS id of the Program's stop rule) "
                        "executed exactly n_exec = min(a_r + LAG, cap) request-relative steps; a request served to its cap executed cap "
                        "(a length stop: no post-EOS row even if the cap token is an EOS). RequiredValues(P, Q_module_v1) stays the "
                        "cap-length manifest of record (fixed population across stopping points, BRIEF 7B; digest unchanged): its identities "
                        "at request-relative steps >= n_exec are the named class not-executed(R17-1, a_r, LAG) -- counted in query_total, "
                        "absent from the sampling pool, never 'not evaluated' and never silently absent; reached + uncovered + not-executed "
                        "== |Q(P)|. CONDITIONS (R17-27): (a) a_r is READ from the Match record of record (G6 lengths_by_request / tokens.json) "
                        "and the Commit's own committed served length must EQUAL it by name -- the committed tokens count only because they "
                        "are linked to the served output and the Match account (C2 boundary linkage + tokens_equal + G6 agree); 'the committer "
                        "says EOS happened here' is never sufficient; (b) LAG is the Program's declared static (workload_lag_static), attested "
                        "declared == record_attests == tokens.json; (c) any committed row of the request at a step >= arrival + n_exec is "
                        "'execution past the stop' and FAILS. A request failing any condition is NOT classified: its prefix carries faults and "
                        "the record's query_checks fail by name. " + NOT_EXECUTED_SCOPE)


def _continuation_rule() -> Callable[..., dict[str, Any]] | None:
    """Continuation_v1's `executed_prefix_by_rule(served, cap=, A=, eos=)` (pad, `verity.ir/continuation.py`, R17-32) when the unit is on
    the tree -- the ONE mechanism revb S-r17b-20 (R3) asks for: the not-executed class = {t : r_t = 0} of the recurrence over the served
    list.  None when the unit is not on this tree (the caller then records the arithmetic as the mechanism BY NAME, never silently)."""
    try:
        from verity_vllm.program.lifting.continuation import executed_prefix_by_rule  # type: ignore[import-not-found]
    except Exception:
        return None
    return executed_prefix_by_rule


MECH_RECURRENCE = "recurrence(Continuation_v1): n_exec = |{t : r_t = 1}| over the served list (first EOS stops; A = LAG); min(a_r + LAG, cap) is the CHECK"
MECH_ARITH_NO_SERVED = ("arithmetic(a_r, LAG): n_exec = min(a_r + LAG, cap) -- the served list was not supplied, the class is not derived from r_t "
                        "(revb S-r17b-20 R3 open on this record: pass served= / eos_ids=)")
MECH_ARITH_NO_RULE = ("arithmetic(a_r, LAG): n_exec = min(a_r + LAG, cap) -- the served list was supplied but verity_vllm.program.lifting.continuation (Continuation_v1, "
                      "pad R17-32) is not on this tree: the class is not derived from r_t (revb S-r17b-20 R3 open on this tree)")


def executed_prefix_of_record(rid: str, *, a_r: int, lag: int, cap: int, arrive: int = 0, eos_declared: bool | None = None,
                              served_len_commit: int | None = None, linked: bool | None = None, lag_attested: bool | None = None,
                              rows_past_stop: int = 0, a_r_source: str, served: Sequence[int] | None = None, eos_ids: Iterable[int] | None = None,
                              rule: Callable[..., dict[str, Any]] | None = None, account_missing: Sequence[str] | None = None) -> dict[str, Any]:
    """Normalise one request's R17-1 executed prefix for the population.  The CALLER (the Commit) supplies the facts of record -- `a_r` READ
    from the Match record, `served_len_commit` its own committed served length, `linked` (C2 boundary linkage + tokens_equal + G6 agree),
    `lag_attested` (declared == record_attests == tokens.json), `eos_declared` (served token a_r-1 is a declared EOS id), `rows_past_stop`
    (committed rows of the request at steps >= arrival + n_exec, from the Commit's coverage) -- this function only fixes the arithmetic and
    NAMES the faults: `faults` non-empty means the class is NOT applicable to this request and the record fails by name (R17-27 (a)(b)(c),
    F-pm-24).  Every condition is fail-closed: an unasserted fact (None) is a fault, not a default.
    [revb S-r17b-09 n1] `a_r_source` is REQUIRED and names WHERE a_r was read (the Match record file + digest, as moe's executed_prefix_facts
    passes it); it has no default, so a caller passing the Commit's own count cannot put 'match record' on the record by omission -- an own
    count must be labelled as such by the caller, and (a) then faults on `linked`.
    [revb S-r17b-20 (R3) ONE mechanism; pad M-0568] with `served` (the request's served token ids of record) and `eos_ids` (the Program's
    declared EOS ids) the class is DERIVED FROM r_t: `rule` (default Continuation_v1's `executed_prefix_by_rule`) runs the recurrence over
    the served list, `n_exec = |{t : r_t = 1}|`, and the arithmetic `min(a_r + LAG, cap)` is demoted to a CHECK (a disagreement is a fault
    by name); the rule's own problems (served past the first EOS, F-pm-24, over the cap) are faults.  Without `served` -- or on a tree
    without the unit -- the arithmetic is the mechanism and is NAMED as such in `mechanism` (never silently the same thing)."""
    a_r, lag, cap, arrive = int(a_r), int(lag), int(cap), int(arrive)
    faults: list[str] = []
    if account_missing:
        # [r19-int-21, F-r19-int-19] FIRST, so the verdict names the cause: a Match account leg the link needs is not beside the Commit
        # (#39: match/capture/ moved to /vault under a queued Commit, F-r19-int-14).  Still fail-closed -- (a) and (b) below fault too --
        # but an operator reads 'input missing: restore tokens.json' and not 'the Program's executed prefix is wrong'.
        faults.append(f"Match account leg(s) missing beside the Commit: {', '.join(str(x) for x in account_missing)} -- the link cannot hold on a "
                      "missing input (restore the Match record's file(s) and re-run the Commit; not a verdict on the Program) (R17-27 a/b)")
    if not isinstance(a_r_source, str) or not a_r_source.strip():
        faults.append("a_r_source unstated: the record must name where a_r was read (R17-27 (a); S-r17b-09 n1)")
    if lag not in (0, 1):
        faults.append(f"LAG {lag} outside the supported profile domain {{0, 1}} (R17-1 N2)")
    if a_r < 0 or a_r > cap:
        faults.append(f"served length a_r {a_r} outside [0, cap {cap}]")
    n_exec = cap if a_r >= cap else min(a_r + lag, cap)                        # the arithmetic: the mechanism without a served list, else the CHECK
    mechanism, by_rule = MECH_ARITH_NO_SERVED, None
    if served is not None:
        rule = rule or _continuation_rule()
        if rule is None:
            mechanism = MECH_ARITH_NO_RULE
        else:
            served_l = [int(t) for t in served]
            if len(served_l) != a_r:
                faults.append(f"served list of {len(served_l)} token(s) != a_r {a_r} ({a_r_source}): the recurrence would run over another prefix than the record's")
            if lag in (0, 1):
                blk = rule(served_l, cap=cap, A=lag, eos=[int(e) for e in (eos_ids or ())])
                by_rule = {k: blk.get(k) for k in ("executed_prefix_len", "eos_step", "lag_steps", "first_eos_index", "served_past_first_eos", "cap_exhausted", "eos_stop", "problems")}
                for pr in blk.get("problems") or []:
                    faults.append(f"Continuation_v1 rule: {pr}")
                if not blk.get("problems") and len(served_l) == a_r:
                    n_rule = int(blk["executed_prefix_len"])
                    if n_rule != n_exec:
                        faults.append(f"the arithmetic min(a_r + LAG, cap) = {n_exec} disagrees with the recurrence |{{t : r_t = 1}}| = {n_rule} (R3: the rule is the mechanism, "
                                      "the arithmetic the check)")
                    n_exec = n_rule
                mechanism = MECH_RECURRENCE
    if a_r < cap and eos_declared is not True:
        faults.append(f"a_r {a_r} < cap {cap} but served token a_r-1 is not asserted a declared EOS id (eos_declared={eos_declared}): "
                      "an early stop the Program's stop rule does not explain -- not-executed(R17-1) is NOT applicable, FAIL by name")
    if linked is not True:                                                       # R17-27 (a)
        faults.append(f"a_r is not linked to the served output and the Match account (linked={linked}): C2 boundary linkage + tokens_equal + G6 "
                      "must agree before the committer's stop counts (R17-27 a)")
    if served_len_commit is None or int(served_len_commit) != a_r:               # rev M-0308 (2)
        faults.append(f"Commit served length {served_len_commit} != record a_r {a_r} (a Commit that stops earlier or later than the record fails by "
                      "name; it does not shrink or grow its required population)")
    if lag_attested is not True:                                                 # R17-27 (b)
        faults.append(f"LAG rule not attested (declared == record_attests == tokens.json required; lag_attested={lag_attested}) (R17-27 b)")
    if int(rows_past_stop) > 0:                                                  # R17-27 (c)
        faults.append(f"execution past the stop: {int(rows_past_stop)} committed row(s) of the request at steps >= arrival {arrive} + n_exec {n_exec} (R17-27 c)")
    return {"request_id": str(rid), "a_r": a_r, "a_r_source": a_r_source, "lag": lag, "cap": cap, "arrive": arrive, "n_exec": int(n_exec),
            "executed_prefix": int(n_exec), "steps_not_executed": [int(n_exec), cap] if n_exec < cap else None, "eos_declared": eos_declared,
            "served_len_commit": served_len_commit, "linked": linked, "lag_attested": lag_attested, "rows_past_stop": int(rows_past_stop),
            "account_missing": [str(x) for x in (account_missing or ())],
            # [revb M-0589 n-r3-1] a faulted request produced NO class: the mechanism is 'not classified', never 'recurrence' beside a fault
            "mechanism": (mechanism if not faults else f"not classified: {len(faults)} fault(s) -- the class is not applicable to this request"),
            "r3_one_mechanism": (by_rule is not None and not faults), "by_rule": by_rule, "faults": faults}


def split_not_executed(vus: list[dict], executed_prefix: dict[str, dict] | None) -> tuple[list[dict], list[dict]]:
    """(rows the sampler may draw, rows at request-relative step >= n_exec of their request).  No `executed_prefix` -> nothing is split."""
    if not executed_prefix:
        return list(vus), []
    keep, out = [], []
    for v in vus:
        ep = executed_prefix.get(str(v["request_id"]))
        if ep is not None and not ep.get("faults") and int(v["step"]) >= int(ep["n_exec"]):
            out.append(v)
        else:
            keep.append(v)
    return keep, out


def _address_family(op: str, member: str, a: dict[str, Any]) -> str:
    """The manifest family of a `population(..., addresses=)` record: recorded by the address rule that produced it (`manifest_family`,
    the same labels build_manifest assigns).  A record built without the label (a caller's own accumulator, e.g. the tests') is named by
    the address SCHEMA it was written in: the sampler op, a `moe/L<k>` plane address (`_MOE_LAYER_RE`, the moe_block_stream op_path
    vocabulary), else the member's own family (`manifest_format.alias_family`)."""
    fam = a.get("manifest_family")
    if fam:
        return str(fam)
    if op == SAMPLER_OP:
        return "sampled_token_ids"
    return "moe_block_stream" if _MOE_LAYER_RE.match(op) else RM.alias_family(member)


def _manifest_engine_name(manifest: dict[str, Any]) -> str:
    """Which engine built a manifest: the traversal's header carries `engine` (schema + `traversal` = the class path; R16 reqval), the
    inventory's carries none (byte-for-byte the R15 header) -- the guard header names it too where present (F-r17b-09)."""
    eng = manifest.get("engine")
    if isinstance(eng, dict):
        return "traversal" if "traversal" in eng else str(eng.get("name") or "traversal")
    if isinstance(eng, str):
        return eng
    g = manifest.get("guard")
    return str(g.get("engine")) if isinstance(g, dict) and g.get("engine") else "inventory"


def controller_rows_of(programs: dict[str, Any], indices: dict[int, "ProgramIndex"] | None = None) -> dict[str, int]:
    """[revb S-r17b-20 R4 c2] the replay's OWN count of guard / controller rows per request Program (spec family under GUARD_PREFIXES) -- the
    cross-check of the manifest header's `guard_rows`, so a stale or foreign header cannot drop or shrink the controller statement ('the
    collector cannot define its own population', applied to the controller)."""
    out: dict[str, int] = {}
    for rid, inst in programs.items():
        rows = inst.get("rows") if isinstance(inst, dict) else None
        if rows is None and indices is not None and id(inst) in indices:
            rows = indices[id(inst)].rows
        out[str(rid)] = sum(1 for r in (rows or []) if RM.spec_family(str(r.get("spec"))).startswith(GUARD_PREFIXES))
    return out


def query_population(manifest: dict[str, Any] | None, addresses: dict[tuple, dict], evaluated: set[tuple], *, ranks: list[int],
                     strata: dict[str, dict] | None = None, first_n: int = 6, executed_prefix: dict[str, dict] | None = None, controller_rows: dict[str, int] | None = None) -> dict[str, Any]:
    """[R17-3 / reqval] Reconcile the replay's reach against the population of record, Q(P), IDENTITY BY IDENTITY.
    `manifest`: the required-value manifest of the row (Q_module(P): the module-boundary population over the Program, families selected by
    the canonical VU query; the traversal adapter's or the inventory's -- the same identities either way); None = no manifest supplied,
    the reconciliation is over the Program rows' own addresses and says so.
    `addresses`: `population(..., addresses=)` accumulator (manifest address -> class of the rows producing it); `evaluated`: the
    (request, step, op_path) triples with an evaluated pick.
    Returns the three record fields the row must carry (`query_total`, `sampled_strata`, `uncovered_by_name`) + `query_checks`:
      * query_total     -- |Q(P)| and its families (the authority: never the collector's, never the replay's own enumeration)
      * sampled_strata  -- what the exact replay REACHES (identities with an evaluable VU) and what it EVALUATED, as a share of Q(P)
      * uncovered_by_name -- every required identity the exact replay has no evaluable VU for, grouped by (family, spec, class) with the
                             mechanism of record that covers it instead; plus the query families with no row (weights, ids, streams)
      * query_checks    -- the fail-closed reconciliation: a VU whose address is NOT a required identity (the replay enumerating
                           something Q(P) does not require), a required identity NO row produces, a row count that disagrees with the
                           manifest's `rows`, a rank set that disagrees.  Any non-zero count is a finding on the record (`ok` False).
      * not_executed_by_rule [COORD R17-26] -- `executed_prefix` (request -> `executed_prefix_of_record`): the identities of Q(P) at
                           request-relative steps >= n_exec are the third bucket beside reached / uncovered (query_total unchanged;
                           `accounted` = reached + uncovered + not-executed == |Q(P)|); a request whose prefix carries faults is NOT
                           classified and fails the check by name."""
    by_key: dict[tuple, dict] = {}
    fam_total: Counter = Counter()
    ranks_seen: set[int] = set()
    # shared modules (vLLM's one RotaryEmbedding instance called by every layer): the manifest addresses them at the FIRST layer's path
    # with invocation = layer order and records the per-layer paths in `op_path_aliases` (alias -> [canonical, invocation]) -- resolved
    # through the ONE table `CommittedStore.words` reads through as well (rel M-0367)
    aliases = alias_table(manifest)

    def _key(rid: str, st: int, op: str, mem: str) -> tuple:
        canon, inv = resolve_op_path(aliases, op)
        return (rid, st, canon, mem, inv)

    non_row_idents: list[dict] = []                                                # [F-moe-r17-05] the no-row families, classified below
    if manifest is not None:
        for ident in manifest.get("identities") or []:
            fam = str(ident.get("family"))
            fam_total[fam] += 1
            ranks_seen.add(int(ident.get("rank", 0) or 0))
            if fam in QUERY_FAMILIES_WITH_ROWS and ident.get("step") != "*" and ident.get("request_id") != "*":
                by_key[(str(ident.get("request_id")), int(ident["step"]), str(ident.get("op_path")), str(ident.get("output_member")),
                        int(ident.get("invocation", 0) or 0))] = ident
            elif executed_prefix:
                non_row_idents.append(ident)
    else:
        for (rid, st, op, mem), a in addresses.items():
            fam_total[_address_family(op, mem, a)] += 1
    total = sum(fam_total.values())
    # --- reach: identities with >= 1 evaluable VU; evaluated: identities under an evaluated (request, step, module) -------------------
    reached: Counter = Counter()
    reached_rows: Counter = Counter()                                              # Program-row addresses (a 2-member row addresses 2 identities)
    evald: Counter = Counter()
    not_exec: Counter = Counter()                                                  # [R17-26] identities at steps >= n_exec, by family
    not_exec_req: Counter = Counter()                                              # per request: every classified identity (rows + named inputs)
    not_exec_req_rows: Counter = Counter()                                         # per request: the classified ROW identities only
    req_max_step: dict[str, int] = {}                                              # per request: the last engine step Q(P) carries an identity at
    req_ids: Counter = Counter()                                                   # required row identities per request (manifest rows families)
    past_stop: list[dict] = []                                                     # an EVALUATED VU past a request's executed prefix: FAIL by name
    outside: list[dict] = []
    n_outside = 0
    rows_disagree: list[dict] = []
    n_rows_disagree = 0
    rows_annot: list[dict] = []
    n_rows_annot = 0
    groups: dict[tuple, dict] = {}
    seen_keys: set[tuple] = set()
    for (rid, st, op, mem), a in sorted(addresses.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        k5 = _key(rid, st, op, mem)
        seen_keys.add(k5)
        ident = by_key.get(k5) if manifest is not None else None
        fam = str(ident["family"]) if ident is not None else _address_family(op, mem, a)
        if manifest is not None and ident is None:
            n_outside += 1
            if len(outside) < first_n:
                outside.append({"request_id": rid, "step": st, "op_path": op, "member": mem, "spec": a["spec"], "class": a["class"]})
            # not a required identity: it is NOT counted toward the reach of Q(P) (the replay does not enlarge the population)
        elif ident is not None:
            # the identity's block height = element_range / width (digest-bearing) must be the number of Program rows producing it; the
            # `rows` annotation is compared beside it (a global manifest annotates every identity of the engine step with the request's
            # token count -- false on per-sequence members such as logits_processor/out: reported as an annotation finding, not a reach gap)
            lo, hi = (ident.get("element_range") or [0, None])[:2]
            width = ident.get("width")
            height = ((int(hi) - int(lo)) // int(width)) if (hi is not None and isinstance(width, int) and width > 0) else None
            if height is not None and height != int(a["rows"]):
                n_rows_disagree += 1
                if len(rows_disagree) < first_n:
                    rows_disagree.append({"request_id": rid, "step": st, "op_path": op, "member": mem, "manifest_height": height, "program_rows": a["rows"]})
            if isinstance(ident.get("rows"), int) and height is not None and int(ident["rows"]) != height:
                n_rows_annot += 1
                if len(rows_annot) < first_n:
                    rows_annot.append({"request_id": rid, "step": st, "op_path": op, "member": mem, "rows_annotation": ident["rows"], "height": height})
        if ident is not None:
            req_ids[str(rid)] += 1
            if int(st) > req_max_step.get(str(rid), -1):
                req_max_step[str(rid)] = int(st)
        ep = (executed_prefix or {}).get(str(rid))
        # `st` is the identity's / Program row's OWN step = request-relative already (the traversal annotates `engine_step` = arrive + step
        # beside it); R17-1's n_exec counts the request's own steps, so the comparison is direct -- the arrival enters only for the
        # batch-level planes below, whose one step is the ENGINE's
        if ep is not None and not ep.get("faults") and int(st) >= int(ep["n_exec"]) and (manifest is None or ident is not None):
            not_exec[fam] += 1                                                    # [R17-26] classified, not reached, not uncovered
            not_exec_req[str(rid)] += 1
            not_exec_req_rows[str(rid)] += 1
            if (rid, st, op) in evaluated:
                past_stop.append({"request_id": rid, "step": st, "op_path": op, "member": mem, "n_exec": ep["n_exec"]})
            continue
        if a["class"] == "vu":
            if manifest is None or ident is not None:
                reached[fam] += 1
                reached_rows[fam] += int(a["rows"])
                if (rid, st, op) in evaluated:
                    evald[fam] += 1
            continue
        sf = RM.spec_family(a["spec"])                                           # grouped by the spec FAMILY (Attention_v3), the shapes counted
        g = groups.setdefault((fam, sf, a["class"]), {"family": fam, "spec": sf, "class": a["class"], "why": a["why"], "identities": 0, "row_addresses": 0,
                                                       "shapes": set(), "first": []})
        g["identities"] += 1
        g["row_addresses"] += int(a["rows"])
        g["shapes"].add(a["spec"])
        if len(g["first"]) < 3:
            g["first"].append([rid, st, op, mem])
    # --- required identities no row produces (manifest / replay address disagreement) ----------------------------------------------------
    missing: list[dict] = []
    n_missing = 0
    if manifest is not None:
        for k, ident in by_key.items():
            if k not in seen_keys:
                n_missing += 1
                if len(missing) < first_n:
                    missing.append({"request_id": k[0], "step": k[1], "op_path": k[2], "member": k[3], "invocation": k[4], "family": ident.get("family"),
                                    "spec": ident.get("spec")})
    # --- [R17-26 / moe F-moe-r17-05] the class over the NO-ROW families: the rule names identities of a request at request-relative steps
    # >= n_exec, whatever their family -- a named input (`input_token_ids` at r0's steps past its stop) or any other per-request identity is
    # not executed by the same rule as the rows; a BATCH-LEVEL plane (request_id '*', `requests` list, ONE engine step) is not executed
    # only when EVERY listed request is past its own stop at that engine step (then nothing was committed there); otherwise the identity
    # exists with a smaller composition and is REQUIRED -- the per-request slots past the stop are counted beside it (the Commit's
    # binding coverage expands batch-level rows per request: its count reconciles against `request_slots_not_executed`, never N_excluded).
    kind_by_family: dict[str, str] = {}
    batch_level: dict[str, dict] = {}
    batch_slots_req: Counter = Counter()
    for ident in non_row_idents:
        fam = str(ident.get("family"))
        rid, st = ident.get("request_id"), ident.get("step")
        if rid != "*" and st != "*":
            kind_by_family.setdefault(fam, "named-input" if fam == "input_token_ids" else "per-request")
            ep = executed_prefix.get(str(rid))
            if ep is not None and not ep.get("faults") and int(st) >= int(ep["n_exec"]):
                not_exec[fam] += 1
                not_exec_req[str(rid)] += 1
            elif int(st) > req_max_step.get(str(rid), -1):
                req_max_step[str(rid)] = int(st)
        elif rid == "*" and isinstance(ident.get("requests"), list) and ident.get("engine_step") is not None:
            kind_by_family.setdefault(fam, "batch-level")
            e, listed = int(ident["engine_step"]), [str(r) for r in ident["requests"]]
            past = [r for r in listed if (ep := executed_prefix.get(r)) is not None and not ep.get("faults")
                    and e - int(ep.get("arrive", 0) or 0) >= int(ep["n_exec"])]
            # a (plane, listed request) pair is one CELL (revb M-0452 (ii), SPEC A3.6: the request's slices of the plane); the Commit's
            # per-request expansion counts cells -- `cells_not_executed` is the number it reconciles against
            b = batch_level.setdefault(fam, {"identities": 0, "identities_not_executed": 0, "identities_partially_executed": 0,
                                             "request_slots": 0, "request_slots_not_executed": 0, "cells_not_executed": 0})
            b["identities"] += 1
            b["request_slots"] += len(listed)
            b["request_slots_not_executed"] += len(past)
            b["cells_not_executed"] += len(past)
            for r in past:
                batch_slots_req[r] += 1
            if listed and len(past) == len(listed):
                b["identities_not_executed"] += 1
                not_exec[fam] += 1
            elif past:
                b["identities_partially_executed"] += 1
        else:
            kind_by_family.setdefault(fam, "static" if st == "*" else "unclassified")
    # --- the uncovered list: row groups by class, then the families with no rows at all -------------------------------------------------
    mech_of_class = {
        "state-linked": "C1 coverage + C3 openings + form (B) where present; the exact VU (K/V history from committed qkv outputs) is an engineering gap, not a domain (R17-3)",
        "no-evaluator": "C1 coverage + C3 openings + form (B) where present; a registered evaluator is the engineering gap (form (A) required, R17-3)",
        "plane-fed": "the plane is read as its consumers' committed input; the producer's own relation awaits MAN-06f v3 unit 2 (form (B) required meanwhile)",
        "interior-composed-v2": "C1 / C3 + form (B) required (MAN-06g composed evaluator is the v2 follow-up)",
        "mixed": "rows of two classes under one identity: named, resolve before any claim",
    }
    uncovered = []
    for g in sorted(groups.values(), key=lambda g: (-g["identities"], g["family"], g["spec"])):
        shapes = sorted(g.pop("shapes"))
        uncovered.append(dict(g, shapes=len(shapes), shape_example=shapes[0], mechanism=mech_of_class.get(g["class"], "unstated (fail closed)")))
    for fam, n in sorted(fam_total.items()):
        if fam in QUERY_FAMILIES_WITH_ROWS:
            continue
        cls = "mechanism" if fam in QUERY_FAMILY_MECHANISM else "unstated"
        # the family's not-executed identities left the uncovered count for the class (reached + uncovered + not-executed == |Q(P)|)
        uncovered.append({"family": fam, "spec": None, "class": cls, "identities": int(n) - int(not_exec.get(fam, 0)), "row_addresses": 0,
                          "why": "no VU row of its own (the query requires it; the exact replay has no relation to evaluate for it)",
                          "mechanism": QUERY_FAMILY_MECHANISM.get(fam, "UNSTATED: a required family with no named mechanism -- fail closed on the record")})
    # [revb S-r17b-16 (R4) / pad R17-32] a LIFTED Program's controller rows (LStepLive / LSelect / LBot; Continuation_v1's LIsEOS<E> `is_eos`,
    # LEmitNext[H] `emit`, LSelect<w> `out` = the bits b_t / e_t / r_t / o_t / u_t) carry NO identity under Q_module_v1's rule (so
    # RequiredValues(Lift(P), Q) == RequiredValues(P, Q) row for row) -- the population statement NAMES them as uncovered with their
    # mechanism of record (the lifted record's R8 controller replay through the registered evaluators; the executed prefix from r_t, R3)
    # instead of letting them be silently absent.  identities 0 by rule, rows counted; a decision to give the bits identities is a Q change
    # (Q_module_v2: digest moves), recorded in Q-EXPANSION -- not made here.
    mq = (manifest or {}).get("query") if isinstance(manifest, dict) else None
    controller = None
    hdr_rows = int(mq.get("guard_rows") or 0) if isinstance(mq, dict) else 0
    own_rows = {str(k): int(v) for k, v in (controller_rows or {}).items()}          # the replay's own count per request Program (c2)
    # [c2; revb F-r17b-36] the header vs the replay's own count of guard-prefixed rows: the required manifest of record is the traversal of
    # the WORKLOAD Program (manifest_verify.used_program_digest == the workload digest; one manifest, query_total.requests = all of them), so
    # its `guard_rows` is the SUM over the request Programs -- the compare is made AT THE HEADER'S SCOPE (sum(own) vs header), never per
    # request (per-request equality is a semantics the header does not have: B2 {r0: 7, r1: 7} under 14 AGREES; {10, 4} under 14 agrees;
    # under 7 both disagree).  A disagreement is a fault by name (a stale / foreign header cannot drop or shrink the statement); the
    # statement is emitted when EITHER side names controller rows, and it says which scope the compare used.
    own_sum = sum(own_rows.values())
    ctrl_scope = "header = the workload Program's traversal: guard_rows is the SUM over the request Programs (compared as sum(program_rows_by_request) vs header_rows)"
    ctrl_disagree = ([{"program_rows_sum": own_sum, "header_rows": hdr_rows, "program_rows_by_request": own_rows, "scope": ctrl_scope}]
                     if (mq is not None and own_rows and own_sum != hdr_rows) else [])
    if (isinstance(mq, dict) and (mq.get("lifted") or hdr_rows > 0)) or own_sum > 0:
        gf = mq.get("guard_families") if isinstance(mq, dict) and isinstance(mq.get("guard_families"), dict) else None
        controller = {"family": "controller", "spec": None, "class": "controller", "identities": 0, "row_addresses": 0,
                      "program_rows": (own_sum if own_rows else hdr_rows), "header_rows": hdr_rows, "program_rows_by_request": own_rows,
                      "compare_scope": ctrl_scope,
                      "rows_agree": ((not ctrl_disagree) if (own_rows and mq is not None) else None),   # None: no header to agree with
                      "families": (gf if gf is not None else "unstated (older manifest header: guard_rows only)"),
                      "why": ("the lifted Program's guard / controller rows carry no identity under Q_module_v1's rule: RequiredValues(Lift(P), Q) == "
                              "RequiredValues(P, Q) row for row; the bits are not sampled by C2 -- said here BY NAME (revb S-r17b-16 R4)"),
                      "mechanism": ("the lifted record's R8 controller replay through the registered evaluators (LIsEOS / LEmitNext / LSelect; pad "
                                    "Continuation_v1) + the executed prefix derived from r_t (R3, executed_prefix_of_record served=)")}
        uncovered.append(controller)
    n_reached, n_evald = sum(reached.values()), sum(evald.values())
    n_uncovered = sum(int(u["identities"]) for u in uncovered)
    n_not_exec = sum(not_exec.values())
    ep_faults = [{"request_id": r, "faults": e.get("faults")} for r, e in sorted((executed_prefix or {}).items()) if e.get("faults")]
    # [R17-26 / rel M-0410 (a)] the class is an EXCLUSION over the cap-length manifest of record: a classified request whose Q(P) carries NO
    # identity at its not-executed steps was CUT before it reached the population (a restricted manifest handed in as Q(P)) -- N_manifest
    # would shrink silently and the class count 0.  Named here, fails the reconciliation: the manifest of record is never restricted.
    manifest_cut: list[dict] = []
    if manifest is not None:
        for r, e in sorted((executed_prefix or {}).items()):
            if e.get("faults") or int(e["n_exec"]) >= int(e["cap"]):
                continue
            last = int(e["cap"]) - 1                                              # request-relative, like the identities' own `step`
            if req_max_step.get(str(r), -1) < last or int(not_exec_req.get(str(r), 0)) == 0:
                manifest_cut.append({"request_id": str(r), "manifest_last_step": req_max_step.get(str(r)), "cap_last_step": last,
                                     "n_exec": int(e["n_exec"]), "identities_at_not_executed_steps": int(not_exec_req.get(str(r), 0)),
                                     "why": "Q(P) carries no identity at the request's not-executed steps: the manifest was restricted before the "
                                            "population (R17-26: the manifest of record stays cap-length; the class is the exclusion)"})
    # [R17 revb F-r17b-38] an identity at invocation >= 1 is addressed ONLY through `op_path_aliases` (a shared module's k-th call, at the
    # first layer's path); one that is the image of no alias entry is unreachable from every Program row -- the 960 + 960 on the TP2 B1 row
    # of record were this (the rank merge had dropped the table).  The CAUSE named beside the two symptom counts, fails the reconciliation.
    alias_images = {(str(c), int(k)) for c, k in aliases.values()}
    unreachable: list[dict] = []
    n_unreachable = 0
    if manifest is not None:
        for ident in manifest.get("identities") or []:
            inv = int(ident.get("invocation", 0) or 0)
            if inv >= 1 and (str(ident.get("op_path")), inv) not in alias_images:
                n_unreachable += 1
                if len(unreachable) < first_n:
                    unreachable.append({"request_id": ident.get("request_id"), "step": ident.get("step"), "op_path": ident.get("op_path"),
                                        "invocation": inv, "family": ident.get("family"), "rank": ident.get("rank", 0)})
    checks = {"vus_outside_query": {"n": n_outside, "first": outside}, "identities_without_rows": {"n": n_missing, "first": missing},
              "identities_unreachable_by_alias": {"n": n_unreachable, "first": unreachable, "alias_entries": len(aliases),
                                                  "why": ("invocation >= 1 identities that no `op_path_aliases` entry maps a Program path to: the manifest "
                                                          "carries no (or a partial) alias table; the Program's per-layer rows of the shared module then "
                                                          "read as vus_outside_query and these identities as identities_without_rows (F-r17b-38)"
                                                          if n_unreachable else None)},
              "rows_disagree": {"n": n_rows_disagree, "first": rows_disagree},
              "rows_annotation_inconsistent": {"n": n_rows_annot, "first": rows_annot,
                                               "note": "manifest annotation only (`rows` != element_range / width); digest-neutral, not a reach gap"},
              "ranks": {"replay": [int(r) for r in ranks], "manifest": sorted(ranks_seen) if manifest is not None else None,
                        "equal": (sorted(ranks_seen) == sorted(int(r) for r in ranks)) if manifest is not None else None},
              "accounted": (n_reached + n_uncovered + n_not_exec == total) if manifest is not None else None,
              "unstated_families": [u["family"] for u in uncovered if u["class"] == "unstated"],
              "controller_rows_disagree": {"n": len(ctrl_disagree), "first": ctrl_disagree[:first_n]},   # [R4 c2] header vs the replay's own count
              "executed_prefix_rule": {"applied": bool(executed_prefix), "faults": ep_faults, "evaluated_past_stop": {"n": len(past_stop), "first": past_stop[:first_n]},
                                       "manifest_cut_before_classification": {"n": len(manifest_cut), "first": manifest_cut[:first_n]}}}
    checks["ok"] = (None if manifest is None else                                  # None = not reconciled (no Q(P) supplied), stated
                    bool(n_outside == 0 and n_missing == 0 and n_unreachable == 0 and n_rows_disagree == 0 and checks["ranks"]["equal"] is not False
                         and checks["accounted"] is not False and not checks["unstated_families"]
                         and not ep_faults and not past_stop and not manifest_cut and not ctrl_disagree))
    _qid0 = ((manifest or {}).get("query") or {}).get("query_id") or "Q_module_v1"
    _noun = {"Q_module_v1": ("the module-boundary query of record", "Program row = module-boundary VU"),
             "Q_module_pad_v1": ("the declared padded-boundary query (revb S-r17b-51 (b))", "Program row = module-boundary VU + every slot unit of a padded MoE row"),
             "Q_fine_v1": ("the fine query (vu_canonical) through the RequiredValues path", "fine VU = relation-level partition unit"),
             "Q_fine_direct_v1": ("the directly-committed fine query (R17-43 s8)", "fine VU ending at a materialised word + the module rows"),
             }.get(_qid0, ("the query of record", (((manifest or {}).get("query") or {}).get("unit") or "per its header")))
    src = ((f"required manifest: {_qid0}(P) -- {_noun[0]} over the Program (unit: {_noun[1]}; "
            "Q_fine_v1 selects the families, not the unit; R17-3 authority, R17-9 (ii), R17-10)" if manifest is not None
            else "NO required manifest supplied: the Program rows' own manifest addresses (the v2 query's population; NOT a Q(P) statement -- "
                 "weights / ids / hidden streams absent)"))
    # [R17 revb F-r17b-09] the guard's state travels with Q(P) onto the record: a manifest whose dataflow guard was not armed says so here
    guard = (manifest or {}).get("guard") if manifest is not None else None
    guard_line = (("guard armed" if guard.get("armed") else "guard NOT armed") + f" ({guard.get('engine')})" if isinstance(guard, dict)
                  else ("guard state not on the manifest (pre-R17 header)" if manifest is not None else "no manifest"))
    # [R17 reqval task 3] the one-release cross-check (traversal of record, inventory beside it) travels onto the record and the scope line
    cc = (manifest or {}).get("cross_check") if manifest is not None else None
    if isinstance(cc, dict):
        guard_line += ("; cross-check " + ("AGREED" if cc.get("agree") else "DISAGREED") +
                       f" ({cc.get('primary_engine')} vs {cc.get('other_engine')}: {(cc.get('identities') or {}).get(cc.get('primary_engine'))} / "
                       f"{(cc.get('identities') or {}).get(cc.get('other_engine'))} identities)")
    elif manifest is not None:
        guard_line += f"; no cross-check on the manifest (one engine asked for by name: {_manifest_engine_name(manifest)})"
    # [COORD R17-10] the query of record BY ID on the scope line (a manifest predating the id: named as such, never assumed)
    qid = (((manifest or {}).get("query") or {}).get("query_id") or "Q_module (pre-R17-10 header, no query_id)") if manifest is not None else "Q"
    if total:
        scope = (f"{qid}(P) = {total} identities ({', '.join(f'{k} {v}' for k, v in sorted(fam_total.items()))}; {guard_line}); exact replay reaches {n_reached} "
                 f"({100.0 * n_reached / total:.1f}%), under an evaluated module {n_evald}")
    else:
        scope = f"{qid}(P) = 0 identities ({guard_line})"
    if uncovered:
        scope += "; uncovered by name: " + ", ".join(f"{u['spec'] or u['family']} x{u['identities']} ({u['class']})" for u in uncovered)
    if n_not_exec or ep_faults:
        scope += (f"; not-executed(R17-1) {n_not_exec} identities (" + ", ".join(f"{r}: a_r {e['a_r']} LAG {e['lag']} -> steps [{e['n_exec']}, {e['cap']})"
                                                                             for r, e in sorted((executed_prefix or {}).items()) if e.get('steps_not_executed')) + ")"
                  + ("; executed-prefix FAULTS: " + "; ".join(f"{f['request_id']}: {f['faults'][0]}" for f in ep_faults) if ep_faults else ""))
    return {"query_total": {"source": src, "identities": int(total), "by_family": {k: int(v) for k, v in sorted(fam_total.items())},
                            "manifest_digest": (manifest or {}).get("manifest_digest"), "program_digest": (manifest or {}).get("program_digest"),
                            "query_digest": ((manifest or {}).get("query") or {}).get("query_digest"),
                            "query_id": ((manifest or {}).get("query") or {}).get("query_id"),                     # [R17-10] Q_module_v1
                            "query_id_digest": ((manifest or {}).get("query") or {}).get("query_id_digest"),
                            "engine": (_manifest_engine_name(manifest) if manifest is not None else None),
                            "implemented_query": ((manifest or {}).get("query") or {}).get("implemented_query"),
                            "unit": "Program row = module-boundary VU under Q_module_v1 (a `vus` count above is a count of those rows)",
                            "requests": (manifest or {}).get("requests"),
                            "guard": ({k: guard.get(k) for k in ("engine", "armed", "available", "fully_wired", "flags", "named_input") if k in guard}
                                      if isinstance(guard, dict) else None),
                            "cross_check": ({k: cc.get(k) for k in ("agree", "primary_engine", "other_engine", "digests", "identities", "complete", "why") if k in cc}
                                            if isinstance(cc, dict) else None),
                            "guard_rule": ("the 'missing family cannot vanish from both sides' guard (unnamed crossing / unnamed input / hidden-BN "
                                           "disagreement) is ARMED only when the traversal engine audited every request Program's operand runs; "
                                           "not armed = the population is the vocabulary's word alone, said here rather than read as clean")},
            "sampled_strata": {"identities_reached": int(n_reached), "identities_under_evaluated_module": int(n_evald), "row_addresses_reached": int(sum(reached_rows.values())),
                               "share_of_query": (round(n_reached / total, 4) if total else None),
                               "by_family_reached": {k: int(v) for k, v in sorted(reached.items())}, "by_family_under_evaluated_module": {k: int(v) for k, v in sorted(evald.items())},
                               "strata": (len(strata) if strata is not None else None),
                               "strata_evaluated": (sum(1 for v in strata.values() if int(v.get("evaluated") or 0) > 0) if strata is not None else None),
                               "rule": ("'complete' grades the sampled strata over the REACHED identities only; every required identity outside the "
                                        "reach is in uncovered_by_name (with its mechanism of record), never inside the grade (R17-3). "
                                        "`identities_under_evaluated_module` counts every identity of a (request, step, module) with >= 1 evaluated pick "
                                        "-- one pick under a 2-member module counts 2; it is NOT a count of identities whose own elements were evaluated "
                                        "(revb S-r17b-04 label)")},
            "uncovered_by_name": uncovered, "uncovered_identities": int(n_uncovered),
            "controller": controller,   # [R4] None on an un-lifted Program; on a lifted one the named statement above (also in uncovered_by_name)
            # [revb F-r17b-22 / report M-0303] the two flat keys the readers wait for, emitted from the SAME computation as the block below
            "not_executed_by_family": {k: int(v) for k, v in sorted(not_exec.items())},
            "not_executed_rule": {"rule": "not-executed(R17-1, a_r, LAG)", "applied": bool(executed_prefix), "n": int(n_not_exec),
                                  "requests": {r: {"a_r": e["a_r"], "lag": e["lag"], "n_exec": e["n_exec"], "cap": e["cap"], "step_range": e.get("steps_not_executed"),
                                                   "n": int(not_exec_req.get(r, 0)), "faults": e.get("faults"), "mechanism": e.get("mechanism")}
                                               for r, e in sorted((executed_prefix or {}).items())},
                                  # [revb S-r17b-20 (R3)] WHICH mechanism produced the class, per request, BY NAME: the recurrence (one code path with
                                  # Continuation_v1) or the arithmetic (the served list / the unit absent) -- a reader sees the tie's status on the record
                                  "mechanism": {m: int(n) for m, n in sorted(Counter(str(e.get("mechanism") or "unstated").split(":")[0]
                                                                                                 for e in (executed_prefix or {}).values()).items())},
                                  "r3_one_mechanism": (bool(executed_prefix) and all(e.get("r3_one_mechanism") is True for e in executed_prefix.values())),
                                  "scope": NOT_EXECUTED_SCOPE},
            "not_executed_by_rule": {"rule": EXECUTED_PREFIX_RULE, "scope": NOT_EXECUTED_SCOPE, "applied": bool(executed_prefix), "identities": int(n_not_exec),
                                     "by_family": {k: int(v) for k, v in sorted(not_exec.items())},
                                     "totals": {"N_manifest": int(total) if manifest is not None else None,
                                                "N_required": (int(total) - int(n_not_exec)) if manifest is not None else None, "N_excluded": int(n_not_exec),
                                                # a batch-level family carries its plane-cell statement beside the identity-level counts (the key moe's
                                                # reconciler reads, M-0468): {identities: planes fully not executed, cells: (plane, request) cells past the stop}
                                                "by_family": {k: {"N_manifest": int(v), "N_required": int(v) - int(not_exec.get(k, 0)), "N_excluded": int(not_exec.get(k, 0)),
                                                                  **({"plane_cells_not_executed": {"identities": int(batch_level[k]["identities_not_executed"]),
                                                                                                   "cells": int(batch_level[k]["cells_not_executed"])}} if k in batch_level else {})}
                                                              for k, v in sorted(fam_total.items())} if manifest is not None else None},
                                     "by_request": {r: dict(e, identities=int(not_exec_req.get(r, 0)), excluded_unexecuted_identities=int(not_exec_req.get(r, 0)),
                                                        excluded_row_identities=int(not_exec_req_rows.get(r, 0)),
                                                        required_row_identities=int(req_ids.get(r, 0)) - int(not_exec_req_rows.get(r, 0)),
                                                        batch_level_slots_not_executed=int(batch_slots_req.get(r, 0)))
                                                    for r, e in sorted((executed_prefix or {}).items())},
                                     # [moe F-moe-r17-05] the REACH of the class over every family of Q(P): kind per family (row | named-input |
                                     # per-request | batch-level | static) and the batch-level planes' per-request slots (identity-level
                                     # N_excluded counts a plane only when every listed request is past its stop; the Commit's per-request
                                     # expansion reconciles against request_slots_not_executed)
                                     "kind_by_family": {**{k: "row" for k in sorted(fam_total) if k in QUERY_FAMILIES_WITH_ROWS}, **dict(sorted(kind_by_family.items()))},
                                     "batch_level": {k: dict(v) for k, v in sorted(batch_level.items())}},
            "query_checks": checks, "scope": scope}


_T_STATIC = re.compile(r"\bT=\d+,")


def _stratum_spec(fam: str, spec: str, statics: dict[str, Any]) -> str:
    """The spec as the STRATUM key.  [R17 dense, ruling R17-11] an attention row's spec carries its history length T (one value per token of
    the request), so 'spec incl. shape' would make every attention row of the run its own stratum -- an exhaustive replay of the family
    (~0.1 ms x T per row; B8 1024/128: 8 x 32k rows x up to 1152 committed reads each), not a sample.  The stratum keeps the numerical
    MECHANISM the length selects -- NB = ceil(T / BN), the number of key blocks visited, and [revb F-r17b-16] `last=full|partial`: whether
    the last block is FULL (T mod BN == 0: no masked lane) or PARTIAL (masked lanes -> exact zeros through exp2 and the lane sums) -- three
    chains (single block / multi-block / partial last block) that NB alone does not keep apart: measured on llama L40S B8 256/32, 32 of 80
    NB-only strata mixed the one full-block row (T = 128) with 127 partial ones, so the full-block chain was picked with probability
    per_stratum / 128 where the spec-as-stratum key had made it certain.  The phase axis keeps prefill / first-decode / mid / last:
    `Attention_v2{NB=9,last=partial,NH=..,...}` x request x phase x module x lifetime; <= +1 stratum per formerly mixed stratum.  Every
    other family's spec is its own key (Gemm{K,N}, RMSNorm{N}, ... carry no per-row length)."""
    if fam not in ATTENTION_ROW_FAMILIES or "T" not in statics or "BN" not in statics:
        return spec
    try:
        t, bn = int(statics["T"]), max(1, int(statics["BN"]))
    except (TypeError, ValueError):
        return spec
    nb = -(-t // bn)
    return _T_STATIC.sub(f"NB={nb},last={'full' if t % bn == 0 else 'partial'},", spec, count=1)


def _uncovered_mechanisms(all_vus: list[dict], not_ev_fam: Counter) -> list[dict]:
    """[R17 dense, ruling R17-11] The Program's attention rows are evaluated on their committed boundary (the Q_module_v1 unit); the
    AttnBlock interiors inside them (Q_fine_v1 units) are not read -- the record names that with the row count, so a PASS bounds its claim."""
    n = sum(1 for v in all_vus if v["family"] in ATTENTION_ROW_FAMILIES) + sum(int(not_ev_fam.get(f, 0)) for f in ATTENTION_ROW_FAMILIES)
    if n == 0:
        return []
    return [{"mechanism": "attention block interiors (Q_fine_v1 units inside the evaluated Attention rows)", "rows": int(n),
             "query_of_record": "Q_module_v1", "unit": "the Attention row (module boundary): committed q + committed K/V history -> committed out",
             "why": ATTENTION_INTERIOR_MECHANISM,
             # [revb M-0148 (3)] what happens to the committed interior words meanwhile -- stated, not implied
             "committed_words_status": "fa2_hidden_m1_stream / fa3_hidden_m1_stream rows are COMMITTED (coverage + openings) and value-UNCHECKED by "
                                       "this replay; form (B) does not compare them either unless oracle_compare.compared_by_family names "
                                       "hidden_stream (the record's not_compared_by_family.hidden_stream is the count) -- Stage B (Q_fine_v1)"}]


def sample(vus: list[dict], per_stratum: int, seed: int, *, targets: list[str] | None = None) -> tuple[list[dict], dict[str, dict[str, int]], dict]:
    """Seeded stratified sample, emitted ROUND-ROBIN across strata (every stratum's first pick before any stratum's second) so that a
    caller's cap keeps >= 1 VU per stratum.  `targets`: `<op_path>[@<step>]` -- every VU of that module (at that request-local step, if
    given) is added to the sample and recorded as targeted (the negatives of record: a single-identity fault at a mid-phase step)."""
    by: dict[str, list[dict]] = {}
    for v in vus:
        by.setdefault(v["stratum"], []).append(v)
    rng = random.Random(int(seed))
    rounds: list[list[dict]] = []
    strata = {}
    for k in sorted(by):
        pool = by[k]
        n = min(int(per_stratum), len(pool))
        chosen = rng.sample(pool, n) if n < len(pool) else list(pool)
        for j, c in enumerate(chosen):
            while len(rounds) <= j:
                rounds.append([])
            rounds[j].append(c)
        strata[k] = {"population": len(pool), "sampled": n}
    picked = [c for rnd in rounds for c in rnd]
    n_strat = len(picked)
    targeted: dict[str, int] = {}
    if targets:
        seen = {id(v) for v in picked}
        for t in targets:
            op, _, stp = str(t).partition("@")
            hit = [v for v in vus if (v["op_path"] == op or v["op_path"].endswith("." + op) or v["op_path"].endswith(op)) and (not stp or int(v["step"]) == int(stp))]
            targeted[str(t)] = len(hit)
            for v in hit:
                if id(v) not in seen:
                    seen.add(id(v))
                    picked.append(v)
                    strata[v["stratum"]]["sampled"] = strata[v["stratum"]].get("sampled", 0) + 1
    return picked, strata, {"hits": targeted, "added": len(picked) - n_strat, "stratified": n_strat}


class _Unresolved(Exception):
    """An input of the VU is not available to the replay (partial, never PASS): the message names the slice."""


def form_b_families(manifest: dict[str, Any] | None) -> dict[str, int]:
    """Producer family -> the manifest's required `instance_outputs` identities this replay has no registered evaluator for (outside
    NOT_YET, not a token selector).  The Commit value-checks those by form (B) only, which is complete only over a Match that snapshotted
    every step (commit_verdict._form_b_complete_over_values): the Match reads this to pick its snapshot steps."""
    out: Counter = Counter()
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") != "instance_outputs" or not r.get("spec"):
            continue
        fam = str(r["spec"]).split("{", 1)[0].strip()
        if fam not in EVALUATORS and fam not in NOT_YET and not RM.is_token_select(fam):
            out[fam] += 1
    return dict(out)


INTERIOR_DEPTH = 4     # composed interior producers per VU (Gemm under a BiasAdd = 1); a deeper chain is 'not evaluable', named


def _evaluate_row(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], r: dict, rid: str, arrive: int,
                  inputs_rec: list, *, depth: int, interior: dict[int, dict[str, np.ndarray]] | None = None) -> dict[str, np.ndarray]:
    """F_row on the row's resolved inputs: committed member words, registered weight slices, committed prompt ids -- or, for an arg that
    names an INTERIOR producer row (a member-less row such as the pre-bias Gemm of a biased linear, MAN-07, or the Fp8GroupQuant of an
    FP8 linear whose two ports the GEMM reads as two arg slices), that row's OWN evaluation from ITS committed inputs (bounded depth;
    `interior` memoises it per VU).  Raises _Unresolved with the slice named when an input is not available."""
    if interior is None:
        interior = {}
    d = pi.describe(r)
    if d is None:
        raise _Unresolved(f"row {r.get('i')} ({RM.spec_family(r['spec'])}) is unmodelled")
    fam, st = d["family"], d["statics"]
    inputs: list[list[np.ndarray]] = []
    wslices: list[np.ndarray | None] = []
    for group in r.get("args") or []:
        words: list[np.ndarray] = []
        wsl = None
        for sl in group:
            kind, a, lo, hi = sl[0], int(sl[1]), int(sl[2]), int(sl[3])
            if kind == "p":
                pname = pi.pnames[a] if a < len(pi.pnames) else str(a)
                if pname == "weights":
                    ws = pi.weight_slice(lo, hi)
                    if ws is None:
                        raise _Unresolved(f"weights slice [{lo}, {hi}) spans fields")
                    fname, flo, fhi, _shape = ws
                    arr = weights(fname, flo, fhi)
                    if arr is None:
                        raise _Unresolved(f"registered weight field {fname} not available to the replay")
                    # [R16 fp8] the slice is read in the field's DECLARED leaf width (bf16 u16 / e4m3 u8 / f32 u32): a 16-bit cast of an
                    # f32 block scale or of an e4m3 byte would be a silent corruption of a load-bearing input
                    bits = pi.weight_bits(fname)
                    arr = np.asarray(arr).reshape(-1)
                    if arr.dtype.itemsize * 8 != bits:
                        raise _Unresolved(f"registered weight field {fname} is provided as {arr.dtype} but the Program declares Value<{bits}>")
                    wsl = np.asarray(arr, dtype=_NP_BY_BITS.get(bits, np.uint16)).reshape(-1)
                    inputs_rec.append({"kind": "weights", "field": fname, "range": [flo, fhi], "bits": bits})
                elif pname in (SEED_PARAM, RM.SPLITS_PARAM):
                    # [R16 stoch, F-r16-20] the sampler's named INPUTS: the committed runner.sampler/seed word of THIS (step, request) and,
                    # under derived S, the committed runner.sampler/splits word of the event the Program's slice names (`splits[s]` = the
                    # request's step s; the identity is per engine step, request-local element 0).  Never the declared / observed value.
                    if hi - lo != 1:
                        raise _Unresolved(f"{pname} slice [{lo}, {hi}) is not one word")
                    step = d["step"] if pname == SEED_PARAM else lo
                    if pname == RM.SPLITS_PARAM and lo != d["step"]:
                        raise _Unresolved(f"splits slice [{lo}, {hi}) names step {lo}, the selector is at step {d['step']}")
                    w, why = store.words(rid, arrive + step, SAMPLER_OP, pname, 0, 1)
                    if w is None:
                        raise _Unresolved(why)
                    words.append(w)
                    inputs_rec.append({"kind": "committed", "identity": [rid, arrive + step, SAMPLER_OP, pname, [0, 1]]})
                else:                                                             # the declared prompt: committed input_token_ids of the arrival step
                    eop = pi.embed_op_path()
                    if eop is None:
                        raise _Unresolved("Program has no Embedding row: prompt ids have no committed identity")
                    w, why = store.words(rid, arrive + 0, eop, IDS_MEMBER, lo, hi)
                    if w is None:
                        raise _Unresolved(why)
                    words.append(w)
                    inputs_rec.append({"kind": "prompt", "identity": [rid, arrive, eop, IDS_MEMBER, [lo, hi]]})
            elif kind == "n":
                m = pi.member_at(a, lo, hi)
                if m is None:
                    pr = pi.by_i.get(a)
                    pd = pi.describe(pr) if pr is not None else None
                    if pd is not None and pd["members"] == [] and pd["family"] in EVALUATORS:      # an interior producer: compose it
                        if depth >= INTERIOR_DEPTH:
                            raise _Unresolved(f"interior chain deeper than {INTERIOR_DEPTH} at row {a} ({pd['family']})")
                        sub_rec: list = []
                        if a in interior:                                              # [R16 fp8] one evaluation per interior row per VU
                            got = interior[a]
                        else:
                            got = interior[a] = _evaluate_row(pi, store, weights, pr, rid, arrive, sub_rec, depth=depth + 1, interior=interior)
                        out = _interior_slice(got, lo, hi)
                        if out is None:
                            raise _Unresolved(f"arg slice ['n', {a}, {lo}, {hi}) exceeds the interior {pd['family']} output "
                                              f"({sum(int(np.asarray(v).size) for v in got.values())} words over {len(got)} member(s))")
                        words.append(out)
                        inputs_rec.append({"kind": "interior", "row": a, "family": pd["family"], "op_path": pd["op_path"], "range": [lo, hi], "inputs": sub_rec})
                        continue
                    cst = const_word(pr) if pr is not None else None
                    if cst is not None:                                                # a Program constant (Const32/64[0x..]_v1: top_p, pos, temperature)
                        if (lo, hi) != (0, 1):
                            raise _Unresolved(f"arg slice ['n', {a}, {lo}, {hi}) of a one-word constant row")
                        words.append(cst)
                        inputs_rec.append({"kind": "const", "row": a, "spec": pr["spec"]})
                        continue
                    raise _Unresolved(f"arg slice ['n', {a}, {lo}, {hi}) is not inside one committed member of row {a}")
                pdsc, member, mlo, mhi, _dt = m
                width = next(int(mm[1]) for mm in pdsc["members"] if mm[0] == member)
                prow = pdsc["row"]
                w, why = store.words(rid, arrive + pdsc["step"], pdsc["op_path"], member, prow * width + mlo, prow * width + mhi)
                if w is None:
                    raise _Unresolved(why)
                words.append(w)
                inputs_rec.append({"kind": "committed", "identity": [rid, arrive + pdsc["step"], pdsc["op_path"], member, [prow * width + mlo, prow * width + mhi]]})
            else:
                raise _Unresolved(f"unknown arg kind {kind!r}")
        inputs.append(words)
        wslices.append(wsl)
    return evaluate(fam, st, inputs, wslices)


def replay_vu(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], vu: dict, *,
              arrive: int) -> dict[str, Any]:
    """One VU: resolve its inputs to committed words / registered weights, evaluate, compare with its committed outputs (`opened_check`)."""
    return opened_check(getattr(store, "committed", None), lambda rec: _replay_vu(pi, store, weights, vu, arrive, rec))


def _replay_vu(pi: ProgramIndex, store: CommittedStore, weights: Callable[[str, int, int], np.ndarray | None], vu: dict, arrive: int,
               rec: dict[str, Any]) -> dict[str, Any]:
    r = pi.by_i[vu["i"]]
    d = pi.describe(r)
    rid = vu["request_id"]
    fam, st = d["family"], d["statics"]
    rec.update({"i": vu["i"], "spec": r["spec"], "request_id": rid, "step": d["step"], "engine_step": arrive + d["step"], "op_path": d["op_path"],
                "row": d["row"], "phase": vu["phase"], "inputs": []})
    try:
        got = _evaluate_row(pi, store, weights, r, rid, arrive, rec["inputs"], depth=0, interior={})
    except _Unresolved as u:
        rec["why"] = str(u)
        return rec | {"result": None}
    except OpeningNotVerified:
        raise
    except Exception as e:  # noqa: BLE001
        rec["why"] = f"evaluator error: {e!r}"[:300]
        return rec | {"result": None}
    if d.get("plane") and d["plane"] not in got and len(got) == 1:
        got = {d["plane"]: next(iter(got.values()))}                             # [MAN-06f v3] a dense relation's `out` IS this row's plane (act = SiluMul out)
        rec["plane"] = d["plane"]
    # the committed outputs: every member the manifest requires of this row
    outs, mism = [], []
    for m in d["members"]:
        member, width, dtype = m[0], int(m[1]), str(m[2])
        if member not in got:
            # [tp TP-08d] a rank Program's row-parallel Gemm / lm_head shard Gemm: the relation's ONE output is this rank's PARTIAL / SHARD
            # member (part_rank<r> / shard_rank<r>, the collective's INPUT the rank committed) -- the evaluator names it "0"/"out";
            # [MOE-TP-01 / moe] the MoeSum relation names its one output `sum`: on a TP rank that sum of the rank's committed `down`
            # words (the rank's K-sharded expert partials, routed weight applied) IS the MoE block's part_rank<r>, the all-reduce input
            alias = _tp_member_alias(got, member)
            if alias is None and d.get("copies") and len(got) == 1:
                alias = next(iter(got.values()))                                 # [F-r19-int-24] `<experts>/out` = the MoeSum relation's one output
            if alias is not None:
                got[member] = alias
        if member not in got:
            rec["why"] = f"evaluator produced no member {member!r}"
            return rec | {"result": None}
        elo, ehi = d["row"] * width, (d["row"] + 1) * width
        w, why = store.words(rid, arrive + d["step"], d["op_path"], member, elo, ehi)
        if w is None:
            rec["why"] = why
            return rec | {"result": None}
        want = np.asarray(got[member]).reshape(-1)
        have = np.asarray(w).reshape(-1)
        if RM.is_token_select(fam):                                              # TokenSelect_v1 / GumbelTopPTokenSelect_v1: the one sampled id
            eq = int(have[0]) == int(want[0])
            outs.append({"member": member, "committed": int(have[0]), "evaluated": int(want[0]), "equal": eq})
        elif dtype in RM.LIFTED_DTYPES:
            # [revb F-r17b-52 (ii)] a LIFTED member of the padded MoE construction (Q_module_pad_v1): the slot enable a_e is derived from the
            # row's COMMITTED router_ids (never the twin's), the expected words are enc(bot) exactly for a disabled slot and the Active word of
            # the twin's value for an enabled one; the router members are Active in every executed step
            active, why_a = padded_slot_enable(member, got, store, rid, arrive + d["step"], d["op_path"], d["row"], int(st["TOPK"]))
            if active is None:
                rec["why"] = why_a
                return rec | {"result": None}
            if _ITEMSIZE.get(str(why), 0) != RM.LIFTED_DTYPES[dtype][1]:
                # (iii) the plane must be COMMITTED at the lifted word's width (u32 / u64): a bf16 plane whose values happen to equal the Active
                # payloads is a different representation of the identity -- refused by name, never read as Active words
                rec["why"] = (f"{member}: committed as {why!r} ({_ITEMSIZE.get(str(why))} B words), the query requires the lifted word {dtype} "
                              f"({RM.LIFTED_DTYPES[dtype][1]} B: Value<{RM.LIFTED_DTYPES[dtype][0] + 1}>) -- representation mismatch (F-r17b-52 (iii))")
                return rec | {"result": None}
            eq, nd, first, verdict = lifted_compare(have, want, active, dtype)
            outs.append({"member": member, "width": width, "equal": eq, "differing_elements": nd, "first_diff": first, "lifted": dtype,
                         "slot_enable": int(active), "verdict": verdict})
        else:
            if _ITEMSIZE.get(dtype, 2) == 4:                                     # [MAN-06f v3] f32 / i32 plane members (router_w / router_ids): whole words
                have16 = have.astype(np.uint32) if have.dtype != np.uint32 else have
                want = np.asarray(want, dtype=np.uint32)
            else:
                have16 = have.astype(np.uint16) if have.dtype != np.uint16 else have
            eq = have16.shape == want.shape and bool(np.array_equal(have16, want))
            nd = int(np.count_nonzero(have16 != want)) if have16.shape == want.shape else -1
            outs.append({"member": member, "width": width, "equal": eq, "differing_elements": nd,
                         "first_diff": (int(np.flatnonzero(have16 != want)[0]) if (not eq and nd > 0) else None)})
        if not eq:
            mism.append(member)
        dump_row_words(fam, rid, arrive + d["step"], d["op_path"], d["row"], member, have, want, eq, store=store, elo=elo, ehi=ehi)
    for c in d.get("copies") or []:
        # [MOE-TP-02] the same output committed under its second required name (the TP rank's moe/L<k>/sum plane beside part_rank<r>):
        # read back at the plane address and compared against the SAME evaluated words -- a plane copy that disagrees is a mismatch by name
        want = np.asarray(got[c["of"]]).reshape(-1)
        elo, ehi = int(c["row"]) * int(c["width"]), (int(c["row"]) + 1) * int(c["width"])
        w, why = store.words(rid, arrive + d["step"], c["op_path"], c["member"], elo, ehi)
        label = f"{c['op_path']}/{c['member']} (copy of {c['of']})"
        if w is None:
            rec["why"] = why
            return rec | {"result": None}
        have = np.asarray(w).reshape(-1)
        if _ITEMSIZE.get(str(c["dtype"]), 2) == 4:                              # [revb n-moe-tp02-1] the primary path's word-width switch, mirrored
            have16 = have.astype(np.uint32) if have.dtype != np.uint32 else have
            want = np.asarray(want, dtype=np.uint32)
        else:
            have16 = have.astype(np.uint16) if have.dtype != np.uint16 else have
        eq = have16.shape == want.shape and bool(np.array_equal(have16, want))
        nd = int(np.count_nonzero(have16 != want)) if have16.shape == want.shape else -1
        outs.append({"member": label, "width": int(c["width"]), "equal": eq, "differing_elements": nd,
                     "first_diff": (int(np.flatnonzero(have16 != want)[0]) if (not eq and nd > 0) else None)})
        if not eq:
            mism.append(label)
        dump_row_words(fam, rid, arrive + d["step"], c["op_path"], int(c["row"]), c["member"], have, want, eq, store=store, elo=elo, ehi=ehi)
    rec["outputs"] = outs
    rec["result"] = not mism
    if mism:
        rec["why"] = f"{fam} {rid} step {d['step']} {d['op_path']}/{','.join(mism)} row {d['row']}: F_V(committed inputs) != committed output"
    return rec


_FORK_FN: Callable | None = None      # the per-VU closure the forked children run (set in the parent right before the fork)
_FORK_VUS: list[dict] = []


class _DeviceInChild(RuntimeError):
    """[rev-cb F-cb-44] a forked child touched the accelerator (a DEVICE-retained committed store, CUDA after fork): its 'evaluator error' is
    an environment fault of the pool, not a verdict about the VU -- raised so the parent falls back to the serial path (its own CUDA context)."""


_DEVICE_ERROR_RE = re.compile(r"AcceleratorError|CUDA|cuda|hip|HIP", re.ASCII)


def _fork_worker(i: int) -> tuple[dict, dict, float]:
    assert _FORK_FN is not None
    vu, rec, dt = _FORK_FN(_FORK_VUS[i])
    why = str(rec.get("why") or "")
    if rec.get("result") is None and why.startswith("evaluator error:") and _DEVICE_ERROR_RE.search(why):
        raise _DeviceInChild(f"child pid {os.getpid()} hit the device: {why[:200]}")
    return vu, _plain(rec), dt


def _plain(x: Any) -> Any:
    """JSON-plain copy of a replay record (no numpy scalars / arrays cross the pipe)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _prewarm_weights(indices: dict, picked: list[dict], weights: Callable[[str, int, int], np.ndarray | None]) -> None:
    """Read every weight field the sample consumes into the provider's cache in the PARENT (the children must not touch the model) --
    INCLUDING the weights of the interior producers a VU composes (the pre-bias Gemm under a BiasAdd, MAN-07): [exec M-0749, measured on
    the Qwen2.5 L40S B1 int-5 record] the BiasAdd VUs' own args name only the bias slice, the composed Gemm's qkv weight was first read
    in a forked child -> AcceleratorError 'CUDA initialization error' -> every BiasAdd stratum 'not evaluated' -> partial -> FAIL closed."""
    seen: set[str] = set()

    def walk(pi: ProgramIndex, r: dict, depth: int) -> None:
        for group in r.get("args") or []:
            for sl in group:
                if sl[0] == "p" and int(sl[1]) < len(pi.pnames) and pi.pnames[int(sl[1])] == "weights":
                    ws = pi.weight_slice(int(sl[2]), int(sl[3]))
                    if ws is not None and ws[0] not in seen:
                        seen.add(ws[0])
                        weights(ws[0], ws[1], ws[2])
                elif sl[0] == "n" and depth < INTERIOR_DEPTH:
                    a, lo, hi = int(sl[1]), int(sl[2]), int(sl[3])
                    if pi.member_at(a, lo, hi) is None:
                        pr = pi.by_i.get(a)
                        pd = pi.describe(pr) if pr is not None else None
                        if pd is not None and pd["members"] == [] and pd["family"] in EVALUATORS:
                            walk(pi, pr, depth + 1)                                # the interior producer's weights are the VU's too

    for vu in picked:
        pi = indices[vu["request_id"]]
        walk(pi, pi.by_i[vu["i"]], 0)


#: spec family -> the C++ twin libraries its evaluator dlopens (built JIT on first use)
_FAMILY_LIBS = {"RMSNormTriton_v1": ("rms",), "RMSNormFusedCuda_v2": ("rms",), "Gemm_v1": (), "Gemm_v2": (), "RoPE_v1": (), "SiluMul_v1": (), "BiasAdd_v1": (),
                "Embedding_v1": (), "TokenSelect_v1": (), "GumbelTopPTokenSelect_v1": (),
                # [R17 dense, ruling R17-11] the attention rows' softmax reads the measured MUFU.EX2 / MUFU.RCP tables: loaded in the parent
                "Attention_v3": ("fa2",), "Attention_v2": ("fa2",),
                # [MAN-06f v3 unit 2] the router's NvExpf reads the measured MUFU.EX2 table (fa2 tables, ~1.5 s to load): loaded in the parent
                "MoeRouterTopK_v1": ("fa2",), "MoeRouterTopKNorm_v1": ("fa2",), "MoeExpertGemm_v1": (), "MoeExpertGemmW_v1": (), "MoeSum_v1": ()}


def _prewarm_evaluators(picked: list[dict]) -> dict[str, str]:
    """[rev-cb F-cb-44 second finding; MAN-06g] Build + dlopen every twin library the sample's families need in the PARENT before the fork:
    the children inherit the mapping and never race a first-use JIT build (a sibling used to dlopen the half-written .so -> 'file too
    short' -> a stratum 'not evaluated').  The build itself is atomic + locked (`verity_vllm.program.kernels._jit`); this makes the fast path
    the only path.  Returns {library: path} for the record; a failure here is reported, not fatal (the children fall back to their own
    locked build)."""
    fams = {str(vu.get("spec") or "").split("{", 1)[0] for vu in picked}
    need = {lib for f in fams for lib in _FAMILY_LIBS.get(f, ())}
    out: dict[str, str] = {}
    if "rms" in need:
        try:
            from verity_vllm.program.registry import prims as _prims
            rm, _t = _prims._rms()
            rm.lib()
            out["rms_triton_model"] = str(rm.LIB)
        except Exception as e:  # noqa: BLE001
            out["rms_triton_model"] = f"prewarm failed: {type(e).__name__}: {str(e)[:120]}"
    if "fa2" in need:
        try:
            from verity_vllm.program.registry import prims as _prims
            _fa2_model, _tables = _prims._fa2()
            out["fa2_mufu_tables"] = f"loaded ({type(_tables).__name__})"
        except Exception as e:  # noqa: BLE001
            out["fa2_mufu_tables"] = f"prewarm failed: {type(e).__name__}: {str(e)[:120]}"
    return out


#: [R19 admit, F-r18-admit-1 fix (c)] what the last forked pool did to the parent's heap before the fork: `gc_frozen_objects` = the
#: objects moved to the permanent generation (never traversed by a collection in the children), `gc_collected` = what the pre-fork
#: collection freed, `gc_freeze` = the policy (env VERITY_REPLAY_FORK_GC_FREEZE=0 opts out for a measurement, never for a record).
_FORK_STATS: dict[str, Any] = {}
FORK_GC_FREEZE = os.environ.get("VERITY_REPLAY_FORK_GC_FREEZE", "1").strip() not in ("0", "false", "no")


def _parallel_replay(fn: Callable[[dict], tuple[dict, dict, float]], picked: list[dict], workers: int) -> list[tuple[dict, dict, float]]:
    """Forked pool over the picked VUs; bounded by a timeout (a child deadlocked on a lock inherited at fork must not hang the Commit).

    [R19 admit, F-r18-admit-1 fix (c): the fork-COW children term] the children inherit the parent's whole C2 heap (the Programs of
    every instance + the MatchOracle + the binding index: 43.5 GiB at #60 B8 Mistral, ~75 GB at #23 B64 Llama) copy-on-write.  A
    child evaluates a few VUs, but the FIRST cyclic garbage collection its own allocations trigger traverses EVERY container object
    of the inherited heap and writes each one's gc header -> every page of the parent heap is copied into that child.  That is the
    R17 #23 datum ("two ForkPoolWorkers at ~72 GB each", F-r17-24 (b)): 72 GB is the parent's C2 heap (programs 39.5 + oracle
    35.4 GB modelled), not 2.9 x the largest committed tensor -- a VU reads its own rows only (`CommittedStore.words` slices), and the
    hidden-M1 stream the "largest tensor" is has no VU row at all.  `gc.freeze()` before the fork moves every live object into the
    permanent generation, which the children's collections never traverse (they collect only what they allocate after the fork), so a
    child's private pages are its own scratch + the refcount-dirtied pages of the objects it touches (the 250 MiB ordinary-worker term).
    Semantics-free: gc.freeze changes WHEN unreachable cycles are reclaimed, never a value; the parent unfreezes right after the pool
    joins.  The record carries what was frozen (`evaluator.fork`)."""
    global _FORK_FN, _FORK_VUS
    import gc
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    _FORK_FN, _FORK_VUS = fn, picked
    _FORK_STATS.clear()
    _FORK_STATS.update({"gc_freeze": bool(FORK_GC_FREEZE), "workers": int(workers), "picked": len(picked)})
    frozen = False
    try:
        if FORK_GC_FREEZE:
            _FORK_STATS["gc_collected"] = int(gc.collect())          # garbage is not frozen into the permanent generation
            gc.freeze()
            frozen = True
            _FORK_STATS["gc_frozen_objects"] = int(gc.get_freeze_count())
        # the canary line (commit.log): the planner's `compare` reads it to model the record with 250 MiB workers
        print(f"[replay] fork gc.freeze={'on' if frozen else 'OFF'}: {workers} workers over {len(picked)} VUs"
              + (f", {_FORK_STATS['gc_frozen_objects']} objects frozen ({_FORK_STATS['gc_collected']} collected) before the fork" if frozen else
                 " (VERITY_REPLAY_FORK_GC_FREEZE=0: the children may copy the parent's C2 heap on write, F-r17-24 (b))"), flush=True)
        with ctx.Pool(processes=workers) as pool:
            res = pool.map_async(_fork_worker, range(len(picked)), chunksize=max(1, len(picked) // (workers * 8)))
            out = res.get(timeout=max(300.0, 3.0 * len(picked)))
        return out
    finally:
        if frozen:
            gc.unfreeze()
        _FORK_FN, _FORK_VUS = None, []


def challenge_seed(challenge: str, run_root: bytes) -> tuple[int, str]:
    """[rev-cb M-0572, v2] the sample seed under a VERIFIER-ISSUED post-commitment challenge: sha256(challenge || run_root)[:8].  Binding the
    challenge to the root means a challenge issued before the root was fixed (or for another run) selects a different sample, and the
    prover cannot grind the sample by recommitting (the root changes -> the verifier issues a fresh challenge).  Returns (seed, source)."""
    import hashlib
    h = hashlib.sha256(challenge.encode("utf-8") + b"|" + bytes(run_root)).digest()
    return int.from_bytes(h[:8], "big"), RC.challenge_seed_source(challenge)


def _catch_probability(strata: dict[str, dict]) -> dict[str, Any]:
    """[rev-cb M-0572] the HONEST-prover claim of a stratified sample, by stratum kind: the probability that ONE faulted VU is among the
    evaluated ones = evaluated / population of its stratum (a fault in every VU of a stratum -- a relabel, a wrong module, a late read of a
    whole tensor -- is caught with probability 1 as soon as the stratum has one evaluation).  Reported per phase (min / median / mean over
    that phase's strata) and overall; the seed's freshness is `sample.seed_source` -- against an ADAPTIVE prover only a post-commitment
    verifier challenge makes these numbers the catch rate (v2)."""
    import statistics
    by_phase: dict[str, list[float]] = {}
    for k, v in strata.items():
        pop, ev = int(v.get("population", 0)), int(v.get("evaluated", 0))
        if pop <= 0:
            continue
        parts = k.split("|")
        phase = parts[3] if len(parts) > 3 else "?"
        by_phase.setdefault(phase, []).append(min(1.0, ev / pop))
    def summ(ps: list[float]) -> dict[str, float]:
        return {"min": round(min(ps), 4), "median": round(statistics.median(ps), 4), "mean": round(sum(ps) / len(ps), 4),
                "strata": len(ps), "strata_fully_evaluated": sum(1 for p in ps if p >= 1.0)}
    allp = [p for ps in by_phase.values() for p in ps]
    return {"single_vu_fault": {"overall": summ(allp) if allp else None, "by_phase": {ph: summ(ps) for ph, ps in sorted(by_phase.items())}},
            "whole_stratum_fault": "1.0 for every stratum with evaluated >= 1 (see strata_unevaluated for the exceptions)",
            "note": "honest-prover measurement: probability that one faulted VU is evaluated = evaluated/population of its stratum; "
                    "adversarially meaningful only under a post-commitment challenge seed (sample.seed_source)"}


ENGINE_FACTS_RULE = ("the evaluators' semantics are the engine's configuration of record (dense M-0081: on cc 9.0 the Gemm_v2 Hopper chain "
                     "equals cuBLASLt only under VLLM_BATCH_INVARIANT=1, which disables split-K by CUBLAS_WORKSPACE_CONFIG=:16:8 / "
                     "CUBLASLT_WORKSPACE_SIZE=1 / allow_bf16_reduced_precision_reduction False; on cc 8.x it installs the Triton persistent "
                     "matmul) -- MEASURED in the process the replay ran in (the Commit stage runs in the engine's process), never declared; a "
                     "fact this process cannot see is 'unavailable: <why>' by name (a laptop re-read of a stored record has no engine)")


def engine_facts_of_record() -> dict[str, Any]:
    """[revb M-0083 ASK] the batch-invariant configuration beside the evaluator strings: what the replay's pins were conditioned on."""
    facts: dict[str, Any] = {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE")}
    try:
        from vllm.model_executor.determinism import batch_invariant as bi          # type: ignore
        facts["vllm_batch_invariant_mode"] = bool(bi.vllm_is_batch_invariant()) if hasattr(bi, "vllm_is_batch_invariant") else "unavailable: no vllm_is_batch_invariant()"
    except Exception as e:                                                          # noqa: BLE001 -- named, never guessed
        facts["vllm_batch_invariant_mode"] = f"unavailable: {type(e).__name__}"
    try:
        import torch                                                                 # type: ignore
        facts["torch_allow_bf16_reduced_precision_reduction"] = bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        try:
            facts["torch_preferred_blas_library"] = str(torch.backends.cuda.preferred_blas_library()).split(".")[-1]
        except Exception as e:                                                      # noqa: BLE001
            facts["torch_preferred_blas_library"] = f"unavailable: {type(e).__name__}"
        facts["compute_capability"] = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else "unavailable: no CUDA device"
    except Exception as e:                                                          # noqa: BLE001
        facts["torch_allow_bf16_reduced_precision_reduction"] = facts["torch_preferred_blas_library"] = facts["compute_capability"] = f"unavailable: {type(e).__name__}"
    facts["rule"] = ENGINE_FACTS_RULE
    return facts


def _picks_by_request_step(results: list) -> dict[str, dict[str, list[int]]]:
    # [revb n-pad-c-1, M-1581] the picks BY (request, step): the strata carry phases, not steps, so "the other request's mid VUs evaluated
    # beside a stopped request's padding steps" was inferred from the phase arithmetic; this states it -- per request, per step: picked /
    # evaluated / mismatch (a padded record's reader splits `mid` at the other request's transition from these)
    picks_by_step: dict[str, dict[int, list[int]]] = {}
    for vu, rec, _dt in results:
        row = picks_by_step.setdefault(str(vu["request_id"]), {}).setdefault(int(vu["step"]), [0, 0, 0])
        row[0] += 1
        if rec.get("result") is not None:
            row[1] += 1
            if not rec["result"]:
                row[2] += 1
    return {r: {str(s): v for s, v in sorted(d.items())} for r, d in sorted(picks_by_step.items())}


def sampled_replay(programs: dict[str, dict[str, Any]], bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None],
                   weights: Callable[[str, int, int], np.ndarray | None], *, arrivals: dict[str, int] | None = None, per_stratum: int = 2,
                   seed: int, seed_source: str = "caller", max_evaluate: int = 2000, attention_paths: list[str] | None = None, tp: dict[str, Any] | None = None,
                   weights_check: dict[str, Any] | None = None, targets: list[str] | None = None, workers: int | None = None,
                   ring_slots: int | None = None, retention: str | None = None, seed_challenge: str | None = None,
                   manifest: dict[str, Any] | None = None, executed_prefix: dict[str, dict] | None = None,
                   program_dirs: dict[str, str] | None = None) -> dict[str, Any]:
    """C2 of record.  `executed_prefix` [COORD R17-26]: request -> `executed_prefix_of_record(...)` from the Commit's own committed tokens;
    rows at request-relative steps >= n_exec leave the sampling pool into the named class `population.not_executed_by_rule` (never a
    stratum that grades 'not evaluated'); a request with prefix faults is not classified and fails `query_checks` by name.  `programs`: request label -> Program instances doc ({"rows", "params", ...}; requests sharing a shape share it).
    `manifest` [R17-3]: the row's required-value manifest (Q(P)) -- the population of record the replay's reach is reconciled against,
    identity by identity (`population.query_total` / `sampled_strata` / `uncovered_by_name` / `query_checks`); None is recorded as such.
    `weights_check`: the caller's proof that the weight slices the relations consume are the COMMITTED weights ({"equal": bool, ...}; rev-cb
    M-0411 O5) -- `equal` False fails the replay by name; None is recorded as 'by reference, unverified at replay'.
    `max_evaluate` caps the stratified sample but never below one VU per stratum (round-robin order; the effective cap is recorded);
    `targets` (`<op_path>[@<step>]`) add every VU of a module to the sample (targeted negatives; not subject to the cap).
    Addresses: the v2 query's population; a request's Build dir (`program_dirs`: request label -> dir) supplies the correspondence the
    manifest was built with (recorded as `addresses`).
    Result: True = every sampled VU replays bit-exactly and every non-empty stratum was evaluated; False = a mismatch (named);
    None = nothing evaluable / a stratum could not be evaluated (partial, never PASS)."""
    t_start = time.perf_counter()
    store = CommittedStore.of_record(bmap, committed, manifest)                 # [rel M-0367] reads through the manifest's alias table
    arrivals = arrivals or {}
    ranks = sorted({int((e.get("identity") or {}).get("rank", 0) or 0) for e in (bmap.get("entries") or [])}) or [0]
    all_vus, not_ev, not_ev_fam, indices, codes = [], Counter(), Counter(), {}, {}   # codes: not-evaluable reason -> its structured code
    addresses: dict[tuple, dict] = {}                                             # [R17-3] manifest address -> class of the rows producing it
    anchors: dict[str, dict] = {}                                                 # [F-r17b-39] per request: the step `last` is anchored to, named
    addr_rec: dict[str, Any] = {"rule": V2_QUERY, "correspondence": {}}
    for rid, inst in programs.items():
        pi = indices.get(id(inst))
        if pi is None:
            pi = ProgramIndex(inst, attention_paths=attention_paths, tp=tp, program_dir=(program_dirs or {}).get(rid))
            indices[id(inst)] = pi
        addr_rec["correspondence"][str(rid)] = pi.correspondence
        # [tp M-0523 (1)] a per-rank map (tp2_worker: every entry carries identity.rank == r) labels its strata with THAT rank; a
        # multi-rank map is replayed as rank 0's module outputs at v1 (named below)
        # [revb F-r17b-39] the `last` phase is anchored to the request's last EXECUTED step (n_exec - 1 under the executed prefix = the
        # lag step of an EOS-stopped request), not to the cap: the EOS-stopped request has a `last` stratum too, sampled deterministically
        anchors[rid] = phase_anchor(pi.n_steps, (executed_prefix or {}).get(str(rid)))
        vus, ne, nef = population(pi, rid, rank=(ranks[0] if len(ranks) == 1 else 0), arrive=int(arrivals.get(rid, 0)), ring_slots=ring_slots,
                                  addresses=addresses, last_step=int(anchors[rid]["last_step"]), codes=codes)
        all_vus.extend(vus)
        not_ev.update(ne)
        not_ev_fam.update(nef)
        indices[rid] = pi
    pool_vus, not_exec_vus = split_not_executed(all_vus, executed_prefix)          # [R17-26] classified out of the pool, counted below
    picked_all, strata, tgt = sample(pool_vus, per_stratum, seed, targets=targets)
    targeted, n_targeted, n_strat = tgt["hits"], int(tgt["added"]), int(tgt["stratified"])
    cap = max(int(max_evaluate), len(strata))                                     # never below one VU per stratum
    picked = picked_all[:min(cap, n_strat)] + picked_all[n_strat:]                # targeted VUs are outside the cap
    if len(ranks) > 1:                                                            # TP: the map carries rank-local partials this version does not replay
        not_ev[f"multi-rank map (ranks {ranks}): rank-local partials / collectives not replayed at v1 (rank 0 module outputs only)"] += 0
    evaluated, equal, mism, not_evald = 0, 0, [], []
    strata_eval: Counter = Counter()
    strata_mism: Counter = Counter()
    fam_time: Counter = Counter()
    fam_n: Counter = Counter()

    def _one(vu: dict) -> tuple[dict, dict, float]:
        t0 = time.perf_counter()
        rec = replay_vu(indices[vu["request_id"]], store, weights, vu, arrive=int(arrivals.get(vu["request_id"], 0)))
        return vu, rec, time.perf_counter() - t0

    # PARALLEL evaluation: the numpy k16-chain evaluators are GIL-bound (many small array ops), so threads gave ~2x on 16 cores; a FORKED
    # process pool shares the committed store + the weights cache copy-on-write and returns small records.  The children never touch CUDA
    # (weights are pre-read into the cache in the parent, the committed reader slices host tensors); a pool failure / timeout falls back
    # to the serial path.  Results are folded in the picked order below, so the record is independent of the schedule.
    if workers is None:
        workers = max(1, min(32, (os.cpu_count() or 2) // 2))
    workers_requested = int(workers)
    pool_note = None
    # [rev-cb F-cb-44] a DEVICE-retained committed store (`--retain device`) cannot be read by forked children (CUDA after fork ->
    # AcceleratorError; every VU would come back 'not evaluated' and the honest row would FAIL closed): the caller names the retention and
    # the replay runs SERIAL in the parent's own CUDA context.  A child that hits the device anyway is a pool failure -> serial fallback.
    if retention == "device" and int(workers) > 1:
        pool_note = "device-retained committed store: forked evaluator disabled, serial replay in the parent's CUDA context (F-cb-44)"
        workers = 1
    results = None
    prewarmed: dict[str, str] = {}
    if int(workers) > 1 and len(picked) > 1:
        try:
            _prewarm_weights(indices, picked, weights)
            prewarmed = _prewarm_evaluators(picked)
            results = _parallel_replay(_one, picked, int(workers))
        except Exception as e:  # noqa: BLE001
            pool_note = f"parallel evaluator unavailable ({type(e).__name__}: {str(e)[:120]}); serial fallback"
            results = None
    if results is None:
        workers = 1
        results = [_one(vu) for vu in picked]
    unev_by_stratum: dict[str, dict] = {}
    child_reads = opened_in_children(results, forked=int(workers) > 1)
    for vu, rec, dt in results:
        fam_time[vu["family"]] += dt
        fam_n[vu["family"]] += 1
        if rec.get("result") is None:
            ne = {k: rec.get(k) for k in ("spec", "request_id", "step", "op_path", "row", "why")}
            ne["why_class"] = why_class(ne.get("why"))
            not_evald.append(ne)
            u = unev_by_stratum.setdefault(vu["stratum"], {"stratum": vu["stratum"], "spec": vu["spec"], "request_id": vu["request_id"], "phase": vu["phase"],
                                                            "op_path": vu["op_path"], "n": 0, "why": ne.get("why"), "why_class": ne["why_class"], "classes": []})
            u["n"] += 1
            if ne["why_class"] not in u["classes"]:
                u["classes"].append(ne["why_class"])
            continue
        evaluated += 1
        strata_eval[vu["stratum"]] += 1
        if rec["result"]:
            equal += 1
        else:
            mism.append(rec)
            strata_mism[vu["stratum"]] += 1
    for k in strata:
        strata[k]["evaluated"] = int(strata_eval.get(k, 0))
        if strata_mism.get(k):
            strata[k]["mismatch"] = int(strata_mism[k])                       # [moe F-moe-r17-06] the stratum's own count, uncapped
    picks_by_request_step = _picks_by_request_step(results)
    # [moe F-moe-r17-06] EVERY mismatch counted by name, uncapped (the first-12 list below carries the inputs/outputs detail only): a
    # reader of the record learns which mechanisms / requests / phases / modules disagree without re-running the replay
    mism_by = {"by_family": dict(Counter(m["spec"].split("{")[0] for m in mism)),
               "by_request": dict(Counter(str(m["request_id"]) for m in mism)),
               "by_phase": dict(Counter(str(m.get("phase")) for m in mism)),
               "by_family_phase": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m.get('phase')}" for m in mism).items())),
               "by_family_request_phase": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m['request_id']}|{m.get('phase')}" for m in mism).items())),
               "by_module": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m.get('op_path')}" for m in mism).items())),
               "strata": {k: int(v) for k, v in sorted(strata_mism.items())}}
    strata_unevaluated = sorted(k for k, v in strata.items() if v["population"] > 0 and v["evaluated"] == 0)
    fams = sorted({v["family"] for v in all_vus})
    wc = dict(weights_check) if isinstance(weights_check, dict) else None
    if wc is not None and wc.get("equal") is False:
        result, grade = False, "mismatch"
        why = f"weights at replay != committed weights root ({wc.get('why') or 'the registered inputs the relations consume are not the committed ones'})"
    elif mism:
        result, grade = False, "mismatch"
        why = mism[0]["why"] + (f" (+{len(mism) - 1} more)" if len(mism) > 1 else "")
    elif evaluated == 0:
        result, grade = None, "not-evaluated"
        why = "no VU could be evaluated: " + (not_evald[0]["why"] if not_evald else "empty population")
    elif strata_unevaluated:
        result, grade = None, "partial"
        why = f"{len(strata_unevaluated)} strata not evaluated: {strata_unevaluated[:3]}"
    elif not_evald:
        # [R16 rev F-r16-13 / M-0104] picks not evaluated INSIDE strata that still had an evaluated pick used to leave the grade 'complete'
        # with sample.not_evaluated_n > 0 -- a run-time fault (missing committed input, bytes not retained, evaluator error, device) on
        # some picks then rode a PASS unnamed.  'complete' means every picked VU was evaluated: any not-evaluated pick -> partial, the first
        # fault named (fault classes first; a by-name pick inside a mixed stratum is named too and, having no fully unevaluated stratum,
        # cannot be lifted by the named-gap rule -- fail closed).  Real records: 0 complete-with-not-evaluated picks (rev, 38 pairs).
        faults = [ne for ne in not_evald if ne.get("why_class") not in WHY_CLASSES_BY_NAME]
        first = (faults or not_evald)[0]
        by_cls = Counter(ne.get("why_class") or "other" for ne in not_evald)
        result, grade = None, "partial"
        why = (f"{len(not_evald)} pick(s) not evaluated inside evaluated strata ({', '.join(f'{c} x {n}' for c, n in sorted(by_cls.items()))})"
               f"{' -- run-time fault, not a gap on the record' if faults else ''}: {first.get('spec')} {first.get('op_path')} "
               f"(req {first.get('request_id')}, step {first.get('step')}): {str(first.get('why'))[:160]} (F-r16-13)")
    else:
        result, grade, why = True, "complete", None
    steps_seen = sorted({v["step"] for v in picked})
    catch = _catch_probability(strata)
    # [R17-3 / reqval] the population of record is Q(P) (the required manifest), never the replay's own enumeration: reconcile the reach
    # identity by identity and carry query_total / sampled_strata / uncovered_by_name on the record (a required identity the exact replay
    # has no VU for is NAMED with its mechanism of record -- never inside the grade, never silently "qualified")
    evaluated_keys = {(vu["request_id"], int(vu["step"]), vu["op_path"]) for vu, rec, _dt in results if rec.get("result") is not None}
    qpop = query_population(manifest, addresses, evaluated_keys, ranks=ranks, strata=strata, executed_prefix=executed_prefix,
                            controller_rows=controller_rows_of(programs, indices))   # [R4 c2] the replay's own guard-row count per request
    return {"schema": SCHEMA, "method": "sampled-exact-replay", "result": result, "grade": grade, "why": why, "catch_probability": catch,
            "population": {"vus": len(all_vus), "source": "Program instances.json.gz rows (every row; evaluable = families with a registered evaluator)",
                           "requests": sorted(programs), "by_family": dict(Counter(v["family"] for v in all_vus)), "not_evaluable_by_family": dict(not_ev_fam),
                           "not_evaluable": dict(not_ev), "not_evaluable_codes": {k: codes.get(k) or RC.reason_code("other") for k in not_ev}, "strata_n": len(strata),
                           "authority": "Q(P) -- the required manifest (R17-3); `vus` above is the replay's enumeration of the Program rows, reconciled below",
                           "query_total": qpop["query_total"], "sampled_strata": qpop["sampled_strata"], "uncovered_by_name": qpop["uncovered_by_name"],
                           "not_executed_by_family": qpop["not_executed_by_family"], "not_executed_rule": qpop["not_executed_rule"],
                           "not_executed_by_rule": dict(qpop["not_executed_by_rule"], vus=len(not_exec_vus),
                                                        vus_by_request=dict(Counter(v["request_id"] for v in not_exec_vus)),
                                                        strata_excluded={k: {"population": int(n), "label": "not evaluable: unexecuted (R17-1)"}
                                                                         for k, n in sorted(Counter(v["stratum"] for v in not_exec_vus).items())}),
                           "uncovered_identities": qpop["uncovered_identities"], "controller": qpop.get("controller"),   # [R4] the lifted controller, by name
                           "query_checks": qpop["query_checks"], "scope": qpop["scope"]},
            "addresses": addr_rec,
            "sample": {"per_stratum": int(per_stratum), "seed": int(seed), "seed_source": seed_source,
                       # [R16 rev M-0600 / F-r16-23] under the verifier-challenge form the FULL challenge is on the record: a reader recomputes
                       # sha256(challenge|run_root)[:8] (challenge_seed) as a fact -- the source text alone (challenge[:16], repr) cannot be
                       **({"challenge": str(seed_challenge)} if seed_challenge is not None else {}),
                       "picked": len(picked), "evaluated": evaluated, "equal": equal,
                       "mismatch_n": len(mism), "not_evaluated_n": len(not_evald), "max_evaluate": int(max_evaluate), "cap_effective": int(cap),
                       "stratified_picked": min(cap, n_strat), "targeted": targeted, "targeted_n": n_targeted,
                       "order": "round-robin across strata (every stratum's first pick before any second); targeted VUs appended, outside the cap",
                       "picks_by_request_step": picks_by_request_step,      # [revb n-pad-c-1] {request: {step: [picked, evaluated, mismatch]}}
                       # [revb S-r17b-77 nit] the step is the REQUEST-RELATIVE step (R17-1's n_exec counts the request's own steps); the engine
                       # step of a pick is arrival + step -- the arrival per request beside it, so a mixed-arrivals reader adds it, not guesses
                       "picks_by_request_step_key": "[picked, evaluated, mismatch] per (request, request-relative step); engine step = picks_by_request_arrival[request] + step; targeted picks included",
                       "picks_by_request_arrival": {r: int(arrivals.get(r, 0)) for r in picks_by_request_step}},
            "timing": {"wall_s": round(time.perf_counter() - t_start, 3), "workers": int(workers), "cpu_s_sum": round(sum(fam_time.values()), 3),
                       "per_family_s": {f: round(fam_time[f], 3) for f in sorted(fam_time)}, "per_family_n": {f: int(fam_n[f]) for f in sorted(fam_n)}},
            # [rev-cb F-cb-44] how the committed store was read: the collector's retention and the evaluator that ran (forked pool / serial)
            "evaluator": {"retention": retention, "workers_requested": workers_requested, "workers": int(workers),
                          "mode": "forked pool" if int(workers) > 1 else "serial (parent process)", "note": pool_note,
                          "prewarmed_libraries": prewarmed,               # [MAN-06g] twin .so's dlopen'ed in the parent before the fork
                          # [R19 admit, F-r18-admit-1 (c)] gc.freeze before the fork: the children's collections never traverse the
                          # inherited C2 heap (the 72 GB-per-child COW of R17 #23); {} when no pool was forked
                          "fork": dict(_FORK_STATS) if int(workers) > 1 else {}},
            "strata": strata, "strata_unevaluated": strata_unevaluated, **({"value_reads_in_children": child_reads} if child_reads else {}),
            # [rev-cb M-0740 G1] EVERY unevaluated stratum with its reason and class, uncapped (the verdict's named-gap rule decides per stratum)
            "unevaluated_strata": [unev_by_stratum[k] for k in strata_unevaluated if k in unev_by_stratum],
            "why_classes": {"by_name": list(WHY_CLASSES_BY_NAME), "fault": list(WHY_CLASSES_FAULT),
                            "rule": "a by-name class is a gap of this replay version (may be carried on a PASS when form (B) covers the values); "
                                    "a fault class is a run that did not evaluate (never carried)"},
            "strata_keys": {"mechanisms": sorted({k.split("|")[0] for k in strata}), "requests": sorted({k.split("|")[1] for k in strata}),
                            "ranks": (["rank 0 (tp1, or the rank-0 shard of a per-rank map)"] if ranks == [0] else [f"rank {r}" for r in ranks]),
                            "steps": {"early": sorted(s for s in steps_seen if s <= 1), "late": sorted(s for s in steps_seen if s > 1)},
                            "phases": sorted({k.split("|")[3] for k in strata}),
                            "modules": len({k.split("|")[4] for k in strata if k.count("|") >= 4}),
                            "key": "spec | request | rank | phase | module op_path | lifetime (window); attention rows: spec with T -> NB = ceil(T/BN) "
                                   "(the block-count mechanism; the row's own spec keeps T) [R17-11]",
                            "lifetime": {"values": dict(Counter(v["lifetime"] for v in all_vus)), "ring_slots": ring_slots,
                                         "rule": "engine step >= ring_slots -> recycled-window (storage the ring already committed under an earlier step's "
                                                 "identities); else fresh-window; unwindowed collectors: one value, stated",
                                         "sampled": dict(Counter(v["lifetime"] for v in picked))},
                            "phase": ("prefill / first-decode / mid / last EXECUTED step of each request (F-r17b-39: `last` is anchored to n_exec - 1 "
                                      "under the executed prefix -- the lag step of an EOS-stopped request -- else to the cap; steps past the "
                                      "prefix are 'not-executed', split out of the pool by name)"),
                            # [revb F-r17b-39] per request: the step `last` is anchored to, the anchor's rule, and the lag step when there is one
                            "phase_anchor": {r: anchors[r] for r in sorted(anchors)}},
            "mismatch": [{k: m.get(k) for k in ("spec", "request_id", "step", "engine_step", "op_path", "row", "phase", "inputs", "outputs", "why")} for m in mism[:12]],
            "mismatch_by": mism_by,                                       # [moe F-moe-r17-06] every mismatch by name, uncapped
            "not_evaluated": not_evald[:12],
            # [R17 dense, M-0123] mechanisms of the Program's rows this replay does NOT read even when the row IS evaluated (named, counted)
            "uncovered_mechanisms": _uncovered_mechanisms(all_vus, not_ev_fam),
            # report M-0430: not-evaluable families BY NAME WITH COUNTS + reason (footnoted on the row; bounds the claim)
            "unreplayable": [{"family": f, "n": int(not_ev_fam[f]),
                              "why": NOT_YET.get(f.split(" (")[0], "structure / interior: no identity" if "(structure" in f
                                                 else ("interior producer: composed into its consumer's VU (MAN-07)" if "(interior" in f
                                                       else ("MoE plane row -> plane slice mapping failed (MAN-06f v3): see population.not_evaluable" if "(plane mapping failed)" in f
                                                             else "no registered evaluator")))}
                             for f in sorted(not_ev_fam)],
            "weights": (wc if wc is not None else {"equal": None, "note": "by reference, unverified at replay (no weights check supplied)"}),
            # [FLUSH-03 fact; rev M-0494 (3)] the committed up plane is not the down plane's bytes, per (request, step, layer) -- beside the replay
            "moe_planes": {"alias_check": moe_plane_alias_check(store)},
            # [revb M-0083] the engine configuration the evaluators were conditioned on, measured in this process (VLLM_BATCH_INVARIANT & co.)
            "engine_of_record": engine_facts_of_record(),
            # [R16 fp8] the composed INTERIOR producers' evaluators are named too (the quant twin is load-bearing for every FP8 GEMM VU)
            "evaluators": {f: EVALUATORS[f] for f in sorted(set(fams) | {k.split(" (")[0] for k in not_ev_fam if "(interior" in k}) if f in EVALUATORS},
            # [rev M-0100] the fitted twins name what pinned them (fixture sha256, kernels, device, date, tests) -- only for families present
            "evaluator_provenance": {f: EVALUATOR_PROVENANCE[f] for f in sorted(set(fams) | {k.split(" (")[0] for k in not_ev_fam if "(interior" in k})
                                     if f in EVALUATOR_PROVENANCE},
            "applicability": {"families_evaluated": fams, "families_not_yet": {k: v for k, v in NOT_YET.items() if k in not_ev_fam},
                              "ranks": ranks,
                              "note": "honest-implementation qualification of the collector's mapping on a seeded stratified sample; not a proof over the population"}}


# ------------------------------------------------------------------------------------------------------------ boundary linkage

def boundary_linkage(bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None], workload_requests: list[dict[str, Any]],
                     *, embed_op_path: str | None = None, splits_of_record: dict[str, Any] | None = None, store: Any = None,
                     eos_of_record: set[int] | list[int] | None = None, eos_lag_rows: int | None = None) -> dict[str, Any]:
    """Exhaustive ordered boundary linkage: for every declared request, committed `input_token_ids` at the arrival step == the declared
    prompt ids; at each later step == the request's committed `sampled_token_ids` of the previous step (i64 slot 0 of its block).
    `workload_requests`: [{"request_id": r<k>, "arrive_step", "prompt_token_ids", "steps": [engine steps], "seed": declared or None}].
    [R16 pin, ruling 36] + `prescribed_input_linkage` (seed / splits); `splits_of_record` = {request_id: {"derived": num_SMs} | other}.
    `store`: a CommittedStore-like reader (has/words) in place of (bmap, committed) -- tests.
    [rev M-0485] `eos_of_record`: the Program's declared stop rule (global_program.stop_rule eos_token_ids; None = undeclared): a request
    linked over FEWER steps than its declared cap must end on a declared EOS id, else unlinked by name.
    [padrev M-0517, RULING 21] `eos_lag_rows`: the placeholder rows the ATTESTED scheduler runs after an EOS (RunHeader.profile.engine_kwargs
    .async_scheduling True -> 1, False -> 0; None = unattested -> every early stop is refused by name).  The EOS is verified at
    steps[-1 - lag]; the trailing `lag` step(s) are the placeholder(s) fed the EOS (their committed embed input == that EOS id, recorded).
    A read whose opening does not verify against the run root is an unlinked pair by name."""
    store = UnverifiedUnlinked(store if store is not None else CommittedStore(bmap, committed))
    # the embedding op_path: the (only) op with an `in` member in the map
    if embed_op_path is None:
        ops = sorted({k[2] for k in store.idx if k[3] == IDS_MEMBER})
        if len(ops) != 1:
            return {"schema": LINKAGE_SCHEMA, "result": None, "why": f"input_token_ids member bound under {len(ops)} op_paths: {ops[:3]}", "pairs": 0, "linked": 0, "unlinked": []}
        embed_op_path = ops[0]
    pairs, linked, unlinked, prompt_ok = 0, 0, [], {}
    for r in workload_requests:
        rid = str(r["request_id"])
        arrive = int(r.get("arrive_step") or 0)
        prompt = [int(t) for t in (r.get("prompt_token_ids") or [])]
        steps = [int(s) for s in (r.get("steps") or [])]
        if not steps:
            continue
        # arrival step: the prescribed prompt
        pairs += 1
        w, why = store.words(rid, arrive, embed_op_path, IDS_MEMBER, 0, len(prompt))
        if w is None:
            unlinked.append({"request_id": rid, "engine_step": arrive, "kind": "prompt", "why": why})
            prompt_ok[rid] = False
        else:
            got = [int(x) for x in np.asarray(w).astype(np.int64).tolist()]
            ok = got == prompt
            prompt_ok[rid] = ok
            if ok:
                linked += 1
            else:
                first = next((i for i, (a, b) in enumerate(zip(got, prompt)) if a != b), min(len(got), len(prompt)))
                unlinked.append({"request_id": rid, "engine_step": arrive, "kind": "prompt", "why": f"committed prompt ids != declared prompt at position {first}"})
        # each later step: the previous step's designated output
        for s in steps[1:]:
            pairs += 1
            w, why = store.words(rid, s, embed_op_path, IDS_MEMBER, 0, 1)
            sw, swhy = store.words(rid, s - 1, SAMPLER_OP, SAMPLED_MEMBER, 0, 1)
            if w is None or sw is None:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "chain", "why": why if w is None else swhy})
                continue
            tid, sid = int(np.asarray(w).astype(np.int64)[0]), int(np.asarray(sw).astype(np.int64)[0])
            if tid == sid:
                linked += 1
            else:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "chain", "why": f"committed input id {tid} != committed sampled id {sid} of step {s - 1}"})
    # [R16 pin, coord ruling 36 (rev M-0410, reqval M-0417, stoch M-0446)] the PRESCRIBED sampler inputs are linked to their approved
    # source here, independently of whether a replay sample exposes a substitution: the committed seed word == the DECLARED workload seed
    # of the request at every committed step; on a derived-S Program the committed splits word == SplitsFor_v1(committed live count, num_SMs
    # of record).  Their unlinked entries join `unlinked` (the same FAIL-by-name path); the boundary `pairs`/`linked` counts stay the boundary's.
    presc = prescribed_input_linkage(store, workload_requests, splits_of_record=splits_of_record)
    unlinked.extend(presc.pop("unlinked"))
    method = "ordered boundary linkage (declared prompt -> input_token_ids; sampled_token_ids[s-1] -> input_token_ids[s])"
    if presc["seed"].get("checked") or presc["splits"].get("checked"):
        method += " + prescribed-input linkage (" + "; ".join(k for k in ("seed", "splits") if presc[k].get("checked")) + ")"
    # [R16 pin, ruling 35 (sys M-0466)] the horizon each request was linked over, beside its declared cap and its Program of record's steps;
    # [rev M-0485 hardening] a horizon SHORTER than the declared cap is an early stop: the committed sampled id at its EOS step must be one
    # of the Program's declared EOS ids (stop_rule, F-pm-24) -- Commit-side, not leaning on the Match chain's G6; undeclared -> fail closed.
    # [padrev M-0517, RULING 21] the EOS step is steps[-1 - lag] where lag = the ATTESTED scheduler's placeholder rows after an EOS (async
    # scheduling runs ONE step after the EOS whose sample the driver discards: sys M-0402 int-3 earlystop, EOS 0 at step 10, 504 at step 11).
    # lag is the same header field GM-01 G6 (eos_lag_rows) and pad's LAG read; unattested -> refuse by name, never default 0 or 1.
    horizon: dict[str, Any] = {}
    eos_set = None if eos_of_record is None else {int(x) for x in eos_of_record}
    lag = None if eos_lag_rows is None else int(eos_lag_rows)
    for r in workload_requests:
        if r.get("max_tokens") is None and r.get("program_steps") is None:
            continue
        rid = str(r["request_id"])
        steps = [int(s) for s in (r.get("steps") or [])]
        h: dict[str, Any] = {"linked_steps": len(steps), "declared_cap": r.get("max_tokens"), "program_steps": r.get("program_steps")}
        if r.get("max_tokens") is not None and steps and len(steps) < int(r["max_tokens"]):
            last = steps[-1]
            over = f"{rid} linked over {len(steps)} of its declared cap {r['max_tokens']} steps"
            e: dict[str, Any] = {"last_step": last, "lag": lag, "declared_eos": (sorted(eos_set)[:16] if eos_set is not None else None), "verified": False}
            w, why = store.words(rid, last, SAMPLER_OP, SAMPLED_MEMBER, 0, 1)
            e["sampled_id"] = None if w is None else int(np.asarray(w).astype(np.int64)[0])
            if lag is None:
                unlinked.append({"request_id": rid, "engine_step": last, "kind": "eos",
                                 "why": f"{over} and the record does not attest its scheduler's EOS lag (RunHeader.profile.engine_kwargs.async_scheduling unattested, "
                                        f"RULING 21): the EOS step cannot be named, refuse by name"})
            elif lag not in (0, 1):
                unlinked.append({"request_id": rid, "engine_step": last, "kind": "eos", "why": f"{over} but the attested EOS lag {lag} is not in {{0, 1}} (RULING 21): refuse by name"})
            elif len(steps) < lag + 1:
                unlinked.append({"request_id": rid, "engine_step": last, "kind": "eos", "why": f"{over}: fewer steps than the attested EOS lag {lag} + 1, no EOS step exists"})
            else:
                eos_step = steps[-1 - lag]
                e["eos_step"] = eos_step
                ew, ewhy = (w, why) if lag == 0 else store.words(rid, eos_step, SAMPLER_OP, SAMPLED_MEMBER, 0, 1)
                e["eos_id"] = None if ew is None else int(np.asarray(ew).astype(np.int64)[0])
                # the trailing lag step(s) are placeholders fed the EOS: their committed embed input is the EOS id (the chain above linked it)
                ph = steps[-lag:] if lag else []
                e["placeholder_steps"] = ph
                e["placeholder_input_ids"] = []
                for ps in ph:
                    pw, _pwhy = store.words(rid, ps, embed_op_path, IDS_MEMBER, 0, 1)
                    e["placeholder_input_ids"].append(None if pw is None else int(np.asarray(pw).astype(np.int64)[0]))
                if ew is None:
                    unlinked.append({"request_id": rid, "engine_step": eos_step, "kind": "eos", "why": f"{over} but the EOS step {eos_step}'s sampled id is not committed: {ewhy}"})
                elif eos_set is None:
                    unlinked.append({"request_id": rid, "engine_step": eos_step, "kind": "eos",
                                     "why": f"{over} and the Program declares no stop rule (eos_token_ids): the early stop cannot be verified, fail closed (F-pm-24)"})
                elif e["eos_id"] not in eos_set:
                    unlinked.append({"request_id": rid, "engine_step": eos_step, "kind": "eos",
                                     "why": f"{over} but its committed sampled id {e['eos_id']} at the EOS step {eos_step} (= last step {last} - lag {lag}) is not a declared EOS id {sorted(eos_set)[:8]}"})
                elif any(x != e["eos_id"] for x in e["placeholder_input_ids"]):
                    unlinked.append({"request_id": rid, "engine_step": ph[0], "kind": "eos",
                                     "why": f"{over}: the placeholder step(s) {ph} after the EOS at step {eos_step} are fed {e['placeholder_input_ids']}, not the EOS id {e['eos_id']}"})
                else:
                    e["verified"] = True
            h["eos_of_record"] = e
        horizon[rid] = h
    result = (pairs > 0 and not unlinked)
    return {"schema": LINKAGE_SCHEMA, "method": method,
            "result": result if pairs > 0 else None, "pairs": pairs, "linked": linked, "unlinked_n": len(unlinked), "unlinked": unlinked[:12],
            "prompt_linked": prompt_ok, "embed_op_path": embed_op_path, "prescribed": presc, **({"horizon": horizon} if horizon else {}),
            "why": None if result else (unlinked[0]["why"] if unlinked else "no (step, request) pairs to link")}


def arrivals_of_record(declared: dict[str, int], schedule: dict[str, Any] | None) -> tuple[dict[str, int], bool]:
    """[rev M-0570 (1), compiled f3fdaba37] the arrivals every C2 consumer (population placement, sampled replay, boundary linkage,
    CompiledKernelCheck) counts from: the ENGINE schedule of record's observed arrive steps when the population was placed on the observed
    committed-step axis (`schedule.use == "observed"`: every declared request observed, contiguous, exactly its cap), else the declaration.
    Returns (arrivals, moved) -- `moved` iff an arrival differs from the declaration.  A missing / failed schedule leaves the declaration."""
    out = {str(k): int(v) for k, v in (declared or {}).items()}
    if not isinstance(schedule, dict) or schedule.get("use") != "observed":
        return out, False
    obs = {str(k): int(v) for k, v in (schedule.get("observed_arrive_steps") or {}).items() if str(k) in out}
    moved = any(out[k] != v for k, v in obs.items())
    out.update(obs)
    return out, moved


def linkage_horizon(arrive: int, declared_cap: int, program_steps: int | None, executed: int | None = None) -> list[int]:
    """[R16 pin, coord ruling 35 sixth consumer (sys M-0466)] the engine steps a request is linked over: its Program OF RECORD's steps
    (T_r + 1: the served shape at B=1 eos-truncated / GP-01 composed at B>=2), capped by the declared cap; the cap alone when the Program's
    step count is unknown (older records).  A request whose committed steps are FEWER than its Program of record still fails by name
    (the chain rule inside the horizon is unchanged); the cap's never-executed steps are not 'unlinked' -- the Program never had them."""
    n = int(declared_cap) if program_steps is None else min(int(declared_cap), int(program_steps))
    # [R17-1 / F-moe-r17-03] `executed`: the request's executed prefix of record (executed_prefix.executed_prefix_of_record: cap, or
    # served + LAG on an EOS stop under the DECLARED lag) -- at B>=2 GP-01 composes every request to its cap, so the Program's step count
    # is the cap's and the horizon is the executed prefix; boundary_linkage then verifies the EOS at steps[-1 - lag] and the placeholder
    # row(s) fed the EOS exactly as before.  None = the pre-R17-1 horizon.
    if executed is not None:
        n = min(n, max(int(executed), 0))
    return list(range(int(arrive), int(arrive) + max(n, 0)))


SEED_RULE = "committed runner.sampler/seed word == the DECLARED workload seed of the request (i64 exact) at every step with a committed sampled id (constant per request; stoch M-0446)"
SPLITS_RULE = "committed runner.sampler/splits word at step s == SplitsFor_v1(#requests with a committed sampled id at s, num_SMs of record) (REQ-03 derived S)"


def _word_int(w: np.ndarray) -> int:
    a = np.asarray(w).reshape(-1)
    return int(a.view(np.int64)[0]) if a.dtype.itemsize == 8 else int(a.astype(np.int64)[0])


def prescribed_input_linkage(store: Any, workload_requests: list[dict[str, Any]], *, splits_of_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """[R16 pin, coord ruling 36] Commit-side linkage of the stochastic Program's PRESCRIBED inputs to their approved source.
    `workload_requests[i]["seed"]` = the request's DECLARED seed (the SamplingParams.seed the run was asked for; None on a greedy row);
    `splits_of_record` = {request_id: {"derived": num_SMs}} from the Build result's `sampling` (None / non-derived = no splits rule).
    SEED: for every (step, request) that committed a `runner.sampler/sampled_token_ids` word, the committed `runner.sampler/seed` word
    must exist and == the declared seed (i64 exact) -- the kernel reads seeds[req] + pos[req]; the seed is one word per request at every
    step (no per-step derivation).  SPLITS (derived S only): at every step with >= 1 committed sampled id, S* = SplitsFor_v1(live, num_SMs)
    with live = the number of requests with a committed sampled id at that step; every live request's committed splits word == S*.
    Missing word / unequal word / an unresolvable S -> unlinked BY NAME (kind 'seed' / 'splits').  A greedy row (no declared seed, no
    derived S) checks nothing and says so.  Returns {"seed": {...}, "splits": {...}, "unlinked": [...]}."""
    unlinked: list[dict[str, Any]] = []
    live_at: dict[int, list[str]] = {}
    seeded = 0
    seed_checked = seed_linked = 0
    for r in workload_requests:
        rid = str(r["request_id"])
        steps = [int(s) for s in (r.get("steps") or [])]
        live = [s for s in steps if store.has(rid, s, SAMPLER_OP, SAMPLED_MEMBER)]
        for s in live:
            live_at.setdefault(s, []).append(rid)
        declared = r.get("seed")
        if declared is None:
            continue
        seeded += 1
        declared = int(declared)
        if not live:
            unlinked.append({"request_id": rid, "engine_step": steps[0] if steps else None, "kind": "seed",
                             "why": f"{rid}: no committed sampled_token_ids at any of its {len(steps)} declared steps -- the declared seed {declared} is linked at no step"})
            continue
        for s in live:
            seed_checked += 1
            w, why = store.words(rid, s, SAMPLER_OP, SEED_PARAM, 0, 1)
            if w is None:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "seed", "why": f"prescribed seed of {rid} at step {s} not committed: {why}"})
                continue
            got = _word_int(w)
            if got == declared:
                seed_linked += 1
            else:
                unlinked.append({"request_id": rid, "engine_step": s, "kind": "seed",
                                 "why": f"committed runner.sampler/seed {got} != declared workload seed {declared} of {rid} at step {s} (prescribed input, ruling 36)"})
    seed_rec: dict[str, Any] = {"rule": SEED_RULE, "requests_declared": seeded, "checked": seed_checked, "linked": seed_linked}
    if seeded == 0:
        # [rev M-0472 hardening] 'not applicable' only when the record carries NO committed seed word: a stochastic run with no declared
        # seed (vLLM's own generator) that still bound runner.sampler/seed has a prescribed input with no approved source -> unlinked by name
        stray = [(rid, s) for s, rids in sorted(live_at.items()) for rid in rids if store.has(rid, s, SAMPLER_OP, SEED_PARAM)]
        if stray:
            rid0, s0 = stray[0]
            seed_rec["committed_without_source"] = len(stray)
            unlinked.append({"request_id": rid0, "engine_step": s0, "kind": "seed",
                             "why": f"committed runner.sampler/seed word at {len(stray)} (step, request) pair(s) (first {rid0} step {s0}) with NO declared workload seed: a prescribed input without an approved source"})
        else:
            seed_rec["why"] = "no declared seed on any request (greedy workload; no committed seed word): not applicable"
    # splits: derived S of record (one num_SMs across the requests' Programs; a disagreement is named, never averaged)
    nsms = set()
    derived_for: set[str] = set()
    for r in workload_requests:
        rid = str(r["request_id"])
        sp = (splits_of_record or {}).get(rid)
        if isinstance(sp, dict) and sp.get("derived") is not None:
            nsms.add(int(sp["derived"]))
            derived_for.add(rid)
    splits_rec: dict[str, Any] = {"rule": SPLITS_RULE, "requests_derived": len(derived_for), "checked": 0, "linked": 0, "steps": 0}
    if not derived_for:
        splits_rec["why"] = "no request Program declares derived S (constant S / greedy): not applicable"
    elif len(nsms) != 1:
        splits_rec["why"] = f"num_SMs of record disagree across the request Programs: {sorted(nsms)}"
        unlinked.append({"request_id": None, "engine_step": None, "kind": "splits", "why": "num_SMs of record disagree across the request Programs: " + str(sorted(nsms))})
    else:
        from verity_vllm.program.registry.topp_split import splits_for_v1
        nsm = nsms.pop()
        splits_rec["num_sms"] = nsm
        sched: dict[int, int] = {}
        for s in sorted(live_at):
            rids = live_at[s]
            try:
                want = int(splits_for_v1(len(rids), nsm))
            except ValueError as e:
                unlinked.append({"request_id": None, "engine_step": s, "kind": "splits", "why": f"no S of record at step {s}: {e}"})
                continue
            sched[s] = want
            splits_rec["steps"] += 1
            for rid in rids:
                if rid not in derived_for:
                    unlinked.append({"request_id": rid, "engine_step": s, "kind": "splits", "why": f"{rid} is live at step {s} but its Program does not declare derived S"})
                    continue
                splits_rec["checked"] += 1
                w, why = store.words(rid, s, SAMPLER_OP, RM.SPLITS_PARAM, 0, 1)
                if w is None:
                    unlinked.append({"request_id": rid, "engine_step": s, "kind": "splits", "why": f"prescribed splits of {rid} at step {s} not committed: {why}"})
                    continue
                got = _word_int(w)
                if got == want:
                    splits_rec["linked"] += 1
                else:
                    unlinked.append({"request_id": rid, "engine_step": s, "kind": "splits",
                                     "why": f"committed runner.sampler/splits {got} != SplitsFor_v1(live {len(rids)}, num_SMs {nsm}) = {want} for {rid} at step {s} (prescribed input, ruling 36)"})
        splits_rec["schedule"] = {str(s): v for s, v in list(sched.items())[:64]}
    return {"seed": seed_rec, "splits": splits_rec, "unlinked": unlinked}


def model_weights_provider(model, prefix: str) -> Callable[[str, int, int], np.ndarray | None]:
    """Registered-input reader over the model's parameters AND buffers: Program field `model.model.layers.0.mlp.down_proj.weight` -> the
    implementation's `layers.0.mlp.down_proj.weight` (wrapper prefix stripped), as bf16 words; cached per field.
    [R17 dense, int-2 canary] the set read here is the set the committed weights root PINS (`register_weights`: `named_parameters()` +
    `named_buffers()`, one qualname per tensor) -- the rotary table `...rotary_emb.cos_sin_cache` is a BUFFER (vLLM registers it
    non-persistent; one shared RotaryEmbedding instance, the Program names it once at layer 0), so a parameters-only reader left every
    RoPE_v1 row unresolved ('registered weight field ... not available to the replay', 300/300 strata on the smollm2 canary).  A
    parameter shadows a buffer of the same qualname (never the case on a torch Module); a field in neither is still None (unresolved)."""
    params = dict(model.named_buffers())
    params.update(dict(model.named_parameters()))
    cache: dict[str, np.ndarray] = {}

    def read(field: str, lo: int, hi: int) -> np.ndarray | None:
        name = field[len(prefix):] if prefix and field.startswith(prefix) else field
        t = params[name] if name in params else (params[field] if field in params else None)     # never `or` on tensors
        if t is None:
            return None
        if name not in cache:
            import torch
            x = t.detach()
            # [R16 fp8] the words in the parameter's OWN width: bf16 -> u16; the block-scaled FP8 linear's e4m3 `weight` ([N, K], the
            # checkpoint's stored bytes; the cutlass kernel reads it through a `.T` view, the storage is not re-laid) -> u8; its f32
            # `weight_scale_inv` ([N/G, K/G]) -> u32.  The replay checks the width against the Program's declared leaf (weight_bits).
            if x.dtype == torch.bfloat16:
                cache[name] = x.contiguous().view(torch.int16).cpu().numpy().view(np.uint16).reshape(-1)
            elif x.dtype == torch.float32:
                cache[name] = x.contiguous().view(torch.int32).cpu().numpy().view(np.uint32).reshape(-1)
            elif str(x.dtype) in ("torch.float8_e4m3fn", "torch.float8_e4m3fnuz", "torch.uint8", "torch.int8"):
                cache[name] = x.contiguous().view(torch.int8).cpu().numpy().view(np.uint8).reshape(-1)
            else:
                return None
        arr = cache[name]
        if hi > arr.size:
            return None
        return arr[lo:hi]

    return read
