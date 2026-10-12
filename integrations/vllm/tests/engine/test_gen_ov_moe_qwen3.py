"""[R19 qwen3, row #75] `profiles.gen_ov_moe_qwen3`: the Match-grade constructive Qwen3-30B-A3B rank profile.

What is proved here, CPU only (no rank log exists yet -- R17's Qwen3 TP2 rank logs died with cp-tp; the capture recipe is on the R19 board):
  * the rank-local configuration and the weights Record reproduce the Build of record's per-rank parameter shapes EXACTLY (veritor
    `lane/r17-tp` .../r17int9/build_summary.json `tp.sharded_params.by_class`, transcribed below) and the fold's `weights` parameter has the
    Build's element count (+ the rotary table);
  * on synthetic `vllm.moe_forward` scope events at the RANK shapes (E=128, TOPK=8, H=2048, I_r=384; the recorded kernel bodies of the
    OLMoE capture with Qwen3's `renormalize=True`) the scope folds to the six-part pair-major Group over `MoeRouterTopKOrderedNorm_v1{128,8,8}`,
    `MoeExpertGemm_v1{128,2048,768}`, `SiluMul_v2{384}`, `MoeExpertGemmW_v1{128,384,2048}`, `MoeSum_v1{8,2048}` -- with the block's TP
    all-reduce resolved as `AllReduce2_v1{N=2048}` whether vLLM issues it inside the scope or after it, and the router's gate GEMM
    (`mlp.gate`, which vLLM d9105ea80 launches inside the scope) resolved as the profile's `Gemm_v1{2048,128}` ahead of the router -- and
    refuses, with the observed value, an unnormalised router, another tiling, the padded construction, a shared-experts input, a gate
    GEMM that follows the router or writes anything but its logits;
  * on a tiny synthetic configuration (2 layers, 8 experts, small dims) the profile builds, every Definition the fold would bind
    enumerates to a Program with the expected input leaves and no unbound member, and the renormalising router's weights sum to one;
  * `PROFILE_BY_CASE["QWEN3_30B_A3B"]` names this module (the case scan would otherwise find nothing: no other module declares the role).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluate import evaluate_region
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.program import Program
from verity_vllm.program import model as analytic
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, TritonLaunch, ViewDesc
from verity_vllm.observe.fold.patterns import TokenSelectArgmax, TokenSelectGumbelTwoStage
from verity_vllm.engine.profiles import profile_module_for_case
from verity_vllm.engine.profiles.gen_ov_moe_patterns import FUSED_MOE_PINS, MoEFusedExperts
from verity_vllm.collectives.embedding_shard import EmbeddingShardCuda

Q3 = importlib.import_module("verity_vllm.engine.profiles.derived_QWEN3_30B_A3B_tp2")
from verity_vllm.observe.fold.resolver import Compute, Group, StepContext, Transparent, Unsupported, resolve_step
from verity_vllm.observe.fold.patterns.collectives import TpCollective
from verity_vllm.observe.fold.patterns.gemm import GemmLaunch
from verity_vllm.observe.fold.patterns.target import gemm_target_of, pinned_num_sms
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import moe as MOE
from verity_vllm.program.registry import prims as P

C = Q3.C
H, I, E, TOPK, NL = C["H"], C["I"], C["NUM_EXPERTS"], C["TOPK"], C["NL"]
CUDA = "cuda:0"

#: the Build of record's per-rank parameters (r17int9 build_summary.json, `tp.sharded_params.by_class`, rank 0 == rank 1 in shape)
BUILD_OF_RECORD_SHARDED_PARAMS = {
    "VocabParallelEmbedding": {"weight(75968, 2048)": 1}, "QKVParallelLinear": {"weight(2560, 2048)": 48}, "RowParallelLinear": {"weight(2048, 2048)": 48},
    "RMSNorm": {"weight(128,)": 96, "weight(2048,)": 97}, "ReplicatedLinear": {"weight(128, 2048)": 48},
    "RoutedExperts": {"w13_weight(128, 768, 2048)": 48, "w2_weight(128, 2048, 384)": 48}, "ParallelLMHead": {"weight(75968, 2048)": 1},
}
BUILD_OF_RECORD_N_ELEMENTS = 15272458240
BUILD_OF_RECORD_COLLECTIVES_PER_STEP = {"all_reduce(M, 2048)": 97, "all_gather(R, 75968)": 1}


# --- the rank configuration and the weights Record ------------------------------------------------------------------------------------

def test_rank_profile_states_the_shard_configuration_and_the_rank_patterns():
    assert Q3.WORLD == 2 and "tp2-rank" in Q3.PROFILE_ID and Q3.PROFILE.notes["case"] == "QWEN3_30B_A3B"
    assert {k: C[k] for k in ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "NUM_EXPERTS", "TOPK")} == \
        dict(H=2048, NL=48, NH=16, KVH=2, D=128, I=384, V=75968, QKV=2560, GU=768, NUM_EXPERTS=128, TOPK=8)
    assert C["BIAS"] is False and C["tie_word_embeddings"] is False and C["QK_NORM"] == "head" and C["NORM_TOPK_PROB"] is True and C["BN"] == 64
    assert C["MOE_I_FULL"] == 768 and C["EPS"] == 1e-6 and C["ROPE_THETA"] == 1e6
    names = [type(x).__name__ for x in Q3.PROFILE.patterns]
    assert names[:4] == ["EmbeddingShardCuda", "TpCollective", "GemmLaunchTuned", "MoEFusedExpertsOrdered"], names[:5]
    assert type(Q3.PROFILE.patterns[0]) is EmbeddingShardCuda and Q3.PROFILE.patterns[1].world == 2
    moe = Q3.PROFILE.patterns[3]
    assert moe.renormalize is True and moe.config is C and len(moe.scope_patterns) == 1 and isinstance(moe.scope_patterns[0], TpCollective)
    assert moe.fused_pins == FUSED_MOE_PINS and (moe.fused_num_warps, moe.fused_num_stages) == (4, 3)
    sel = [x for x in Q3.PROFILE.patterns if isinstance(x, (TokenSelectArgmax, TokenSelectGumbelTwoStage))]
    assert sel and all(x.vocab == 151936 for x in sel), "the sampler runs over the all-gathered logits (full V), not the rank's shard"
    assert Q3.PROFILE.accepted is False
    assert "moe_block_scope" not in [getattr(p, "name", "") for p in Q3.PROFILE.patterns]     # constructive: no wave-1 Unsupported reading


def _shape_of(t) -> tuple[int, ...]:
    out = []
    while hasattr(t, "n"):
        out.append(t.n)
        t = t.elem
    return tuple(out)


def test_weights_record_reproduces_the_build_of_record_per_rank_parameters():
    wt = analytic.weights_type(C)
    layer = wt.fields[wt.index("layers")][1].elem
    shapes = {n: _shape_of(t) for n, t in layer.fields}
    assert shapes == {"ln1": (2048,), "qkv_w": (2560, 2048), "q_norm": (128,), "k_norm": (128,), "o_w": (2048, 2048), "ln2": (2048,),
                      "gate_w": (128, 2048), "w13": (128, 768, 2048), "w2": (128, 2048, 384)}, shapes
    assert "qkv_b" not in shapes and "gate_up_w" not in shapes and "down_w" not in shapes
    top = {n: _shape_of(t) for n, t in wt.fields if n != "layers"}
    assert top["embed"] == (75968, 2048) and top["lm_head"] == (75968, 2048) and top["norm"] == (2048,) and top["cos_sin"] == (40960, 128)
    # the Build's by_class census, re-derived from the Record: every class, every shape, every count
    got = {"VocabParallelEmbedding": {"weight(75968, 2048)": 1}, "ParallelLMHead": {"weight(75968, 2048)": 1},
           "QKVParallelLinear": {f"weight{shapes['qkv_w']}": NL}, "RowParallelLinear": {f"weight{shapes['o_w']}": NL},
           "RMSNorm": {"weight(128,)": 2 * NL, "weight(2048,)": 2 * NL + 1}, "ReplicatedLinear": {f"weight{shapes['gate_w']}": NL},
           "RoutedExperts": {f"w13_weight{shapes['w13']}": NL, f"w2_weight{shapes['w2']}": NL}}
    assert got == BUILD_OF_RECORD_SHARDED_PARAMS
    cos_sin = 40960 * 128
    assert wt.leaves - cos_sin == BUILD_OF_RECORD_N_ELEMENTS


def test_weight_field_binds_every_qwen3_moe_parameter_name_and_refuses_the_dense_ones():
    wf = Q3.weight_field
    assert wf("model.embed_tokens.weight") == ("embed",) and wf("lm_head.weight") == ("lm_head",) and wf("model.norm.weight") == ("norm",)
    for k in range(NL):
        assert wf(f"model.layers.{k}.self_attn.q_norm.weight") == ("layers", k, "q_norm")
        assert wf(f"model.layers.{k}.self_attn.k_norm.weight") == ("layers", k, "k_norm")
        assert wf(f"model.layers.{k}.mlp.gate.weight") == ("layers", k, "gate_w")
        assert wf(f"model.layers.{k}.mlp.experts.routed_experts.w13_weight") == ("layers", k, "w13")
        assert wf(f"model.layers.{k}.mlp.experts.routed_experts.w2_weight") == ("layers", k, "w2")
        assert wf(f"model.layers.{k}.self_attn.qkv_proj.weight") == ("layers", k, "qkv_w") and wf(f"model.layers.{k}.self_attn.o_proj.weight") == ("layers", k, "o_w")
    assert wf("model.layers.0.mlp.gate_up_proj.weight") is None and wf("model.layers.0.self_attn.qkv_proj.bias") is None
    assert Q3.root_class("model.layers.0.self_attn.rotary_emb.cos_sin_cache") == "weight" and Q3.root_class("model.layers.7.self_attn.rotary_emb.cos_sin_cache") == "alias-of-cos_sin"
    assert Q3.root_class("model.layers.0.mlp.shared_expert.gate_up_proj.weight") == "unmodelled-moe-variant"
    assert Q3.root_class("model.layers.0.self_attn.attn.k_scale") == "ignored"


def test_check_header_reshards_the_recorded_full_config_and_checks_the_moe_keys():
    hdr = RunHeader("verity-capture/log/v1", "2026-09-22T00:00:00Z", "x", {}, {"repo": Q3.MODEL, "config": dict(Q3.HF_CONFIG), "tp": 2}, {"requests": []})
    assert Q3.check_header(hdr) == []
    bad = dict(Q3.HF_CONFIG, norm_topk_prob=False, num_experts=64, moe_intermediate_size=1536)
    out = Q3.check_header(RunHeader("s", "t", "x", {}, {"repo": Q3.MODEL, "config": bad, "tp": 1}, {"requests": []}))
    assert any("norm_topk_prob" in m for m in out) and any("num_experts=64" in m for m in out) and any("moe_intermediate_size" in m for m in out)
    assert any("tensor_parallel_size=1" in m for m in out)


def test_tp1_profile_is_the_same_module_at_world_one():
    p1 = Q3.build_profile(1)
    c1 = p1.config
    assert (c1["NH"], c1["KVH"], c1["I"], c1["V"], c1["QKV"], c1["NUM_EXPERTS"]) == (32, 4, 768, 151936, 5120, 128)
    names = [type(x).__name__ for x in p1.patterns]
    assert names[:2] == ["GemmLaunchTuned", "MoEFusedExpertsOrdered"] and "TpCollective" not in names and "EmbeddingShardCuda" not in names
    assert p1.patterns[1].scope_patterns == () and p1.patterns[1].renormalize is True
    assert p1.id != Q3.PROFILE.id and "tp" not in p1.id


def test_profile_module_for_case_derives_the_rank_module_name():
    assert profile_module_for_case("QWEN3_30B_A3B", 2) == "derived_QWEN3_30B_A3B_tp2" and profile_module_for_case("QWEN3_30B_A3B") == "derived_QWEN3_30B_A3B"


def test_injected_fused_moe_pins_are_the_pattern_s_pins_and_are_recorded():
    pins = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    p = Q3.build_profile(2, fused_moe_pins=pins, fused_moe_num_warps=8, fused_moe_num_stages=2)
    moe = next(x for x in p.patterns if isinstance(x, MoEFusedExperts))
    assert moe.fused_pins == pins and (moe.fused_num_warps, moe.fused_num_stages) == (8, 2)
    assert p.notes["moe"]["pins"]["fused_moe_kernel"]["BLOCK_SIZE_M"] == 16 and "injected" in p.notes["moe"]["pins"]["fused_moe_kernel"]["source"]
    assert Q3.PROFILE.patterns[3].fused_pins == FUSED_MOE_PINS       # the module's profile is untouched


# --- the synthetic scope at the rank shapes -------------------------------------------------------------------------------------------

def _contig(shape):
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return tuple(reversed(out))


def view(alloc, shape, dtype="bfloat16", strides=None, offset=0):
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides or _contig(shape)), dtype, CUDA)


CONSTS = {"group_n": 0, "group_k": 0, "naive_block_assignment": False, **FUSED_MOE_PINS, "SPLIT_K": 1, "MUL_ROUTED_WEIGHT": False,
          "top_k": TOPK, "compute_type": "bf16", "use_fp8_w8a8": False, "use_int8_w8a8": False, "use_int8_w8a16": False,
          "per_channel_quant": False, "HAS_BIAS": False, "SWAP_AB": False, "USE_TD": False}
LAYER = 5
GEMM_FN = "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent"
A_GATE_W = 9


def gate_tiling(M: int) -> tuple[dict, int, int]:
    """The tiling (constexprs with NUM_SMS, num_warps, num_stages) the profile's GEMM patterns pin for the gate GEMM [M,H] x [H,E] on the
    test header's target: the registered target's `launch_config`, over the base `GemmLaunch`'s constexprs."""
    ctx = _ctx(_start(M))
    gt, refusal = gemm_target_of(ctx)
    assert gt is not None, refusal
    base = next(p for p in Q3.PROFILE.patterns if type(p) is GemmLaunch)
    tconsts, warps, stages, _src = gt.launch_config(M, E, H)
    consts = dict(base.constexprs)
    consts.update({k: v for k, v in tconsts.items() if k in consts or k.startswith("BLOCK_SIZE_")})
    consts["NUM_SMS"] = pinned_num_sms(ctx, base.num_sms)
    return consts, warps, stages


def moe_scope_events(M: int, *, step: int = 0, naive: bool = False, renormalize: bool = True, all_reduce: str | None = "inside",
                     consts_up: dict | None = None, shared_experts_input=None, num_stages: int = 3, gate: str | None = None,
                     gate_consts: dict | None = None):
    """One `vllm.moe_forward` scope of layer LAYER at the rank shapes (cf. test_gen_ov_moe.moe_scope_events).  `all_reduce`: "inside" = the
    collective is a child of the scope (FusedMoE.forward_impl, reduce_results=True); "after" = a sibling after the scope (Qwen3MoeSparseMoeBlock
    with reduce_results=False); None = no collective (TP1 shape).  `gate`: None = the router logits are an input of the scope (OLMoE);
    "inside" = vLLM d9105ea80's Qwen3MoeSparseMoeBlock (`FusedMoE(gate=...)`, `router_logits=hidden_states`): the `mlp.gate` GEMM launches
    in the scope ahead of topk_softmax and writes the logits; "after-router" = that launch follows topk_softmax; "elsewhere" = it writes
    another allocation; "twice" = two such launches ahead of the router."""
    nid = [1000]

    def new():
        nid[0] += 1
        return nid[0]

    hidden, logits, out = view(295, (M, H)), view(296, (M, E)), view(300, (M, H))
    received = view(305, (M, H))
    pad = view(259, (M,), "bool")
    evs = []
    scope = new()
    mod = f"model.layers.{LAYER}.mlp.experts"
    evs.append(DispatchEnter(scope, None, step, "forward", "vllm.moe_forward.default",
                             "vllm::moe_forward(Tensor(a0!) hidden_states, Tensor router_logits, Tensor? shared_experts_input, Tensor? input_ids, PyObject layer_name, SymInt hidden_dim_unpadded) -> Tensor",
                             [hidden, hidden if gate else logits, shared_experts_input, None, "layer", H], {}, 7, mod))

    def gate_gemm(c):
        gmod = f"model.layers.{LAYER}.mlp.gate"
        wt = ViewDesc(A_GATE_W, 0, (H, E), (1, H), "bfloat16", CUDA)
        i = new()
        evs.append(DispatchEnter(i, scope, step, "forward", "aten.t.default", "aten::t(Tensor(a) self) -> Tensor(a)", [view(A_GATE_W, (E, H))], {}, 7, gmod))
        evs.append(DispatchExit(i, wt))
        i = new()
        evs.append(DispatchEnter(i, scope, step, "forward", "aten.empty.memory_format", "aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
                                 [[M, E], None, None, None, None, None], {}, 7, gmod))
        evs.append(DispatchExit(i, c))
        consts, warps, stages = gate_tiling(M)
        consts.update(gate_consts or {})
        args = {"a_ptr": hidden, "b_ptr": wt, "c_ptr": c, "bias_ptr": None, "M": M, "N": E, "K": H, "stride_am": H, "stride_ak": 1,
                "stride_bk": 1, "stride_bn": H, "stride_cm": E, "stride_cn": 1, **consts}
        grid = [min(consts["NUM_SMS"], -(-M // consts["BLOCK_SIZE_M"]) * -(-E // consts["BLOCK_SIZE_N"]))]
        evs.append(TritonLaunch(new(), scope, step, "forward", GEMM_FN, "4a7e9290", args, consts, grid, warps, stages, 7, gmod))

    if gate in ("inside", "twice"):
        gate_gemm(logits)
    if gate == "twice":
        gate_gemm(view(311, (M, E)))
    if gate == "elsewhere":
        gate_gemm(view(312, (M, E)))
    w_, ids = view(297, (M, TOPK), "float32"), view(298, (M, TOPK), "int32")
    for al, shp, dt in ((297, (M, TOPK), "float32"), (298, (M, TOPK), "int32"), (299, (M, TOPK), "int32")):
        i = new()
        evs.append(DispatchEnter(i, scope, step, "forward", "aten.empty.memory_format", "aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
                                 [list(shp), None, None, None, None, None], {}, 7, None))
        evs.append(DispatchExit(i, view(al, shp, dt)))
    tok = view(299, (M, TOPK), "int32")
    i = new()
    evs.append(DispatchEnter(i, scope, step, "forward", "_moe_C.topk_softmax.default",
                             "_moe_C::topk_softmax(Tensor($0! -> ) topk_weights, Tensor($1! -> ) topk_indices, Tensor($2! -> ) token_expert_indices, Tensor gating_output, bool renormalize, Tensor? bias, Tensor? is_padding) -> ()",
                             [w_, ids, tok, logits, renormalize, None, pad], {}, 7, None))
    evs.append(DispatchExit(i, None))
    if gate == "after-router":
        gate_gemm(logits)
    i = new()
    evs.append(DispatchEnter(i, scope, step, "forward", "aten.empty_like.default", "aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
                             [hidden, None, None, None, None, None], {}, 7, None))
    evs.append(DispatchExit(i, out))
    P_ = M * TOPK
    cache1 = view(301, (M, TOPK, 2 * I), offset=P_ * I)
    cache2 = view(301, (P_, I))
    cache3 = view(301, (M, TOPK, H), offset=P_ * I)
    if naive:
        sorted_ids, expert_ids, npad = None, view(298, (P_,), "int32"), view(304, (1,), "int32")
        EM = P_ * 64
    else:
        EM = min(P_ * 64, P_ + E * 63)
        sorted_ids, expert_ids, npad = view(302, (EM,), "int32"), view(303, (EM // 64,), "int32"), view(304, (1,), "int32")
        i = new()
        evs.append(DispatchEnter(i, scope, step, "forward", "_moe_C.moe_align_block_size.default",
                                 "_moe_C::moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor($0! -> ) sorted_token_ids, Tensor($1! -> ) experts_ids, Tensor($2! -> ) num_tokens_post_pad, Tensor? maybe_expert_map) -> ()",
                                 [ids, E, 64, sorted_ids, expert_ids, npad, None], {}, 7, None))
        evs.append(DispatchExit(i, None))
    w13, w2 = view(7, (E, 2 * I, H)), view(8, (E, H, I))
    c_up = dict(CONSTS, naive_block_assignment=naive)
    c_up.update(consts_up or {})
    c_dn = dict(CONSTS, naive_block_assignment=naive, MUL_ROUTED_WEIGHT=True, top_k=1)

    def launch(a, b, c, weights, N, K, consts):
        args = {"a_ptr": a, "b_ptr": b, "c_ptr": c, "b_bias_ptr": None, "a_scale_ptr": None, "b_scale_ptr": None, "topk_weights_ptr": weights,
                "sorted_token_ids_ptr": sorted_ids, "expert_ids_ptr": expert_ids, "num_tokens_post_padded_ptr": npad, "N": N, "K": K, "EM": EM,
                "num_valid_tokens": P_, "stride_am": a.strides[0], "stride_ak": 1, "stride_be": N * K, "stride_bk": 1, "stride_bn": K,
                "stride_cm": N, "stride_cn": 1, **consts}
        return TritonLaunch(new(), scope, step, "forward", "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel", "sha", args, consts,
                            [(EM // 64) * (N // 64)], 4, num_stages, 7)

    evs.append(launch(hidden, w13, cache1, None, 2 * I, H, c_up))
    i = new()
    evs.append(DispatchEnter(i, scope, step, "forward", "_C.silu_and_mul.default", "_C::silu_and_mul(Tensor($0! -> ) result, Tensor input) -> ()",
                             [cache2, view(301, (P_, 2 * I), offset=P_ * I)], {}, 7, None))
    evs.append(DispatchExit(i, None))
    evs.append(launch(cache2, w2, cache3, w_, H, I, c_dn))
    i = new()
    evs.append(DispatchEnter(i, scope, step, "forward", "_moe_C.moe_sum.default", "_moe_C::moe_sum(Tensor input, Tensor($0! -> ) output, Tensor? topk_ids=None, Tensor? expert_map=None) -> ()",
                             [cache3, out, None, None], {}, 7, None))
    evs.append(DispatchExit(i, None))

    def ar(parent):
        i = new()
        evs.append(DispatchEnter(i, parent, step, "forward", "vllm.all_reduce.default", "vllm::all_reduce(Tensor tensor, str group_name) -> Tensor",
                                 [out, "tp:0"], {}, 7, mod))
        evs.append(DispatchExit(i, received))

    if all_reduce == "inside":
        ar(scope)
        evs.append(DispatchExit(scope, received))
    else:
        evs.append(DispatchExit(scope, out))
        if all_reduce == "after":
            ar(None)
    return evs


def _start(M: int, step: int = 0, padded: int | None = None) -> StepStart:
    rows = [("r0", p) for p in range(M)]
    return StepStart(step, rows, list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0, 1, 2, 3]}, 16,
                     list(range(M)), [("r0", M - 1)], [M - 1], 1, M, padded if padded is not None else M, None, None, "prefill")


def _ctx(start: StepStart, profile=None) -> StepContext:
    header = RunHeader("verity-capture/log/v1", "2026-09-22T00:00:00Z", "x", {"vllm": "d9105ea80"}, {"repo": Q3.MODEL, "config": {}}, {"requests": []})
    roots = {7: [Root(view(7, (E, 2 * I, H)), "weight", f"model.layers.{LAYER}.mlp.experts.routed_experts.w13_weight")],
             8: [Root(view(8, (E, H, I)), "weight", f"model.layers.{LAYER}.mlp.experts.routed_experts.w2_weight")],
             A_GATE_W: [Root(view(A_GATE_W, (E, H)), "weight", f"model.layers.{LAYER}.mlp.gate.weight")]}
    return StepContext(header, start, None, [], roots, C, profile or Q3.PROFILE)


def _resolve(M: int, profile=None, **kw):
    start_kw = {k: kw.pop(k) for k in ("padded",) if k in kw}
    evs = moe_scope_events(M, **kw)
    start = _start(M, **start_kw)
    se = step_events([start, *evs])
    return resolve_step(se, _ctx(start, profile)), se


EXPECTED_SPECS = [f"MoeRouterTopKOrderedNorm_v1{{E={E},TOPK={TOPK},VPT=8}}", f"MoeExpertGemm_v1{{E={E},K={H},N={2 * I}}}", f"SiluMul_v2{{I={I}}}",
                  f"MoeExpertGemmW_v1{{E={E},K={I},N={H}}}", f"MoeSum_v1{{TOPK={TOPK},H={H}}}"]


@pytest.mark.parametrize("M,naive,where", [(2, True, "inside"), (2, True, "after"), (3, False, "inside"), (32, False, "after"), (1, True, None)])
def test_rank_scope_folds_to_the_renormalising_pair_major_group_plus_the_block_all_reduce(M, naive, where):
    res, se = _resolve(M, naive=naive, all_reduce=where)
    groups = [r for r in res if isinstance(r, Group)]
    assert len(groups) == 1 and not [r for r in res if isinstance(r, Unsupported)], res
    g = groups[0]
    assert g.consumed == tuple(sorted(se.roots[0].ids()))
    computes = [p for p in g.parts if isinstance(p, Compute)]
    specs = [c.spec.id for c in computes]
    in_scope = [f"AllReduce2_v1{{N={H}}}"] if where == "inside" else []
    assert specs == EXPECTED_SPECS + in_scope, specs
    assert [c.rows for c in computes[:5]] == [M, M * TOPK, M * TOPK, M * TOPK, M]
    router, up, silu, down, msum = computes[:5]
    assert router.spec.name == "MoeRouterTopKOrderedNorm" and router.outputs[0].view.dtype == "float32" and router.outputs[1].view.dtype == "int32"
    assert up.operands[0].indices == tuple(p // TOPK for p in range(M * TOPK)) and up.operands[2].view.shape == (E, 2 * I, H)
    assert down.operands[3].view.shape == (E, H, I) and down.operands[2].view.alloc == router.outputs[0].view.alloc
    assert msum.outputs[0].view.shape == (M, H)
    assert up.hint.endswith("routed_experts.w13_weight") and down.hint.endswith("routed_experts.w2_weight")     # the record's weight names
    kernels = [k for p in g.parts for k in p.kernels]
    assert kernels.count("fused_moe_kernel") == 2 and "topkGating" in kernels and "moe_sum_vec_kernel" in kernels and "act_and_mul_kernel" in kernels
    aligns = [p for p in g.parts if isinstance(p, Transparent) and any("moe_align_block_size" in k for k in p.kernels)]
    assert len(aligns) == (0 if naive else 1)
    # the block's TP all-reduce: ONE AllReduce2_v1{N=H} over (own routed sum, peer partial), the received storage its output -- wherever issued
    ars = [c for r in res for c in (r.parts if isinstance(r, Group) else (r,)) if isinstance(c, Compute) and c.spec.name == "AllReduce2"]
    assert len(ars) == (0 if where is None else 1)
    if ars:
        ar = ars[0]
        assert ar.spec.id == f"AllReduce2_v1{{N={H}}}" and ar.rows == M and ar.operands[0].own.alloc == msum.outputs[0].view.alloc
        assert ar.outputs[0].view.alloc == 305 and ar.operands[0].site == f"model.layers.{LAYER}.mlp.experts"
        assert (where == "inside") == (ar in g.parts)
    assert len(res) == (2 if where == "after" else 1)


@pytest.mark.parametrize("M,naive,where", [(2, True, "inside"), (3, False, "after"), (32, False, "inside"), (1, True, None)])
def test_in_scope_gate_gemm_resolves_as_the_profile_s_gemm_ahead_of_the_router(M, naive, where):
    """vLLM d9105ea80's Qwen3MoeSparseMoeBlock hands FusedMoE its gate and passes router_logits=hidden_states, so the 30B and 235B captures
    launch `mlp.gate`'s matmul_kernel_persistent inside `vllm.moe_forward`: it is the Group's first Compute, the profile's own GEMM, and
    writes the logits the router reads."""
    res, se = _resolve(M, naive=naive, all_reduce=where, gate="inside")
    groups = [r for r in res if isinstance(r, Group)]
    assert len(groups) == 1 and not [r for r in res if isinstance(r, Unsupported)], res
    g = groups[0]
    assert g.consumed == tuple(sorted(se.roots[0].ids()))
    computes = [p for p in g.parts if isinstance(p, Compute)]
    in_scope = [f"AllReduce2_v1{{N={H}}}"] if where == "inside" else []
    assert [c.spec.id for c in computes] == [f"Gemm_v1{{K={H},N={E}}}"] + EXPECTED_SPECS + in_scope
    gemm, router = computes[0], computes[1]
    assert gemm.rows == M and gemm.pattern.startswith("gemm.matmul_kernel_persistent") and gemm.kernels == ("matmul_kernel_persistent",)
    assert gemm.operands[0].view.alloc == 295 and gemm.operands[1].view.alloc == A_GATE_W and gemm.operands[1].view.shape == (E, H)
    assert gemm.outputs[0].view.alloc == router.operands[0].view.alloc == 296
    assert "mlp.gate" in gemm.hint


@pytest.mark.parametrize("kw,needle", [
    (dict(gate="after-router"), "follows topk_softmax"),
    (dict(gate="elsewhere"), "does not write the router's gating logits"),
    (dict(gate="twice"), "more than one matmul_kernel_persistent"),
    (dict(gate="inside", gate_consts={"BLOCK_SIZE_K": 32}), "refused the router's gate GEMM"),   # the gate's tiling is the profile's pin
])
def test_in_scope_gate_gemm_declines_unless_it_is_the_router_s_logits_launch(kw, needle):
    res, _ = _resolve(2, naive=True, all_reduce=None, **kw)
    bad = [r for r in res if isinstance(r, Unsupported)]
    assert bad and not [r for r in res if isinstance(r, Group)], res
    assert any(needle in r.reason for r in bad), [r.reason for r in bad]


@pytest.mark.parametrize("construction", ["indexed-read", "indexed-read-ordered"])
def test_scope_activation_is_the_one_the_moe_construction_declares(construction):
    """The Build binds the activation `TargetProfile.MOE_CONSTRUCTIONS` declares; a fold binding another version fails the Match's compare."""
    res, _ = _resolve(2, naive=True, all_reduce=None)
    g = next(r for r in res if isinstance(r, Group))
    silu = next(p for p in g.parts if isinstance(p, Compute) and p.spec.name == "SiluMul")
    declared = [d for d in TargetProfile.MOE_CONSTRUCTIONS[construction]["definitions"] if d.startswith("SiluMul_")]
    assert [d.split("{")[0] for d in declared] == [silu.spec.id.split("{")[0]], (declared, silu.spec.id)


def test_one_step_of_the_row_has_the_build_of_record_collective_count():
    """98 collectives per engine step on the Build of record (r17int9 build_summary.json `tp.collectives`): 48 o_proj + 48 MoE + 1 embedding
    all_reduce(M, 2048) and 1 all_gather(R, 75968) -- the profile's AllReduce2_v1{N=2048} is the MoE half of the 97."""
    assert BUILD_OF_RECORD_COLLECTIVES_PER_STEP["all_reduce(M, 2048)"] == 2 * NL + 1
    assert Q3.PROFILE.notes["moe"]["definitions"][-1].startswith(f"AllReduce2_v1{{N={H}}}")


@pytest.mark.parametrize("kw,needle", [
    (dict(renormalize=False), "renormalize=True"),                          # Qwen3 states the renormalising router; the OLMoE value refuses
    (dict(consts_up={"BLOCK_SIZE_M": 16}), "tiling"),
    (dict(num_stages=4), "tiling"),
    (dict(consts_up={"use_fp8_w8a8": True}), "numeric constexprs"),
    (dict(shared_experts_input=view(310, (2, H))), "shared experts are not modelled"),   # Qwen3-30B-A3B has none; a shared branch refuses by name
    (dict(padded=8), "padding rows"),
])
def test_rank_scope_declines_with_the_observed_values(kw, needle):
    M = kw.pop("M", 2)
    res, _ = _resolve(M, **kw)
    bad = [r for r in res if isinstance(r, Unsupported)]
    assert bad, res
    assert any(needle in r.reason for r in bad), [r.reason for r in bad]


def test_padded_construction_refuses_under_the_renormalising_router():
    p = Q3.build_profile(2, "padded")
    res, _ = _resolve(2, profile=p)
    bad = [r for r in res if isinstance(r, Unsupported)]
    assert bad and "renormalize=True" in bad[0].reason and "MoeFfnFixedL" in bad[0].reason
    assert Q3.with_moe_construction(Q3.PROFILE, None) is Q3.PROFILE and Q3.with_moe_construction(Q3.PROFILE, "indexed-read-ordered") is Q3.PROFILE


def test_olmoe_profile_is_unchanged_by_the_renormalize_and_scope_pattern_fields():
    OV = importlib.import_module("verity_vllm.engine.profiles.derived_OLMOE")
    moe = next(x for x in OV.PROFILE.patterns if isinstance(x, MoEFusedExperts))
    assert moe.renormalize is False and moe.scope_patterns == () and moe.fused_pins == FUSED_MOE_PINS


# --- the tiny synthetic configuration: the Definitions enumerate, the router renormalises --------------------------------------------

TINY_HF = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "attention_bias": False, "decoder_sparse_step": 1, "head_dim": 16,
           "hidden_act": "silu", "hidden_size": 64, "intermediate_size": 256, "max_position_embeddings": 64, "mlp_only_layers": [],
           "moe_intermediate_size": 64, "norm_topk_prob": True, "num_attention_heads": 4, "num_experts": 8, "num_experts_per_tok": 2,
           "num_hidden_layers": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": False, "vocab_size": 128}


def _eval(prog: Program, inputs: list[list[int]]) -> list[int]:
    Cc = prog.circuit
    t: dict[int, int] = {}
    for pi, vals in enumerate(inputs):
        coll = prog.input_coll(pi)
        assert coll.type.leaves == len(vals), (pi, coll.type.leaves, len(vals))
        for j, v in enumerate(vals):
            t[resolve(Cc.root_scope, coll.refs[j])] = v
    evaluate_region(Cc, Cc[0:prog.gates], t)
    return [t[g] for g in Cc.root_outputs()]


def _bf16_bits(a) -> list[int]:
    return [int(x) >> 16 for x in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


def test_tiny_config_builds_and_every_bound_definition_enumerates_with_no_unbound_member():
    """2 layers, 8 experts, H=64, I=64 (rank: 32), V=128 (rank: 64), 4 heads / 2 kv heads (rank: 2 / 1): the rank configuration derives,
    the weights Record has the shard shapes, and each Definition the MoE fold binds -- at the TINY bindings -- is a closed Program whose
    input leaves are exactly the operand words (no free member)."""
    Ct, prov = Q3.derive_config(TINY_HF, 2)
    assert (Ct["H"], Ct["NL"], Ct["NH"], Ct["KVH"], Ct["D"], Ct["I"], Ct["V"], Ct["NUM_EXPERTS"], Ct["TOPK"]) == (64, 2, 2, 1, 16, 32, 64, 8, 2)
    assert prov["QK_NORM"] == "hand" and prov["I"] == "hf-config" and Ct["QK_NORM"] == "head"
    wt = analytic.weights_type(Ct)
    layer = wt.fields[wt.index("layers")][1].elem
    assert {n: _shape_of(t) for n, t in layer.fields} == {"ln1": (64,), "qkv_w": (64, 64), "q_norm": (16,), "k_norm": (16,), "o_w": (64, 32),
                                                          "ln2": (64,), "gate_w": (8, 64), "w13": (8, 64, 64), "w2": (8, 64, 32)}
    Et, K_, VPT, Ht, It = 8, 2, 4, Ct["H"], Ct["I"]
    for spec, leaves in ((bind(MOE.MoeRouterTopKNorm, E=Et, TOPK=K_, VPT=VPT), [Et]),
                         (bind(MOE.MoeExpertGemm, E=Et, K=Ht, N=2 * It), [Ht, 1, Et * 2 * It * Ht]),
                         (bind(MOE.MoeExpertGemmW, E=Et, K=It, N=Ht), [It, 1, 1, Et * Ht * It]),
                         (bind(MOE.MoeSum, TOPK=K_, H=Ht), [K_ * Ht])):
        prog = Program(spec)
        assert prog.gates > prog.input_gates > 0
        assert [prog.input_coll(i).type.leaves for i in range(len(leaves))] == leaves, spec.id
        assert prog.input_gates == sum(leaves), spec.id           # every input word is a leaf, nothing unbound


def test_renormalising_router_rescales_the_top_k_to_sum_one_and_keeps_the_ids():
    rng = np.random.default_rng(3)
    Et, K_, VPT = 8, 2, 4
    logits = np.array([b << 16 for b in _bf16_bits(rng.standard_normal(Et).astype(np.float32) * 2)], dtype=np.uint32).view(np.float32)
    raw = _eval(Program(bind(MOE.MoeRouterTopK, E=Et, TOPK=K_, VPT=VPT)), [_bf16_bits(logits)])
    nrm = _eval(Program(bind(MOE.MoeRouterTopKNorm, E=Et, TOPK=K_, VPT=VPT)), [_bf16_bits(logits)])
    assert raw[K_:] == nrm[K_:]                                                              # same ids
    w_raw = np.array(raw[:K_], dtype=np.uint32).view(np.float32)
    w_nrm = np.array(nrm[:K_], dtype=np.uint32).view(np.float32)
    s = np.float32(0)
    for k in range(K_):
        s = np.float32(s + w_raw[k])
    scale = P.f32_of(P.F32Div.evaluate(P.bits_of(np.float32(1.0)), P.bits_of(s)))
    assert list(w_nrm) == [np.float32(w * scale) for w in w_raw]                             # w_k * (1.f / sum), one div, one mul per slot
    assert abs(float(w_nrm.sum()) - 1.0) < 1e-6
