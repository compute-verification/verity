"""The block-FP8 fused-MoE fold scope (`gen_ov_moe_patterns.MoEFusedExpertsFp8`) on synthetic `vllm.moe_forward` events shaped as
vLLM d9105ea80's `TritonExperts.apply` issues them at block [64, 64] (the input quant, the w13 launch, `silu_and_mul`, the SiLU output's
quant, the w2 launch), at the Qwen3-30B-A3B TP2 rank shapes, on a cc 12.0 target whose header attests `TritonExperts`.

It checks that the scope claims every event, binds both quants as `Fp8GroupQuant_v1{K,64}` and both launches as the block-FP8 pair forms
with the target's e4m3 step, with operand rows the Definitions' parameters fit; and that it refuses, by name, a missing attestation, a
bf16 launch, another block, another quant group, column-major or 128-grid scales, the fused SiLU-quant, a missing quant, and a target
with no e4m3 step.
"""
from __future__ import annotations

import dataclasses
import importlib
import math

import pytest

from verity_vllm.engine.profiles import generic
from verity_vllm.engine.profiles.gen_ov_moe_patterns import (FUSED_MOE_PINS, MOE_FP8_CONSTRUCTION_PATTERNS, TRITON_EXPERTS, MoEFusedExperts,
                                                             MoEFusedExpertsFp8, MoEFusedExpertsFp8Ordered)
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, TritonLaunch, ViewDesc
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Shared, StepContext, Unsupported, resolve_step
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.frontend import targets as T

Q3 = importlib.import_module("verity_vllm.engine.profiles.derived_QWEN3_30B_A3B_tp2")
C = Q3.C
H, I, E, TOPK = C["H"], C["I"], C["NUM_EXPERTS"], C["TOPK"]
G = 64
CUDA = "cuda:0"
E4M3 = "float8_e4m3fn"
LAYER = 5
A_W13, A_W2, A_W13S, A_W2S = 7, 8, 17, 18
SM120 = TargetProfile(compute_capability=(12, 0), num_sms=188, moe_backend="triton")
FN = "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel"
EMPTY = ("aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, "
         "MemoryFormat? memory_format=None) -> Tensor")
QUANT = ("_C::per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! output_s, int group_size, float eps, float fp8_min, "
         "float fp8_max, bool scale_ue8m0, bool column_major_scales, bool tma_aligned) -> ()")
CONSTS = {**FUSED_MOE_PINS, "SPLIT_K": 1, "naive_block_assignment": False, "MUL_ROUTED_WEIGHT": False, "top_k": TOPK, "compute_type": "bf16",
          "use_fp8_w8a8": True, "use_int8_w8a8": False, "use_int8_w8a16": False, "per_channel_quant": False, "HAS_BIAS": False,
          "SWAP_AB": False, "USE_TD": False, "group_n": G, "group_k": G}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _contig(shape):
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return tuple(reversed(out))


def view(alloc, shape, dtype="bfloat16", strides=None, offset=0):
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides or _contig(shape)), dtype, CUDA)


def scale_shapes(g: int = G) -> dict[str, tuple[int, int, int]]:
    return {"w13": (E, cdiv(2 * I, g), cdiv(H, g)), "w2": (E, cdiv(H, g), cdiv(I, g))}


def fp8_scope_events(M: int, *, naive: bool = False, consts_up: dict | None = None, quant_group: int = G, x_s_strides=None,
                     w13_scale_shape=None, input_quant: bool = True, fused_silu_quant: bool = False):
    """One `vllm.moe_forward` scope of layer LAYER at the rank shapes with block-FP8 experts (cf. test_gen_ov_moe_qwen3.moe_scope_events)."""
    nid = [1000]

    def new():
        nid[0] += 1
        return nid[0]

    hidden, logits, out, received = view(295, (M, H)), view(296, (M, E)), view(300, (M, H)), view(305, (M, H))
    evs = []
    scope = new()

    def op(name, schema, args, outs=None):
        i = new()
        evs.append(DispatchEnter(i, scope, 0, "forward", name, schema, list(args), {}, 7, None))
        evs.append(DispatchExit(i, outs))

    def quant(x, xq, xs, group):
        op("aten.empty.memory_format", EMPTY, [list(xq.shape), None, None, None, None, None], xq)
        op("aten.empty.memory_format", EMPTY, [list(xs.shape), None, None, None, None, None], xs)
        op("_C.per_token_group_fp8_quant.default", QUANT, [x, xq, xs, group, 1e-10, -448.0, 448.0, False, False, False])

    evs.append(DispatchEnter(scope, None, 0, "forward", "vllm.moe_forward.default",
                             "vllm::moe_forward(Tensor(a0!) hidden_states, Tensor router_logits, Tensor? shared_experts_input, Tensor? input_ids, PyObject layer_name, SymInt hidden_dim_unpadded) -> Tensor",
                             [hidden, logits, None, None, "layer", H], {}, 7, f"model.layers.{LAYER}.mlp.experts"))
    w_, ids, tok = view(297, (M, TOPK), "float32"), view(298, (M, TOPK), "int32"), view(299, (M, TOPK), "int32")
    for v in (w_, ids, tok):
        op("aten.empty.memory_format", EMPTY, [list(v.shape), None, None, None, None, None], v)
    op("_moe_C.topk_softmax.default",
       "_moe_C::topk_softmax(Tensor($0! -> ) topk_weights, Tensor($1! -> ) topk_indices, Tensor($2! -> ) token_expert_indices, Tensor gating_output, bool renormalize, Tensor? bias, Tensor? is_padding) -> ()",
       [w_, ids, tok, logits, True, None, view(259, (M,), "bool")])
    op("aten.empty_like.default", "aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
       [hidden, None, None, None, None, None], out)
    P = M * TOPK
    xq, xs = view(310, (M, H), E4M3), view(311, (M, H // G), "float32", strides=x_s_strides)
    if input_quant:
        quant(hidden, xq, xs, quant_group)
    if naive:
        sorted_ids, expert_ids, npad, EM = None, view(298, (P,), "int32"), view(304, (1,), "int32"), P * 64
    else:
        EM = min(P * 64, P + E * 63)
        sorted_ids, expert_ids, npad = view(302, (EM,), "int32"), view(303, (EM // 64,), "int32"), view(304, (1,), "int32")
        op("_moe_C.moe_align_block_size.default",
           "_moe_C::moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor($0! -> ) sorted_token_ids, Tensor($1! -> ) experts_ids, Tensor($2! -> ) num_tokens_post_pad, Tensor? maybe_expert_map) -> ()",
           [ids, E, 64, sorted_ids, expert_ids, npad, None])
    cache1, cache2, cache3 = view(301, (M, TOPK, 2 * I), offset=P * I), view(301, (P, I)), view(301, (M, TOPK, H), offset=P * I)
    w13, w2 = view(A_W13, (E, 2 * I, H), E4M3), view(A_W2, (E, H, I), E4M3)
    w13s, w2s = view(A_W13S, w13_scale_shape or scale_shapes()["w13"], "float32"), view(A_W2S, scale_shapes()["w2"], "float32")

    def launch(a, a_s, b, b_s, c, weights, N, K, consts):
        args = {"a_ptr": a, "b_ptr": b, "c_ptr": c, "b_bias_ptr": None, "a_scale_ptr": a_s, "b_scale_ptr": b_s, "topk_weights_ptr": weights,
                "sorted_token_ids_ptr": sorted_ids, "expert_ids_ptr": expert_ids, "num_tokens_post_padded_ptr": npad, "N": N, "K": K, "EM": EM,
                "num_valid_tokens": P, "stride_am": a.strides[0], "stride_ak": 1, "stride_be": N * K, "stride_bk": 1, "stride_bn": K,
                "stride_cm": N, "stride_cn": 1, "stride_asm": a_s.strides[0], "stride_ask": a_s.strides[1], "stride_bse": b_s.strides[0],
                "stride_bsk": b_s.strides[2], "stride_bsn": b_s.strides[1], **consts}
        evs.append(TritonLaunch(new(), scope, 0, "forward", FN, "sha", args, consts, [(EM // 64) * (N // 64)], 4, 3, 7))

    c_up = dict(CONSTS, naive_block_assignment=naive, **(consts_up or {}))
    launch(xq, xs, w13, w13s, cache1, None, 2 * I, H, c_up)
    hq, hs = view(312, (P, I), E4M3), view(313, (P, I // G), "float32")
    if fused_silu_quant:
        op("aten.empty.memory_format", EMPTY, [list(hq.shape), None, None, None, None, None], hq)
        op("_C.silu_and_mul_per_block_quant.default",
           "_C::silu_and_mul_per_block_quant(Tensor! out, Tensor input, Tensor! scales, int group_size, Tensor? scale_ub, bool is_scale_transposed) -> ()",
           [hq, view(301, (P, 2 * I), offset=P * I), hs, G, None, False])
    else:
        op("_C.silu_and_mul.default", "_C::silu_and_mul(Tensor($0! -> ) result, Tensor input) -> ()", [cache2, view(301, (P, 2 * I), offset=P * I)])
        quant(cache2, hq, hs, quant_group)
    launch(hq, hs, w2, w2s, cache3, w_, H, I, dict(CONSTS, naive_block_assignment=naive, MUL_ROUTED_WEIGHT=True, top_k=1))
    op("_moe_C.moe_sum.default", "_moe_C::moe_sum(Tensor input, Tensor($0! -> ) output, Tensor? topk_ids=None, Tensor? expert_map=None) -> ()",
       [cache3, out, None, None])
    i = new()
    evs.append(DispatchEnter(i, scope, 0, "forward", "vllm.all_reduce.default", "vllm::all_reduce(Tensor tensor, str group_name) -> Tensor",
                             [out, "tp:0"], {}, 7, f"model.layers.{LAYER}.mlp.experts"))
    evs.append(DispatchExit(i, received))
    evs.append(DispatchExit(scope, received))
    return evs


def fp8_profile(cls=MoEFusedExpertsFp8Ordered, group: int = G):
    """The rank profile with its MoE scope replaced by the block-FP8 one, every other field of the scope kept."""
    old = next(p for p in Q3.PROFILE.patterns if isinstance(p, MoEFusedExperts))
    new = cls(**{f.name: getattr(old, f.name) for f in dataclasses.fields(old)}, group=group)
    return dataclasses.replace(Q3.PROFILE, patterns=[new if p is old else p for p in Q3.PROFILE.patterns])


def _ctx(start: StepStart, *, target: TargetProfile | None = SM120, experts: str | None = TRITON_EXPERTS, profile=None) -> StepContext:
    prof = {"target": target.to_json()} if target is not None else {}
    if experts is not None:
        prof["backends"] = {"moe_experts_class": experts}
    header = RunHeader("verity-capture/log/v1", "2026-10-03T00:00:00Z", "x", prof, {"repo": Q3.MODEL, "config": {}}, {"requests": []})
    sh = scale_shapes()
    pre = f"model.layers.{LAYER}.mlp.experts.routed_experts"
    roots = {A_W13: [Root(view(A_W13, (E, 2 * I, H), E4M3), "weight", f"{pre}.w13_weight")],
             A_W2: [Root(view(A_W2, (E, H, I), E4M3), "weight", f"{pre}.w2_weight")],
             A_W13S: [Root(view(A_W13S, sh["w13"], "float32"), "weight", f"{pre}.w13_weight_scale_inv")],
             A_W2S: [Root(view(A_W2S, sh["w2"], "float32"), "weight", f"{pre}.w2_weight_scale_inv")]}
    return StepContext(header, start, None, [], roots, C, profile or fp8_profile())


def _resolve(M: int, *, ctx_kw: dict | None = None, **kw):
    start = StepStart(0, [("r0", p) for p in range(M)], list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0, 1, 2, 3]}, 16,
                      list(range(M)), [("r0", M - 1)], [M - 1], 1, M, M, None, None, "prefill")
    se = step_events([start, *fp8_scope_events(M, **kw)])
    return resolve_step(se, _ctx(start, **(ctx_kw or {}))), se


def _fits(c: Compute) -> bool:
    """Every operand's row has its parameter's leaf count (the fold's read of each row)."""
    params, _ = c.spec.signature()
    rows = [op.view.numel if isinstance(op, Shared) else math.prod(op.view.shape[1:]) for op in c.operands]
    return len(params) == len(rows) and all(t.leaves == n for (_, t), n in zip(params, rows))


@pytest.mark.parametrize("M,naive", [(2, True), (3, False), (32, False)])
def test_block_fp8_scope_folds_both_quants_and_both_pair_forms_on_the_target_s_e4m3_step(M, naive):
    res, se = _resolve(M, naive=naive)
    assert len(res) == 1 and isinstance(res[0], Group), res
    g = res[0]
    assert g.consumed == tuple(sorted(se.roots[0].ids())) and g.pattern == "moe.fused_experts.pairwise.fp8"
    computes = [p for p in g.parts if isinstance(p, Compute)]
    dot = T.moe_expert_fp8_dot(SM120).id
    assert [c.spec.id for c in computes] == [
        f"MoeRouterTopKOrderedNorm_v1{{E={E},TOPK={TOPK},VPT=8}}",
        f"Fp8GroupQuant_v1{{K={H},G={G}}}", f"MoeExpertFp8Gemm_v1{{E={E},K={H},N={2 * I},G={G},KB=32,SWAP=False,DOT={dot}}}",
        f"SiluMul_v2{{I={I}}}",
        f"Fp8GroupQuant_v1{{K={I},G={G}}}", f"MoeExpertFp8GemmW_v1{{E={E},K={I},N={H},G={G},KB=32,SWAP=False,DOT={dot}}}",
        f"MoeSum_v1{{TOPK={TOPK},H={H}}}", f"AllReduce2_v1{{N={H}}}"]
    P = M * TOPK
    assert [c.rows for c in computes] == [M, M, P, P, P, P, M, M]
    router, q1, up, silu, q2, down, msum, _ar = computes
    assert all(_fits(c) for c in (q1, up, q2, down))
    # the input quant reads hidden_states; the w13 launch gathers its token's quantised row and group scales
    assert q1.operands[0].view.alloc == 295 and q1.attribution == tuple(("r0", p) for p in range(M))
    assert [type(o) for o in up.operands] == [Gather, Gather, PerRow, Shared, Shared]
    assert up.operands[0].indices == up.operands[1].indices == tuple(p // TOPK for p in range(P))
    assert (up.operands[0].view, up.operands[1].view) == (q1.outputs[0].view, q1.outputs[1].view)
    assert up.operands[2].view.alloc == router.outputs[1].view.alloc
    assert up.operands[3].view.dtype == E4M3 and up.operands[4].view.shape == scale_shapes()["w13"]
    # the SiLU output's quant on the pair rows feeds the w2 launch with the router's weights
    assert q2.operands[0].view == silu.outputs[0].view and q2.attribution == ()
    assert [type(o) for o in down.operands] == [PerRow, PerRow, PerRow, PerRow, Shared, Shared]
    assert (down.operands[0].view, down.operands[1].view) == (q2.outputs[0].view, q2.outputs[1].view)
    assert down.operands[3].view.alloc == router.outputs[0].view.alloc and down.operands[5].view.shape == scale_shapes()["w2"]
    pre = f"model.layers.{LAYER}.mlp.experts.routed_experts"
    assert up.roots == (None, None, None, f"{pre}.w13_weight", f"{pre}.w13_weight_scale_inv")
    assert down.roots == (None, None, None, None, f"{pre}.w2_weight", f"{pre}.w2_weight_scale_inv")
    assert (msum.operands[0].view.alloc, msum.operands[0].view.offset) == (down.outputs[0].view.alloc, down.outputs[0].view.offset)
    kernels = [k for p in g.parts for k in p.kernels]
    assert kernels.count("per_token_group_quant_8bit_kernel") == 2 and kernels.count("fused_moe_kernel") == 2


@pytest.mark.parametrize("kw,ctx_kw,needle", [
    ({}, {"experts": None}, f"does not attest {TRITON_EXPERTS}"),
    ({}, {"experts": "CutlassExpertsFp8"}, f"does not attest {TRITON_EXPERTS}"),
    ({"consts_up": {"use_fp8_w8a8": False}}, {}, "differ from the modelled block-FP8 kernel"),
    ({"consts_up": {"group_n": 128, "group_k": 128}}, {}, "differ from the modelled block-FP8 kernel"),
    ({"consts_up": {"BLOCK_SIZE_K": 64}}, {}, "tiling differs"),
    ({"quant_group": 128}, {}, "did not resolve as Fp8GroupQuant_v1"),
    ({"x_s_strides": (1, 3)}, {}, "a_scale is not the row-major f32"),
    ({"w13_scale_shape": scale_shapes(128)["w13"]}, {}, "b_scale is not the contiguous f32"),
    ({"fused_silu_quant": True}, {}, "fused_experts shape"),
    ({"input_quant": False}, {}, "two per_token_group_fp8_quant calls"),
    ({}, {"target": TargetProfile(compute_capability=(9, 0), num_sms=132, moe_backend="triton")}, "no registered e4m3 step"),
    ({}, {"target": TargetProfile(compute_capability=(8, 9), num_sms=128, moe_backend="triton")}, "no registered e4m3 step"),
])
def test_block_fp8_scope_refuses_by_name(kw, ctx_kw, needle):
    res, _ = _resolve(3, ctx_kw=ctx_kw, **kw)
    bad = [r for r in res if isinstance(r, Unsupported)]
    assert bad and needle in bad[0].reason, [getattr(r, "reason", r) for r in res]


def test_the_bf16_scope_does_not_claim_a_block_fp8_scope():
    res, _ = _resolve(3, ctx_kw={"profile": Q3.PROFILE})
    assert [r for r in res if isinstance(r, Unsupported)] and not [r for r in res if isinstance(r, Group)]


def test_the_block_fp8_constructions_are_the_pairwise_ones():
    assert MOE_FP8_CONSTRUCTION_PATTERNS == {"indexed-read": MoEFusedExpertsFp8, "indexed-read-ordered": MoEFusedExpertsFp8Ordered}
    res, _ = _resolve(2, naive=True, ctx_kw={"profile": fp8_profile(MoEFusedExpertsFp8)})
    assert isinstance(res[0], Group) and [p.spec.name for p in res[0].parts if isinstance(p, Compute)][0] == "MoeRouterTopKNorm"


@pytest.mark.parametrize("world,construction,cls,g", [(1, None, MoEFusedExpertsFp8Ordered, 128), (2, "indexed-read", MoEFusedExpertsFp8, 128),
                                                      (4, "indexed-read-ordered", MoEFusedExpertsFp8Ordered, 64)])
def test_a_block_fp8_moe_profile_folds_with_the_fp8_scope_on_the_rank_s_scale_grid(world, construction, cls, g):
    """`profiles.generic` gives Qwen3-30B-A3B-FP8 the block-FP8 scope of its declared construction, its group the rank's scale grid
    (`model.fp8_moe_block`: the TP4 rank's 192 expert columns refine 128 to 64), and pins the launch's FP8 arguments in its notes."""
    prof = generic.build_profile("QWEN3_30B_A3B_FP8", world, moe_construction=construction)
    moe = [p for p in prof.patterns if isinstance(p, MoEFusedExperts)]
    assert [type(p) for p in moe] == [cls] and moe[0].group == g and moe[0].name == "moe.fused_experts.pairwise.fp8"
    Ir, Hr = prof.config["I"], prof.config["H"]
    assert {"use_fp8_w8a8": True, "group_n": g, "group_k": g}.items() <= prof.notes["moe"]["pins"]["fused_moe_kernel"].items()
    defs = prof.notes["moe"]["definitions"]
    assert f"Fp8GroupQuant_v1{{K={Hr},G={g}}}" in defs and f"Fp8GroupQuant_v1{{K={Ir},G={g}}}" in defs
    assert any(d.startswith(f"MoeExpertFp8GemmW_v1{{E=128,K={Ir},N={Hr},G={g},KB=32,") for d in defs)


def test_a_block_fp8_moe_profile_refuses_the_padded_construction_and_a_bf16_one_keeps_the_bf16_scope():
    with pytest.raises(ValueError, match=r"'padded' is not one of \['indexed-read', 'indexed-read-ordered'\] \(block-FP8 experts\)"):
        generic.build_profile("QWEN3_30B_A3B_FP8", 1, moe_construction="padded")
    bf16 = generic.build_profile("QWEN3_30B_A3B", 1)
    assert [type(p).__name__ for p in bf16.patterns if isinstance(p, MoEFusedExperts)] == ["MoEFusedExpertsOrdered"]
    assert bf16.notes["moe"]["pins"]["fused_moe_kernel"]["use_fp8_w8a8"] is False
