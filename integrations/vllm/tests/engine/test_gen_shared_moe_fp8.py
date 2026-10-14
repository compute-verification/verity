"""The block-FP8 shared-expert MoE scope (`gen_shared_moe_patterns.MoEFusedSharedExpertsFp8`) on synthetic `vllm.moe_forward_shared`
events shaped as vLLM d9105ea80 issues them for GLM-5.3 at a TP8 rank on the H200 (H 6144, routed and shared intermediate 256 per rank,
256 experts, top 8, block 128, routed scaling 2.5), without batch invariance, under a header that attests `TritonExperts` and the
cutlass block-FP8 linears.

It checks that the scope claims every event and binds the gate (both tiers, through the profile's `gate_spec`), the shared expert's
two quantised linears and SiLU, the sigmoid router, the routed input quant, both pair forms on Hopper's tuned wgmma tiles, the fused
SiLU quant and `moe_sum`, with operand rows the Definitions' parameters fit, and that the combine after the op is `ResidualAdd_v1`;
and that it refuses, by name, a missing gate Definition, an unattested cuteDSL gate, a missing attestation, a non-wgmma tile, another
router, another block, the unfused SiLU quant and shared experts that read another input; and that `run_facts` attests the cuteDSL
gate tier only when a gate is eligible for it.
"""
from __future__ import annotations

import dataclasses
import importlib
import math

import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity_vllm.engine.profiles.gen_ln_patterns import ResidualAddPattern
from verity_vllm.engine.profiles.gen_ov_moe_patterns import TRITON_EXPERTS, MoEFusedExpertsFp8
from verity_vllm.engine.profiles.gen_shared_moe_patterns import GATE_CUBLAS, GATE_CUTEDSL, MoEFusedSharedExpertsFp8
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, TritonLaunch, ViewDesc
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Shared, StepContext, Transparent, Unsupported, resolve_step
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.program.frontend import targets as T
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import moe as MOE
from verity_vllm.program.registry.prims import BF16, F32

Q3 = importlib.import_module("verity_vllm.engine.profiles.derived_QWEN3_30B_A3B_tp2")
H, I, SI, E, TOPK, G, RSF = 6144, 256, 256, 256, 8, 128, 2.5
C = {"NUM_EXPERTS": E, "TOPK": TOPK, "H": H, "I": I, "SHARED_I": SI, "ROUTED_SCALING_FACTOR": RSF}
CUDA = "cuda:0"
E4M3 = "float8_e4m3fn"
LAYER = 5
PRE = f"model.layers.{LAYER}.mlp"
H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, moe_backend="triton", batch_invariant=False)
#: the H200's tuned `fused_moe_kernel` configs at E=256, N=256, fp8_w8a8, block [128, 128] (BLOCK_SIZE_M/N/K)
TUNED = {1: (8, 32, 128), 2: (8, 32, 128), 4: (8, 64, 128), 8: (16, 64, 128), 32: (16, 128, 128), 1024: (64, 128, 128)}
A = {"gate_w": 7, "bias": 8, "sgu": 9, "sgu_s": 10, "sd": 11, "sd_s": 12, "w13": 13, "w13_s": 14, "w2": 15, "w2_s": 16}
FN = "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel"
EMPTY = ("aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, "
         "MemoryFormat? memory_format=None) -> Tensor")
QUANT = ("_C::per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! output_s, int group_size, float eps, float fp8_min, "
         "float fp8_max, bool scale_ue8m0, bool column_major_scales, bool tma_aligned) -> ()")
SCALED_MM = "_C::cutlass_scaled_mm(Tensor! out, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()"
SILU = "_C::silu_and_mul(Tensor($0! -> ) result, Tensor input) -> ()"
SILU_QUANT = ("_C::silu_and_mul_per_block_quant(Tensor! out, Tensor input, Tensor! scales, int group_size, Tensor? scale_ub, "
              "bool is_scale_transposed) -> ()")
ROUTER = ("_moe_C::grouped_topk(Tensor scores, int n_group, int topk_group, int topk, bool renormalize, float routed_scaling_factor, "
          "Tensor bias, int scoring_func=0) -> (Tensor, Tensor)")
SCOPE = ("vllm::moe_forward_shared(Tensor hidden_states, Tensor router_logits, Tensor? shared_experts_input, Tensor? input_ids, "
         "PyObject layer_name, SymInt hidden_dim_unpadded) -> (Tensor, Tensor)")
CONSTS = {"SPLIT_K": 1, "MUL_ROUTED_WEIGHT": False, "top_k": TOPK, "compute_type": "bf16", "use_fp8_w8a8": True, "use_int8_w8a8": False,
          "use_int8_w8a16": False, "per_channel_quant": False, "HAS_BIAS": False, "USE_TD": False, "group_n": G, "group_k": G}

GateStandIn = CompositeDefinition("GateStandIn", 1, ("K", "N", "TIER"),
                                  lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, F32)),
                                  lambda B, S, x, w: None, register=False)


def gate_spec(K, N, M, tier):
    return bind(GateStandIn, K=K, N=N, TIER=tier), (f"{tier}_gate_kernel",)


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


def weights() -> dict[str, ViewDesc]:
    return {"gate_w": view(A["gate_w"], (E, H)), "bias": view(A["bias"], (E,), "float32"),
            "sgu": view(A["sgu"], (2 * SI, H), E4M3), "sgu_s": view(A["sgu_s"], (2 * SI // G, H // G), "float32"),
            "sd": view(A["sd"], (H, SI), E4M3), "sd_s": view(A["sd_s"], (H // G, SI // G), "float32"),
            "w13": view(A["w13"], (E, 2 * I, H), E4M3), "w13_s": view(A["w13_s"], (E, cdiv(2 * I, G), H // G), "float32"),
            "w2": view(A["w2"], (E, H, I), E4M3), "w2_s": view(A["w2_s"], (E, H // G, I // G), "float32")}


NAMES = {"gate_w": f"{PRE}.gate.weight", "bias": f"{PRE}.gate.e_score_correction_bias",
         "sgu": f"{PRE}.shared_experts.gate_up_proj.weight", "sgu_s": f"{PRE}.shared_experts.gate_up_proj.weight_scale_inv",
         "sd": f"{PRE}.shared_experts.down_proj.weight", "sd_s": f"{PRE}.shared_experts.down_proj.weight_scale_inv",
         "w13": f"{PRE}.experts.w13_weight", "w13_s": f"{PRE}.experts.w13_weight_scale_inv",
         "w2": f"{PRE}.experts.w2_weight", "w2_s": f"{PRE}.experts.w2_weight_scale_inv"}


def scope_events(M: int, *, gate: str = GATE_CUBLAS, tile=None, router_args=None, fused_silu_quant: bool = True,
                 shared_input=None, quant_group: int = G):
    """One `vllm.moe_forward_shared` of layer LAYER, then the combine `aten.add.Tensor(shared, routed)` outside it."""
    nid = [1000]

    def new():
        nid[0] += 1
        return nid[0]

    W = weights()
    hidden = view(295, (M, H))
    shared_out, routed_out, y = view(330, (M, H)), view(300, (M, H)), view(340, (M, H))
    evs = []
    scope = new()

    def op(name, schema, args, outs=None, parent=None):
        i = new()
        evs.append(DispatchEnter(i, scope if parent is None else parent, 0, "forward", name, schema, list(args), {}, 7, None))
        evs.append(DispatchExit(i, outs))
        return i

    def empty(v):
        op("aten.empty.memory_format", EMPTY, [list(v.shape), None, None, None, None, None], v)

    def quant(x, xq, xs, column_major):
        empty(xq)
        empty(xs)
        op("_C.per_token_group_fp8_quant.default", QUANT, [x, xq, xs, quant_group, 1e-10, -448.0, 448.0, False, column_major, False])

    def scaled_mm(out, xq, xs, w, ws):
        K, N = xq.shape[1], out.shape[1]
        empty(out)
        op("_C.cutlass_scaled_mm.default", SCALED_MM, [out, xq, view(w.alloc, (K, N), E4M3, strides=(1, K)), xs,
                                                       view(ws.alloc, (K // G, cdiv(N, G)), "float32", strides=(1, K // G)), None])

    sx = shared_input or hidden
    evs.append(DispatchEnter(scope, None, 0, "forward", "vllm.moe_forward_shared.default", SCOPE, [hidden, hidden, sx, None, "layer", 0], {}, 7,
                             f"{PRE}.experts"))
    scores = view(320, (M, E), "float32")
    if gate == GATE_CUBLAS:
        op("aten.mm.dtype", "aten::mm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor",
           [hidden, view(A["gate_w"], (H, E), strides=(1, H)), "torch.float32"], scores)
    else:
        empty(scores)
    # the shared expert
    xq1, xs1 = view(321, (M, H), E4M3), view(322, (M, H // G), "float32", strides=(1, M))
    quant(sx, xq1, xs1, True)
    gu = view(323, (M, 2 * SI))
    scaled_mm(gu, xq1, xs1, W["sgu"], W["sgu_s"])
    h = view(324, (M, SI))
    empty(h)
    op("_C.silu_and_mul.default", SILU, [h, gu])
    xq2, xs2 = view(325, (M, SI), E4M3), view(326, (M, SI // G), "float32", strides=(1, M))
    quant(h, xq2, xs2, True)
    scaled_mm(shared_out, xq2, xs2, W["sd"], W["sd_s"])
    # the router
    w_, ids = view(297, (M, TOPK), "float32"), view(298, (M, TOPK), "int32")
    ra = {"n_group": 1, "topk_group": 1, "topk": TOPK, "renormalize": True, "routed_scaling_factor": RSF, "scoring_func": 1, **(router_args or {})}
    op("_moe_C.grouped_topk.default", ROUTER, [scores, ra["n_group"], ra["topk_group"], ra["topk"], ra["renormalize"], ra["routed_scaling_factor"],
                                              W["bias"], ra["scoring_func"]], [w_, ids])
    # the routed experts
    P = M * TOPK
    xq, xs = view(310, (M, H), E4M3), view(311, (M, H // G), "float32")
    quant(hidden, xq, xs, False)
    bm, bn, bk = tile or TUNED[min(TUNED, key=lambda k: abs(k - M))]
    naive = P <= E // 4
    if naive:
        sorted_ids, expert_ids, npad, EM = None, view(298, (P,), "int32"), view(304, (1,), "int32"), P * bm
    else:
        EM = min(P * bm, P + E * (bm - 1))
        sorted_ids, expert_ids, npad = view(302, (EM,), "int32"), view(303, (cdiv(EM, bm),), "int32"), view(304, (1,), "int32")
        op("_moe_C.moe_align_block_size.default",
           "_moe_C::moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor($0! -> ) sorted_token_ids, Tensor($1! -> ) "
           "experts_ids, Tensor($2! -> ) num_tokens_post_pad, Tensor? maybe_expert_map) -> ()", [ids, E, bm, sorted_ids, expert_ids, npad, None])
    cache1, cache3 = view(301, (M, TOPK, 2 * I), offset=P * I), view(301, (M, TOPK, H), offset=P * I)
    tile_c = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 1, "SWAP_AB": bm < 64 and bn >= 64,
              "naive_block_assignment": naive}

    def launch(a, a_s, b, b_s, c, wts, N, K, consts):
        args = {"a_ptr": a, "b_ptr": b, "c_ptr": c, "b_bias_ptr": None, "a_scale_ptr": a_s, "b_scale_ptr": b_s, "topk_weights_ptr": wts,
                "sorted_token_ids_ptr": sorted_ids, "expert_ids_ptr": expert_ids, "num_tokens_post_padded_ptr": npad, "N": N, "K": K, "EM": EM,
                "num_valid_tokens": P, "stride_am": a.strides[0], "stride_ak": 1, "stride_be": N * K, "stride_bk": 1, "stride_bn": K,
                "stride_cm": N, "stride_cn": 1, "stride_asm": a_s.strides[0], "stride_ask": a_s.strides[1], "stride_bse": b_s.strides[0],
                "stride_bsk": b_s.strides[2], "stride_bsn": b_s.strides[1], **consts}
        evs.append(TritonLaunch(new(), scope, 0, "forward", FN, "sha", args, consts, [cdiv(EM, bm) * cdiv(N, bn)], 4, 4, 7))

    launch(xq, xs, W["w13"], W["w13_s"], cache1, None, 2 * I, H, dict(CONSTS, **tile_c))
    hq, hs = view(312, (P, I), E4M3), view(313, (P, I // G), "float32")
    empty(hq)
    empty(hs)
    if fused_silu_quant:
        op("_C.silu_and_mul_per_block_quant.default", SILU_QUANT, [hq, view(301, (P, 2 * I), offset=P * I), hs, G, None, False])
    else:
        cache2 = view(301, (P, I))
        op("_C.silu_and_mul.default", SILU, [cache2, view(301, (P, 2 * I), offset=P * I)])
        quant(cache2, hq, hs, False)
    launch(hq, hs, W["w2"], W["w2_s"], cache3, w_, H, I, dict(CONSTS, **tile_c, MUL_ROUTED_WEIGHT=True, top_k=1))
    empty(routed_out)
    op("_moe_C.moe_sum.default", "_moe_C::moe_sum(Tensor input, Tensor($0! -> ) output, Tensor? topk_ids=None, Tensor? expert_map=None) -> ()",
       [cache3, routed_out, None, None])
    evs.append(DispatchExit(scope, [shared_out, routed_out]))
    i = new()
    evs.append(DispatchEnter(i, None, 0, "forward", "aten.add.Tensor", "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
                             [shared_out, routed_out], {}, 7, f"{PRE}"))
    evs.append(DispatchExit(i, y))
    return evs


def scope(**kw) -> MoEFusedSharedExpertsFp8:
    return MoEFusedSharedExpertsFp8(C, group=G, **{"gate_spec": gate_spec, **kw})


def _ctx(start, *, target=H200, experts=TRITON_EXPERTS, fp8_kernel="CutlassFp8BlockScaledMMKernel", ll_bf16=None, pattern=None):
    prof = {"target": target.to_json(), "backends": {}}
    if experts is not None:
        prof["backends"]["moe_experts_class"] = experts
    if fp8_kernel is not None:
        prof["backends"]["fp8_block_gemm_kernel"] = fp8_kernel
    if ll_bf16 is not None:
        prof["backends"]["moe_gate_ll_bf16_gemm"] = ll_bf16
    header = RunHeader("verity-capture/log/v1", "2026-10-12T00:00:00Z", "x", prof, {"repo": "zai-org/GLM-5.3-FP8", "config": {}}, {"requests": []})
    roots = {v.alloc: [Root(v, "weight", NAMES[k])] for k, v in weights().items()}
    profile = dataclasses.replace(Q3.PROFILE, config=C, patterns=[pattern or scope(), ResidualAddPattern()])
    return StepContext(header, start, None, [], roots, C, profile)


def _resolve(M: int, *, ctx_kw: dict | None = None, **kw):
    start = StepStart(0, [("r0", p) for p in range(M)], list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0, 1, 2, 3]}, 16,
                      list(range(M)), [("r0", M - 1)], [M - 1], 1, M, M, None, None, "prefill")
    se = step_events([start, *scope_events(M, **kw)])
    return resolve_step(se, _ctx(start, **(ctx_kw or {}))), se


def _fits(c: Compute) -> bool:
    """Every operand's row has its parameter's leaf count (the fold's read of each row)."""
    params, _ = c.spec.signature()
    rows = [op.view.numel if isinstance(op, Shared) else math.prod(op.view.shape[1:]) for op in c.operands]
    return len(params) == len(rows) and all(t.leaves == n for (_, t), n in zip(params, rows))


def _reasons(res) -> list:
    return [getattr(r, "reason", r) for r in res if isinstance(r, Unsupported)]


@pytest.mark.parametrize("M,gate", [(4, GATE_CUBLAS), (32, GATE_CUBLAS), (4, GATE_CUTEDSL), (16, GATE_CUTEDSL)])
def test_the_shared_scope_folds_the_gate_the_shared_expert_the_router_and_the_routed_pairs_on_the_tuned_wgmma_tiles(M, gate):
    res, se = _resolve(M, gate=gate, ctx_kw={"ll_bf16": True} if gate == GATE_CUTEDSL else None)
    assert len(res) == 2 and isinstance(res[0], Group), _reasons(res)
    g = res[0]
    assert g.consumed == tuple(sorted(se.roots[0].ids())) and g.pattern == "moe.fused_shared_experts.pairwise.fp8"
    computes = [p for p in g.parts if isinstance(p, Compute)]
    bm, bn, kb = TUNED[min(TUNED, key=lambda k: abs(k - M))]
    fp8 = f"G={G},KB={kb},SWAP={bm < 64 and bn >= 64},DOT=HopperE4m3QgmmaDot32_v1"
    assert [c.spec.id for c in computes] == [
        bind(GateStandIn, K=H, N=E, TIER=gate).id,
        f"Fp8GroupQuant_v1{{K={H},G={G}}}", T.scaled_mm_fp8_block_spec(H200, H, 2 * SI, G).id, f"SiluMul_v2{{I={SI}}}",
        f"Fp8GroupQuant_v1{{K={SI},G={G}}}", T.scaled_mm_fp8_block_spec(H200, SI, H, G).id,
        bind(MOE.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=RSF, KERNEL="block").id,
        f"Fp8GroupQuant_v1{{K={H},G={G}}}", f"MoeExpertFp8Gemm_v1{{E={E},K={H},N={2 * I},{fp8}}}", f"SiluMulFp8BlockQuant_v1{{I={I},G={G}}}",
        f"MoeExpertFp8GemmW_v1{{E={E},K={I},N={H},{fp8}}}", f"MoeSum_v1{{TOPK={TOPK},H={H}}}"]
    P = M * TOPK
    assert [c.rows for c in computes] == [M] * 8 + [P, P, P, M]
    assert all(_fits(c) for c in computes)
    gc, q1, m1, silu, q2, m2, router, qr, up, sq, down, msum = computes
    # the gate reads hidden_states and the gate weight, and writes the scores the router reads
    assert gc.operands[0].view.alloc == 295 and gc.operands[1].view == weights()["gate_w"] and gc.roots == (None, f"{PRE}.gate.weight")
    assert router.operands[0].view == gc.outputs[0].view and router.roots == (None, f"{PRE}.gate.e_score_correction_bias")
    assert gc.kernels == (f"{gate}_gate_kernel",) and router.kernels == ("single_group_topk_block_kernel",)
    # the shared expert: hidden -> quant -> gate_up -> SiLU -> quant -> down, into the op's first output
    assert q1.operands[0].view.alloc == 295 and m1.operands[0].view == q1.outputs[0].view and silu.operands[0].view == m1.outputs[0].view
    assert q2.operands[0].view == silu.outputs[0].view and m2.operands[0].view == q2.outputs[0].view and m2.outputs[0].view.alloc == 330
    assert m1.roots[2:] == (NAMES["sgu"], NAMES["sgu_s"]) and m2.roots[2:] == (NAMES["sd"], NAMES["sd_s"])
    # the routed pairs: w13 gathers its token's quantised row; the fused SiLU quant writes w2's A and scales
    assert qr.operands[0].view.alloc == 295 and [type(o) for o in up.operands] == [Gather, Gather, PerRow, Shared, Shared]
    assert up.operands[0].indices == tuple(p // TOPK for p in range(P)) and up.operands[2].view.alloc == router.outputs[1].view.alloc
    assert sq.operands[0].view.shape == (P, 2 * I) and (sq.operands[0].view.alloc, sq.operands[0].view.offset) == (301, P * I)
    assert (down.operands[0].view, down.operands[1].view) == (sq.outputs[0].view, sq.outputs[1].view)
    assert down.operands[3].view.alloc == router.outputs[0].view.alloc
    assert up.roots[3:] == (NAMES["w13"], NAMES["w13_s"]) and down.roots[4:] == (NAMES["w2"], NAMES["w2_s"])
    assert msum.outputs[0].view.alloc == 300
    assert sum(isinstance(p, Transparent) and p.effect == "noop" for p in g.parts) == (0 if P <= E // 4 else 1)
    # the combine outside the op
    assert isinstance(res[1], Compute) and res[1].spec.id == f"ResidualAdd_v1{{N={H}}}"
    assert [o.view.alloc for o in res[1].operands] == [330, 300]


def test_the_cutedsl_gate_binds_the_gate_weight_root_by_the_empty_it_fills():
    res, se = _resolve(4, gate=GATE_CUTEDSL, ctx_kw={"ll_bf16": True})
    gc = next(p for p in res[0].parts if isinstance(p, Compute))
    empty = next(n for n in se.roots[0].children if n.op.startswith("aten.empty") and n.output_views()[0].alloc == 320)
    assert gc.consumed == (empty.id,) and gc.operands[1].view == weights()["gate_w"]


@pytest.mark.parametrize("M,kw,ctx_kw,needle", [
    (4, {"pattern": None}, {}, "no registered bf16 x bf16 -> f32 gate Definition for the cublas tier"),
    (4, {"gate": GATE_CUTEDSL}, {}, "does not attest the cuteDSL ll_bf16_gemm tier"),
    (32, {"gate": GATE_CUTEDSL}, {"ll_bf16": True}, "does not attest the cuteDSL ll_bf16_gemm tier at M <= 16"),
    (4, {}, {"experts": None}, f"does not attest {TRITON_EXPERTS}"),
    (4, {}, {"fp8_kernel": None}, "the shared expert's cutlass_scaled_mm did not resolve at block 128"),
    (2, {}, {}, "no wgmma tile"),
    (4, {"tile": (8, 64, 128)}, {"target": TargetProfile(compute_capability=(12, 0), num_sms=188, moe_backend="triton", batch_invariant=False)},
     "modelled on cc 9.0 only"),
    (4, {"router_args": {"n_group": 8, "topk_group": 4}}, {}, "one group, renormalized, sigmoid"),
    (4, {"router_args": {"routed_scaling_factor": 1.0}}, {}, "at the profile's routed_scaling_factor"),
    (4, {"fused_silu_quant": False}, {}, "block-FP8 shared-expert shape"),
    (4, {"shared_input": ViewDesc(296, 0, (4, H), (H, 1), "bfloat16", CUDA)}, {}, "does not take the hidden states"),
    (4, {"quant_group": 64}, {}, "did not resolve at block 128"),
])
def test_the_shared_scope_refuses_by_name(M, kw, ctx_kw, needle):
    if "pattern" in kw:
        kw.pop("pattern")
        ctx_kw = {**ctx_kw, "pattern": scope(gate_spec=None)}
    res, _ = _resolve(M, ctx_kw=ctx_kw, **kw)
    bad = _reasons(res)
    assert bad and needle in bad[0], bad


def test_the_plain_block_fp8_scope_refuses_the_shared_op_and_the_shared_scope_leaves_moe_forward_alone():
    plain = MoEFusedExpertsFp8(C, group=G)
    res, _ = _resolve(4, ctx_kw={"pattern": plain})
    assert _reasons(res) and "shared_experts_input" in _reasons(res)[0]
    sc = scope()
    node = step_events([StepStart(0, [("r0", 0)], [0], [0], {"r0": 1}, {"r0": 0}, {"r0": [0]}, 16, [0], [("r0", 0)], [0], 1, 1, 1, None, None,
                                  "decode"),
                        DispatchEnter(1, None, 0, "forward", "vllm.moe_forward.default", "", [], {}, 7, None), DispatchExit(1, None)]).roots[0]
    assert sc.match(node, None) is None


def test_the_run_facts_read_back_attests_the_cutedsl_gate_only_when_a_gate_is_eligible():
    from types import SimpleNamespace

    from verity_vllm.engine.run_facts import moe_gate_ll_bf16_gemm_of

    def model(*flags):
        mods = [SimpleNamespace()] + [SimpleNamespace(allow_ll_bf16_gemm=f) for f in flags]
        return SimpleNamespace(modules=lambda: iter(mods))

    assert moe_gate_ll_bf16_gemm_of(model(False, True)) is True
    assert moe_gate_ll_bf16_gemm_of(model(False, False)) is False
    assert moe_gate_ll_bf16_gemm_of(model()) is False
    assert moe_gate_ll_bf16_gemm_of(model(1)) is False
