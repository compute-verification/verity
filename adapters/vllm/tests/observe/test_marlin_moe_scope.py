"""The MXFP4 Marlin shared-expert MoE scope (`gen_marlin_moe_patterns.MoEMarlinSharedExperts`) on synthetic `vllm.moe_forward_shared`
events of one request, under the registered DeepSeek-V4-Pro TP8 rank profile (its config and its patterns) and a header that declares
the H200.

The scope is written from vLLM d9105ea80, not from a served log, which the tree does not hold: each op's schema is the one vLLM
declares (`csrc/libtorch_stable/moe/torch_bindings.cpp`, `csrc/libtorch_stable/torch_bindings.cpp`), in the form the dispatcher
gives it; the two `moe_wna16_marlin_gemm` calls pass what `_fused_marlin_moe` passes through `ops.moe_wna16_marlin_gemm`
(`fused_moe/experts/marlin_moe.py`); the hash router's call is `ops.topk_hash_softplus_sqrt`'s, and the scored router's launch has
`_dsv4_topk_kernel`'s parameters (`fused_moe/router/dsv4_topk.py`) at their pins.  Allocation ids and event ids are made up.  So the
tests show what the pattern does with a scope of that shape, and not that a served scope has it.

They check that a hash layer's scope and a scored layer's fold whole, on the gate tier vLLM dispatches at the launch's tokens, to the
Build's Definitions in `_forward_impl`'s order, with operand rows the Definitions' parameters fit and the Marlin experts' weights read
from their Record fields; that the hash router takes a padding mask only where the step filled it False; and that the scope refuses,
by name, each read that is not the modelled call's.
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import dataclasses
import importlib
import math
import re

import pytest
from verity.definitions.numerics import marlin_moe_rows as MR

from verity_vllm.observe.events import (
    DispatchEnter,
    DispatchExit,
    Root,
    RunHeader,
    StepStart,
    TritonLaunch,
    ViewDesc,
)
from verity_vllm.observe.fold.patterns import vocab_kind
from verity_vllm.observe.fold.patterns.ops import schema_param_names
from verity_vllm.observe.fold.profiles import gen_marlin_moe_patterns
from verity_vllm.observe.fold.profiles.gen_marlin_moe_patterns import (
    MXFP4_TYPE_ID,
    MoEMarlinSharedExperts,
)
from verity_vllm.observe.fold.resolver import (
    Compute,
    Const,
    Gather,
    Group,
    PerRow,
    Pieces,
    Shared,
    StepContext,
    Unsupported,
    WeightField,
    resolve_step,
)
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.program.frontend.rules.vllm_bindings.pins import DSV4_TOPK_PINS
from verity_vllm.program.frontend.rules.vllm_marlin_moe import served_config
from verity_vllm.program.frontend.target_profile import TargetProfile

RANK = importlib.import_module("verity_vllm.observe.fold.profiles.derived_DSV4_PRO_tp8")
C = RANK.C
H, INTER, SI, E, TOPK, VOCAB = C["H"], C["I"], C["SHARED_I"], C["NUM_EXPERTS"], C["TOPK"], C["VOCAB"]
RSF, LIMIT, HASH_LAYERS = C["ROUTED_SCALING_FACTOR"], C["SWIGLU_LIMIT"], C["HASH_LAYERS"]
#: a hash layer (below HASH_LAYERS) and a scored one
HASHED, SCORED = 2, 5
#: tokens at which vLLM's gate takes the cuteDSL tier (no `mm` in the scope) and the cuBLAS tier (an `aten.mm`)
FEW, MANY = 4, 17
CUDA = "cuda:0"
E4M3, E8M0 = "float8_e4m3fn", "float8_e8m0fnu"
H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, moe_backend="triton", batch_invariant=False)
PATTERN = "moe.marlin_shared_experts.mxfp4"

SCOPE = ("vllm::moe_forward_shared(Tensor hidden_states, Tensor router_logits, Tensor? shared_experts_input, Tensor? input_ids, "
         "PyObject layer_name, SymInt hidden_dim_unpadded) -> (Tensor, Tensor)")
EMPTY = ("aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, "
         "MemoryFormat? memory_format=None) -> Tensor")
FILL = "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)"
MM = "aten::mm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor"
DYN_MM = ("vllm::dynamic_flashinfer_deepgemm_blockscale_gemm(Tensor input, Tensor weight, Tensor weight_scale, int group_size, "
          "bool use_deep_gemm_e8m0) -> Tensor")
CLAMP = "_C::silu_and_mul_with_clamp(Tensor($0! -> ) result, Tensor input, float limit, float alpha=1., float beta=0.) -> ()"
HASH_ROUTER = ("_moe_C::topk_softplus_sqrt(Tensor($0! -> ) topk_weights, Tensor($1! -> ) topk_indices, Tensor($2! -> ) token_expert_indices, "
               "Tensor gating_output, bool renormalize, float routed_scaling_factor, Tensor? bias, Tensor? input_ids, Tensor? tid2eid, "
               "Tensor? is_padding, Tensor? bias_vl=None, int image_sentinel_lo=0) -> ()")
ALIGN = ("_moe_C::moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor($0! -> ) sorted_token_ids, Tensor($1! -> ) "
         "experts_ids, Tensor($2! -> ) num_tokens_post_pad, Tensor? maybe_expert_map) -> ()")
MARLIN = ("_moe_C::moe_wna16_marlin_gemm(Tensor($0! -> ) a, Tensor? c_or_none, Tensor($1! -> ) b_q_weight, Tensor? b_bias_or_none, "
          "Tensor($2! -> ) b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? g_idx_or_none, "
          "Tensor? perm_or_none, Tensor($3! -> ) workspace, Tensor sorted_token_ids, Tensor($4! -> ) expert_ids, "
          "Tensor($5! -> ) num_tokens_past_padded, Tensor($6! -> ) topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, "
          "int b_type_id, int size_m, int size_n, int size_k, bool is_full_k, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float, "
          "int thread_k, int thread_n, int blocks_per_sm) -> Tensor")
SUM = "_moe_C::moe_sum(Tensor input, Tensor($0! -> ) output, Tensor? topk_ids=None, Tensor? expert_map=None) -> ()"
ADD = "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
#: allocation ids of what the scope reads and writes
HIDDEN, TOKENS, SCORES, SHARED_OUT, ROUTED_OUT, WEIGHTS, IDS, CACHE13, CACHE2, PADDING = 295, 290, 320, 330, 300, 297, 298, 301, 305, 291


def _strides(shape) -> tuple[int, ...]:
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return tuple(reversed(out))


def view(alloc: int, shape, dtype: str = "bfloat16", strides=None, offset: int = 0) -> ViewDesc:
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides or _strides(shape)), dtype, CUDA)


def names(layer: int) -> dict[str, str]:
    ffn = f"model.layers.{layer}.ffn"
    return {"gate": f"{ffn}.gate.weight", "tid2eid": f"{ffn}.gate.tid2eid", "bias": f"{ffn}.gate.e_score_correction_bias",
            "sgu": f"{ffn}.shared_experts.gate_up_proj.weight", "sgu_s": f"{ffn}.shared_experts.gate_up_proj.weight_scale_inv",
            "sd": f"{ffn}.shared_experts.down_proj.weight", "sd_s": f"{ffn}.shared_experts.down_proj.weight_scale_inv",
            "w13": f"{ffn}.experts.routed_experts.w13_weight", "w13_s": f"{ffn}.experts.routed_experts.w13_weight_scale",
            "w2": f"{ffn}.experts.routed_experts.w2_weight", "w2_s": f"{ffn}.experts.routed_experts.w2_weight_scale"}


def weights() -> dict[str, ViewDesc]:
    """The layer's weights as served: the experts' as Marlin repacks them, i32 [E, K/16, 2N] codes and e8m0 [E, K/32, N] scales."""
    return {"gate": view(7, (E, H)), "tid2eid": view(8, (VOCAB, TOPK), "int32"), "bias": view(17, (E,), "float32"),
            "sgu": view(9, (2 * SI, H), E4M3), "sgu_s": view(10, (2 * SI // 128, H // 128), "float32"),
            "sd": view(11, (H, SI), E4M3), "sd_s": view(12, (H // 128, SI // 128), "float32"),
            "w13": view(13, (E, H // 16, 4 * INTER), "int32"), "w13_s": view(14, (E, H // 32, 2 * INTER), E8M0),
            "w2": view(15, (E, INTER // 16, 2 * H), "int32"), "w2_s": view(16, (E, INTER // 32, H), E8M0)}


def scope_events(M: int, layer: int, *, padding: ViewDesc | None = None, module: str | None = None) -> list:
    """One `vllm.moe_forward_shared` of `layer` over M tokens, then the combine `aten.add.Tensor(shared, routed)` outside it."""
    nid = [1000]

    def new() -> int:
        nid[0] += 1
        return nid[0]

    W, ffn = weights(), f"model.layers.{layer}.ffn"
    hidden, tokens = view(HIDDEN, (M, H)), view(TOKENS, (M,), "int32")
    shared_out, routed_out = view(SHARED_OUT, (M, H)), view(ROUTED_OUT, (M, H))
    evs: list = []
    scope = new()

    def op(name: str, schema: str, args: list, outs=None) -> None:
        i = new()
        evs.append(DispatchEnter(i, scope, 0, "forward", name, schema, list(args), {}, 7, None))
        evs.append(DispatchExit(i, outs))

    def empty(v: ViewDesc) -> None:
        op("aten.empty.memory_format", EMPTY, [list(v.shape), None, None, None, None, None], v)

    def dyn_mm(out: ViewDesc, x: ViewDesc, w: str) -> None:
        op("vllm.dynamic_flashinfer_deepgemm_blockscale_gemm.default", DYN_MM, [x, W[w], W[f"{w}_s"], 128, True], out)

    evs.append(DispatchEnter(scope, None, 0, "forward", "vllm.moe_forward_shared.default", SCOPE, [hidden, hidden, hidden, tokens, "layer", 0],
                             {}, 7, f"{ffn}.experts" if module is None else module))
    scores = view(SCORES, (M, E), "float32")
    if M > 16:
        op("aten.mm.dtype", MM, [hidden, view(7, (H, E), strides=(1, H)), "torch.float32"], scores)
    else:
        empty(scores)
    # the shared expert
    gate_up, act = view(323, (M, 2 * SI)), view(324, (M, SI))
    dyn_mm(gate_up, hidden, "sgu")
    empty(act)
    op("_C.silu_and_mul_with_clamp.default", CLAMP, [act, gate_up, LIMIT, 1.0, 0.0])
    dyn_mm(shared_out, act, "sd")
    # the router
    w, ids, tei = view(WEIGHTS, (M, TOPK), "float32"), view(IDS, (M, TOPK), "int32"), view(299, (M, TOPK), "int32")
    if layer < HASH_LAYERS:
        for v in (w, ids, tei):
            empty(v)
        op("_moe_C.topk_softplus_sqrt.default", HASH_ROUTER, [w, ids, tei, scores, True, RSF, None, tokens, W["tid2eid"], padding, None, 0])
    else:
        empty(w)
        empty(ids)
        consts = {**DSV4_TOPK_PINS["constexprs"], "launch_pdl": True}
        args = {"gating_output_ptr": scores, "correction_bias_ptr": W["bias"], "topk_weights_ptr": w, "topk_ids_ptr": ids,
                "routed_scaling_factor": RSF, "input_ids_ptr": None, "bias_vl_ptr": None, "image_sentinel_lo": 0, **consts}
        evs.append(TritonLaunch(new(), scope, 0, "forward", DSV4_TOPK_PINS["kernel"], DSV4_TOPK_PINS["src_sha256"], args, consts, [M],
                                DSV4_TOPK_PINS["num_warps"], 3, 7))
    # the routed experts
    P, BM = M * TOPK, MR.moe_block(M, TOPK, E)
    padded = P + E * (BM - 1)
    sorted_ids, expert_ids, post_pad = view(302, (padded,), "int32"), view(303, (math.ceil(padded / BM),), "int32"), view(304, (1,), "int32")
    for v in (sorted_ids, expert_ids, post_pad):
        empty(v)
    op("_moe_C.moe_align_block_size.default", ALIGN, [ids, E, BM, sorted_ids, expert_ids, post_pad, None])
    cache1, cache2, cache3 = view(CACHE13, (P, 2 * INTER)), view(CACHE2, (P, INTER)), view(CACHE13, (P, H))
    workspace = view(306, (H200.num_sms * 4,), "int32")

    def gemm(a: ViewDesc, c: ViewDesc, what: str, top_k: int, size_m: int, N: int, K: int) -> None:
        op("_moe_C.moe_wna16_marlin_gemm.default", MARLIN,
           [a, c, W[what], None, W[f"{what}_s"], None, None, None, None, None, workspace, sorted_ids, expert_ids, post_pad, w, BM, top_k,
            top_k == 1, MXFP4_TYPE_ID, size_m, N, K, True, False, True, False, -1, -1, -1], c)

    gemm(hidden, cache1, "w13", TOPK, M, 2 * INTER, H)
    op("_C.silu_and_mul_with_clamp.default", CLAMP, [cache2, cache1, LIMIT, 1.0, 0.0])
    gemm(cache2, cache3, "w2", 1, P, H, INTER)
    empty(routed_out)
    op("_moe_C.moe_sum.default", SUM, [view(CACHE13, (M, TOPK, H)), routed_out, None, None])
    evs.append(DispatchExit(scope, [shared_out, routed_out]))
    i = new()
    evs.append(DispatchEnter(i, None, 0, "forward", "aten.add.Tensor", ADD, [shared_out, routed_out], {}, 7, ffn))
    evs.append(DispatchExit(i, view(340, (M, H))))
    return evs


def _calls(evs: list, op: str) -> list[int]:
    """Where `op`'s calls are in `evs`, in order: a dispatcher op by its name, the scored router's launch by its kernel's."""
    return [k for k, e in enumerate(evs)
            if (isinstance(e, DispatchEnter) and e.op.split(".")[-2] == op) or (isinstance(e, TritonLaunch) and e.fn.rsplit(".", 1)[-1] == op)]


def edited(evs: list, op: str, nth: int = 0, **changed) -> list:
    """`evs` with the `nth` call of `op` given `changed` arguments, each by its name in the op's schema."""
    k = _calls(evs, op)[nth]
    call = evs[k]
    if isinstance(call, TritonLaunch):
        new = dataclasses.replace(call, args={**call.args, **changed})
    else:
        at = schema_param_names(call.schema)
        args = list(call.args)
        for name, value in changed.items():
            args[at.index(name)] = value
        new = dataclasses.replace(call, args=args)
    return [*evs[:k], new, *evs[k + 1:]]


def without(evs: list, op: str, nth: int = 0) -> list:
    """`evs` without the `nth` call of `op`."""
    k = _calls(evs, op)[nth]
    gone = evs[k].id
    return [e for e in evs if getattr(e, "id", None) != gone]


def written(evs: list, op: str, nth: int, out: ViewDesc, **changed) -> list:
    """`edited`, the call returning `out` as well."""
    call = evs[_calls(evs, op)[nth]].id
    return [dataclasses.replace(e, outs=out) if isinstance(e, DispatchExit) and e.id == call else e for e in edited(evs, op, nth, **changed)]


def swapped(evs: list, k: int, a: int) -> list:
    """`evs` with the dispatcher calls at `k` and `a`, each an enter and the exit after it, in each other's place and under each other's
    id: the one logged first is then the later of the two."""
    out = list(evs)
    out[k], out[a] = dataclasses.replace(evs[k], id=evs[a].id), dataclasses.replace(evs[a], id=evs[k].id)
    out[k + 1], out[a + 1] = dataclasses.replace(evs[k + 1], id=evs[a].id), dataclasses.replace(evs[a + 1], id=evs[k].id)
    return out


def inside(evs: list, *more) -> list:
    """`evs` with the events `more` ahead of the scope's own exit."""
    scope = evs[0].id
    k = next(k for k, e in enumerate(evs) if isinstance(e, DispatchExit) and e.id == scope)
    return [*evs[:k], *more, *evs[k:]]


def start_of(M: int, requests: int = 1) -> StepStart:
    """A step of M token rows, `requests` requests' in turn, whose input ids are the view the hash router reads."""
    rows = [(f"r{p % requests}", p // requests) for p in range(M)]
    reqs = sorted({r for r, _ in rows})
    return StepStart(0, rows, list(range(M)), [p for _, p in rows], {r: M for r in reqs}, {r: 0 for r in reqs}, {r: [0, 1, 2, 3] for r in reqs},
                     16, list(range(M)), [(r, M - 1) for r in reqs], [M - 1] * len(reqs), len(reqs), M, M, view(TOKENS, (M,), "int32"), None,
                     "prefill")


def ctx_of(start: StepStart, layer: int, *, target: TargetProfile = H200, pattern: MoEMarlinSharedExperts | None = None,
           attested: bool = True) -> StepContext:
    """The rank profile's context under a header that declares `target` and the block-FP8 kernel, and attests the cuteDSL gate tier
    unless `attested` is False."""
    backends = {"fp8_block_gemm_kernel": "FlashInferFp8DeepGEMMDynamicBlockScaledKernel", "moe_gate_ll_bf16_gemm": attested}
    header = RunHeader("verity-capture/log/v1", "2026-10-18T00:00:00Z", "x", {"target": target.to_json(), "backends": backends},
                       {"repo": RANK.REPO_ID, "config": RANK.HF_CONFIG, "tp": 8}, {"requests": []})
    served, named = weights(), names(layer)
    roots = {v.alloc: [Root(v, "weight", named[k])] for k, v in served.items()}
    profile = RANK.PROFILE
    if pattern is not None:
        profile = dataclasses.replace(profile, patterns=[pattern if isinstance(p, MoEMarlinSharedExperts) else p for p in profile.patterns])
    return StepContext(header, start, None, [], roots, C, profile)


def _resolve(evs: list, layer: int, *, requests: int = 1, before: tuple = (), **ctx_kw):
    M = next(e for e in evs if isinstance(e, DispatchEnter)).args[0].shape[0]
    start = start_of(M, requests)
    se = step_events([start, *before, *evs])
    return resolve_step(se, ctx_of(start, layer, **ctx_kw)), se


def _reasons(res) -> list:
    return [r.reason for r in res if isinstance(r, Unsupported)]


def _row_leaves(operand) -> int:
    """How many leaves one row of `operand` reads."""
    if isinstance(operand, Shared):
        return operand.view.numel
    if isinstance(operand, (PerRow, Gather)):
        return math.prod(operand.view.shape[1:])
    if isinstance(operand, Pieces):
        return sum(p.count if isinstance(p, Const) else p.numel for p in operand.rows[0])
    assert isinstance(operand, WeightField)
    record = RANK.PROFILE.weights_type
    return sum(record.fields[record.index(n)][1].leaves for n in operand.path)


def _fits(c: Compute) -> bool:
    """Every operand's row has its parameter's leaf count."""
    params, _ = c.spec.signature()
    return [t.leaves for _, t in params] == [_row_leaves(o) for o in c.operands]


def _fill(v: ViewDesc, value) -> tuple:
    """The runner's `fill_` of `v`, a call outside the scope and ahead of it."""
    return (DispatchEnter(900, None, 0, "forward", "aten.fill_.Scalar", FILL, [v, value], {}, 7, None), DispatchExit(900, v))


@pytest.mark.parametrize("layer", [HASHED, SCORED])
@pytest.mark.parametrize("M", [FEW, MANY])
def test_the_marlin_scope_folds_the_gate_the_shared_expert_the_router_and_the_routed_pairs(layer, M):
    res, se = _resolve(scope_events(M, layer), layer)
    assert len(res) == 2 and isinstance(res[0], Group), _reasons(res)
    g, named, W, P, BM = res[0], names(layer), weights(), M * TOPK, MR.moe_block(M, TOPK, E)
    assert g.consumed == tuple(sorted(se.roots[0].ids())) and g.pattern == PATTERN and g.hint == f"model.layers.{layer}.ffn.experts"
    computes = [p for p in g.parts if isinstance(p, Compute)]
    assert all(c.pattern == PATTERN for c in computes)
    gate = (f"CublasSplitKGemmF32Out_v1{{K={H},N={E},SPLIT_K=5,DOT=HopperBF16WgmmaDot16_v1}}" if M > 16 else
            f"LlBf16DotprodGemm_v1{{N={E},K={H},BS=128,VEC=8,FMA=1}}")
    router = ([f"GatherI32x{VOCAB}_v1", f"MoeRouterHash_v1{{E={E},SCALE={RSF}}}"] if layer < HASH_LAYERS else
              [f"MoeRouterSqrtSoftplus_v1{{E={E},SCALE={RSF}}}"])
    served = "TK=64,TN=128,GRID=528,DOT=HopperBF16WgmmaDot16_v1"
    assert [c.spec.id for c in computes] == [
        gate, f"FlashInferFp8Linear_v1{{K={H},N={2 * SI}}}", f"SiluMulClamp_v1{{I={SI},LIMIT={LIMIT}}}", f"FlashInferFp8Linear_v1{{K={SI},N={H}}}",
        *router, f"MoeAlignRanks_v1{{E={E},P={P},BM={BM}}}", f"MoeAlignBlocks_v1{{P={P},BM={BM}}}",
        f"MarlinMoeExpertGemm_v1{{E={E},K={H},N={2 * INTER},{served}}}", f"SiluMulClamp_v1{{I={INTER},LIMIT={LIMIT}}}",
        f"MarlinMoeExpertGemmW_v1{{E={E},K={INTER},N={H},{served}}}", f"MoeSum_v1{{TOPK={TOPK},H={H}}}"]
    assert all(_fits(c) for c in computes)
    gc, m1, silu, m2, *route, ranks, blocks, up, rsilu, down, msum = computes
    assert [c.rows for c in computes] == [M] * 4 + ([P, M] if layer < HASH_LAYERS else [M]) + [1, 1, P, P, P, M]
    # the gate reads the hidden states and the gate weight, and writes the scores the router reads
    assert gc.operands[0].view.alloc == HIDDEN and gc.operands[1].view == W["gate"] and gc.roots == (None, named["gate"])
    assert gc.outputs[0].view.alloc == SCORES
    # the shared expert: hidden -> gate_up -> the clamped SiLU -> down, into the op's first output
    assert m1.operands[0].view.alloc == HIDDEN and silu.operands[0].view == m1.outputs[0].view
    assert m2.operands[0].view == silu.outputs[0].view and m2.outputs[0].view.alloc == SHARED_OUT
    assert m1.roots[1:] == (named["sgu"], named["sgu_s"]) and m2.roots[1:] == (named["sd"], named["sd_s"])
    # the router writes the weights and ids of the pairs
    if layer < HASH_LAYERS:
        ids, routed = route
        tokens, table = ids.operands
        assert (tokens.view.alloc, tokens.indices) == (TOKENS, tuple(p // TOPK for p in range(P)))
        assert (table.view.alloc, table.view.shape, table.view.strides) == (W["tid2eid"].alloc, (TOPK, VOCAB), (1, TOPK))
        assert table.indices == tuple(p % TOPK for p in range(P)) and ids.roots == (None, named["tid2eid"])
        assert ids.outputs[0].view.alloc == IDS and [o.view.alloc for o in routed.operands] == [SCORES, IDS]
        assert routed.outputs[0].view.alloc == WEIGHTS and routed.kernels == ("softplus_sqrt", "SoftplusSqrt")
    else:
        routed, = route
        assert [type(o) for o in routed.operands] == [PerRow, Shared] and routed.operands[1].view == W["bias"]
        assert routed.operands[0].view.alloc == SCORES and [o.view.alloc for o in routed.outputs] == [WEIGHTS, IDS]
        assert routed.roots == (None, named["bias"]) and routed.kernels == ("_dsv4_topk_kernel",)
    # the align's schedule is fold-private: the ranks and starts feed the blocks, whose (blk, par) both launches read
    rank, begin = (o.view for o in ranks.outputs)
    blk, par = (o.view for o in blocks.outputs)
    assert all(v.alloc < 0 for v in (rank, begin, blk, par)) and len({v.alloc for v in (rank, begin, blk, par)}) == 4
    assert ranks.operands[0].view.alloc == IDS and [type(o) for o in ranks.operands[1:]] == [Pieces, Pieces]
    # the pair indices are the Build's cached i32 constants, one instance a value for the whole resolution
    assert ranks.operands[1].rows[0] == ranks.operands[2].rows[0] == tuple(Const(32, p, "i32", shared=True) for p in range(P))
    assert [o.view for o in blocks.operands] == [ranks.operands[0].view, rank, begin] and (blk.shape, par.shape) == ((P,), (1,))
    # w13 gathers its token's hidden row; the clamped SiLU reads its output; w2 multiplies the router's weight in
    assert [type(o) for o in up.operands] == [Gather, PerRow, PerRow, Shared, WeightField, WeightField]
    assert up.operands[0].view.alloc == HIDDEN and up.operands[0].indices == tuple(p // TOPK for p in range(P))
    assert [o.view for o in up.operands[1:4]] == [ranks.operands[0].view, blk, par]
    assert [o.path for o in up.operands[4:]] == [(named["w13"],), (named["w13_s"],)] and up.roots == (None,) * 4 + (named["w13"], named["w13_s"])
    assert up.outputs[0].view == view(CACHE13, (P, 2 * INTER)) and rsilu.operands[0].view == up.outputs[0].view
    assert [type(o) for o in down.operands] == [PerRow, PerRow, PerRow, PerRow, Shared, WeightField, WeightField]
    assert down.operands[0].view == rsilu.outputs[0].view == view(CACHE2, (P, INTER))
    assert down.operands[2].view == view(WEIGHTS, (P,), "float32") and [o.view for o in down.operands[3:5]] == [blk, par]
    assert [o.path for o in down.operands[5:]] == [(named["w2"],), (named["w2_s"],)] and down.roots == (None,) * 5 + (named["w2"], named["w2_s"])
    assert up.kernels == down.kernels == ("Marlin",) and down.outputs[0].view == view(CACHE13, (P, H))
    # moe_sum reads w2's output by token and writes the op's second output
    assert msum.operands[0].view == view(CACHE13, (M, TOPK, H)) and msum.outputs[0].view.alloc == ROUTED_OUT
    # the combine outside the op
    assert isinstance(res[1], Compute) and res[1].spec.id == f"ResidualAdd_v1{{N={H}}}"
    assert [o.view.alloc for o in res[1].operands] == [SHARED_OUT, ROUTED_OUT]


def test_the_mxfp4_type_id_is_the_packing_of_float4_e2m1f():
    """`ScalarType.id` (vLLM's `scalar_type.py`) packs, from the low bit, the exponent (8 bits), the mantissa (8), signed (1), the bias
    (32), finite values only (1) and the NaN representation (8).  `float4_e2m1f` is exponent 2, mantissa 1, signed, no bias, finite
    values only, and no NaN representation (0)."""
    packed, at = 0, 0
    for value, bits in ((2, 8), (1, 8), (1, 1), (0, 32), (1, 1), (0, 8)):
        packed |= value << at
        at += bits
    assert at <= 64 and packed == MXFP4_TYPE_ID


@pytest.mark.parametrize("schema", [EMPTY, FILL, MM, DYN_MM, CLAMP, HASH_ROUTER, ALIGN, MARLIN, SUM, ADD], ids=lambda s: s.split("(")[0])
def test_each_schema_is_written_as_torch_prints_it(schema):
    """The observer records `str(func._schema)`, so a schema here is the string torch prints, in which a default of 1.0 is `1.`.  torch
    prints a `Tensor!` as `Tensor($n! -> )` and does not parse that back, so each is parsed as the `Tensor!` vLLM declares.  `SCOPE` is
    not held: torch's parser does not read its `PyObject`."""
    torch = pytest.importorskip("torch")
    assert str(torch._C.parse_schema(re.sub(r"\(\$\d+! -> \)", "!", schema))) == schema


def _specs(group) -> list[str]:
    return [p.spec.id for p in group.parts if isinstance(p, Compute)]


def test_the_hash_router_takes_a_padding_mask_the_step_filled_false():
    buffer, mask = view(PADDING, (64,), "bool"), view(PADDING, (FEW,), "bool")
    res, _ = _resolve(scope_events(FEW, HASHED, padding=mask), HASHED, before=_fill(buffer, False))
    assert isinstance(res[1], Group), _reasons(res)
    plain, _ = _resolve(scope_events(FEW, HASHED), HASHED)
    assert _specs(res[1]) == _specs(plain[0])


def test_the_hash_router_reads_only_the_fills_over_the_masks_rows():
    """Later fills of the mask's allocation that end where its rows begin, begin where they end, or hold no element are not fills of
    the mask: the fill that covers its rows decides."""
    lo = 8
    mask = view(PADDING, (FEW,), "bool", offset=lo)
    fills = [(view(PADDING, (64,), "bool"), False), (view(PADDING, (lo,), "bool"), True), (view(PADDING, (4,), "bool", offset=lo + FEW), True),
             (view(PADDING, (0,), "bool", offset=lo + 1), True)]
    before: list = []
    for k, (v, value) in enumerate(fills):
        enter, leave = _fill(v, value)
        before += [dataclasses.replace(enter, id=900 + k), dataclasses.replace(leave, id=900 + k)]
    res, _ = _resolve(scope_events(FEW, HASHED, padding=mask), HASHED, before=tuple(before))
    assert isinstance(res[len(fills)], Group), _reasons(res)
    plain, _ = _resolve(scope_events(FEW, HASHED), HASHED)
    assert _specs(res[len(fills)]) == _specs(plain[0])


def test_the_hash_router_folds_a_call_recorded_without_its_two_defaulted_arguments():
    """`topk_softplus_sqrt`'s last two parameters have defaults (`bias_vl=None`, `image_sentinel_lo=0`): a call recorded with the ten
    arguments ahead of them is the text-token route as well."""
    evs = scope_events(FEW, HASHED)
    k = _calls(evs, "topk_softplus_sqrt")[0]
    assert schema_param_names(evs[k].schema)[10:] == ["bias_vl", "image_sentinel_lo"]
    res, _ = _resolve([*evs[:k], dataclasses.replace(evs[k], args=evs[k].args[:10]), *evs[k + 1:]], HASHED)
    assert isinstance(res[0], Group), _reasons(res)
    plain, _ = _resolve(evs, HASHED)
    assert _specs(res[0]) == _specs(plain[0])


@pytest.mark.parametrize("mask,fills,needle", [
    (view(PADDING, (FEW,), "bool"), [], "is_padding is not the step's False prefix"),
    (view(PADDING, (FEW,), "bool"), [(view(PADDING, (64,), "bool"), True)], "is_padding is not the step's False prefix"),
    (view(PADDING, (FEW,), "bool"), [(view(PADDING, (FEW - 1,), "bool"), False)], "is_padding is not the step's False prefix"),
    (view(PADDING, (FEW,), "bool"), [(view(PADDING, (64,), "bool"), False), (view(PADDING, (1,), "bool", offset=1), True)],
     "is_padding is not the step's False prefix"),
    (view(PADDING, (FEW,), "bool"), [(view(PADDING, (64,), "bool"), 0)], "is_padding is not the step's False prefix"),
    (view(PADDING, (FEW,), "int32"), [(view(PADDING, (64,), "int32"), False)], "is_padding is not a contiguous bool [M] view"),
    (view(PADDING, (FEW + 1,), "bool"), [(view(PADDING, (64,), "bool"), False)], "is_padding is not a contiguous bool [M] view"),
    (view(PADDING, (FEW,), "bool", strides=(2,)), [(view(PADDING, (64,), "bool"), False)], "is_padding is not a contiguous bool [M] view"),
    (7, [(view(PADDING, (64,), "bool"), False)], "is_padding is not a contiguous bool [M] view"),
])
def test_the_hash_router_refuses_any_other_padding_mask(mask, fills, needle):
    before: list = []
    for k, (v, value) in enumerate(fills):
        enter, leave = _fill(v, value)
        before += [dataclasses.replace(enter, id=900 + k), dataclasses.replace(leave, id=900 + k)]
    res, _ = _resolve(scope_events(FEW, HASHED, padding=mask), HASHED, before=tuple(before))
    bad = _reasons(res)
    assert bad and needle in bad[0], bad


#: refusals an argument decides: (layer, the op, which of its calls, the arguments it gets, a part of the reason)
BY_ARGUMENT = [
    (HASHED, "moe_forward_shared", 0, {"router_logits": view(296, (FEW, H))}, "does not take the hidden states as router_logits"),
    (HASHED, "topk_softplus_sqrt", 0, {"token_expert_indices": None}, "topk_softplus_sqrt arguments not bound as expected"),
    (HASHED, "topk_softplus_sqrt", 0, {"renormalize": False}, "is not the renormalised text-token table route"),
    (HASHED, "topk_softplus_sqrt", 0, {"routed_scaling_factor": 1.0}, "is not the renormalised text-token table route"),
    (HASHED, "topk_softplus_sqrt", 0, {"routed_scaling_factor": True}, "is not the renormalised text-token table route"),
    (HASHED, "topk_softplus_sqrt", 0, {"bias": view(17, (E,), "float32")}, "is not the renormalised text-token table route"),
    (HASHED, "topk_softplus_sqrt", 0, {"bias_vl": view(17, (E,), "float32")}, "is not the renormalised text-token table route"),
    (HASHED, "topk_softplus_sqrt", 0, {"image_sentinel_lo": 5}, "is not the renormalised text-token table route"),
    (HASHED, "topk_softplus_sqrt", 0, {"gating_output": view(SCORES, (FEW, E), "bfloat16")}, "gating_output is not a row-major f32 [M,E] view"),
    (HASHED, "topk_softplus_sqrt", 0, {"gating_output": view(SCORES, (FEW, E), "float32", strides=(1, FEW))},
     "gating_output is not a row-major f32 [M,E] view"),
    (HASHED, "topk_softplus_sqrt", 0, {"input_ids": view(TOKENS + 1, (FEW,), "int32")}, "input_ids are not the step's i32 input_ids view"),
    (HASHED, "topk_softplus_sqrt", 0, {"input_ids": view(TOKENS, (FEW,), "int64")}, "input_ids are not the step's i32 input_ids view"),
    (HASHED, "topk_softplus_sqrt", 0, {"tid2eid": view(8, (VOCAB, TOPK), "int64")}, "tid2eid is not the i32 [VOCAB, TOPK] token table weight"),
    (HASHED, "topk_softplus_sqrt", 0, {"tid2eid": view(99, (VOCAB, TOPK), "int32")}, "tid2eid is not the i32 [VOCAB, TOPK] token table weight"),
    (HASHED, "topk_softplus_sqrt", 0, {"topk_weights": view(WEIGHTS, (FEW, TOPK), "bfloat16")},
     "topk_weights is not a contiguous float32 [M,TOPK] view"),
    (HASHED, "topk_softplus_sqrt", 0, {"topk_indices": view(IDS, (FEW, TOPK), "int64")}, "topk_indices is not a contiguous int32 [M,TOPK] view"),
    (HASHED, "topk_softplus_sqrt", 0, {"token_expert_indices": view(299, (FEW, TOPK + 1), "int32")},
     "token_expert_indices is not a contiguous int32 [M,TOPK] view"),
    (SCORED, "_dsv4_topk_kernel", 0, {"topk_ids_ptr": None}, "_dsv4_topk_kernel arguments not bound as expected"),
    (SCORED, "_dsv4_topk_kernel", 0, {"routed_scaling_factor": 1.0}, "is not the (M,) launch at the profile's routed_scaling_factor"),
    (SCORED, "_dsv4_topk_kernel", 0, {"bias_vl_ptr": view(18, (E,), "float32")}, "is not the (M,) launch at the profile's routed_scaling_factor"),
    (SCORED, "_dsv4_topk_kernel", 0, {"gating_output_ptr": view(SCORES, (FEW, E), "float32", strides=(1, FEW))},
     "gating_output and correction_bias are not packed f32 [M,E] and the [E] bias weight"),
    (SCORED, "_dsv4_topk_kernel", 0, {"gating_output_ptr": view(SCORES, (FEW + 1, E), "float32")},
     "gating_output and correction_bias are not packed f32 [M,E] and the [E] bias weight"),
    (SCORED, "_dsv4_topk_kernel", 0, {"gating_output_ptr": view(SCORES, (FEW, E), "float64")},
     "gating_output and correction_bias are not packed f32 [M,E] and the [E] bias weight"),
    (SCORED, "_dsv4_topk_kernel", 0, {"correction_bias_ptr": view(99, (E,), "float32")},
     "gating_output and correction_bias are not packed f32 [M,E] and the [E] bias weight"),
    (SCORED, "_dsv4_topk_kernel", 0, {"correction_bias_ptr": view(17, (E + 1,), "float32")},
     "gating_output and correction_bias are not packed f32 [M,E] and the [E] bias weight"),
    (SCORED, "_dsv4_topk_kernel", 0, {"correction_bias_ptr": view(17, (E,), "bfloat16")},
     "gating_output and correction_bias are not packed f32 [M,E] and the [E] bias weight"),
    (SCORED, "_dsv4_topk_kernel", 0, {"correction_bias_ptr": view(17, (E,), "float32", strides=(2,))},
     "gating_output and correction_bias are not packed f32 [M,E] and the [E] bias weight"),
    (SCORED, "_dsv4_topk_kernel", 0, {"topk_weights_ptr": view(WEIGHTS, (FEW, TOPK), "bfloat16")},
     "topk_weights_ptr is not a packed float32 [M,TOPK] view"),
    (SCORED, "_dsv4_topk_kernel", 0, {"topk_ids_ptr": view(IDS, (FEW, TOPK), "int32", strides=(1, FEW))},
     "topk_ids_ptr is not a packed int32 [M,TOPK] view"),
    (HASHED, "silu_and_mul_with_clamp", 0, {"limit": 7.0}, "the shared expert's silu_and_mul_with_clamp did not resolve"),
    (HASHED, "dynamic_flashinfer_deepgemm_blockscale_gemm", 1, {"group_size": 64},
     "the shared expert's down dynamic_flashinfer_deepgemm_blockscale_gemm did not resolve"),
    (HASHED, "moe_align_block_size", 0, {"topk_ids": view(IDS + 1, (FEW, TOPK), "int32")}, "does not sort the router's ids over E experts"),
    (HASHED, "moe_align_block_size", 0, {"num_experts": E // 2}, "does not sort the router's ids over E experts"),
    (HASHED, "moe_align_block_size", 0, {"maybe_expert_map": view(307, (E,), "int32")}, "does not sort the router's ids over E experts"),
    (HASHED, "moe_align_block_size", 0, {"sorted_token_ids": None}, "does not sort the router's ids over E experts"),
    (HASHED, "moe_align_block_size", 0, {"block_size": 16}, "block 16 is not fused_marlin_moe's at 4 tokens (8)"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"use_atomic_add": True}, "the w13 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"b_type_id": 0}, "the w13 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"thread_k": 64}, "the w13 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"mul_topk_weights": True}, "the w13 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"size_m": FEW * TOPK}, "the w13 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"a_scales": view(308, (FEW,), "float32")},
     "the w13 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 1, {"top_k": TOPK}, "the w2 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 1, {"b_bias_or_none": view(308, (E, H))}, "the w2 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"global_scale": view(308, (1,), "float32")},
     "the w13 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"b_zeros_or_none": view(308, (E, H // 32, 2 * INTER), "int32")},
     "the w13 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 1, {"g_idx_or_none": view(308, (INTER,), "int32")},
     "the w2 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 1, {"perm_or_none": view(308, (INTER,), "int32")},
     "the w2 moe_wna16_marlin_gemm launch is not fused_marlin_moe's W4A16"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"expert_ids": view(309, (1,), "int32")}, "the w13 launch does not read moe_align_block_size's schedule"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"a": view(HIDDEN + 1, (FEW, H))}, "the w13 launch does not read its input and the router's weights"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"topk_weights": view(WEIGHTS + 1, (FEW, TOPK), "float32")},
     "the w13 launch does not read its input and the router's weights"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"b_scales": None}, "the w13 launch does not read its input and the router's weights"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"c_or_none": None}, "the w13 launch does not write a row-major bf16 [P, N] C"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"b_q_weight": view(15, (E, H // 16, 4 * INTER), "int32")},
     "the w13 launch's weights are not the layer's Marlin-repacked w13_weight and _weight_scale roots"),
    (HASHED, "moe_wna16_marlin_gemm", 0, {"b_scales": view(14, (E, H // 32, 2 * INTER), "float32")},
     "the w13 launch's weights are not the layer's Marlin-repacked w13_weight and _weight_scale roots"),
    (HASHED, "moe_wna16_marlin_gemm", 1, {"a": view(CACHE2 + 1, (FEW * TOPK, INTER))}, "the w2 launch does not read its input and the router's weights"),
    (HASHED, "silu_and_mul_with_clamp", 1, {"alpha": 2.0}, "the routed silu_and_mul_with_clamp did not resolve as SiluMulClamp"),
    (HASHED, "silu_and_mul_with_clamp", 1, {"limit": 7.0}, "the routed silu_and_mul_with_clamp did not resolve as SiluMulClamp"),
    (HASHED, "silu_and_mul_with_clamp", 1, {"input": view(CACHE13 + 1, (FEW * TOPK, 2 * INTER))},
     "the routed silu_and_mul_with_clamp does not read the w13 output as [P, 2I]"),
    (HASHED, "moe_sum", 0, {"topk_ids": view(IDS, (FEW, TOPK), "int32")}, "moe_sum is not the modelled (input, output) kernel"),
    (HASHED, "moe_sum", 0, {"expert_map": view(307, (E,), "int32")}, "moe_sum is not the modelled (input, output) kernel"),
    (HASHED, "moe_sum", 0, {"output": None}, "moe_sum is not the modelled (input, output) kernel"),
    (HASHED, "moe_sum", 0, {"input": view(CACHE13, (FEW * TOPK, H))}, "moe_sum input is not the contiguous bf16 [M,TOPK,H] w2 output"),
    (HASHED, "moe_sum", 0, {"input": view(CACHE13 + 1, (FEW, TOPK, H))}, "moe_sum input is not the contiguous bf16 [M,TOPK,H] w2 output"),
    (HASHED, "moe_sum", 0, {"input": view(CACHE13, (FEW, TOPK, H), "float32")}, "moe_sum input is not the contiguous bf16 [M,TOPK,H] w2 output"),
    (HASHED, "moe_sum", 0, {"input": view(CACHE13, (FEW, TOPK, H), strides=(2 * TOPK * H, H, 1))},
     "moe_sum input is not the contiguous bf16 [M,TOPK,H] w2 output"),
    (HASHED, "moe_sum", 0, {"output": view(ROUTED_OUT + 1, (FEW, H))}, "moe_sum does not write the op's second output"),
    (HASHED, "moe_sum", 0, {"output": view(ROUTED_OUT, (FEW, H), "float32")}, "moe_sum does not write the op's second output"),
    (HASHED, "moe_sum", 0, {"output": view(ROUTED_OUT, (FEW + 1, H))}, "moe_sum does not write the op's second output"),
    (HASHED, "moe_sum", 0, {"output": view(ROUTED_OUT, (FEW, H), strides=(1, FEW))}, "moe_sum does not write the op's second output"),
]


@pytest.mark.parametrize("layer,op,nth,changed,needle", BY_ARGUMENT)
def test_the_marlin_scope_refuses_a_call_that_is_not_the_modelled_one(layer, op, nth, changed, needle):
    res, _ = _resolve(edited(scope_events(FEW, layer), op, nth, **changed), layer)
    bad = _reasons(res)
    assert bad and needle in bad[0], bad


@pytest.mark.parametrize("c", [view(CACHE13, (FEW * TOPK + 1, 2 * INTER)), view(CACHE13, (FEW * TOPK, 2 * INTER), "float32"),
                               view(CACHE13, (FEW * TOPK, 2 * INTER), strides=(1, FEW * TOPK))])
def test_the_marlin_scope_refuses_a_launch_whose_output_has_another_shape_dtype_or_layout(c):
    """The launch returns the C it is given: each of the shape, the dtype and the strides, alone, makes it another output."""
    res, _ = _resolve(written(scope_events(FEW, HASHED), "moe_wna16_marlin_gemm", 0, c, c_or_none=c), HASHED)
    assert _reasons(res) == ["the w13 launch does not write a row-major bf16 [P, N] C"]


def test_the_scored_router_is_held_to_its_pins_and_its_grid():
    evs = scope_events(FEW, SCORED)
    k = _calls(evs, "_dsv4_topk_kernel")[0]
    for change, needle in (({"src_sha": "0" * 64}, "_dsv4_topk_kernel differs from its pins (DSV4_TOPK_PINS)"),
                           ({"num_warps": 4}, "_dsv4_topk_kernel differs from its pins (DSV4_TOPK_PINS)"),
                           ({"constexprs": {**evs[k].constexprs, "HAS_VL": True}}, "_dsv4_topk_kernel differs from its pins (DSV4_TOPK_PINS)"),
                           ({"grid": [FEW, 1]}, "is not the (M,) launch at the profile's routed_scaling_factor")):
        res, _ = _resolve([*evs[:k], dataclasses.replace(evs[k], **change), *evs[k + 1:]], SCORED)
        assert _reasons(res) and needle in _reasons(res)[0], (change, _reasons(res))


def test_the_marlin_scope_refuses_a_scope_of_another_shape_order_layer_or_launch():
    evs = scope_events(FEW, HASHED)
    # a kernel short, and a kernel out of `_forward_impl`'s order
    res, _ = _resolve(without(evs, "moe_wna16_marlin_gemm", 1), HASHED)
    assert "does not have the MXFP4 Marlin shared-expert shape" in _reasons(res)[0]
    res, _ = _resolve(swapped(evs, _calls(evs, "moe_align_block_size")[0], _calls(evs, "topk_softplus_sqrt")[0]), HASHED)
    assert "scope runs out of order" in _reasons(res)[0]
    # a second gate mm
    many = scope_events(MANY, HASHED)
    k = _calls(many, "mm")[0]
    again = [dataclasses.replace(e, id=5000) for e in many[k:k + 2]]
    res, _ = _resolve(inside(many, *again), HASHED)
    assert "does not have the MXFP4 Marlin shared-expert shape" in _reasons(res)[0]
    # a launch of another kernel is no second router: the scope has its shape, and the launch is what it does not explain
    for layer in (HASHED, SCORED):
        scope = scope_events(FEW, layer)
        other = TritonLaunch(5000, scope[0].id, 0, "forward", "vllm.model_executor.other_kernel", "0" * 64, {}, {}, [FEW], 4, 3, 7)
        res, _ = _resolve(inside(scope, other), layer)
        assert _reasons(res) == ["unexplained child vllm.model_executor.other_kernel inside moe_forward_shared"]
    # each router at the other kind's layer
    res, _ = _resolve(scope_events(FEW, HASHED, module=f"model.layers.{SCORED}.ffn.experts"), HASHED)
    assert f"layer {SCORED}'s router is not its kind" in _reasons(res)[0]
    res, _ = _resolve(scope_events(FEW, SCORED, module=f"model.layers.{HASHED}.ffn.experts"), SCORED)
    assert f"layer {HASHED}'s router is not its kind" in _reasons(res)[0]
    # a scope under no decoder layer, and a launch of two requests' tokens
    res, _ = _resolve(scope_events(FEW, HASHED, module="model.ffn.experts"), HASHED)
    assert "moe_forward_shared is not under a decoder layer's module" in _reasons(res)[0]
    res, _ = _resolve(evs, HASHED, requests=2)
    assert "the launch routes several requests' tokens" in _reasons(res)[0]


def test_the_marlin_scope_refuses_a_config_a_target_and_a_token_count_it_has_no_numerics_for():
    evs = scope_events(FEW, HASHED)
    for key, value in (("SWIGLU_LIMIT", None), ("ROUTED_SCALING_FACTOR", True), ("HASH_LAYERS", 3.0), ("VOCAB", None)):
        res, _ = _resolve(evs, HASHED, pattern=MoEMarlinSharedExperts({**C, key: value}))
        assert "profile config lacks NUM_EXPERTS / TOPK / H / I / SHARED_I / HASH_LAYERS / VOCAB / ROUTED_SCALING_FACTOR / SWIGLU_LIMIT" in \
            _reasons(res)[0], key
    # the Marlin configs are the capture's target's: another target has none, and neither has a launch past what its sweep determines
    other = dataclasses.replace(H200, num_sms=H200.num_sms + 1)
    res, _ = _resolve(evs, HASHED, target=other)
    assert _reasons(res) and "numerics_unregistered" in _reasons(res)[0], _reasons(res)
    beyond = next(M for M in range(MANY, 4096) if isinstance(served_config(E, TOPK, H, 2 * INTER, TOPK, M, target=H200), str))
    res, _ = _resolve(scope_events(beyond, HASHED), HASHED)
    assert f"numerics_unregistered: the capture's sweep does not determine the Marlin config at {beyond} tokens" in _reasons(res)[0]


def test_the_marlin_scope_refuses_a_gate_the_record_does_not_attest_or_the_build_does_not_bind():
    evs = scope_events(FEW, HASHED)
    res, _ = _resolve(evs, HASHED, attested=False)
    assert "moe_forward_shared has no gate mm, and the record does not attest the cuteDSL ll_bf16_gemm tier" in _reasons(res)[0]
    res, _ = _resolve(evs, HASHED, target=dataclasses.replace(H200, batch_invariant=True))
    assert "no registered bf16 x bf16 -> f32 gate Definition for the cutedsl tier" in _reasons(res)[0]
    # the gate's refusals name the router whose scores it writes
    for layer, router in ((HASHED, "topk_softplus_sqrt"), (SCORED, "_dsv4_topk_kernel")):
        res, _ = _resolve(edited(scope_events(MANY, layer), "mm", 0, mat2=view(99, (H, E), strides=(1, H))), layer)
        assert _reasons(res) == [f"the gate weight is not a bf16 [E,H] weight Root (<gate>.weight beside the {router} weight)"]
    # the cuteDSL gate's output is made ahead of the shared expert's first GEMM, not of its second
    res, _ = _resolve(swapped(evs, _calls(evs, "empty")[0], _calls(evs, "dynamic_flashinfer_deepgemm_blockscale_gemm")[0]), HASHED)
    late = "the scores topk_softplus_sqrt reads are not one in-scope aten.empty ahead of the shared expert (the cuteDSL gate's output)"
    assert _reasons(res) == [late]


@pytest.mark.parametrize("layer,role,reason", [
    (HASHED, "moe_router_hash", "the record target's vocabulary binds no 'moe_router_hash'"),
    (SCORED, "moe_router_sqrtsoftplus", "the record target's vocabulary binds no 'moe_router_sqrtsoftplus'"),
    (HASHED, "marlin_moe_align", "the record target's vocabulary binds no 'marlin_moe_align'"),
    (HASHED, "marlin_moe_gemm_w", "the w2 launch: the record target's vocabulary binds no 'marlin_moe_gemm_w'"),
    (HASHED, "moe_sum", "the record target's vocabulary binds no 'moe_sum'"),
])
def test_the_marlin_scope_refuses_a_role_the_targets_vocabulary_does_not_bind(monkeypatch, layer, role, reason):
    """Every target's vocabulary binds the scope's roles today, so `vocab_kind`'s refusal of `role` is stood in for, in its words."""
    def kind(ctx, asked, **statics):
        if asked == role:
            raise ValueError(f"the record target's vocabulary binds no {asked!r}")
        return vocab_kind(ctx, asked, **statics)

    monkeypatch.setattr(gen_marlin_moe_patterns, "vocab_kind", kind)
    res, _ = _resolve(scope_events(FEW, layer), layer)
    assert _reasons(res) == [reason]


def test_the_marlin_scope_leaves_another_op_and_a_launch_alone():
    evs = scope_events(FEW, SCORED)
    start = start_of(FEW)
    se = step_events([start, *evs])
    scope, pattern = se.roots[0], MoEMarlinSharedExperts(C)
    launch = next(n for n in scope.children if n.is_triton)
    assert pattern.match(launch, ctx_of(start, SCORED)) is None and pattern.match(se.roots[1], ctx_of(start, SCORED)) is None
