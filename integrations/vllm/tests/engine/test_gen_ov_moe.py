"""Lane `ov-moe`: the pair-major MoE definitions (`verity_vllm.program.registry.quarantine.ov_moe.moe`), the `MoEFusedExperts`
recognizer (positive and negative cases on synthetic events shaped exactly like the pinned vLLM's `vllm.moe_forward`
body as recorded in out/gen/lanes/moe/runs/olmoe_3x1_1req), and — when wave 1's real captures are present in the
worktree — the end-to-end resolve / fold / VU-population facts quoted in out/gen/lanes/ov-moe/REPORT.md."""

from __future__ import annotations

import importlib

import math
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, TritonLaunch, ViewDesc
from verity_vllm.engine.profiles.gen_ov_moe_patterns import FUSED_MOE_PINS, MoEFusedExperts

OV = importlib.import_module("verity_vllm.engine.profiles.derived_OLMOE")
from verity_vllm.observe.fold.resolver import Compute, Group, StepContext, Transparent, Unsupported, resolve_step
from verity_vllm.observe.fold.tree import step_events
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluate import evaluate_region
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.program import Program
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.quarantine.ov_moe import moe as Q

C = OV.C
H, I, E, TOPK = C["H"], C["I"], C["NUM_EXPERTS"], C["TOPK"]
CUDA = "cuda:0"
REPO = Path(__file__).resolve().parents[2]
W1_RUNS = REPO / "out" / "gen" / "lanes" / "moe" / "runs"


# --- definitions -----------------------------------------------------------------------------------------------------

def _eval(prog: Program, inputs: list[list[int]]) -> list[int]:
    Cc = prog.circuit
    t: dict[int, int] = {}
    for pi, vals in enumerate(inputs):
        coll = prog.input_coll(pi)
        assert coll.type.leaves == len(vals)
        for j, v in enumerate(vals):
            t[resolve(Cc.root_scope, coll.refs[j])] = v
    evaluate_region(Cc, Cc[0:prog.gates], t)
    return [t[g] for g in Cc.root_outputs()]


def _bf16_bits(a) -> list[int]:
    return [int(x) >> 16 for x in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


def _f32_bits(a) -> list[int]:
    return [int(x) for x in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


def _bf16_round(a):
    return np.array([b << 16 for b in _bf16_bits(a)], dtype=np.uint32).view(np.float32).reshape(np.shape(a))


def test_pair_gemm_with_expert_index_operand_equals_dense_gemm_on_the_selected_expert():
    """`MoeExpertGemm_v1(x, e, W)` == `Gemm_v1(x, W[e])` for every e (the gather over the expert axis is inside the body);
    `MoeExpertGemmW_v1` multiplies the routing weight into the f32 accumulator before the single bf16 rounding."""
    rng = np.random.default_rng(0)
    Ee, K, N = 4, 32, 6
    x = _bf16_round(rng.standard_normal(K).astype(np.float32))
    w = _bf16_round(rng.standard_normal((Ee, N, K)).astype(np.float32))
    prog = Program(bind(Q.MoeExpertGemm, E=Ee, K=K, N=N))
    progw = Program(bind(Q.MoeExpertGemmW, E=Ee, K=K, N=N))
    dense = Program(bind(b1.Gemm, K=K, N=N))
    dot = Program(bind(b1.DotBf16, K=K))
    for e in range(Ee):
        assert _eval(prog, [_bf16_bits(x), [e], _bf16_bits(w)]) == _eval(dense, [_bf16_bits(x), _bf16_bits(w[e])])
        g = np.float32(rng.random())
        ref = [P.F2fpBf16.evaluate(P.F32Mul.evaluate(_eval(dot, [_bf16_bits(x), _bf16_bits(w[e, n])])[0], _f32_bits([g])[0])) for n in range(N)]
        assert _eval(progw, [_bf16_bits(x), [e], _f32_bits([g]), _bf16_bits(w)]) == ref
    # per coordinate: K gathers + K/16 mma steps + zero + cvt (+ the routed-weight mul); the expert table is K*N*E input leaves
    assert prog.gates - prog.input_gates == N * (K + K // 16 + 2) and progw.gates - progw.input_gates == N * (K + K // 16 + 3)


def test_moe_sum_is_the_ascending_f32_chain_then_one_rounding():
    rng = np.random.default_rng(1)
    Hh = 8
    y = _bf16_round(rng.standard_normal((TOPK, Hh)).astype(np.float32))
    out = _eval(Program(bind(Q.MoeSum, TOPK=TOPK, H=Hh)), [_bf16_bits(y)])
    ref = []
    for h in range(Hh):
        acc = np.float32(0)
        for k in range(TOPK):
            acc = np.float32(acc + y[k, h])
        ref.append(P.F2fpBf16.evaluate(_f32_bits([acc])[0]))
    assert out == ref


def test_router_stand_in_ids_and_tie_rule():
    """Top-k ids agree with a float64 softmax argsort (stable), weights agree to f32 rounding; ties go to the lower expert."""
    rng = np.random.default_rng(2)
    Ee, K_, VPT = 16, 4, 4
    logits = _bf16_round(rng.standard_normal(Ee).astype(np.float32) * 2)
    prog = Program(bind(Q.MoeRouterTopK, E=Ee, TOPK=K_, VPT=VPT))
    out = _eval(prog, [_bf16_bits(logits)])
    wts = np.array(out[:K_], dtype=np.uint32).view(np.float32)
    p = np.exp(logits.astype(np.float64) - logits.max())
    p /= p.sum()
    order = [int(i) for i in np.argsort(-p, kind="stable")[:K_]]
    assert out[K_:] == order and np.allclose(wts, p[order], rtol=2e-6)
    tie = logits.copy()
    tie[5] = tie[2] = np.float32(10.0)
    assert _eval(prog, [_bf16_bits(tie)])[K_:K_ + 2] == [2, 5]


def test_quarantine_identity_and_conformance_labels():
    assert Q.SEMANTIC_ID == "ov-moe/MoeExpertGemm@v1"
    assert Q.MoeRouterTopK.conformance == "not-yet-established" and Q.NvExpf.id == "NvExpf_v1"   # cov-moe: the CUDA expf sequence, no stand-in
    for d in (Q.MoeExpertGemm, Q.MoeExpertGemmW, Q.MoeSum):
        assert d.conformance == "not-yet-established" and d.version == 1


# --- NvExpf: the CUDA math-library expf as nvcc emits it (cov-moe) -------------------------------------------------------

def test_nv_expf_composite_equals_the_direct_sequence_and_is_total():
    prog = Program(bind(Q.NvExpf))
    specials = [0, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x80000001, 0x42B17218,
                0xC2CFF1B5, 0xC2AEAC50, 0x7F7FFFFF, 0xFF7FFFFF, 0x41200000, 0xC1200000, 0x00800000]
    rng = np.random.default_rng(0)
    for x in specials + [int(v) for v in rng.integers(0, 2**32, size=200, dtype=np.uint64)]:
        assert _eval(prog, [[x]])[0] == Q.nv_expf_bits(x)
    f = lambda v: P.f32_of(Q.nv_expf_bits(P.bits_of(np.float32(v))))
    assert f(0.0) == 1.0 and f(-0.0) == 1.0 and f(np.inf) == np.inf and f(-np.inf) == 0.0 and np.isnan(f(np.nan))
    assert f(89.0) == np.inf and f(-104.0) == 0.0 and f(-87.33655) == np.float32(1.1754907e-38)     # over/underflow, no ftz on the product
    # 1 ulp off the correctly rounded value, as libdevice is: exp(-1) = 0.367879441171... whose nearest f32 is 0x3EBC5AB2
    # (from the double, which is far from the f32 midpoint, so no double rounding).  NOT `np.exp(np.float32)` as the reference:
    # numpy's single-precision exp is platform code (macOS gives 0x3EBC5AB2, the pod's SIMD path 0x3EBC5AB1).
    assert Q.nv_expf_bits(P.bits_of(np.float32(-1.0))) == 0x3EBC5AB3 and P.bits_of(np.float32(math.exp(-1.0))) == 0x3EBC5AB2


def test_nv_expf_helpers_are_the_ptx_ops():
    # cvt.sat.f32.f32
    assert Q.F32Sat.evaluate(0x7FC00000) == 0 and Q.F32Sat.evaluate(0x80000000) == 0 and Q.F32Sat.evaluate(0xBF800000) == 0
    assert Q.F32Sat.evaluate(0x3F800001) == 0x3F800000 and Q.F32Sat.evaluate(0x3F000000) == 0x3F000000 and Q.F32Sat.evaluate(0x7F800000) == 0x3F800000
    # fma.rm.f32: one rounding toward -inf (RN would round 12582913 + 252*s up when the fraction is >= .5)
    s = P.bits_of(np.float32(0.9999))
    exact = 12582913 + 252 * float(np.float32(0.9999))
    assert P.f32_of(Q.F32FmaRm.evaluate(s, P.bits_of(252.0), P.bits_of(12582913.0))) == np.float32(int(exact))
    assert P.f32_of(P.F32Fma.evaluate(s, P.bits_of(252.0), P.bits_of(12582913.0))) == np.float32(round(exact))
    assert Q.F32FmaRm.evaluate(P.bits_of(-1.0), P.bits_of(1e-45), 0) == 0x80000001                      # -tiny rounds down to -min_subnormal
    assert Q.F32FmaRm.evaluate(P.bits_of(1.0), P.bits_of(1e-45), 0) == 0x00000001
    # neg / exponent-field scaling
    assert Q.F32Neg.evaluate(0) == 0x80000000 and Q.F32Neg.evaluate(0x7FC00000) == 0xFFC00000
    assert P.f32_of(Q.F32BitsShl23.evaluate(P.bits_of(12582912.0 + 127))) == 1.0 and P.f32_of(Q.F32BitsShl23.evaluate(P.bits_of(12582912.0 + 1))) == np.float32(2.0 ** -126)


REC_ROUTER = REPO / "out" / "gen" / "lanes" / "ov-moe" / "difftest" / "moe_router_topk"


@pytest.mark.skipif(not (REC_ROUTER / "manifest.json").exists(), reason="lane ov-moe's recorded topk_softmax cases not in this checkout")
def test_router_program_is_bit_exact_on_the_recorded_kernel_cases():
    """The 24 `_moe_C::topk_softmax` cases recorded on the pod (bf16 logits in, f32 weights + i32 ids out): weights AND ids
    bit-exact with `NvExpf` (the correctly rounded stand-in matched 160/192 weight words, REPORT.md §4)."""
    import json
    man = json.loads((REC_ROUTER / "manifest.json").read_text())
    prog = Program(bind(Q.MoeRouterTopK, E=64, TOPK=8, VPT=8))
    for rec in man["cases"]:
        cdir = REC_ROUTER / f"case_{rec['i']}"
        xin = [int(b) for b in np.load(cdir / "in_0.npy").reshape(-1).astype(np.uint16)]
        w = [int(b) for b in np.load(cdir / "out_0.npy").reshape(-1).view(np.uint32)]
        ids = [int(b) for b in np.load(cdir / "out_1.npy").reshape(-1).view(np.uint32)]
        out = _eval(prog, [xin])
        assert out[:8] == w and out[8:] == ids, (rec["i"], rec["inputs"][0].get("kind"))


# --- synthetic scope (the recorded shapes of olmoe_3x1_1req, layer 0) ----------------------------------------------------

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


def moe_scope_events(M: int, *, step: int = 0, naive: bool = False, renormalize: bool = False, consts_up: dict | None = None,
                     consts_down: dict | None = None, swap_launches: bool = False, num_warps: int = 4, num_stages: int = 3):
    nid = [1000]

    def new():
        nid[0] += 1
        return nid[0]

    hidden, logits, out = view(295, (M, H)), view(296, (M, E)), view(300, (M, H))
    pad = view(259, (M,), "bool")
    evs = []
    scope = new()
    evs.append(DispatchEnter(scope, None, step, "forward", "vllm.moe_forward.default",
                             "vllm::moe_forward(Tensor(a0!) hidden_states, Tensor router_logits, Tensor? shared_experts_input, Tensor? input_ids, PyObject layer_name, SymInt hidden_dim_unpadded) -> Tensor",
                             [hidden, logits, None, None, "layer", 0], {}, 7, "model.layers.0.mlp.experts"))
    w_, ids, tok = view(297, (M, TOPK), "float32"), view(298, (M, TOPK), "int32"), view(299, (M, TOPK), "int32")
    for al, shp, dt in ((297, (M, TOPK), "float32"), (298, (M, TOPK), "int32"), (299, (M, TOPK), "int32")):
        i = new()
        evs.append(DispatchEnter(i, scope, step, "forward", "aten.empty.memory_format", "aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
                                 [list(shp), None, None, None, None, None], {}, 7, None))
        evs.append(DispatchExit(i, view(al, shp, dt)))
    i = new()
    evs.append(DispatchEnter(i, scope, step, "forward", "_moe_C.topk_softmax.default",
                             "_moe_C::topk_softmax(Tensor($0! -> ) topk_weights, Tensor($1! -> ) topk_indices, Tensor($2! -> ) token_expert_indices, Tensor gating_output, bool renormalize, Tensor? bias, Tensor? is_padding) -> ()",
                             [w_, ids, tok, logits, renormalize, None, pad], {}, 7, None))
    evs.append(DispatchExit(i, None))
    i = new()
    evs.append(DispatchEnter(i, scope, step, "forward", "aten.empty_like.default", "aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
                             [hidden, None, None, None, None, None], {}, 7, None))
    evs.append(DispatchExit(i, out))
    P_ = M * TOPK
    cache1 = view(301, (M, TOPK, 2 * I), offset=P_ * I)      # intermediate_cache1 after cache2's slab, as in the recorded workspace
    cache2 = view(301, (P_, I))
    cache3 = view(301, (M, TOPK, H), offset=P_ * I)
    if naive:
        ids_flat = view(298, (P_,), "int32")
        npad = view(304, (1,), "int32")
        sorted_ids, expert_ids = None, ids_flat
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
    c_dn.update(consts_down or {})

    def launch(a, b, c, weights, N, K, consts):
        args = {"a_ptr": a, "b_ptr": b, "c_ptr": c, "b_bias_ptr": None, "a_scale_ptr": None, "b_scale_ptr": None, "topk_weights_ptr": weights,
                "sorted_token_ids_ptr": sorted_ids, "expert_ids_ptr": expert_ids, "num_tokens_post_padded_ptr": npad, "N": N, "K": K, "EM": EM,
                "num_valid_tokens": P_, "stride_am": a.strides[0], "stride_ak": 1, "stride_be": N * K, "stride_bk": 1, "stride_bn": K,
                "stride_cm": N, "stride_cn": 1, **consts}
        return TritonLaunch(new(), scope, step, "forward", "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel", "sha", args, consts,
                            [(EM // 64) * (N // 64)], num_warps, num_stages, 7)

    up = launch(hidden, w13, cache1, None, 2 * I, H, c_up)
    down = launch(cache2, w2, cache3, w_, H, I, c_dn)
    evs.append(down if swap_launches else up)
    i = new()
    evs.append(DispatchEnter(i, scope, step, "forward", "_C.silu_and_mul.default", "_C::silu_and_mul(Tensor($0! -> ) result, Tensor input) -> ()",
                             [cache2, view(301, (P_, 2 * I), offset=P_ * I)], {}, 7, None))
    evs.append(DispatchExit(i, None))
    evs.append(up if swap_launches else down)
    i = new()
    evs.append(DispatchEnter(i, scope, step, "forward", "_moe_C.moe_sum.default", "_moe_C::moe_sum(Tensor input, Tensor($0! -> ) output, Tensor? topk_ids=None, Tensor? expert_map=None) -> ()",
                             [cache3, out, None, None], {}, 7, None))
    evs.append(DispatchExit(i, None))
    evs.append(DispatchExit(scope, out))
    return evs


def _start(M: int, step: int = 0, padded: int | None = None) -> StepStart:
    rows = [("r0", p) for p in range(M)]
    return StepStart(step, rows, list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0, 1, 2, 3]}, 16,
                     list(range(M)), [("r0", M - 1)], [M - 1], 1, M, padded if padded is not None else M, None, None, "prefill")


def _ctx(start: StepStart) -> StepContext:
    header = RunHeader("verity-capture/log/v1", "2026-09-17T00:00:00Z", "672e5db02", {"vllm": "d9105ea80"}, {"repo": OV.MODEL, "config": {}}, {"requests": []})
    roots = {7: [Root(view(7, (E, 2 * I, H)), "weight", "model.layers.0.mlp.experts.routed_experts.w13_weight")],
             8: [Root(view(8, (E, H, I)), "weight", "model.layers.0.mlp.experts.routed_experts.w2_weight")]}
    return StepContext(header, start, None, [], roots, C, OV.PROFILE)


def _resolve(M: int, **kw):
    start_kw = {k: kw.pop(k) for k in ("padded",) if k in kw}
    evs = moe_scope_events(M, **kw)
    start = _start(M, **start_kw)
    se = step_events([start, *evs])
    return resolve_step(se, _ctx(start)), se


@pytest.mark.parametrize("M,naive", [(3, False), (32, False), (1, True)])
def test_moe_scope_resolves_as_the_six_part_pair_major_group(M, naive):
    res, se = _resolve(M, naive=naive)
    assert len(res) == 1 and isinstance(res[0], Group)
    g = res[0]
    assert g.consumed == tuple(sorted(se.roots[0].ids()))
    computes = [p for p in g.parts if isinstance(p, Compute)]
    specs = [c.spec.id for c in computes]
    assert specs == [f"MoeRouterTopKOrdered_v1{{E={E},TOPK={TOPK},VPT=8}}", f"MoeExpertGemm_v1{{E={E},K={H},N={2 * I}}}", f"SiluMul_v2{{I={I}}}",
                     f"MoeExpertGemmW_v1{{E={E},K={I},N={H}}}", f"MoeSum_v1{{TOPK={TOPK},H={H}}}"]
    assert [c.rows for c in computes] == [M, M * TOPK, M * TOPK, M * TOPK, M]
    router, up, silu, down, msum = computes
    assert [type(o).__name__ for o in up.operands] == ["Gather", "PerRow", "Shared"]
    assert up.operands[0].indices == tuple(p // TOPK for p in range(M * TOPK))           # A row of pair p = token p // TOPK
    assert up.operands[1].view.alloc == router.outputs[1].view.alloc                      # e operand = the router's topk_ids words
    assert up.operands[2].view.shape == (E, 2 * I, H) and up.outputs[0].view.shape == (M * TOPK, 2 * I)
    assert [type(o).__name__ for o in down.operands] == ["PerRow", "PerRow", "PerRow", "Shared"]
    assert down.operands[2].view.alloc == router.outputs[0].view.alloc and down.operands[2].view.dtype == "float32"
    assert msum.operands[0].view.shape == (M, TOPK, H) and msum.outputs[0].view.shape == (M, H)
    kernels = [k for p in g.parts for k in p.kernels]
    assert kernels.count("fused_moe_kernel") == 2 and "topkGating" in kernels and "moe_sum_vec_kernel" in kernels and "act_and_mul_kernel" in kernels
    aligns = [p for p in g.parts if isinstance(p, Transparent) and any("moe_align_block_size" in k for k in p.kernels)]
    assert len(aligns) == (0 if naive else 1) and all(t.effect == "noop" for t in aligns)
    if aligns:
        # the claim must cover both of vLLM's dispatch paths as Kineto names them: the small-batch kernel (decode) and the
        # sorted pair (prefill; lane cov-moe: G1 residue 32 kernels on a 155-token prompt under the single-name claim)
        from verity_vllm.properties.census import claims_for
        declared = set(aligns[0].kernels)
        for kineto in ("void vllm::moe::moe_align_block_size_small_batch_expert_kernel<int>(int const*, int*, int*, int*, unsigned long, int, int, int)",
                       "void vllm::moe::moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int*, int, int, int, int, unsigned long)",
                       "void vllm::moe::count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, int*, unsigned long, int, int, int, bool)"):
            assert len(claims_for(kineto, declared)) == 1, kineto


@pytest.mark.parametrize("kw,needle", [
    (dict(renormalize=True), "renormalize=False"),
    (dict(consts_up={"use_fp8_w8a8": True}), "numeric constexprs"),
    (dict(consts_down={"SPLIT_K": 2}), "numeric constexprs"),
    (dict(consts_up={"BLOCK_SIZE_M": 16}), "tiling"),
    (dict(num_stages=4), "tiling"),
    (dict(swap_launches=True), "MUL_ROUTED_WEIGHT"),
    (dict(padded=8), "padding rows"),
])
def test_moe_scope_declines_with_the_observed_values(kw, needle):
    res, _ = _resolve(3, **kw)
    assert len(res) == 1 and isinstance(res[0], Unsupported), res
    assert needle in res[0].reason, res[0].reason


def test_profile_is_the_tuned_gemm_plus_the_pairwise_pattern_and_not_accepted():
    from verity_vllm.engine.profiles import vllm_d9105ea80_sm89_eager as BASE
    names = [getattr(p, "name", type(p).__name__) for p in OV.PROFILE.patterns]
    assert names[:2] == ["gemm.matmul_kernel_persistent.tuned", "moe.fused_experts.pairwise"] and "moe_block_scope" not in names
    assert OV.PROFILE.accepted is False and OV.PROFILE.notes["case"] == "OLMOE"
    assert OV.PROFILE.kernel_allowlist == BASE.PROFILE.kernel_allowlist                     # no profile-side allowlisting of MoE kernels
    assert OV.weight_field("model.layers.3.mlp.experts.routed_experts.w13_weight") == ("layers", 3, "w13")
    assert isinstance(OV.PROFILE.patterns[1], MoEFusedExperts)


# --- wave 1's real captures (present in the lane worktree, not in a fresh checkout) ----------------------------------------

def _w1_log(name: str) -> Path:
    p = W1_RUNS / name / "capture" / "log.jsonl.gz"
    if not p.is_file():
        pytest.skip(f"wave-1 capture {p} not present")
    return p


@pytest.mark.parametrize("name,steps", [("olmoe_3x1_1req", 1), ("olmoe_1x1_1req", 1)])
def test_real_capture_resolves_without_unsupported(name, steps):
    from verity_vllm.observe.fold.resolve_log import resolve_log
    from verity_vllm.observe.fold.resolver import parts_of
    _, results, _, _ = resolve_log(_w1_log(name), OV.PROFILE)
    assert len(results) == steps
    for sr in results:
        assert sr.error is None
        parts = [p for r in sr.resolutions for p in parts_of(r)]
        assert not [p for p in parts if isinstance(p, Unsupported)]
        specs = [p.spec.id for p in parts if isinstance(p, Compute)]
        assert specs.count(f"MoeExpertGemm_v1{{E={E},K={H},N={2 * I}}}") == C["NL"] and specs.count(f"MoeSum_v1{{TOPK={TOPK},H={H}}}") == C["NL"]


def test_real_capture_folds_and_every_moe_gemm_instance_has_coordinate_units():
    """The fold binds the whole 3-token prefill (no unresolved read); under the 8M-leaf budget every oversize pair-GEMM
    instance decomposes into N coordinate units that tile its returns — the VU coverage line of REPORT.md."""
    from verity_vllm.check.replay import vu_query as vq
    from verity_vllm.pipeline.match import fold_summary_result
    summ, result = fold_summary_result(_w1_log("olmoe_3x1_1req"), "gen_ov_moe_olmoe", with_program=True)
    assert summ["unsupported"] == 0 and summ["unresolved"] == 0 and summ["errors"] == 0 and summ["n_instances"] == 1736
    ix = vq.index(result, max_arg_leaves=8_000_000, depth=3)
    s = ix.summary()
    assert s["instances_without_units"] == 0
    up = s["specs"][f"MoeExpertGemm_v1{{E={E},K={H},N={2 * I}}}"]
    assert up["instances"] == 3 * TOPK * C["NL"] and up["vus"] == up["instances"] * 2 * I and up["granularity"] == [f"child:MoeExpertCoordinate_v1{{E={E},K={H}}}"]
    kk = next(k for k, m in ix.by_node.items() if m["spec"] == f"MoeExpertGemm_v1{{E={E},K={H},N={2 * I}}}")
    covered = set()
    for vu in ix.of_instance(kk):
        covered.update(vq.returns_of(result, vu))
    assert len(covered) == 2 * I
