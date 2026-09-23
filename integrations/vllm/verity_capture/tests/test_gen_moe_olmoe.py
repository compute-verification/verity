"""Lane `moe`: the OLMOE profile module (`profiles.gen_moe_olmoe`) and the recognised-Unsupported MoE scope pattern
(`profiles.gen_moe_patterns.MoEBlockScope`), on synthetic events shaped like the pinned vLLM's `vllm.moe_forward` body
(the real log's shapes are in out/gen/lanes/moe/MECHANISM_MOE.md)."""

from __future__ import annotations

from verity_capture import analytic
from verity_capture.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, TritonLaunch, ViewDesc
from verity_capture.profiles import gen_moe_olmoe as OLMOE
from verity_capture.profiles import profile_for_case, profile_module_for_case
from verity_capture.profiles.gen_moe_patterns import MOE_KERNELS_EXPECTED, MoEBlockScope, moe_subtree_census
from verity_capture.resolver import StepContext, Unsupported, resolve_step
from verity_capture.tree import step_events

C = OLMOE.C
H, I, E, TOPK, V = C["H"], C["I"], C["NUM_EXPERTS"], C["TOPK"], C["V"]
CUDA = "cuda:0"


def _contig(shape):
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return tuple(reversed(out))


def view(alloc, shape, dtype="bfloat16", strides=None, offset=0):
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides or _contig(shape)), dtype, CUDA)


# --- the profile module ------------------------------------------------------------------------------------------

def test_config_facts_of_olmoe():
    assert (C["H"], C["NL"], C["NH"], C["KVH"], C["D"], C["I"], C["V"], C["MAX_POS"]) == (2048, 16, 16, 16, 128, 1024, 50304, 4096)
    assert (C["NUM_EXPERTS"], C["TOPK"], C["NORM_TOPK_PROB"]) == (64, 8, False)
    assert C["EPS"] == 1e-5 and OLMOE.PROV["EPS"] == "hand"                 # config.json carries no epsilon
    assert C["QK_NORM"] is True and C["RENORMALIZE"] is False and OLMOE.PROV["QK_NORM"] == "hand"
    assert C["BIAS"] is False and C["tie_word_embeddings"] is False and C["BN"] == 64 and C["QKV"] == 3 * 2048
    assert OLMOE.PROFILE.notes["case"] == "OLMOE" and OLMOE.PROFILE.accepted is False
    # coverage campaign (lane cov-moe): the role maps to the constructive ov-moe profile (PROFILE_BY_CASE), which builds on this one
    assert profile_module_for_case("OLMOE") == "gen_ov_moe_olmoe"
    assert profile_for_case("OLMOE").notes["wave1_profile"] == OLMOE.PROFILE_ID


def test_pattern_list_is_lane_specializations_then_base():
    from verity_capture.profiles import vllm_d9105ea80_sm89_eager as base
    names = [getattr(p, "name", type(p).__name__) for p in OLMOE.PROFILE.patterns]
    base_names = [getattr(p, "name", type(p).__name__) for p in base.make_patterns(C)]
    assert names[:2] == ["gemm.matmul_kernel_persistent.tuned_ada", "moe_block_scope"] and names[2:] == base_names
    assert OLMOE.PROFILE.kernel_allowlist == base.PROFILE.kernel_allowlist      # no profile-side allowlisting


def test_weight_fields_dense_parts_and_unbound_moe_weights():
    wf = OLMOE.weight_field
    assert wf("model.embed_tokens.weight") == ("embed",)
    assert wf("lm_head.weight") == ("lm_head",)                                # untied: a field of its own
    assert wf("model.norm.weight") == ("norm",)
    assert wf("model.layers.7.input_layernorm.weight") == ("layers", 7, "ln1")
    assert wf("model.layers.7.self_attn.qkv_proj.weight") == ("layers", 7, "qkv_w")
    assert wf("model.layers.7.self_attn.o_proj.weight") == ("layers", 7, "o_w")
    assert wf("model.layers.7.post_attention_layernorm.weight") == ("layers", 7, "ln2")
    assert wf("model.layers.0.self_attn.rotary_emb.cos_sin_cache") == ("cos_sin",)
    assert wf("model.layers.7.self_attn.qkv_proj.bias") is None                # OLMoE has no qkv bias
    assert wf("model.layers.7.mlp.gate_up_proj.weight") is None                # dense-only names never bind
    # the OLMoE-specific weights bind iff the (PROPOSED) Record fields exist
    wt = analytic.weights_type(C)
    layer_fields = {k for k, _ in wt.fields[wt.index("layers")][1].elem.fields}
    for name, field in OLMOE.MOE_LAYER_FIELDS.items():
        got = wf(f"model.layers.3.{name}")
        if field in layer_fields:
            assert got == ("layers", 3, field)
            assert OLMOE.root_class(f"model.layers.3.{name}") == "weight"
        else:
            assert got is None
            assert OLMOE.root_class(f"model.layers.3.{name}") == "unbound-moe-weight"
    assert OLMOE.root_class("model.layers.9.self_attn.rotary_emb.cos_sin_cache") == "alias-of-cos_sin"
    assert OLMOE.root_class("model.layers.9.self_attn.attn.k_scale") == "ignored"
    assert OLMOE.root_class("model.layers.9.mlp.gate_up_proj.weight") == "unknown"


def _header(cfg: dict) -> RunHeader:
    return RunHeader("verity-capture/log/v1", "2026-09-17T00:00:00Z", "e1ed15745", {"vllm": "d9105ea80"},
                     {"repo": OLMOE.MODEL, "config": cfg}, {"requests": []})


def test_check_header_accepts_untied_and_checks_moe_keys():
    import json
    cfg = json.loads(OLMOE.OLMOE_CONFIG.read_text())
    assert OLMOE.check_header(_header(cfg)) == []
    bad = dict(cfg, num_experts=8, num_experts_per_tok=2, clip_qkv=8.0)
    msgs = OLMOE.check_header(_header(bad))
    assert any(m.startswith("num_experts=8") for m in msgs) and any(m.startswith("num_experts_per_tok=2") for m in msgs)
    assert any(m.startswith("clip_qkv=8.0") for m in msgs)
    assert OLMOE.check_header(_header(dict(cfg, hidden_size=1024))) == ["hidden_size=1024 != C[H]=2048"]


# --- the MoE scope on synthetic events ------------------------------------------------------------------------------

def moe_scope_events(M: int, step: int = 0):
    """`vllm.moe_forward(hidden, router_logits, layer_name)` with the body vLLM d9105ea80 runs for an unquantized bf16
    FusedMoE with `renormalize=False`: `_moe_C.topk_softmax` (fp32 weights + int32 ids), `moe_align_block_size`
    (sorted / padded token ids per expert), `fused_moe_kernel` gate|up (N=2I, K=H), `_C.silu_and_mul`, `fused_moe_kernel`
    down (N=H, K=I, weights multiplied in), `_moe_C.moe_sum` over the TOPK slots.  Ids and shapes are those roles; the
    real metadata is in the mechanism note."""
    next_id = [1000]

    def nid():
        next_id[0] += 1
        return next_id[0]

    hidden = view(1, (M, H))
    logits = view(2, (M, E))
    out = view(3, (M, H))
    evs = []
    scope = nid()
    evs.append(DispatchEnter(scope, None, step, "forward", "vllm.moe_forward.default",
                             "vllm::moe_forward(Tensor hidden_states, Tensor router_logits, str layer_name) -> Tensor",
                             [hidden, logits, "model.layers.0.mlp.experts"], {}, 7, "model.layers.0.mlp.experts"))
    topk_w, topk_ids, tok_exp = view(4, (M, TOPK), "float32"), view(5, (M, TOPK), "int32"), view(6, (M, TOPK), "int32")
    i = nid()
    evs.append(DispatchEnter(i, scope, step, "forward", "_moe_C.topk_softmax.default",
                             "_moe_C::topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! token_expert_indices, Tensor gating_output) -> ()",
                             [topk_w, topk_ids, tok_exp, logits], {}, 7, None))
    evs.append(DispatchExit(i, None))
    block_m = 16
    max_pad = M * TOPK + E * (block_m - 1)
    sorted_ids, expert_ids, num_pad = view(7, (max_pad,), "int32"), view(8, ((max_pad + block_m - 1) // block_m,), "int32"), view(9, (1,), "int32")
    i = nid()
    evs.append(DispatchEnter(i, scope, step, "forward", "_moe_C.moe_align_block_size.default",
                             "_moe_C::moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! experts_ids, Tensor! num_tokens_post_pad) -> ()",
                             [topk_ids, E, block_m, sorted_ids, expert_ids, num_pad], {}, 7, None))
    evs.append(DispatchExit(i, None))
    w13, w2 = view(20, (E, 2 * I, H)), view(21, (E, H, I))
    inter = view(10, (M, TOPK, 2 * I))
    consts = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1, "MUL_ROUTED_WEIGHT": False,
              "top_k": TOPK, "compute_type": "bf16", "use_fp8_w8a8": False, "use_int8_w8a8": False, "use_int8_w8a16": False,
              "use_int4_w4a16": False, "per_channel_quant": False, "even_Ks": True}
    grid1 = [((max_pad + block_m - 1) // block_m) * ((2 * I + 63) // 64)]
    evs.append(TritonLaunch(nid(), scope, step, "forward", "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel", "sha",
                            {"a_ptr": hidden, "b_ptr": w13, "c_ptr": inter, "topk_weights_ptr": topk_w, "sorted_token_ids_ptr": sorted_ids,
                             "expert_ids_ptr": expert_ids, "num_tokens_post_padded_ptr": num_pad, "N": 2 * I, "K": H, **consts},
                            consts, grid1, 4, 3, 7))
    act = view(11, (M, TOPK, I))
    i = nid()
    evs.append(DispatchEnter(i, scope, step, "forward", "_C.silu_and_mul.default", "_C::silu_and_mul(Tensor! result, Tensor input) -> ()",
                             [view(11, (M * TOPK, I)), view(10, (M * TOPK, 2 * I))], {}, 7, None))
    evs.append(DispatchExit(i, None))
    down = view(12, (M, TOPK, H))
    consts2 = dict(consts, MUL_ROUTED_WEIGHT=True, top_k=1)
    grid2 = [((max_pad + block_m - 1) // block_m) * ((H + 63) // 64)]
    evs.append(TritonLaunch(nid(), scope, step, "forward", "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel", "sha",
                            {"a_ptr": act, "b_ptr": w2, "c_ptr": down, "topk_weights_ptr": topk_w, "sorted_token_ids_ptr": sorted_ids,
                             "expert_ids_ptr": expert_ids, "num_tokens_post_padded_ptr": num_pad, "N": H, "K": I, **consts2},
                            consts2, grid2, 4, 3, 7))
    i = nid()
    evs.append(DispatchEnter(i, scope, step, "forward", "_moe_C.moe_sum.default", "_moe_C::moe_sum(Tensor input, Tensor! output) -> ()",
                             [down, out], {}, 7, None))
    evs.append(DispatchExit(i, None))
    evs.append(DispatchExit(scope, out))
    return evs, hidden, logits, out


def _start(M: int, step: int = 0) -> StepStart:
    rows = [("r0", p) for p in range(M)]
    return StepStart(step, rows, list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0, 1, 2, 3]}, 16,
                     list(range(M)), [("r0", M - 1)], [M - 1], 1, M, M, None, None, "prefill")


def _ctx(start: StepStart) -> StepContext:
    roots = {20: [Root(view(20, (E, 2 * I, H)), "weight", "model.layers.0.mlp.experts.w13_weight")],
             21: [Root(view(21, (E, H, I)), "weight", "model.layers.0.mlp.experts.w2_weight")]}
    return StepContext(_header({}), start, None, [], roots, C, OLMOE.PROFILE)


def test_moe_scope_is_one_recognised_unsupported_consuming_its_subtree():
    M = 32
    evs, hidden, logits, out = moe_scope_events(M)
    start = _start(M)
    se = step_events([start, *evs])
    res = resolve_step(se, _ctx(start))
    assert len(res) == 1 and isinstance(res[0], Unsupported)
    u = res[0]
    assert u.op == "vllm.moe_forward.default" and "data-dependent" in u.reason and "MECHANISM_MOE.md" in u.reason
    node = se.roots[0]
    assert u.consumed == tuple(sorted(node.ids())) and len(u.consumed) == 1 + 4 + 2       # scope + 4 ops + 2 launches
    d = u.detail
    assert d["rows"] == M and d["config"] == {"E": E, "TOPK": TOPK, "RENORMALIZE": False, "NORM_TOPK_PROB": False, "I": I, "H": H}
    assert d["hidden_states"]["shape"] == [M, H] and d["router_logits"]["shape"] == [M, E] and d["output"][0]["shape"] == [M, H]
    assert d["children"] == ["_moe_C.topk_softmax.default", "_moe_C.moe_align_block_size.default",
                             "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel", "_C.silu_and_mul.default",
                             "vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel", "_moe_C.moe_sum.default"]
    census = d["census"]
    assert [l["fn"] for l in census["triton_launches"]] == ["fused_moe_kernel", "fused_moe_kernel"]
    assert census["triton_launches"][0]["constexprs"]["MUL_ROUTED_WEIGHT"] is False
    assert census["triton_launches"][1]["constexprs"]["MUL_ROUTED_WEIGHT"] is True
    assert census["triton_launches"][0]["args"]["N"] == 2 * I and census["triton_launches"][1]["args"]["K"] == I
    assert [o["op"] for o in census["dispatcher_ops"]] == ["_moe_C.topk_softmax.default", "_moe_C.moe_align_block_size.default",
                                                            "_C.silu_and_mul.default", "_moe_C.moe_sum.default"]
    assert census["dispatcher_ops"][0]["in"][0]["dtype"] == "float32"            # topk weights are fp32
    assert census["n_events"] == 6
    import json
    json.dumps(d)                                                                # survives the resolution report


def test_moe_scope_declines_everything_else():
    pat = MoEBlockScope(C)
    M = 4
    evs, *_ = moe_scope_events(M)
    start = _start(M)
    se = step_events([start, *evs])
    ctx = _ctx(start)
    for child in se.roots[0].children:
        assert pat.match(child, ctx) is None
    assert set(MOE_KERNELS_EXPECTED) >= {"fused_moe_kernel", "topk_softmax", "moe_align_block_size", "moe_sum"}
    assert moe_subtree_census(se.roots[0])["n_events"] == 6


def test_moe_scope_shared_variant_and_decode_row():
    evs, *_ = moe_scope_events(1, step=3)
    evs[0] = DispatchEnter(evs[0].id, None, 3, "forward", "vllm.moe_forward_shared.default", evs[0].schema, evs[0].args, {}, 7, None)
    start = _start(1, step=3)
    se = step_events([start, *evs])
    res = resolve_step(se, _ctx(start))
    assert len(res) == 1 and isinstance(res[0], Unsupported) and res[0].detail["rows"] == 1


# --- the tuned sm89 GEMM tilings ---------------------------------------------------------------------------------------

from verity_capture.profiles import vllm_d9105ea80_sm89_eager as BASE
from verity_capture.profiles.gen_moe_patterns import ADA_TUNED_MATMUL, GemmLaunchTunedAda, tuned_matmul_config
from verity_capture.resolver import Compute, PerRow, Shared


def test_tuned_matmul_config_follows_vllm_bucket_rule():
    # (N, K) = (2048, 2048): first bucket with max_m >= M; above the range the last bucket; untuned shapes -> None
    assert tuned_matmul_config(1, 2048, 2048) == {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 5}
    assert tuned_matmul_config(32, 2048, 2048)["BLOCK_SIZE_M"] == 16 and tuned_matmul_config(32, 2048, 2048)["num_stages"] == 5
    assert tuned_matmul_config(33, 2048, 2048) == tuned_matmul_config(64, 2048, 2048)
    assert tuned_matmul_config(100000, 2048, 2048) == {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 5}
    assert tuned_matmul_config(1, 6144, 2048) is None and tuned_matmul_config(1, 64, 2048) is None and tuned_matmul_config(1, 50304, 2048) is None
    assert tuned_matmul_config(4, 2048, 6144)["BLOCK_SIZE_K"] == 128                       # the (2048, 6144) key has block_k 128
    assert all(len(b) == 10 for _, b in ADA_TUNED_MATMUL.values())


def gemm_events(M: int, N: int, K: int, consts: dict, num_warps: int, num_stages: int, step: int = 1):
    """`aten.t(weight)` alias then the root-level batch-invariant launch (as the M1 logs show it)."""
    w = view(30, (N, K))
    wt = ViewDesc(30, 0, (K, N), (1, K), "bfloat16", CUDA)
    a = view(31, (M, K))
    c = view(32, (M, N))
    evs = [DispatchEnter(2001, None, step, "forward", "aten.t.default", "aten::t(Tensor(a) self) -> Tensor(a)", [w], {}, 7, None),
           DispatchExit(2001, wt)]
    args = {"a_ptr": a, "b_ptr": wt, "c_ptr": c, "bias_ptr": None, "M": M, "N": N, "K": K, "stride_am": K, "stride_ak": 1,
            "stride_bk": 1, "stride_bn": K, "stride_cm": N, "stride_cn": 1, **consts}
    grid = [min(consts["NUM_SMS"], -(-M // consts["BLOCK_SIZE_M"]) * -(-N // consts["BLOCK_SIZE_N"]))]
    evs.append(TritonLaunch(2002, None, step, "forward", "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent",
                            "sha", args, dict(consts), grid, num_warps, num_stages, 7))
    return evs, a, w, c


def _gemm_ctx(start: StepStart) -> StepContext:
    roots = {30: [Root(view(30, (2048, 2048)), "weight", "model.layers.0.self_attn.o_proj.weight")]}
    return StepContext(_header({}), start, None, [], roots, C, OLMOE.PROFILE)


def _tuned_consts(M, N, K):
    cfg = tuned_matmul_config(M, N, K)
    return {**BASE.GEMM_CONSTEXPRS, "NUM_SMS": BASE.GEMM_NUM_SMS,
            **{k: cfg[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")}}, cfg["num_warps"], cfg["num_stages"]


def test_tuned_o_proj_launch_resolves_to_gemm_v1_decode_and_prefill():
    for M in (1, 32):
        consts, warps, stages = _tuned_consts(M, 2048, 2048)
        evs, a, w, c = gemm_events(M, 2048, 2048, consts, warps, stages)
        start = _start(M)
        res = resolve_step(step_events([start, *evs]), _gemm_ctx(start))
        comps = [r for r in res if isinstance(r, Compute)]
        assert len(comps) == 1, res
        g = comps[0]
        assert g.spec.id == "Gemm_v1{K=2048,N=2048}" and g.rows == M
        assert g.pattern == "gemm.matmul_kernel_persistent.tuned_ada" and g.kernels == ("matmul_kernel_persistent",)
        assert g.operands == (PerRow(a), Shared(w)) and g.outputs == (PerRow(c),)
        assert "o_proj" in g.hint and len(g.attribution) == M


def test_tuned_pattern_declines_untuned_shapes_and_wrong_pins():
    pat = OLMOE.PROFILE.patterns[0]
    assert isinstance(pat, GemmLaunchTunedAda)
    # an untuned shape with the base pins: the tuned pattern declines, the base GemmLaunch resolves it
    consts = {**BASE.GEMM_CONSTEXPRS, "NUM_SMS": BASE.GEMM_NUM_SMS}
    evs, *_ = gemm_events(1, 6144, 2048, consts, BASE.GEMM_NUM_WARPS, BASE.GEMM_NUM_STAGES)
    start = _start(1)
    se = step_events([start, *evs])
    ctx = _gemm_ctx(start)
    assert pat.match(se.roots[1], ctx) is None
    res = resolve_step(se, ctx)
    g = [r for r in res if isinstance(r, Compute)][0]
    assert g.spec.id == "Gemm_v1{K=2048,N=6144}" and g.pattern == "gemm.matmul_kernel_persistent"
    # a tuned shape whose launch does NOT carry the bucket's pins: declined -> the base reports the mismatch
    consts, warps, stages = _tuned_consts(1, 2048, 2048)
    evs, *_ = gemm_events(1, 2048, 2048, {**consts, "BLOCK_SIZE_N": 64}, warps, stages)
    se = step_events([start, *evs])
    assert pat.match(se.roots[1], ctx) is None
    res = resolve_step(se, ctx)
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and "pinned" in uns[0].reason
    # the bucket's tiles with the wrong pipeline depth: declined as well
    evs, *_ = gemm_events(1, 2048, 2048, consts, warps, 3)
    se = step_events([start, *evs])
    assert pat.match(se.roots[1], ctx) is None
    # the base 128x128 pins on a tuned shape: not what vLLM launches for it -> declined by the tuned pattern too
    evs, *_ = gemm_events(1, 2048, 2048, {**BASE.GEMM_CONSTEXPRS, "NUM_SMS": 128}, 8, 3)
    se = step_events([start, *evs])
    assert pat.match(se.roots[1], ctx) is None
