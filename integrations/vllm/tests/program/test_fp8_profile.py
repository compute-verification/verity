"""The block-scaled FP8 / Qwen3 eager Match profile: `dense_generic` derives it from the checkpoint's config.json (no
per-model module), the weights Record carries the e4m3 weights + f32 block scales + per-head q/k norm weights, and the two FP8
patterns resolve the recorded custom ops (`_C.per_token_group_fp8_quant`, `_C.cutlass_scaled_mm` blockwise) into the registered
families -- and refuse every knob / target / layout the pinned Definitions do not cover.  Op names, schemas and argument orders are
those of the H100 capture (Qwen3-4B-Instruct-2507-FP8, vLLM d9105ea80, VLLM_USE_DEEP_GEMM=0)."""
from __future__ import annotations

import hashlib
import importlib
import json

import pytest

from verity_vllm.program import model as analytic
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, ViewDesc
from verity_vllm.observe.fold.patterns import ALIAS_OPS, classify_transparent
from verity_vllm.observe.fold.patterns.fp8 import CutlassScaledMmFp8Block, PerTokenGroupFp8Quant
from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.observe.fold.profiles import generic
from verity_vllm.observe.fold.profiles import profile_for_case, profile_module_for_case
from verity_vllm.observe.fold.resolver import Compute, PerRow, Shared, StepContext, Transparent, Unsupported, spec_label
from verity_vllm.observe.fold.tree import build_trees

ROLE = "QWEN3_4B_FP8"
CFG_PATH = dg.HF_CONFIG_DIR / f"{ROLE}.config.json"
CUDA = "cuda:0"
M, K, N, G = 11, 2560, 6144, 128


def _contig(shape):
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return tuple(reversed(out))


def view(alloc, shape, dtype, strides=None, offset=0):
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides or _contig(shape)), dtype, CUDA)


# ---------------------------------------------------------------- profile / record


def test_shipped_fp8_config_is_the_manifest_bytes():
    entry = dg.checkpoint_entry(ROLE)
    assert entry is not None and entry["repo"] == "Qwen/Qwen3-4B-Instruct-2507-FP8"
    assert hashlib.sha256(CFG_PATH.read_bytes()).hexdigest() == entry["config_sha256"]


def test_qwen3_fp8_role_derives_a_profile_with_the_fp8_patterns_first():
    assert profile_module_for_case(ROLE) == f"derived_{ROLE}"
    p = profile_for_case(ROLE)
    assert p.id == "vllm-d9105ea80-sm89-eager-fp8-block128-qwen3_4b_fp8"
    C = p.config
    assert C["family"] == "qwen3" and C["QK_NORM"] == "head" and C["FP8_BLOCK"] == 128
    assert (C["H"], C["NL"], C["NH"], C["KVH"], C["D"], C["I"], C["V"]) == (2560, 36, 32, 8, 128, 9728, 151936)
    assert C["BIAS"] is False and C["tie_word_embeddings"] is True and C["ROPE_THETA"] == 5_000_000
    names = [type(x).__name__ for x in p.patterns]
    assert names[:2] == ["PerTokenGroupFp8Quant", "CutlassScaledMmFp8Block"]      # specific kernels first
    assert p.patterns[0].group == 128 and p.patterns[1].group == 128
    assert "RMSNormTritonLaunch" in names and "GemmLaunch" in names
    assert p.notes["fp8"]["families"] == ["Fp8GroupQuant_v1{K,G}", "ScaledMmFp8BlockSharedScale_v1{K,N,G} (fp8_scale_construction per-coordinate: "
                                          "ScaledMmFp8Block_v1{K,N,G})"]
    mod = importlib.import_module(f"verity_vllm.observe.fold.profiles.dense_{ROLE}")
    assert mod.PROVENANCE["QK_NORM"] == "hand" and mod.PROVENANCE["FP8_BLOCK"] == "hf-config"


def test_weights_record_carries_e4m3_weights_block_scales_and_head_norms():
    C = profile_for_case(ROLE).config
    layer = dict(analytic.weights_type(C).fields)["layers"].elem
    fields = [n for n, _ in layer.fields]
    assert fields == ["ln1", "qkv_w", "qkv_ws", "q_norm", "k_norm", "o_w", "o_ws", "ln2", "gate_up_w", "gate_up_ws", "down_w", "down_ws"]
    f = dict(layer.fields)
    assert str(f["qkv_w"]) == "Array<6144,Array<2560,Value<8>>>" and str(f["qkv_ws"]) == "Array<48,Array<20,Value<32>>>"
    assert str(f["down_w"]) == "Array<2560,Array<9728,Value<8>>>" and str(f["down_ws"]) == "Array<20,Array<76,Value<32>>>"
    assert str(f["q_norm"]) == "Array<128,Value<16>>" and str(f["ln1"]) == "Array<2560,Value<16>>"     # norms stay bf16
    # D = 68: qkv's N = 48 * 68 = 3264 ends mid-block (cdiv = 26 scale rows); o_proj's K = 32 * 68 = 2176 = 17 groups
    f = dict(dict(analytic.weights_type(dict(C, D=68, QKV=48 * 68)).fields)["layers"].elem.fields)
    assert str(f["qkv_w"]) == "Array<3264,Array<2560,Value<8>>>" and str(f["qkv_ws"]) == "Array<26,Array<20,Value<32>>>"
    assert str(f["o_ws"]) == "Array<20,Array<17,Value<32>>>"
    with pytest.raises(ValueError, match="K is not a multiple"):                # D = 66: o_proj's K = 2112 is not
        analytic.weights_type(dict(C, D=66, QKV=48 * 66))
    # a bf16 Llama-architecture configuration is untouched by the two facts
    C0 = dict(C)
    C0.pop("QK_NORM"), C0.pop("FP8_BLOCK")
    from verity.ml.definitions.registry import b1
    assert analytic.weights_type(C0) == analytic.weights_type(C0) and dict(analytic.weights_type(C0).fields)["layers"].elem == b1.layer_weights_type(C0)


def test_weight_map_names_the_scales_and_head_norms_only_under_the_facts():
    p = profile_for_case(ROLE)
    wf = p.weight_field
    assert wf("model.layers.5.self_attn.qkv_proj.weight") == ("layers", 5, "qkv_w")
    assert wf("model.layers.5.self_attn.qkv_proj.weight_scale_inv") == ("layers", 5, "qkv_ws")
    assert wf("model.layers.5.mlp.down_proj.weight_scale_inv") == ("layers", 5, "down_ws")
    assert wf("model.layers.5.self_attn.q_norm.weight") == ("layers", 5, "q_norm")
    assert wf("model.layers.5.self_attn.k_norm.weight") == ("layers", 5, "k_norm")
    assert wf("lm_head.weight") == ("embed",)                       # tied checkpoint
    assert wf("model.layers.5.self_attn.attn._k_scale") is None     # kv_cache_dtype=auto placeholders: ignored roots
    # the same names on a bf16 Mistral profile are unknown roots, not silently mapped
    m = profile_for_case("MISTRAL7B").weight_field
    assert m("model.layers.5.self_attn.qkv_proj.weight_scale_inv") is None and m("model.layers.5.self_attn.q_norm.weight") is None


def test_other_quantization_schemes_refuse_fail_closed():
    cfg = json.loads(CFG_PATH.read_text())
    assert generic.structural_refusal(cfg) is None
    bad = dict(cfg, quantization_config=dict(cfg["quantization_config"], weight_block_size=None))
    assert "weight_block_size" in generic.structural_refusal(bad)
    bad = dict(cfg, quantization_config=dict(cfg["quantization_config"], activation_scheme="static"))
    assert "activation_scheme" in generic.structural_refusal(bad)
    bad = dict(cfg, quantization_config={"quant_method": "compressed-tensors"})
    assert "quant_method" in generic.structural_refusal(bad)
    bad = dict(cfg, quantization_config=dict(cfg["quantization_config"], modules_to_not_convert=["lm_head", "model.layers.3.mlp.down_proj"]))
    assert "mixed-precision" in generic.structural_refusal(bad)
    with pytest.raises(ValueError):
        generic.derive_config(bad, repo=None, revision=None)
    assert "head_dim" in generic.structural_refusal({k: v for k, v in cfg.items() if k != "head_dim"})


# ---------------------------------------------------------------- patterns

A_X, A_XQ, A_XS, A_OUT, A_W, A_WS = 500, 501, 502, 503, 2, 3
QUANT_SCHEMA = ("_C::per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! output_s, int group_size, float eps, float fp8_min, "
                "float fp8_max, bool scale_ue8m0, bool column_major_scales, bool tma_aligned) -> ()")
MM_SCHEMA = "_C::cutlass_scaled_mm(Tensor! out, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()"


def _roots(n: int = N):
    return [Root(view(A_W, (n, K), "float8_e4m3fn"), "weight", "model.layers.0.self_attn.qkv_proj.weight"),
            Root(view(A_WS, (-(-n // G), K // G), "float32"), "weight", "model.layers.0.self_attn.qkv_proj.weight_scale_inv")]


def _ctx(cc=(9, 0), roots=None, kernel="CutlassFp8BlockScaledMMKernel", target=None):
    p = profile_for_case(ROLE)
    backends = {} if kernel is None else {"fp8_block_gemm_kernel": kernel}
    hdr = RunHeader("verity-capture/log/v1", "2026-09-19T00:00:00Z", "pid",
                    {"gpu": {"name": "NVIDIA H100 80GB HBM3", "compute_capability": list(cc), "multi_processor_count": 132,
                             "tuned_gemm_arch_family": "hopper" if cc[0] == 9 else "ada"}, "backends": backends,
                     **({} if target is None else {"target": target.to_json()})},
                    {"repo": "Qwen/Qwen3-4B-Instruct-2507-FP8", "config": {}}, {"requests": []})
    rows = [("r0", i) for i in range(M)]
    start = StepStart(0, rows, list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0]}, 16, list(range(M)), [("r0", M - 1)], [M - 1], 1, M, M)
    rmap: dict[int, list[Root]] = {}
    for r in (roots if roots is not None else _roots()):
        rmap.setdefault(r.view.alloc, []).append(r)
    return StepContext(hdr, start, None, [], rmap, p.config, p)


def _node(op, schema, args, outs=None, module="model.layers.0.self_attn.qkv_proj"):
    ev = [DispatchEnter(1, None, 0, "forward", op, schema, list(args), {}, 7, module), DispatchExit(1, outs)]
    roots, _ = build_trees(ev)
    return roots[0]


def quant_args(**over):
    a = dict(x=view(A_X, (M, K), "bfloat16"), xq=view(A_XQ, (M, K), "float8_e4m3fn"), xs=view(A_XS, (M, K // G), "float32", strides=(1, M)),
             G=G, eps=1e-10, lo=-448.0, hi=448.0, ue8m0=False, col=True, tma=False)
    a.update(over)
    return [a["x"], a["xq"], a["xs"], a["G"], a["eps"], a["lo"], a["hi"], a["ue8m0"], a["col"], a["tma"]]


@pytest.mark.parametrize("ue8m0, family", [(False, "Fp8GroupQuant_v1"), (True, "Fp8GroupQuantUe8m0_v1")])
def test_per_token_group_quant_resolves_to_the_registered_family_with_both_outputs(ue8m0, family):
    """`scale_ue8m0` selects the Definition: False the pinned Fp8GroupQuant_v1, True the power-of-two-scale Fp8GroupQuantUe8m0_v1
    (registry.deepgemm, what `moe_kernel_quantize_input` launches on sm_90 with DeepGEMM at its default)."""
    ctx = _ctx()
    r = PerTokenGroupFp8Quant(group=G).match(_node("_C.per_token_group_fp8_quant.default", QUANT_SCHEMA, quant_args(ue8m0=ue8m0)), ctx)
    assert isinstance(r, Compute)
    assert spec_label(r.spec) == f"{family}{{K={K},G={G}}}"
    assert r.rows == M and len(r.operands) == 1 and isinstance(r.operands[0], PerRow)
    assert len(r.outputs) == 2 and all(isinstance(o, PerRow) for o in r.outputs)
    assert r.outputs[0].view.dtype == "float8_e4m3fn" and r.outputs[1].view.shape == (M, K // G)
    assert r.attribution == tuple(ctx.start.rows)


@pytest.mark.parametrize("over, why", [
    ({"G": 64}, "group size"), ({"eps": 1e-6}, "knobs"), ({"hi": 224.0}, "knobs"),
    ({"xs": view(A_XS, (M, K // 64), "float32")}, "[M, K/G]"),
    ({"xq": view(A_XQ, (M, K), "float8_e5m2")}, "e4m3"),
])
def test_per_token_group_quant_refuses_other_knobs_and_formats(over, why):
    r = PerTokenGroupFp8Quant(group=G).match(_node("_C.per_token_group_fp8_quant.default", QUANT_SCHEMA, quant_args(**over)), _ctx())
    assert isinstance(r, Unsupported) and why in r.reason


def mm_args(**over):
    a = dict(out=view(A_OUT, (M, N), "bfloat16"), xq=view(A_XQ, (M, K), "float8_e4m3fn"),
             b=view(A_W, (K, N), "float8_e4m3fn", strides=(1, K)),                 # weight.T
             xs=view(A_XS, (M, K // G), "float32", strides=(1, M)),
             bs=view(A_WS, (K // G, N // G), "float32", strides=(1, K // G)),     # weight_scale_inv.T
             bias=None)
    a.update(over)
    return [a["out"], a["xq"], a["b"], a["xs"], a["bs"], a["bias"]]


def test_cutlass_scaled_mm_blockwise_resolves_with_both_weights_named_per_operand():
    ctx = _ctx()
    r = CutlassScaledMmFp8Block(group=G).match(_node("_C.cutlass_scaled_mm.default", MM_SCHEMA, mm_args()), ctx)
    assert isinstance(r, Compute), r
    assert spec_label(r.spec) == f"ScaledMmFp8BlockSharedScale_v1{{K={K},N={N},G={G}}}" and r.rows == M
    assert [type(o).__name__ for o in r.operands] == ["PerRow", "PerRow", "Shared", "Shared"]
    w, ws = r.operands[2].view, r.operands[3].view
    assert w.shape == (N, K) and w.strides == (K, 1) and ws.shape == (N // G, K // G) and ws.strides == (K // G, 1)   # the parameters, not their .T
    assert r.roots == (None, None, "model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.self_attn.qkv_proj.weight_scale_inv")
    assert r.hint.endswith("/model.layers.0.self_attn.qkv_proj.weight")
    assert isinstance(r.outputs[0], PerRow) and r.outputs[0].view.dtype == "bfloat16"


def test_cutlass_scaled_mm_blockwise_resolves_over_a_ragged_n():
    """GLM-5.3's `fused_qkv_a_proj`: N = 2048 + 576 = 2624 output rows over cdiv(2624, 128) = 21 weight-scale rows, the last block
    64 rows.  A [K/G, N // G] = [.., 20] b_scales is not that weight's scale tensor, and K must still be a multiple of the group."""
    n, nb = 2624, 21
    ragged = dict(out=view(A_OUT, (M, n), "bfloat16"), b=view(A_W, (K, n), "float8_e4m3fn", strides=(1, K)))
    args = mm_args(**ragged, bs=view(A_WS, (K // G, nb), "float32", strides=(1, K // G)))
    r = CutlassScaledMmFp8Block(group=G).match(_node("_C.cutlass_scaled_mm.default", MM_SCHEMA, args), _ctx(roots=_roots(n)))
    assert isinstance(r, Compute), r
    assert spec_label(r.spec) == f"ScaledMmFp8BlockSharedScale_v1{{K={K},N={n},G={G}}}" and r.rows == M
    w, ws = r.operands[2].view, r.operands[3].view
    assert w.shape == (n, K) and ws.shape == (nb, K // G) and ws.strides == (K // G, 1)
    assert r.roots[2:] == ("model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.self_attn.qkv_proj.weight_scale_inv")
    floor = mm_args(**ragged, bs=view(A_WS, (K // G, n // G), "float32", strides=(1, K // G)))
    r = CutlassScaledMmFp8Block(group=G).match(_node("_C.cutlass_scaled_mm.default", MM_SCHEMA, floor), _ctx(roots=_roots(n)))
    assert isinstance(r, Unsupported) and "as_weight_view" in r.reason
    k = K + 64
    ragged_k = mm_args(xq=view(A_XQ, (M, k), "float8_e4m3fn"), b=view(A_W, (k, N), "float8_e4m3fn", strides=(1, k)))
    r = CutlassScaledMmFp8Block(group=G).match(_node("_C.cutlass_scaled_mm.default", MM_SCHEMA, ragged_k), _ctx())
    assert isinstance(r, Unsupported) and "K is not a multiple" in r.reason


def test_cutlass_scaled_mm_binds_the_declared_scale_construction():
    from verity_vllm.program.frontend.target_profile import TargetProfile

    node = _node("_C.cutlass_scaled_mm.default", MM_SCHEMA, mm_args())
    specs = {}
    for scale in (None, "shared-per-block", "per-coordinate"):
        tp = TargetProfile(compute_capability=(9, 0), num_sms=132, fp8_block_gemm="cutlass", fp8_scale_construction=scale)
        r = CutlassScaledMmFp8Block(group=G).match(node, _ctx(target=tp))
        assert isinstance(r, Compute), r
        specs[scale] = spec_label(r.spec)
    assert specs[None] == specs["shared-per-block"] == f"ScaledMmFp8BlockSharedScale_v1{{K={K},N={N},G={G}}}"
    assert specs["per-coordinate"] == f"ScaledMmFp8Block_v1{{K={K},N={N},G={G}}}"


def test_cutlass_scaled_mm_refuses_non_hopper_targets_bias_and_other_scale_shapes():
    node = _node("_C.cutlass_scaled_mm.default", MM_SCHEMA, mm_args())
    r = CutlassScaledMmFp8Block(group=G).match(node, _ctx(cc=(8, 9)))          # L40S / Ada: another FP8 GEMM, no Definition
    assert isinstance(r, Unsupported) and r.detail["compute_capability"] == [8, 9]
    assert r.reason == "blockwise FP8 linear on ada (8.9): this target's _C.cutlass_scaled_mm blockwise kernel has no registered Definition"
    # the record must attest the kernel class: a header without the read-back (an older capture) or with another class is refused
    r = CutlassScaledMmFp8Block(group=G).match(node, _ctx(kernel=None))
    assert isinstance(r, Unsupported) and "does not attest CutlassFp8BlockScaledMMKernel" in r.reason and r.detail["fp8_block_gemm_kernel"] is None
    r = CutlassScaledMmFp8Block(group=G).match(node, _ctx(kernel="FlashInferFp8DeepGEMMDynamicBlockScaledKernel"))
    assert isinstance(r, Unsupported) and r.detail["fp8_block_gemm_kernel"] == "FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
    r = CutlassScaledMmFp8Block(group=G).match(_node("_C.cutlass_scaled_mm.default", MM_SCHEMA, mm_args(bias=view(600, (N,), "bfloat16"))), _ctx())
    assert isinstance(r, Unsupported) and "bias" in r.reason
    per_tensor = mm_args(xs=view(A_XS, (1, 1), "float32"), bs=view(A_WS, (1, 1), "float32"))
    r = CutlassScaledMmFp8Block(group=G).match(_node("_C.cutlass_scaled_mm.default", MM_SCHEMA, per_tensor), _ctx())
    assert isinstance(r, Unsupported) and "per-token-group" in r.reason
    # the weight must be a checkpoint parameter (a Root): an activation in `b`'s place is refused
    r = CutlassScaledMmFp8Block(group=G).match(node, _ctx(roots=[]))
    assert isinstance(r, Unsupported) and "weight Roots" in r.reason


def test_reshape_that_copies_is_a_reshaping_move_and_numpy_T_is_an_alias():
    """vLLM's `rms_norm_batch_invariant` reshapes the strided `[M, NH, D]` q slice of the fused QKV output to `[M*NH, D]`: at M > 1 torch
    copies (a fresh allocation) -- an element-order-preserving move, not an alias violation; `Tensor.T` (aten.numpy_T) is a view."""
    assert "aten.numpy_T" in ALIAS_OPS
    ctx = _ctx()
    src = view(A_OUT, (M, 32, 128), "bfloat16", strides=(N, 128, 1))
    dst = view(700, (M * 32, 128), "bfloat16")
    node = _node("aten.reshape.default", "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)", [src, [-1, 128]], outs=dst)
    r = classify_transparent(node, ctx)
    assert isinstance(r, Transparent) and r.effect == "move" and isinstance(r.src, Shared) and isinstance(r.dst, Shared) and r.rows == 1
    # a view-able reshape stays an alias; a reshape to a fresh allocation with another element count stays refused
    same = view(A_OUT, (M * 32, 128), "bfloat16", strides=(128, 1))
    r = classify_transparent(_node("aten.reshape.default", "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)", [same, [-1, 128]], outs=same), ctx)
    assert isinstance(r, Transparent) and r.effect == "alias"
    r = classify_transparent(_node("aten.reshape.default", "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)", [src, [-1, 128]],
                                   outs=view(701, (M * 16, 128), "bfloat16")), ctx)
    assert isinstance(r, Unsupported)
    wT = view(A_W, (K, N), "float8_e4m3fn", strides=(1, K))
    r = classify_transparent(_node("aten.numpy_T.default", "aten::numpy_T(Tensor(a) self) -> Tensor(a)", [view(A_W, (N, K), "float8_e4m3fn")], outs=wT), ctx)
    assert isinstance(r, Transparent) and r.effect == "alias"
