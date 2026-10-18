"""The exact-int8 Llama-3.1-8B role's capture profile: its shipped config, its weights record and weight map, and the two fold patterns
(`observe.fold.patterns.int8`) over synthetic launches of the named ops (`engine.int8_ops`)."""
from __future__ import annotations

import hashlib

import pytest

from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.engine.profiles import profile_for_case
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, ViewDesc
from verity_vllm.observe.fold.patterns.int8 import Int8AttentionScope, Int8Linear
from verity_vllm.observe.fold.resolver import Compute, Const, GatherMany, PerRow, Shared, StepContext, Unsupported, spec_label
from verity_vllm.observe.fold.tree import build_trees
from verity_vllm.program import model as analytic

ROLE = "LLAMA31_8B_INT8"
CUDA = "cuda:0"


def _contig(shape):
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= s
    return tuple(reversed(out))


def view(alloc, shape, dtype, strides=None, offset=0):
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides or _contig(shape)), dtype, CUDA)


def test_shipped_int8_config_is_the_manifest_bytes():
    entry = dg.checkpoint_entry(ROLE)
    assert entry is not None and entry["repo"] == "verity-int8/Llama-3.1-8B-INT8"
    assert hashlib.sha256((dg.HF_CONFIG_DIR / f"{ROLE}.config.json").read_bytes()).hexdigest() == entry["config_sha256"]


def test_the_int8_role_derives_a_profile_with_the_int8_patterns_first():
    p = profile_for_case(ROLE)
    assert p.config["INT8_EXACT"] == 127 and not p.config.get("FP8_BLOCK")
    assert [type(x).__name__ for x in p.patterns][:2] == ["Int8Linear", "Int8AttentionScope"]
    assert "AttentionScope" in [type(x).__name__ for x in p.patterns]
    assert "Int8Linear" not in [type(x).__name__ for x in profile_for_case("LLAMA31_8B").patterns]


def test_weights_record_carries_s8_weights_and_row_scales_and_the_map_names_them():
    p = profile_for_case(ROLE)
    layer = dict(analytic.weights_type(p.config).fields)["layers"].elem
    assert [n for n, _ in layer.fields] == ["ln1", "qkv_w", "qkv_ws", "o_w", "o_ws", "ln2", "gate_up_w", "gate_up_ws", "down_w", "down_ws"]
    f = dict(layer.fields)
    assert str(f["qkv_w"]) == "Array<6144,Array<4096,Value<8>>>" and str(f["qkv_ws"]) == "Array<6144,Array<1,Value<32>>>"
    assert str(f["down_w"]) == "Array<4096,Array<14336,Value<8>>>" and str(f["ln1"]) == "Array<4096,Value<16>>"
    wf = p.weight_field
    assert wf("model.layers.5.self_attn.qkv_proj.weight_scale") == ("layers", 5, "qkv_ws")
    assert wf("model.layers.5.mlp.down_proj.weight_scale") == ("layers", 5, "down_ws")
    assert wf("lm_head.weight") == ("lm_head",)
    assert profile_for_case("LLAMA31_8B").weight_field("model.layers.5.mlp.down_proj.weight_scale") is None


# ---------------------------------------------------------------- patterns

M, K, N = 5, 64, 16
A_X, A_XQ, A_SA, A_Y, A_W, A_WS = 500, 501, 502, 503, 2, 3
LIN_SCHEMA = "verity_int8::linear(Tensor x, Tensor w, Tensor s_w) -> (Tensor, Tensor, Tensor)"
WIN_SCHEMA = ("verity_int8::window_attention(Tensor q, Tensor key_cache, Tensor value_cache, Tensor blocks, SymInt s, Tensor k_scales, "
              "float scale) -> Tensor")
UA_SCHEMA = ("vllm::unified_attention_with_output(Tensor query, Tensor key, Tensor value, Tensor(a!) output, str layer_name, "
             "Tensor? output_scale=None, Tensor? output_block_scale=None, Tensor? kv_cache_dummy_dep=None) -> ()")
COPY_SCHEMA = "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)"
LAYER = "model.layers.0.self_attn.attn"
KBITS = [0x3D000000, 0x3E000000]


def _header(backends):
    return RunHeader("verity-capture/log/v1", "2026-10-17T00:00:00Z", "pid",
                     {"gpu": {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "compute_capability": [12, 0], "multi_processor_count": 188},
                      "backends": backends}, {"repo": "verity-int8/Llama-3.1-8B-INT8", "config": {}}, {"requests": []})


def _ctx(rows, seq_lens, block_tables, roots=(), backends=None, block_size=16):
    p = profile_for_case(ROLE)
    T = len(rows)
    reqs = sorted({r for r, _ in rows})
    start = StepStart(0, rows, list(range(T)), [p_ for _, p_ in rows], seq_lens, {r: 0 for r in reqs}, block_tables, block_size,
                      list(range(T)), [(r, max(p_ for q, p_ in rows if q == r)) for r in reqs], [T - 1], len(reqs), T, T)
    rmap: dict[int, list[Root]] = {}
    for r in roots:
        rmap.setdefault(r.view.alloc, []).append(r)
    return StepContext(_header(backends if backends is not None else {"int8_k_scales": {LAYER: KBITS}}), start, None, [], rmap, p.config, p)


def _tree(events):
    roots, _ = build_trees(events)
    return roots[0]


def _lin_roots():
    return [Root(view(A_W, (N, K), "int8"), "weight", "model.layers.0.self_attn.qkv_proj.weight"),
            Root(view(A_WS, (N, 1), "float32"), "weight", "model.layers.0.self_attn.qkv_proj.weight_scale")]


def _lin(w_dtype="int8", sw_shape=(N, 1)):
    args = [view(A_X, (M, K), "bfloat16"), view(A_W, (N, K), w_dtype), view(A_WS, sw_shape, "float32")]
    outs = (view(A_XQ, (M, K), "int8"), view(A_SA, (M,), "float32"), view(A_Y, (M, N), "bfloat16"))
    return _tree([DispatchEnter(1, None, 0, "forward", "verity_int8.linear.default", LIN_SCHEMA, args, {}, 7, "model.layers.0.self_attn.qkv_proj"),
                  DispatchExit(1, outs)])


def test_int8_linear_resolves_to_the_quantizer_then_the_product_with_both_weights_named():
    r = Int8Linear().match(_lin(), _ctx([("r0", i) for i in range(M)], {"r0": M}, {"r0": [0]}, roots=_lin_roots()))
    assert isinstance(r, Compute), r
    assert spec_label(r.spec) == f"Int8RowQuant_v1{{K={K}}}" and r.rows == M and r.consumed
    assert [type(o) for o in r.operands] == [PerRow] and [type(o) for o in r.outputs] == [PerRow, PerRow]
    (mm,) = r.also
    assert spec_label(mm.spec) == f"Int8ScaledMm_v1{{K={K},N={N}}}" and mm.rows == M and mm.consumed == ()
    assert mm.operands[:2] == r.outputs and [type(o) for o in mm.operands[2:]] == [Shared, Shared] and [type(o) for o in mm.outputs] == [PerRow]
    assert mm.roots == (None, None, "model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.self_attn.qkv_proj.weight_scale")


def test_int8_linear_refuses_other_dtypes_and_non_parameter_weights():
    ctx = _ctx([("r0", i) for i in range(M)], {"r0": M}, {"r0": [0]}, roots=_lin_roots())
    assert "Int8Linear_v1 is" in Int8Linear().match(_lin(w_dtype="uint8"), ctx).reason
    r = Int8Linear().match(_lin(), _ctx([("r0", i) for i in range(M)], {"r0": M}, {"r0": [0]}))
    assert isinstance(r, Unsupported) and "weight Roots" in r.reason
    assert Int8Linear().match(_tree([DispatchEnter(1, None, 0, "forward", "aten.linear.default", "aten::linear(Tensor input, Tensor weight, "
                                                  "Tensor? bias=None) -> Tensor", [], {}, 7, None), DispatchExit(1, None)]), ctx) is None


H, G, D, BS, NB = 4, 2, 32, 16, 8
A_Q, A_OUT, A_KV, A_BT, A_KS, A_T0, A_T1 = 600, 601, 602, 603, 604, 605, 606


def _cache(which):
    return view(A_KV, (NB, BS, G, D), "bfloat16", strides=(BS * G * 2 * D, G * 2 * D, 2 * D, 1), offset=0 if which == "k" else D)


def _attention(T, wins):
    """`unified_attention_with_output` over T query rows with one window per (q0, n, s, result alloc), each copied into its rows."""
    q, out = view(A_Q, (T, H, D), "bfloat16"), view(A_OUT, (T, H, D), "bfloat16")
    ev, i = [DispatchEnter(1, None, 0, "forward", "vllm.unified_attention_with_output.default", UA_SCHEMA,
                           [q, view(700, (T, G, D), "bfloat16"), view(701, (T, G, D), "bfloat16"), out, LAYER], {}, 7, LAYER)], 2
    for q0, n, s, res in wins:
        args = [view(A_Q, (n, H, D), "bfloat16", strides=(H * D, D, 1), offset=q0 * H * D), _cache("k"), _cache("v"),
                view(A_BT, (4,), "int32", offset=0), s, view(A_KS, (G,), "float32"), D ** -0.5]
        ev += [DispatchEnter(i, 1, 0, "forward", "verity_int8.window_attention.default", WIN_SCHEMA, args, {}, 7, LAYER),
               DispatchExit(i, view(res, (n, H, D), "bfloat16")),
               DispatchEnter(i + 1, 1, 0, "forward", "aten.copy_.default", COPY_SCHEMA,
                             [view(A_OUT, (n, H, D), "bfloat16", strides=(H * D, D, 1), offset=q0 * H * D), view(res, (n, H, D), "bfloat16")], {}, 7, LAYER),
               DispatchExit(i + 1, None)]
        i += 2
    return _tree(ev + [DispatchExit(1, None)])


def test_int8_attention_resolves_one_window_per_request_with_the_headers_k_scales():
    # r0 prefills 3 rows (positions 0..2), r1 decodes position 5: rows [r0:0, r0:1, r0:2, r1:5]
    rows = [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 5)]
    ctx = _ctx(rows, {"r0": 3, "r1": 6}, {"r0": [2], "r1": [5]})
    r = Int8AttentionScope().match(_attention(4, [(0, 3, 3, A_T0), (3, 1, 6, A_T1)]), ctx)
    assert isinstance(r, Compute), r
    computes = [r, *r.also]
    assert [spec_label(c.spec) for c in computes] == [f"Int8LutAttentionWindow_v1{{N=3,S=3,H={H},G={G},D={D},SCALE={D ** -0.5}}}",
                                                       f"Int8LutAttentionWindow_v1{{N=1,S=6,H={H},G={G},D={D},SCALE={D ** -0.5}}}"]
    c0, c1 = computes
    assert c0.rows == c1.rows == 1
    assert c0.operands[1].indices == ((32, 33, 34),) and c1.operands[1].indices == ((80, 81, 82, 83, 84, 85),)
    assert c0.operands[3] == Const(32, tuple(KBITS), "k_scale")
    assert isinstance(c1.outputs[0], GatherMany) and c1.outputs[0].view.offset == 3 * H * D and c1.outputs[0].indices == ((0,),)
    assert c0.operands[0].view.offset == 0 and c1.operands[0].view.offset == 3 * H * D
    assert set(r.consumed) == {1, 2, 3, 4, 5}


def test_int8_attention_refuses_without_the_headers_k_scales_or_a_wrong_window():
    rows = [("r0", 0), ("r0", 1), ("r0", 2)]
    r = Int8AttentionScope().match(_attention(3, [(0, 3, 3, A_T0)]), _ctx(rows, {"r0": 3}, {"r0": [2]}, backends={}))
    assert isinstance(r, Unsupported) and "int8_k_scales" in r.reason
    r = Int8AttentionScope().match(_attention(3, [(0, 3, 4, A_T0)]), _ctx(rows, {"r0": 3}, {"r0": [2]}))
    assert isinstance(r, Unsupported) and "last rows" in r.reason


@pytest.mark.parametrize("T", [1])
def test_an_fa2_scope_is_not_the_int8_patterns(T):
    ev = [DispatchEnter(1, None, 0, "forward", "vllm.unified_attention_with_output.default", UA_SCHEMA,
                        [view(A_Q, (T, H, D), "bfloat16"), view(700, (T, G, D), "bfloat16"), view(701, (T, G, D), "bfloat16"),
                         view(A_OUT, (T, H, D), "bfloat16"), LAYER], {}, 7, LAYER), DispatchExit(1, None)]
    assert Int8AttentionScope().match(_tree(ev), _ctx([("r0", 0)], {"r0": 1}, {"r0": [0]})) is None
