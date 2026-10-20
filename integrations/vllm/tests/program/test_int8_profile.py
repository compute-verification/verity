"""The exact-int8 Llama-3.1-8B role's capture profile: its shipped config, its weights record and weight map, and the two fold patterns
(`observe.fold.patterns.int8`) over synthetic launches of the named ops (`engine.int8_ops`), the attention's under the plain read and
under a credited read (`engine.int8_window`)."""
from __future__ import annotations

import hashlib

import pytest

from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.engine.profiles import profile_for_case
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, ViewDesc
from verity_vllm.observe.fold.patterns.int8 import Int8AttentionScope, Int8RowQuant, Int8ScaledMm
from verity_vllm.observe.fold.resolver import Compute, Const, GatherMany, Group, PerRow, Shared, StepContext, Unsupported, spec_label
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
    assert [type(x).__name__ for x in p.patterns][:3] == ["Int8RowQuant", "Int8ScaledMm", "Int8AttentionScope"]
    assert "AttentionScope" in [type(x).__name__ for x in p.patterns]
    assert "Int8ScaledMm" not in [type(x).__name__ for x in profile_for_case("LLAMA31_8B").patterns]


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
QUANT_SCHEMA = "verity_int8::row_quant(Tensor x) -> (Tensor, Tensor)"
MM_SCHEMA = "verity_int8::scaled_mm(Tensor xq, Tensor s_a, Tensor w, Tensor s_w) -> Tensor"
KV_SCHEMA = ("verity_int8::kv_quant(Tensor key_cache, Tensor value_cache, Tensor blocks, SymInt s, Tensor k_scales) "
             "-> (Tensor, Tensor, Tensor, Tensor)")
ROWS_SCHEMA = "verity_int8::window_rows(Tensor q, Tensor kq, Tensor vq, Tensor sv, Tensor v0, Tensor k_scales, float scale) -> Tensor"
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


QKV = "model.layers.0.self_attn.qkv_proj"


def _quant(x_dtype="bfloat16"):
    outs = (view(A_XQ, (M, K), "int8"), view(A_SA, (M,), "float32"))
    return _tree([DispatchEnter(1, None, 0, "forward", "verity_int8.row_quant.default", QUANT_SCHEMA, [view(A_X, (M, K), x_dtype)], {}, 7, QKV),
                  DispatchExit(1, outs)])


def _mm(w_dtype="int8", sw_shape=(N, 1)):
    args = [view(A_XQ, (M, K), "int8"), view(A_SA, (M,), "float32"), view(A_W, (N, K), w_dtype), view(A_WS, sw_shape, "float32")]
    return _tree([DispatchEnter(1, None, 0, "forward", "verity_int8.scaled_mm.default", MM_SCHEMA, args, {}, 7, QKV),
                  DispatchExit(1, view(A_Y, (M, N), "bfloat16"))])


def test_int8_linear_resolves_to_the_quantizer_then_the_product_with_both_weights_named():
    ctx = _ctx([("r0", i) for i in range(M)], {"r0": M}, {"r0": [0]}, roots=_lin_roots())
    q, mm = Int8RowQuant().match(_quant(), ctx), Int8ScaledMm().match(_mm(), ctx)
    assert isinstance(q, Compute) and isinstance(mm, Compute), (q, mm)
    assert spec_label(q.spec) == f"Int8RowQuant_v1{{K={K}}}" and q.rows == M and q.consumed and q.hint == f"{QKV}/quant"
    assert [type(o) for o in q.operands] == [PerRow] and [type(o) for o in q.outputs] == [PerRow, PerRow]
    assert spec_label(mm.spec) == f"ScaledMmS8_v1{{K={K},N={N},DOT=BlackwellS8ImmaDot32_v1}}" and mm.rows == M and mm.consumed
    assert mm.operands[:2] == q.outputs and [type(o) for o in mm.operands[2:]] == [Shared, Shared] and [type(o) for o in mm.outputs] == [PerRow]
    assert mm.roots == (None, None, f"{QKV}.weight", f"{QKV}.weight_scale")


def test_int8_linear_refuses_other_dtypes_and_non_parameter_weights():
    ctx = _ctx([("r0", i) for i in range(M)], {"r0": M}, {"r0": [0]}, roots=_lin_roots())
    assert "Int8RowQuant_v1 is" in Int8RowQuant().match(_quant("float16"), ctx).reason
    assert "ScaledMmS8_v1 is" in Int8ScaledMm().match(_mm(w_dtype="uint8"), ctx).reason
    r = Int8ScaledMm().match(_mm(), _ctx([("r0", i) for i in range(M)], {"r0": M}, {"r0": [0]}))
    assert isinstance(r, Unsupported) and "weight Roots" in r.reason
    other = _tree([DispatchEnter(1, None, 0, "forward", "aten.linear.default", "aten::linear(Tensor input, Tensor weight, "
                                 "Tensor? bias=None) -> Tensor", [], {}, 7, None), DispatchExit(1, None)])
    assert Int8RowQuant().match(other, ctx) is None and Int8ScaledMm().match(other, ctx) is None


H, G, D, BS, NB = 4, 2, 32, 16, 8
A_Q, A_OUT, A_KV, A_BT, A_KS, A_T0, A_T1 = 600, 601, 602, 603, 604, 605, 606


def _cache(which):
    return view(A_KV, (NB, BS, G, D), "bfloat16", strides=(BS * G * 2 * D, G * 2 * D, 2 * D, 1), offset=0 if which == "k" else D)


def _attention(T, wins, rows_reads=None):
    """`unified_attention_with_output` over T query rows with one kv_quant / window_rows pair per (q0, n, s, result alloc): kv_quant's
    four results at allocs res + 10..13, window_rows reading them (or `rows_reads`) and its result copied into its rows."""
    q, out = view(A_Q, (T, H, D), "bfloat16"), view(A_OUT, (T, H, D), "bfloat16")
    ev, i = [DispatchEnter(1, None, 0, "forward", "vllm.unified_attention_with_output.default", UA_SCHEMA,
                           [q, view(700, (T, G, D), "bfloat16"), view(701, (T, G, D), "bfloat16"), out, LAYER], {}, 7, LAYER)], 2
    ks = view(A_KS, (G,), "float32")
    for q0, n, s, res in wins:
        kv = (view(res + 10, (G, s, D), "int8"), view(res + 11, (G, s, D), "int8"), view(res + 12, (G, D), "float32"),
              view(res + 13, (G, D), "float32"))
        args = [view(A_Q, (n, H, D), "bfloat16", strides=(H * D, D, 1), offset=q0 * H * D), *(rows_reads or kv), ks, D ** -0.5]
        ev += [DispatchEnter(i, 1, 0, "forward", "verity_int8.kv_quant.default", KV_SCHEMA,
                             [_cache("k"), _cache("v"), view(A_BT, (4,), "int32", offset=0), s, ks], {}, 7, LAYER),
               DispatchExit(i, kv),
               DispatchEnter(i + 1, 1, 0, "forward", "verity_int8.window_rows.default", ROWS_SCHEMA, args, {}, 7, LAYER),
               DispatchExit(i + 1, view(res, (n, H, D), "bfloat16")),
               DispatchEnter(i + 2, 1, 0, "forward", "aten.copy_.default", COPY_SCHEMA,
                             [view(A_OUT, (n, H, D), "bfloat16", strides=(H * D, D, 1), offset=q0 * H * D), view(res, (n, H, D), "bfloat16")], {}, 7, LAYER),
               DispatchExit(i + 2, None)]
        i += 3
    return _tree(ev + [DispatchExit(1, None)])


def test_int8_attention_resolves_kv_quant_per_kv_head_then_a_row_per_row_and_head():
    # r0 prefills 3 rows (positions 0..2), r1 decodes position 5: rows [r0:0, r0:1, r0:2, r1:5]
    rows = [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 5)]
    ctx = _ctx(rows, {"r0": 3, "r1": 6}, {"r0": [2], "r1": [5]})
    r = Int8AttentionScope().match(_attention(4, [(0, 3, 3, A_T0), (3, 1, 6, A_T1)]), ctx)
    assert isinstance(r, Group), r
    computes = list(r.parts)
    sc = D ** -0.5
    kvq = lambda s: [f"Int8KvQuant_v1{{S={s},D={D}}}"] * G  # noqa: E731
    row = lambda p: [f"Int8LutAttentionRow_v1{{D={D},S={p},SCALE={sc}}}"] * H  # noqa: E731
    assert [spec_label(c.spec) for c in computes] == kvq(3) + row(1) + row(2) + row(3) + kvq(6) + row(6)
    assert all(c.rows == 1 for c in computes) and set(r.consumed) == set(range(1, 8))
    # each KV quantization is stamped at its own kv_quant call (where the capture snapshots its results), the rows at the scope
    assert [c.consumed for c in computes if spec_label(c.spec).startswith("Int8KvQuant")] == [(2,)] * G + [(5,)] * G
    assert {c.consumed for c in computes if spec_label(c.spec).startswith("Int8LutAttentionRow")} == {()}
    k1 = computes[1]                              # r0's KV head 1: its slots of head 1 in the caches, its K scale, kv_quant's head-1 slabs
    assert k1.operands[0].indices == ((32, 33, 34),) and k1.operands[0].view.offset == 2 * D and k1.operands[0].view.shape == (NB * BS, D)
    assert k1.operands[1].view.offset == 3 * D and k1.operands[2] == Const(32, KBITS[1], "k_scale")
    assert [(o.view.alloc, o.view.offset, o.view.shape) for o in k1.outputs] == [(A_T0 + 10, 3 * D, (3, D)), (A_T0 + 11, 3 * D, (3, D)),
                                                                                 (A_T0 + 12, D, (D,)), (A_T0 + 13, D, (D,))]
    assert computes[G + 1 * H + 2].operands[1].view.shape == (2, D)      # r0's row 1 (position 1), head 2: keys 0..1
    last = computes[-1]                           # r1's position 5, head 3 (KV head 1): all six keys, its output row and head
    assert [type(o) for o in last.operands] == [Shared, Shared, Const, Shared, Shared, Shared] and last.operands[2] == Const(32, KBITS[1], "k_scale")
    assert (last.operands[0].view.offset, last.operands[1].view.alloc, last.operands[1].view.offset, last.operands[1].view.shape) == (
        3 * H * D + 3 * D, A_T1 + 10, 6 * D, (6, D))
    assert last.outputs == (Shared(view(A_OUT, (D,), "bfloat16", offset=3 * H * D + 3 * D)),)


def test_int8_attention_refuses_without_the_headers_k_scales_or_a_wrong_window():
    rows = [("r0", 0), ("r0", 1), ("r0", 2)]
    r = Int8AttentionScope().match(_attention(3, [(0, 3, 3, A_T0)]), _ctx(rows, {"r0": 3}, {"r0": [2]}, backends={}))
    assert isinstance(r, Unsupported) and "int8_k_scales" in r.reason
    r = Int8AttentionScope().match(_attention(3, [(0, 3, 4, A_T0)]), _ctx(rows, {"r0": 3}, {"r0": [2]}))
    assert isinstance(r, Unsupported) and "last rows" in r.reason
    other = (view(900, (G, 3, D), "int8"), view(901, (G, 3, D), "int8"), view(902, (G, D), "float32"), view(903, (G, D), "float32"))
    r = Int8AttentionScope().match(_attention(3, [(0, 3, 3, A_T0)], rows_reads=other), _ctx(rows, {"r0": 3}, {"r0": [2]}))
    assert isinstance(r, Unsupported) and "does not read its kv_quant" in r.reason


KVW_SCHEMA = "verity_int8::kv_quant_window(Tensor k, Tensor v, Tensor k_scales) -> (Tensor, Tensor, Tensor, Tensor)"
CS_SCHEMA = ("verity_int8::credited_step(Tensor seq_lens, SymInt batch, SymInt kv_heads, SymInt head_dim, SymInt read) "
             "-> (Tensor, Tensor, Tensor, Tensor)")
CW_SCHEMA = ("verity_int8::credited_window(Tensor kv_cache, Tensor block_table, Tensor? slot_mapping, Tensor lens, Tensor wo, "
             "Tensor(a1!) k, Tensor(a2!) v, SymInt layer, SymInt read) -> ()")
A_WK, A_WV = 610, 611


def _credited_attention(T, wins, first_layer=True, k_rows=None):
    """`_attention` under a credited read: the step's credited_step (at its first layer), the layer's credited_window over the
    LBNHC cache [NB, G, BS, 2D] into K and V [rows, G, D], and per window kv_quant_window over its rows (`k_rows` overrides each
    window's first row)."""
    rows = sum(s for _, _, s, _ in wins)
    q, out = view(A_Q, (T, H, D), "bfloat16"), view(A_OUT, (T, H, D), "bfloat16")
    kbuf, vbuf = view(A_WK, (rows, G, D), "bfloat16"), view(A_WV, (rows, G, D), "bfloat16")
    kv = view(A_KV, (NB, G, BS, 2 * D), "bfloat16", strides=(BS * G * 2 * D, 2 * D, G * 2 * D, 1))
    ev = [DispatchEnter(1, None, 0, "forward", "vllm.unified_attention_with_output.default", UA_SCHEMA,
                        [q, view(700, (T, G, D), "bfloat16"), view(701, (T, G, D), "bfloat16"), out, LAYER], {}, 7, LAYER)]
    if first_layer:
        ev += [DispatchEnter(2, 1, 0, "forward", "verity_int8.credited_step.default", CS_SCHEMA,
                             [view(620, (len(wins),), "int32"), len(wins), G, D, 1], {}, 7, LAYER),
               DispatchExit(2, (view(621, (len(wins),), "int32"), view(622, (len(wins) + 1,), "int32"), kbuf, vbuf))]
    ev += [DispatchEnter(3, 1, 0, "forward", "verity_int8.credited_window.default", CW_SCHEMA,
                         [kv, view(A_BT, (len(wins), 4), "int32"), None, view(621, (len(wins),), "int32"),
                          view(622, (len(wins) + 1,), "int32"), kbuf, vbuf, 0, 1], {}, 7, LAYER), DispatchExit(3, None)]
    ks, i, row0 = view(A_KS, (G,), "float32"), 4, 0
    for j, (q0, n, s, res) in enumerate(wins):
        r0 = row0 if k_rows is None else k_rows[j]
        res4 = (view(res + 10, (G, s, D), "int8"), view(res + 11, (G, s, D), "int8"), view(res + 12, (G, D), "float32"),
                view(res + 13, (G, D), "float32"))
        args = [view(A_Q, (n, H, D), "bfloat16", strides=(H * D, D, 1), offset=q0 * H * D), *res4, ks, D ** -0.5]
        ev += [DispatchEnter(i, 1, 0, "forward", "verity_int8.kv_quant_window.default", KVW_SCHEMA,
                             [view(A_WK, (s, G, D), "bfloat16", offset=r0 * G * D), view(A_WV, (s, G, D), "bfloat16", offset=r0 * G * D), ks],
                             {}, 7, LAYER),
               DispatchExit(i, res4),
               DispatchEnter(i + 1, 1, 0, "forward", "verity_int8.window_rows.default", ROWS_SCHEMA, args, {}, 7, LAYER),
               DispatchExit(i + 1, view(res, (n, H, D), "bfloat16")),
               DispatchEnter(i + 2, 1, 0, "forward", "aten.copy_.default", COPY_SCHEMA,
                             [view(A_OUT, (n, H, D), "bfloat16", strides=(H * D, D, 1), offset=q0 * H * D), view(res, (n, H, D), "bfloat16")], {}, 7, LAYER),
               DispatchExit(i + 2, None)]
        i, row0 = i + 3, row0 + s
    return _tree(ev + [DispatchExit(1, None)])


def _bindings(group):
    return [(spec_label(c.spec), [(type(o).__name__, getattr(o, "indices", None), getattr(getattr(o, "view", None), "offset", None),
                                   getattr(getattr(o, "view", None), "shape", None), getattr(getattr(o, "view", None), "strides", None))
                                  for o in c.operands], c.outputs) for c in group.parts]


@pytest.mark.parametrize("first_layer", [True, False])
def test_a_credited_read_folds_to_the_plain_reads_computes(first_layer):
    # the same step as the plain test: kv_quant_window over the credited window's rows binds the same slots of the same cache
    rows = [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 5)]
    ctx = _ctx(rows, {"r0": 3, "r1": 6}, {"r0": [2], "r1": [5]})
    wins = [(0, 3, 3, A_T0), (3, 1, 6, A_T1)]
    plain = Int8AttentionScope().match(_attention(4, wins), ctx)
    cred = Int8AttentionScope().match(_credited_attention(4, wins, first_layer=first_layer), ctx)
    assert isinstance(plain, Group) and isinstance(cred, Group), cred
    assert _bindings(cred) == _bindings(plain)
    kvq = [c for c in cred.parts if spec_label(c.spec).startswith("Int8KvQuant")]
    assert [c.consumed for c in kvq] == [(4,)] * G + [(7,)] * G
    assert set(cred.consumed) == set(range(1, 10)) - ({2} if not first_layer else set())


def test_a_credited_read_refuses_rows_other_than_its_requests_window():
    rows = [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 5)]
    ctx = _ctx(rows, {"r0": 3, "r1": 6}, {"r0": [2], "r1": [5]})
    wins = [(0, 3, 3, A_T0), (3, 1, 6, A_T1)]
    r = Int8AttentionScope().match(_credited_attention(4, wins, k_rows=[0, 2]), ctx)
    assert isinstance(r, Unsupported) and "rows of the credited window" in r.reason
    r = Int8AttentionScope().match(_credited_attention(4, wins, k_rows=[3, 0]), ctx)
    assert isinstance(r, Unsupported) and "rows of the credited window" in r.reason
    tree = _credited_attention(4, wins)
    tree.children = [c for c in tree.children if "credited_window" not in str(c.op)]
    r = Int8AttentionScope().match(tree, ctx)
    assert isinstance(r, Unsupported) and "exactly one credited_window" in r.reason


@pytest.mark.parametrize("T", [1])
def test_an_fa2_scope_is_not_the_int8_patterns(T):
    ev = [DispatchEnter(1, None, 0, "forward", "vllm.unified_attention_with_output.default", UA_SCHEMA,
                        [view(A_Q, (T, H, D), "bfloat16"), view(700, (T, G, D), "bfloat16"), view(701, (T, G, D), "bfloat16"),
                         view(A_OUT, (T, H, D), "bfloat16"), LAYER], {}, 7, LAYER), DispatchExit(1, None)]
    assert Int8AttentionScope().match(_tree(ev), _ctx([("r0", 0)], {"r0": 1}, {"r0": [0]})) is None
