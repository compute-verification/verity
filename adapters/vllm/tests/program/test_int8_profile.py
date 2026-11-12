"""The exact-int8 Llama-3.1-8B role's capture profile: its shipped config, its weights record and weight map, and the two fold patterns
(`observe.fold.patterns.int8`) over synthetic launches of the named ops (`engine.int8_ops`), the attention's under the plain read and
under a credited read (`engine.int8_window`)."""
from __future__ import annotations

import hashlib

import pytest

from verity_vllm.engine.profiles import dense_generic as dg
from verity_vllm.observe.fold.profiles import profile_for_case
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
    assert "INT8_LM_HEAD" not in p.config and wf("lm_head.weight_scale") is None


def test_the_v2_role_carries_an_s8_lm_head_and_its_row_scales():
    role = "LLAMA31_8B_INT8_V2"
    entry = dg.checkpoint_entry(role)
    assert entry is not None and entry["recipe"]["id"] == "verity-int8-channel127/v2"
    assert hashlib.sha256((dg.HF_CONFIG_DIR / f"{role}.config.json").read_bytes()).hexdigest() == entry["config_sha256"]
    p = profile_for_case(role)
    assert p.config["INT8_EXACT"] == 127 and p.config["INT8_LM_HEAD"]
    fields = analytic.weights_type(p.config).fields
    names = [n for n, _ in fields]
    assert names[names.index("lm_head") + 1] == "lm_head_ws"
    f = dict(fields)
    assert str(f["lm_head"]) == "Array<128256,Array<4096,Value<8>>>" and str(f["lm_head_ws"]) == "Array<128256,Array<1,Value<32>>>"
    assert str(f["embed"]) == str(dict(analytic.weights_type(profile_for_case(ROLE).config).fields)["embed"])
    assert dict(fields)["layers"].elem == dict(analytic.weights_type(profile_for_case(ROLE).config).fields)["layers"].elem
    wf = p.weight_field
    assert wf("lm_head.weight") == ("lm_head",) and wf("lm_head.weight_scale") == ("lm_head_ws",)


def test_int8_exact_config_reads_whether_the_lm_head_is_kept():
    pytest.importorskip("vllm")
    from verity_vllm.engine.int8_vllm import Int8ExactConfig
    base = {"quant_method": "verity_int8", "weight_max": 127}
    assert not Int8ExactConfig.from_config({**base, "modules_to_not_convert": ["lm_head"]}).lm_head
    assert not Int8ExactConfig.from_config(base).lm_head
    assert Int8ExactConfig.from_config({**base, "modules_to_not_convert": []}).lm_head
    assert Int8ExactConfig.from_config({**base, "modules_to_not_convert": ["model.norm"]}).lm_head
    with pytest.raises(ValueError, match="weight_max"):
        Int8ExactConfig.from_config({**base, "weight_max": 63})


def test_int8_exact_config_keeps_the_linears_its_list_names_bf16():
    """A base's ``modules_to_not_convert`` (GLM-5.3's keeps kv_b_proj, the indexer's wk, its routers and norms bf16): each
    linear it names, a fused one by all its shards, is vLLM's unquantized method marked kept, which the readback and H1 leave
    out; every other linear is the plain arm; a fused layer with some shards kept, or kept experts, are refused."""
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from verity_vllm.engine.int8_vllm import Int8ExactConfig, Int8KeptLinearMethod, Int8PlainLinearMethod
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of, kept

    class Linear(LinearBase):
        def __init__(self):
            torch.nn.Module.__init__(self)

    class Experts(RoutedExperts):
        def __init__(self):
            torch.nn.Module.__init__(self)

    cfg = Int8ExactConfig.from_config({"quant_method": "verity_int8", "weight_max": 127, "modules_to_not_convert": [
        "lm_head", "model.norm", "model.layers.0.self_attn.kv_b_proj", "model.layers.0.self_attn.indexer.wk",
        "model.layers.0.mlp.gate", "model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj", "model.layers.2.self_attn.q_proj", "model.layers.3.mlp.experts"]})
    cfg.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    assert not cfg.lm_head
    method = {p: cfg.get_quant_method(Linear(), p) for p in (
        "model.layers.0.self_attn.kv_b_proj", "model.layers.0.self_attn.indexer.wk", "model.layers.0.mlp.gate",
        "model.layers.1.self_attn.qkv_proj", "model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.kv_a_proj")}
    assert {p for p, m in method.items() if isinstance(m, Int8KeptLinearMethod)} == {
        "model.layers.0.self_attn.kv_b_proj", "model.layers.0.self_attn.indexer.wk", "model.layers.0.mlp.gate",
        "model.layers.1.self_attn.qkv_proj"}
    assert all(isinstance(m, Int8PlainLinearMethod) for p, m in method.items() if not isinstance(m, Int8KeptLinearMethod))
    assert isinstance(method["model.layers.0.mlp.gate"], UnquantizedLinearMethod)
    with pytest.raises(ValueError, match="some but not all shards"):
        cfg.get_quant_method(Linear(), "model.layers.2.self_attn.qkv_proj")
    with pytest.raises(ValueError, match="no bf16 variant"):
        cfg.get_quant_method(Experts(), "model.layers.3.mlp.experts")

    class Model:
        def __init__(self, *methods):
            self.mods = [type("M", (), {"quant_method": m})() for m in methods]

        def modules(self):
            return iter(self.mods)

    plain, kept_bf16 = method["model.layers.0.self_attn.o_proj"], method["model.layers.0.self_attn.kv_b_proj"]
    model = Model(plain, kept_bf16)
    assert kept(model) == {id(model.mods[1])} and int8_exact_of(model) == "linears"
    assert int8_exact_of(Model(plain, UnquantizedLinearMethod())) == "<mixed>"


# ---------------------------------------------------------------- patterns

M, K, N = 5, 64, 16
A_X, A_XQ, A_SA, A_Y, A_W, A_WS = 500, 501, 502, 503, 2, 3
QUANT_SCHEMA = "verity_int8::row_quant(Tensor x) -> (Tensor, Tensor)"
MM_SCHEMA = "verity_int8::scaled_mm(Tensor xq, Tensor s_a, Tensor w, Tensor s_w) -> Tensor"
KV_SCHEMA = ("verity_int8::kv_quant_batch(Tensor key_cache, Tensor value_cache, Tensor block_table, SymInt[] lens, Tensor k_scales) "
             "-> (Tensor, Tensor, Tensor, Tensor)")
ROWS_SCHEMA = ("verity_int8::window_rows_batch(Tensor query, Tensor kq, Tensor vq, Tensor sv, Tensor v0, Tensor k_scales, SymInt[] qsl, "
               "SymInt[] lens, float scale) -> Tensor")
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


def _qsl(reqs):
    qsl = [0]
    for n, _ in reqs:
        qsl.append(qsl[-1] + n)
    return qsl


def _results(B, s_max):
    """kv_quant_batch's four results [B, G, s_max, D] / [B, G, D] at allocs A_T0 + 10..13."""
    return (view(A_T0 + 10, (B, G, s_max, D), "int8"), view(A_T0 + 11, (B, G, s_max, D), "int8"), view(A_T0 + 12, (B, G, D), "float32"),
            view(A_T0 + 13, (B, G, D), "float32"))


def _scope(T, events):
    q, out = view(A_Q, (T, H, D), "bfloat16"), view(A_OUT, (T, H, D), "bfloat16")
    return _tree([DispatchEnter(1, None, 0, "forward", "vllm.unified_attention_with_output.default", UA_SCHEMA,
                                [q, view(700, (T, G, D), "bfloat16"), view(701, (T, G, D), "bfloat16"), out, LAYER], {}, 7, LAYER),
                  *events, DispatchExit(1, None)])


def _rows(i, reqs, kv, rows_reads=None):
    """window_rows_batch (event i) over the scope's first qsl[-1] query rows and kv_quant_batch's results (or `rows_reads`), and
    the copy of its result (at A_T0) into the output's rows (event i + 1)."""
    qsl, lens = _qsl(reqs), [s for _, s in reqs]
    t1 = qsl[-1]
    rows = view(A_Q, (t1, H, D), "bfloat16")
    return [DispatchEnter(i, 1, 0, "forward", "verity_int8.window_rows_batch.default", ROWS_SCHEMA,
                          [rows, *(rows_reads or kv), view(A_KS, (G,), "float32"), qsl, lens, D ** -0.5], {}, 7, LAYER),
            DispatchExit(i, view(A_T0, (t1, H, D), "bfloat16")),
            DispatchEnter(i + 1, 1, 0, "forward", "aten.copy_.default", COPY_SCHEMA,
                          [view(A_OUT, (t1, H, D), "bfloat16"), view(A_T0, (t1, H, D), "bfloat16")], {}, 7, LAYER),
            DispatchExit(i + 1, None)]


def _attention(T, reqs, rows_reads=None, kv_lens=None):
    """`unified_attention_with_output` over T query rows, `engine.int8_ops.paged_attention`'s: one kv_quant_batch over the step's
    requests (n rows of an s-key window each, in `reqs`' order), one window_rows_batch reading its results (or `rows_reads`), and
    its result copied into the output's rows."""
    lens = [s for _, s in reqs]
    kv = _results(len(reqs), max(lens))
    return _scope(T, [DispatchEnter(2, 1, 0, "forward", "verity_int8.kv_quant_batch.default", KV_SCHEMA,
                                    [_cache("k"), _cache("v"), view(A_BT, (len(reqs), 4), "int32"), kv_lens or lens, view(A_KS, (G,), "float32")],
                                    {}, 7, LAYER),
                      DispatchExit(2, kv), *_rows(3, reqs, kv, rows_reads)])


def test_int8_attention_resolves_kv_quant_per_kv_head_then_a_row_per_row_and_head():
    # r0 prefills 3 rows (positions 0..2), r1 decodes position 5: rows [r0:0, r0:1, r0:2, r1:5]
    rows = [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 5)]
    ctx = _ctx(rows, {"r0": 3, "r1": 6}, {"r0": [2], "r1": [5]})
    r = Int8AttentionScope().match(_attention(4, [(3, 3), (1, 6)]), ctx)
    assert isinstance(r, Group), r
    computes = list(r.parts)
    sc = D ** -0.5
    kvq = lambda s: [f"Int8KvQuant_v1{{S={s},D={D}}}"] * G  # noqa: E731
    row = lambda p: [f"Int8LutAttentionRow_v1{{D={D},S={p},SCALE={sc}}}"] * H  # noqa: E731
    assert [spec_label(c.spec) for c in computes] == kvq(3) + row(1) + row(2) + row(3) + kvq(6) + row(6)
    assert all(c.rows == 1 for c in computes) and set(r.consumed) == set(range(1, 5))
    # every KV quantization is stamped at the one kv_quant_batch call (where the capture snapshots its results), the rows at the scope
    assert [c.consumed for c in computes if spec_label(c.spec).startswith("Int8KvQuant")] == [(2,)] * (2 * G)
    assert {c.consumed for c in computes if spec_label(c.spec).startswith("Int8LutAttentionRow")} == {()}
    k1 = computes[1]                              # r0's KV head 1: its slots of head 1 in the caches, its K scale, its slabs of the padded results
    assert k1.operands[0].indices == ((32, 33, 34),) and k1.operands[0].view.offset == 2 * D and k1.operands[0].view.shape == (NB * BS, D)
    assert k1.operands[1].view.offset == 3 * D and k1.operands[2] == Const(32, KBITS[1], "k_scale")
    assert [(o.view.alloc, o.view.offset, o.view.shape) for o in k1.outputs] == [(A_T0 + 10, 6 * D, (3, D)), (A_T0 + 11, 6 * D, (3, D)),
                                                                                 (A_T0 + 12, D, (D,)), (A_T0 + 13, D, (D,))]
    assert computes[G + 1 * H + 2].operands[1].view.shape == (2, D)      # r0's row 1 (position 1), head 2: keys 0..1
    last = computes[-1]                           # r1's position 5, head 3 (KV head 1): all six keys, its output row and head
    assert [type(o) for o in last.operands] == [Shared, Shared, Const, Shared, Shared, Shared] and last.operands[2] == Const(32, KBITS[1], "k_scale")
    assert (last.operands[0].view.offset, last.operands[1].view.alloc, last.operands[1].view.offset, last.operands[1].view.shape) == (
        3 * H * D + 3 * D, A_T0 + 10, (G * 6 + 6) * D, (6, D))
    assert last.outputs == (Shared(view(A_OUT, (D,), "bfloat16", offset=3 * H * D + 3 * D)),)


def test_a_request_without_rows_in_the_step_has_no_window():
    # r_idle sits between r0 and r1 in the block table with no rows (lens 0): it gets no Int8KvQuant and r1's slabs are its third
    rows = [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 5)]
    ctx = _ctx(rows, {"r0": 3, "r1": 6}, {"r0": [2], "r1": [5]})
    plain = Int8AttentionScope().match(_attention(4, [(3, 3), (1, 6)]), ctx)
    idle = Int8AttentionScope().match(_attention(4, [(3, 3), (0, 0), (1, 6)]), ctx)
    assert isinstance(idle, Group), idle
    assert [spec_label(c.spec) for c in idle.parts] == [spec_label(c.spec) for c in plain.parts]
    assert idle.parts[-1].operands[1].view.offset == (2 * G * 6 + 6) * D


def test_int8_attention_refuses_without_the_headers_k_scales_or_a_wrong_window():
    rows = [("r0", 0), ("r0", 1), ("r0", 2)]
    r = Int8AttentionScope().match(_attention(3, [(3, 3)]), _ctx(rows, {"r0": 3}, {"r0": [2]}, backends={}))
    assert isinstance(r, Unsupported) and "int8_k_scales" in r.reason
    r = Int8AttentionScope().match(_attention(3, [(3, 4)]), _ctx(rows, {"r0": 3}, {"r0": [2]}))
    assert isinstance(r, Unsupported) and "last rows" in r.reason
    r = Int8AttentionScope().match(_attention(3, [(3, 2)]), _ctx(rows, {"r0": 3}, {"r0": [2]}))
    assert isinstance(r, Unsupported) and "last rows" in r.reason
    r = Int8AttentionScope().match(_attention(3, [(3, 3)], kv_lens=[4]), _ctx(rows, {"r0": 3}, {"r0": [2]}))
    assert isinstance(r, Unsupported) and "host lists of one batch" in r.reason
    other = (view(900, (1, G, 3, D), "int8"), view(901, (1, G, 3, D), "int8"), view(902, (1, G, D), "float32"), view(903, (1, G, D), "float32"))
    r = Int8AttentionScope().match(_attention(3, [(3, 3)], rows_reads=other), _ctx(rows, {"r0": 3}, {"r0": [2]}))
    assert isinstance(r, Unsupported) and "does not read its kv_quant_batch's results" in r.reason


KVW_SCHEMA = ("verity_int8::kv_quant_window_batch(Tensor keys, Tensor values, SymInt[] starts, SymInt[] lens, Tensor k_scales) "
              "-> (Tensor, Tensor, Tensor, Tensor)")
CS_SCHEMA = ("verity_int8::credited_step(Tensor seq_lens, SymInt batch, SymInt kv_heads, SymInt head_dim, SymInt read) "
             "-> (Tensor, Tensor, Tensor, Tensor)")
CW_SCHEMA = ("verity_int8::credited_window(Tensor kv_cache, Tensor block_table, Tensor? slot_mapping, Tensor lens, Tensor wo, "
             "Tensor(a1!) k, Tensor(a2!) v, SymInt layer, SymInt read) -> ()")
A_WK, A_WV = 610, 611


def _credited_attention(T, reqs, first_layer=True, starts=None):
    """`_attention` under a credited read: the step's credited_step (at its first layer), the layer's credited_window over the
    LBNHC cache [NB, G, BS, 2D] into K and V [rows, G, D], and one kv_quant_window_batch over each request's rows of them from
    `starts` (default: the windows back to back)."""
    lens = [s for _, s in reqs]
    rows = sum(lens)
    kbuf, vbuf = view(A_WK, (rows, G, D), "bfloat16"), view(A_WV, (rows, G, D), "bfloat16")
    kv = view(A_KV, (NB, G, BS, 2 * D), "bfloat16", strides=(BS * G * 2 * D, 2 * D, G * 2 * D, 1))
    ev = []
    if first_layer:
        ev += [DispatchEnter(2, 1, 0, "forward", "verity_int8.credited_step.default", CS_SCHEMA,
                             [view(620, (len(reqs),), "int32"), len(reqs), G, D, 1], {}, 7, LAYER),
               DispatchExit(2, (view(621, (len(reqs),), "int32"), view(622, (len(reqs) + 1,), "int32"), kbuf, vbuf))]
    ev += [DispatchEnter(3, 1, 0, "forward", "verity_int8.credited_window.default", CW_SCHEMA,
                         [kv, view(A_BT, (len(reqs), 4), "int32"), None, view(621, (len(reqs),), "int32"),
                          view(622, (len(reqs) + 1,), "int32"), kbuf, vbuf, 0, 1], {}, 7, LAYER), DispatchExit(3, None)]
    res = _results(len(reqs), max(lens))
    ev += [DispatchEnter(4, 1, 0, "forward", "verity_int8.kv_quant_window_batch.default", KVW_SCHEMA,
                         [kbuf, vbuf, starts or [0, *[sum(lens[:j + 1]) for j in range(len(lens) - 1)]], lens, view(A_KS, (G,), "float32")],
                         {}, 7, LAYER),
           DispatchExit(4, res), *_rows(5, reqs, res)]
    return _scope(T, ev)


def _bindings(group):
    return [(spec_label(c.spec), [(type(o).__name__, getattr(o, "indices", None), getattr(getattr(o, "view", None), "offset", None),
                                   getattr(getattr(o, "view", None), "shape", None), getattr(getattr(o, "view", None), "strides", None))
                                  for o in c.operands], c.outputs) for c in group.parts]


@pytest.mark.parametrize("first_layer", [True, False])
def test_a_credited_read_folds_to_the_plain_reads_computes(first_layer):
    # the same step as the plain test: kv_quant_window_batch over the credited window's rows binds the same slots of the same cache
    rows = [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 5)]
    ctx = _ctx(rows, {"r0": 3, "r1": 6}, {"r0": [2], "r1": [5]})
    reqs = [(3, 3), (1, 6)]
    plain = Int8AttentionScope().match(_attention(4, reqs), ctx)
    cred = Int8AttentionScope().match(_credited_attention(4, reqs, first_layer=first_layer), ctx)
    assert isinstance(plain, Group) and isinstance(cred, Group), cred
    assert _bindings(cred) == _bindings(plain)
    kvq = [c for c in cred.parts if spec_label(c.spec).startswith("Int8KvQuant")]
    assert [c.consumed for c in kvq] == [(4,)] * (2 * G)
    assert set(cred.consumed) == set(range(1, 7)) - ({2} if not first_layer else set())


def test_a_credited_read_refuses_rows_other_than_its_requests_window():
    rows = [("r0", 0), ("r0", 1), ("r0", 2), ("r1", 5)]
    ctx = _ctx(rows, {"r0": 3, "r1": 6}, {"r0": [2], "r1": [5]})
    reqs = [(3, 3), (1, 6)]
    for starts in ([0, 2], [3, 0], [0, 4]):
        r = Int8AttentionScope().match(_credited_attention(4, reqs, starts=starts), ctx)
        assert isinstance(r, Unsupported) and "rows of the credited window" in r.reason, starts
    tree = _credited_attention(4, reqs)
    tree.children = [c for c in tree.children if "credited_window" not in str(c.op)]
    r = Int8AttentionScope().match(tree, ctx)
    assert isinstance(r, Unsupported) and "exactly one credited_window" in r.reason


@pytest.mark.parametrize("T", [1])
def test_an_fa2_scope_is_not_the_int8_patterns(T):
    ev = [DispatchEnter(1, None, 0, "forward", "vllm.unified_attention_with_output.default", UA_SCHEMA,
                        [view(A_Q, (T, H, D), "bfloat16"), view(700, (T, G, D), "bfloat16"), view(701, (T, G, D), "bfloat16"),
                         view(A_OUT, (T, H, D), "bfloat16"), LAYER], {}, 7, LAYER), DispatchExit(1, None)]
    assert Int8AttentionScope().match(_tree(ev), _ctx([("r0", 0)], {"r0": 1}, {"r0": [0]})) is None
