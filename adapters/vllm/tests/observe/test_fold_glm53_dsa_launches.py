"""GLM-5.3's DSA indexer launches in a multi-request step, folded as vLLM d9105ea80 issues them on cc 9.0 (`sparse_attn_indexer`):

* the fused-q launch is held to its pinned source, geometry and constexprs, then to its RoPE pointers, the step's positions and its
  bf16 head rows; an indexer layer's index-Q is held to its pointers, dimensions, shapes and scales, and an IndexShare layer to its
  dummy dimensions; the launch folds as one RoPE row a head and, on an indexer layer, one index-Q row a token;
* the leading one-row requests are decodes and every request after them a prefill (`split_decodes_and_prefills`, threshold 1);
* each prefill chunk gathers its requests' keys, then DeepGEMM fills an invisible ``[align(rows, 4), width]`` f32 allocation whose row of
  request j holds its logits from column ``cu_j``; the decodes' paged logits ``[rows, width]`` come after, from column 0;
* each top-k writes the run of the top-k allocation's rows its launch covers;
* every row is one `DsaMqaLogits_v3{N=T}` and one `DsaTopKShortRow_v1`, the Build's per row; a launch that is not the next one is refused;
* the request-index copy between the top-k and FA3 is held to its pinned launch, its pointers, its i32 layouts and the shared top-k
  allocation, in that order, and records the page table it writes;
* the FA3 sparse MLA call is held to its pointers, the step's rows, the pinned widths, bf16 and the default arguments, in that order.
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import pytest

from verity_vllm.observe.events import (
    DispatchEnter,
    DispatchExit,
    RunHeader,
    StepStart,
    TritonLaunch,
    ViewDesc,
)
from verity_vllm.observe.fold.patterns import dsa as D
from verity_vllm.observe.fold.profiles import profile_for_case
from verity_vllm.observe.fold.resolver import (
    Compute,
    Gather,
    Group,
    PerRow,
    StepContext,
    Transparent,
    Unsupported,
    spec_label,
)
from verity_vllm.observe.fold.tree import build_trees

CUDA, LAYER = "cuda:0", "model.layers.0.self_attn"
GATHER = ("_C_cache_ops::cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor($0! -> ) dst_k, Tensor($1! -> ) dst_scale, "
          "Tensor block_table, Tensor cu_seq_lens) -> ()")
EMPTY = ("aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
         "bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
PREFILL_TOPK = ("_C::top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, Tensor($0! -> ) indices, int numRows, "
                "int stride0, int stride1, int topK) -> ()")
DECODE_TOPK = "_C::cooperative_topk(Tensor logits, Tensor lengths, Tensor($0! -> ) output, Tensor workspace, int k, int max_seq_len) -> ()"
TOPK_ALLOC = 2293


def view(alloc, shape, dtype="float32", offset=0, strides=None):
    if strides is None:
        strides, acc = [], 1
        for s in reversed(shape):
            strides.insert(0, acc)
            acc *= s
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides), dtype, CUDA)


def _ctx(requests):
    """A step over `requests` = [(request, positions this step)] in row order, each request in its own run of 64-slot blocks."""
    rows = [(r, p) for r, ps in requests for p in ps]
    hdr = RunHeader("verity-capture/log/v1", "2026-10-18T00:00:00Z", "pid", {}, {"repo": "zai-org/GLM-5.3", "config": {}}, {"requests": []})
    seq = {r: ps[-1] + 1 for r, ps in requests}
    tables, nb = {}, 0
    for r, n in seq.items():
        tables[r], nb = list(range(nb, nb + -(-n // 64))), nb + -(-n // 64)
    start = StepStart(0, rows, [0] * len(rows), [p for _, p in rows], seq, {r: ps[0] for r, ps in requests}, tables, 64,
                      [0] * len(rows), [], [], len(requests), len(rows), len(rows))
    p = profile_for_case("QWEN3_4B_FP8")
    ctx = StepContext(hdr, start, None, [], {}, p.config, p)
    M = len(rows)
    D._state(ctx)["topk"] = (TOPK_ALLOC, 0, 2048, (2048, 1))
    D._state(ctx)["modules"][LAYER] = {
        "index_q": view(10, (M, 32, 128), "float8_e4m3fn"), "index_weights": view(11, (M, 32)),
        "index_cache": D._index_cache(view(12, (nb, 64, 132), "uint8"), view(12, (nb, 64, 33)), 64, 132, 128),
        "mqa_after": 1, "mqa_launches": D._indexer_launches(ctx)}
    return ctx


def _node(id_, op, schema, args, outs=None):
    return build_trees([DispatchEnter(id_, None, 0, "forward", op, schema, list(args), {}, 7, LAYER), DispatchExit(id_, outs)])[0][0]


def _gather(id_, keys):
    return _node(id_, "_C_cache_ops.cp_gather_indexer_k_quant_cache.default", GATHER,
                 [view(13, (8, 64, 132), "uint8"), view(14, (keys, 128), "float8_e4m3fn"), view(14, (keys, 4), "uint8"),
                  view(15, (1, 4), "int32"), view(16, (2,), "int32")])


def _empty(id_, shape, alloc):
    return _node(id_, "aten.empty.memory_format", EMPTY, [list(shape)], outs=view(alloc, shape))


def _logits(r):
    assert isinstance(r, Group), r
    return [(spec_label(p.spec), p.attribution[0], p.outputs[0].view.offset, p.outputs[0].view.shape) for p in r.parts]


def test_the_step_s_launches_are_its_leading_one_row_requests_then_the_prefills():
    assert D._indexer_launches(_ctx([("a", [19]), ("b", [20]), ("c", range(11)), ("d", [3])])) == \
        [[0, 1], [("c", list(range(2, 13))), ("d", [13])]]
    assert D._indexer_launches(_ctx([("a", [19]), ("b", [20])])) == [[0, 1], []]
    assert D._indexer_launches(_ctx([("c", range(11)), ("a", [19])])) == [[], [("c", list(range(11))), ("a", [11])]]


def test_a_mixed_step_folds_the_prefill_chunk_then_the_decodes():
    """Greedy B8 step 9 on rank 0: a's decode row first, b's 11-row prefill after it; DeepGEMM's prefill logits are [12, 512]."""
    ctx = _ctx([("a", [19]), ("b", range(11))])
    assert isinstance(D.DsaMqaGather().match(_gather(2, 11), ctx), Transparent)
    pre = _logits(D.DsaMqaAllocation().match(_empty(3, (12, 512), 31015), ctx))
    assert pre == [(f"DsaMqaLogits_v3{{N={p + 1},H=32,D=128}}", ("b", p), p * 512, (1, p + 1)) for p in range(11)]
    dec = _logits(D.DsaMqaAllocation().match(_empty(4, (1, 512), 31018), ctx))
    assert dec == [("DsaMqaLogits_v3{N=20,H=32,D=128}", ("a", 19), 0, (1, 20))]
    assert D.DsaMqaAllocation().match(_empty(5, (3, 512), 31020), ctx) is None


def test_a_chunk_of_several_prefills_reads_each_request_s_keys_from_its_first_column():
    ctx = _ctx([("a", range(5)), ("b", range(3))])
    D.DsaMqaGather().match(_gather(2, 8), ctx)
    got = _logits(D.DsaMqaAllocation().match(_empty(3, (8, 512), 40), ctx))
    assert [(a, off) for _s, a, off, _sh in got] == [(("a", p), p * 512) for p in range(5)] + [(("b", p), (5 + p) * 512 + 5) for p in range(3)]


def test_each_prefill_chunk_is_its_own_gather_and_allocation():
    ctx = _ctx([("a", range(5)), ("b", range(3))])
    D.DsaMqaGather().match(_gather(2, 5), ctx)
    assert [a for _s, a, _o, _sh in _logits(D.DsaMqaAllocation().match(_empty(3, (8, 512), 40), ctx))] == [("a", p) for p in range(5)]
    D.DsaMqaGather().match(_gather(4, 3), ctx)
    assert [(a, off) for _s, a, off, _sh in _logits(D.DsaMqaAllocation().match(_empty(5, (4, 512), 41), ctx))] == \
        [(("b", p), p * 512) for p in range(3)]


def test_a_single_prefill_of_256_rows_folds_as_before():
    """B1's prefill: [256, 512], row t at column 0."""
    ctx = _ctx([("a", range(256))])
    D.DsaMqaGather().match(_gather(2, 256), ctx)
    got = _logits(D.DsaMqaAllocation().match(_empty(3, (256, 512), 40), ctx))
    assert len(got) == 256 and got[255] == ("DsaMqaLogits_v3{N=256,H=32,D=128}", ("a", 255), 255 * 512, (1, 256))


@pytest.mark.parametrize("keys,shape", [(11, (11, 512)), (6, (12, 512)), (None, (12, 512))],
                         ids=["unaligned-rows", "partial-request", "no-gather"])
def test_an_allocation_that_is_not_the_next_launch_is_refused(keys, shape):
    ctx = _ctx([("b", range(11)), ("c", range(4))])
    if keys is not None:
        D.DsaMqaGather().match(_gather(2, keys), ctx)
    assert isinstance(D.DsaMqaAllocation().match(_empty(3, shape, 40), ctx), Unsupported)


def test_each_top_k_writes_the_rows_of_its_launch():
    ctx = _ctx([("a", [19]), ("b", range(11))])
    pre = _node(6, "_C.top_k_per_row_prefill.default", PREFILL_TOPK,
                [view(31015, (11, 11), strides=(512, 1)), view(1, (11,), "int32"), view(2, (11,), "int32"),
                 view(TOPK_ALLOC, (11, 2048), "int32", offset=2048), 11, 512, 1, 2048])
    r = D.DsaTopK().match(pre, ctx)
    assert isinstance(r, Group) and [p.attribution[0] for p in r.parts] == [("b", p) for p in range(11)]
    assert [p.outputs[0].view.offset for p in r.parts] == [(1 + p) * 2048 for p in range(11)]
    dec = _node(7, "_C.cooperative_topk.default", DECODE_TOPK,
                [view(31018, (1, 288), strides=(512, 1)), view(3, (1, 1), "int32"), view(TOPK_ALLOC, (1, 2048), "int32"),
                 view(4, (1048576,), "uint8"), 2048, 20])
    r = D.DsaTopK().match(dec, ctx)
    assert isinstance(r, Group) and [(p.attribution[0], p.outputs[0].view.offset) for p in r.parts] == [(("a", 19), 0)]
    off = _node(8, "_C.cooperative_topk.default", DECODE_TOPK,
                [view(31018, (1, 288), strides=(512, 1)), view(3, (1, 1), "int32"), view(TOPK_ALLOC, (1, 2048), "int32", offset=12 * 2048),
                 view(4, (1048576,), "uint8"), 2048, 20])
    assert isinstance(D.DsaTopK().match(off, ctx), Unsupported)


Q_PINS = {"NUM_Q_HEADS": 8, "Q_PE_HALF_ROT_DIM": 32, "QL_NOPE_DIM": 512, "QL_NOPE_BLOCK": 512, "INDEX_ROPE_INTERLEAVE": True,
          "QUANTIZE_MQA": False}
Q_OFF_PIN = "fused-q source or launch geometry differs from the recorded GLM-5.3 pin"


def _q(rows=2, indexer=True, consts=None, warps=1, fn="vllm.models.glm." + D.Q_KERNEL, **args):
    """The fused-q launch over `rows` rows, of an indexer layer (which also rotates and quantizes its index-Q) or of an IndexShare
    layer; `args` replaces bound arguments by name."""
    bound = {"q_pe_ptr": view(60, (rows, 8, 64), "bfloat16"), "q_pe_out_ptr": view(61, (rows, 8, 64), "bfloat16"),
             "q_pe_cos_sin_ptr": view(62, (4096, 64), "bfloat16"), "pos_ptr": view(63, (rows,), "int64"),
             "ql_nope_ptr": view(64, (rows, 8, 512), "bfloat16")}
    dims = {"HAS_INDEXER": False, "NUM_INDEX_Q_HEADS": 1, "INDEX_Q_HEAD_DIM": 1}
    if indexer:
        bound.update({"index_q_ptr": view(65, (rows, 32, 128), "bfloat16"), "index_q_cos_sin_ptr": view(66, (4096, 64), "bfloat16"),
                      "index_weights_ptr": view(67, (rows, 32), "bfloat16"), "index_q_fp8_ptr": view(68, (rows, 32, 128), "float8_e4m3fn"),
                      "index_weights_out_ptr": view(69, (rows, 32)), "index_weights_softmax_scale": 128 ** -0.5,
                      "index_weights_head_scale": 32 ** -0.5})
        dims = {"HAS_INDEXER": True, "NUM_INDEX_Q_HEADS": 32, "INDEX_Q_HEAD_DIM": 128, "INDEX_Q_HALF_ROT_DIM": 32}
    bound.update(args)
    consts = {**Q_PINS, **dims, **(consts or {})}
    return build_trees([TritonLaunch(21, None, 0, "forward", fn, D.Q_SRC_SHA, {**bound, **consts}, consts, [rows], warps, 3, 7, LAYER)])[0][0]


def test_a_fused_q_of_an_indexer_layer_is_one_rope_row_a_head_and_one_index_q_row_a_token():
    ctx = _ctx([("a", [5]), ("b", [9])])
    assert D.DsaQ().match(_q(fn="vllm.models.glm.another_kernel"), ctx) is None, "another kernel is not this pattern's"
    r = D.DsaQ().match(_q(), ctx)
    assert isinstance(r, Group) and (r.consumed, r.hint, r.pattern) == ((21,), LAYER, "dsa.fused_q")
    rope, index_q = r.parts
    assert isinstance(rope, Compute) and (spec_label(rope.spec), rope.rows, rope.consumed) == ("RoPEHeadGptJ_v2{D=64}", 16, (21,))
    assert rope.operands == (PerRow(view(60, (16, 64), "bfloat16")), Gather(view(62, (4096, 64), "bfloat16"), (5,) * 8 + (9,) * 8))
    assert rope.outputs == (PerRow(view(61, (16, 64), "bfloat16")),) and rope.attribution == (("a", 5),) * 8 + (("b", 9),) * 8
    assert (rope.hint, rope.kernels, rope.roots) == (f"{LAYER}/q_pe", (D.Q_KERNEL,), (None, None))
    assert isinstance(index_q, Compute) and (spec_label(index_q.spec), index_q.rows) == ("DsaIndexerQ_v1{H=32,D=128,ROPE=64}", 2)
    assert index_q.operands == (PerRow(view(65, (2, 32, 128), "bfloat16")), Gather(view(66, (4096, 64), "bfloat16"), (5, 9)),
                                PerRow(view(67, (2, 32), "bfloat16")))
    assert index_q.outputs == (PerRow(view(68, (2, 32, 128), "float8_e4m3fn")), PerRow(view(69, (2, 32))))
    assert (index_q.consumed, index_q.hint) == ((), f"{LAYER}/indexer_q")
    state = D._state(ctx)["modules"][LAYER]
    assert (state["q_rope"], state["q_v"]) == (view(61, (2, 8, 64), "bfloat16"), view(64, (2, 8, 512), "bfloat16"))
    assert (state["index_q"], state["index_weights"]) == (view(68, (2, 32, 128), "float8_e4m3fn"), view(69, (2, 32)))
    assert (state["mqa_after"], state["mqa_launches"]) == (21, [[0, 1], []])


def test_a_fused_q_of_an_index_share_layer_is_the_rope_rows_alone():
    ctx = _ctx([("a", [5]), ("b", [9])])
    before = dict(D._state(ctx)["modules"][LAYER])
    r = D.DsaQ().match(_q(indexer=False), ctx)
    assert isinstance(r, Group) and [spec_label(p.spec) for p in r.parts if isinstance(p, Compute)] == ["RoPEHeadGptJ_v2{D=64}"]
    state = D._state(ctx)["modules"][LAYER]
    assert (state["q_rope"], state["q_v"]) == (view(61, (2, 8, 64), "bfloat16"), view(64, (2, 8, 512), "bfloat16"))
    assert {k: state[k] for k in before} == before, "an IndexShare layer leaves the indexer's state as it was"
    dims = D.DsaQ().match(_q(indexer=False, consts={"NUM_INDEX_Q_HEADS": 32}), ctx)
    assert isinstance(dims, Unsupported) and dims.reason == "IndexShare layer has unexpected dummy index-Q dimensions"


@pytest.mark.parametrize("kw,reason,detail", [
    ({"warps": 8}, Q_OFF_PIN, {"src_sha": D.Q_SRC_SHA, "num_warps": 8, "num_stages": 3}),
    ({"consts": {"QUANTIZE_MQA": True}}, "fused-q constexprs differ from the recorded GLM-5.3 pin", {"mismatches": {"QUANTIZE_MQA": (True, False)}}),
    ({"q_pe_out_ptr": None}, "fused-q RoPE pointers are not bound", {}),
    ({"pos_ptr": 3}, "positions argument is not a tensor view", {"positions": "3"}),
    ({"q_pe_cos_sin_ptr": view(62, (4096, 32), "bfloat16")}, "fused-q RoPE views differ from bf16 [M,8,64]",
     {"cs": repr(view(62, (4096, 32), "bfloat16"))}),
    ({"q_pe_ptr": view(60, (2, 8, 64), "bfloat16", strides=(1024, 64, 1))}, "fused-q RoPE views do not flatten to [M*8, 64] head rows",
     {"q": repr(view(60, (2, 8, 64), "bfloat16", strides=(1024, 64, 1)))}),
    ({"index_weights_out_ptr": None}, "indexer-layer fused-q pointers are not bound", {}),
    ({"consts": {"INDEX_Q_HEAD_DIM": 64}}, "index-Q dimensions differ from the recorded GLM-5.3 pin", {}),
    ({"index_weights_ptr": view(67, (2, 16), "bfloat16")}, "index-Q views have the wrong shape",
     {"weights": repr(view(67, (2, 16), "bfloat16"))}),
    ({"index_weights_head_scale": 0.5}, "index-Q scales differ from D^-0.5 and H^-0.5", {"softmax_scale": 128 ** -0.5, "head_scale": 0.5}),
], ids=["warps", "constexpr", "rope-pointer", "positions", "rope-views", "head-rows", "index-pointer", "index-dims", "index-shapes", "scales"])
def test_a_fused_q_off_its_pins_or_its_views_is_refused_with_what_differs(kw, reason, detail):
    off = D.DsaQ().match(_q(**kw), _ctx([("a", [5]), ("b", [9])]))
    assert isinstance(off, Unsupported) and off.reason == reason
    assert {k: off.detail[k] for k in detail} == detail


SHARE_PINS = {"BLOCK_SIZE": 64, "BLOCK_STRIDE_ROWS": 64, "BLOCK_N": 2048, "HAS_PREFILL": False, "COUNT_VALID": True, "SINGLE_TILE": True,
              "COMPACT_TO_FRONT": True, "DCP_SIZE": 1, "DCP_RANK": 0, "DCP_INTERLEAVE": 1}


def _share(rows=2, consts=None, warps=8, fn=D.INDEX_SHARE_KERNEL, **pointers):
    """The request-index copy's launch over `rows` rows: it reads the request ids, the block table and the top-k allocation's rows, and
    writes the page table and each row's valid count."""
    args = {"req_id_ptr": view(50, (rows,), "int32"), "block_table_ptr": view(51, (rows, 4), "int32"),
            "token_indices_ptr": view(TOPK_ALLOC, (rows, 2048), "int32"), "out_ptr": view(52, (rows, 2048), "int32"),
            "valid_count_ptr": view(53, (rows,), "int32"), **pointers}
    consts = {**SHARE_PINS, **(consts or {})}
    launch = TritonLaunch(41, None, 0, "forward", fn, D.INDEX_SHARE_SRC_SHA, {**args, **consts}, consts, [rows], warps, 3, 7, LAYER)
    return build_trees([launch])[0][0]


@pytest.mark.parametrize("kw,warps,mismatches", [({"warps": 4}, 4, {}), ({"consts": {"HAS_PREFILL": True}}, 8, {"HAS_PREFILL": (True, False)})],
                         ids=["warps", "constexpr"])
def test_an_index_share_copy_off_its_pinned_launch_is_refused_with_what_differs(kw, warps, mismatches):
    off = D.DsaIndexShare().match(_share(**kw), _ctx([("a", [5]), ("b", [9])]))
    assert isinstance(off, Unsupported) and off.reason == "IndexShare conversion source or launch geometry differs from the recorded GLM-5.3 pin"
    assert (off.detail["src_sha"], off.detail["num_warps"], off.detail["num_stages"]) == (D.INDEX_SHARE_SRC_SHA, warps, 3)
    assert off.detail["mismatches"] == mismatches


def test_an_index_share_copy_is_held_to_its_pointers_then_its_layouts_then_the_shared_top_k_allocation():
    ctx = _ctx([("a", [5]), ("b", [9])])
    unbound = D.DsaIndexShare().match(_share(out_ptr=None), ctx)
    assert isinstance(unbound, Unsupported) and unbound.reason == "IndexShare conversion pointers are not bound"
    valid = view(53, (3,), "int32")
    shaped = D.DsaIndexShare().match(_share(valid_count_ptr=valid), ctx)
    assert isinstance(shaped, Unsupported) and shaped.reason == "IndexShare conversion views differ from the pinned i32 layouts"
    assert shaped.detail["valid"] == repr(valid) and shaped.detail["source"] == repr(view(TOPK_ALLOC, (2, 2048), "int32"))
    other = D.DsaIndexShare().match(_share(token_indices_ptr=view(99, (2, 2048), "int32")), ctx)
    assert isinstance(other, Unsupported) and other.reason == "IndexShare conversion does not read the shared top-k allocation"
    assert (other.detail["source"], other.detail["established"]) == ((99, 0, 2048, (2048, 1)), (TOPK_ALLOC, 0, 2048, (2048, 1)))
    assert "page_table" not in D._state(ctx)["modules"][LAYER], "a refused copy records no page table"


def test_an_index_share_copy_of_the_shared_top_k_rows_is_staging_and_records_its_page_table():
    ctx = _ctx([("a", [5]), ("b", [9])])
    assert D.DsaIndexShare().match(_share(fn="pkg.another.kernel"), ctx) is None, "another kernel is not this pattern's"
    r = D.DsaIndexShare().match(_share(), ctx)
    assert isinstance(r, Transparent) and (r.effect, r.consumed, r.kernels) == ("noop", (41,), ("ConvertReqIndexToGlobalIndexKernel",))
    assert r.hint == f"{LAYER}/IndexShare top-k staging" and r.pattern == "dsa.index_share_copy"
    assert D._state(ctx)["modules"][LAYER]["page_table"] == (52, 0, 2048, (2048, 1))


MLA = "_vllm_fa3_C::fwd(Tensor q, Tensor k, Tensor v, Tensor? q_v) -> Tensor[]"


def _mla(rows=2, heads=8, out_width=512, dtype="bfloat16", q_v=True, outs=True, op="_vllm_fa3_C.fwd.default"):
    """An FA3 sparse MLA call: `q` [rows, heads, 64], `q_v` and the output [rows, heads, 512], `k` and `v` of the paged cache."""
    q, qv, out = view(20, (rows, heads, 64), dtype), view(21, (rows, heads, 512), "bfloat16"), view(22, (rows, heads, out_width), "bfloat16")
    args = [q, view(23, (4, 64, 1, 64), "bfloat16"), view(24, (4, 64, 1, 512), "bfloat16"), qv if q_v else None]
    return _node(40, op, MLA, args, outs=[out] if outs else None), q, qv, out


@pytest.mark.parametrize("kw", [{"q_v": False}, {"outs": False}])
def test_a_sparse_mla_without_one_of_its_pointers_is_refused(kw):
    off = D.DsaSparseMla().match(_mla(**kw)[0], _ctx([("a", [5]), ("b", [9])]))
    assert isinstance(off, Unsupported) and off.reason == "FA3 sparse MLA q/q_v/k/v/out pointers are not bound" and off.consumed == (40,)


@pytest.mark.parametrize("kw", [{"rows": 3}, {"heads": 4}, {"out_width": 64}])
def test_a_sparse_mla_off_the_steps_rows_or_the_pinned_widths_is_refused_with_its_views(kw):
    node, q, qv, out = _mla(**kw)
    off = D.DsaSparseMla().match(node, _ctx([("a", [5]), ("b", [9])]))
    assert isinstance(off, Unsupported) and off.reason == "FA3 sparse MLA views differ from [M,8,64]/[M,8,512]"
    assert (off.detail["q"], off.detail["q_v"], off.detail["out"]) == (repr(q), repr(qv), repr(out))


def test_a_sparse_mla_at_the_pinned_views_is_held_to_bf16_and_then_to_the_default_arguments():
    ctx = _ctx([("a", [5]), ("b", [9])])
    wide = D.DsaSparseMla().match(_mla(dtype="float32")[0], ctx)
    assert isinstance(wide, Unsupported) and wide.reason == "FA3 sparse MLA views are not bf16"
    off = D.DsaSparseMla().match(_mla()[0], ctx)
    assert isinstance(off, Unsupported) and off.reason == "FA3 sparse MLA arguments differ from the GLM-5.3 default path"
    assert off.detail["mismatches"]["max_seqlen_k"] == (None, 2048) and off.detail["mismatches"]["is_causal"] == (None, True)
    assert D.DsaSparseMla().match(_mla(op="_vllm_fa2_C.fwd.default")[0], ctx) is None, "another library's call is not this pattern's"
