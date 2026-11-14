"""GLM-5.3's DSA indexer launches in a multi-request step, folded as vLLM d9105ea80 issues them on cc 9.0 (`sparse_attn_indexer`):

* the fused norm/RoPE launch is held to its pinned source, geometry and constexprs, then to its pointers, the step's positions and
  rows, bf16, its weights' and rotary table's shapes, the latent cache's slot rows, the step's slots and its two static epsilons; an
  indexer layer's index-K is held to its pointers, dimensions, shapes and the index cache's layout and slots, and an IndexShare layer
  to its dummy dimensions; last the top-k allocation is held to its layout and to the one an earlier layer established; the launch
  folds as the two RMSNorms and the k rotation a token and, on an indexer layer, one index-K row a token;
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
* the FA3 sparse MLA call is held to its pointers, the step's rows, the pinned widths, bf16 and the default arguments, then to the
  arguments it may not carry, to `k` and `v` as the halves of the latent cache its layer's norm launch wrote, to the page table its
  layer's index copy wrote, to its split count, to the record target's 132 SMs and to the step's block tables, in that order; it folds
  as one `SparseMlaSplit_v1` a row over the request's latent rows, at the split count the launch's heuristic gives the row;
* a logits allocation the indexer cannot fill, a gather with no keys destination and a top-k of another width or past the short row
  are refused;
* a refusal of several conditions has a case off each of them alone, and each argument the FA3 call may not carry has its own.
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import dataclasses
import math

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
from verity_vllm.observe.fold.patterns.geometry import GeometryError
from verity_vllm.observe.fold.profiles import profile_for_case
from verity_vllm.observe.fold.resolver import (
    Compute,
    Const,
    Gather,
    GatherMany,
    Group,
    PerRow,
    Shared,
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


def _ctx(requests, slots=None):
    """A step over `requests` = [(request, positions this step)] in row order, each request in its own run of 64-slot blocks; `slots`
    are its rows' KV slots, slot 0 for each when not given."""
    rows = [(r, p) for r, ps in requests for p in ps]
    hdr = RunHeader("verity-capture/log/v1", "2026-10-18T00:00:00Z", "pid", {}, {"repo": "zai-org/GLM-5.3", "config": {}}, {"requests": []})
    seq = {r: ps[-1] + 1 for r, ps in requests}
    tables, nb = {}, 0
    for r, n in seq.items():
        tables[r], nb = list(range(nb, nb + -(-n // 64))), nb + -(-n // 64)
    start = StepStart(0, rows, [0] * len(rows), [p for _, p in rows], seq, {r: ps[0] for r, ps in requests}, tables, 64,
                      [0] * len(rows) if slots is None else list(slots), [], [], len(requests), len(rows), len(rows))
    p = profile_for_case("QWEN3_4B_FP8")
    ctx = StepContext(hdr, start, None, [], {}, p.config, p)
    M = len(rows)
    D._state(ctx)["topk"] = (TOPK_ALLOC, 0, 2048, (2048, 1))
    D._state(ctx)["modules"][LAYER] = {
        "index_q": view(10, (M, 32, 128), "float8_e4m3fn"), "index_weights": view(11, (M, 32)),
        "index_cache": D._index_cache(view(12, (nb, 64, 132), "uint8"), view(12, (nb, 64, 33)), 64, 132, 128),
        "mqa_after": 1, "mqa_launches": D._indexer_launches(ctx)}
    return ctx


def _node(id_, op, schema, args, outs=None, kwargs=None):
    enter = DispatchEnter(id_, None, 0, "forward", op, schema, list(args), dict(kwargs or {}), 7, LAYER)
    return build_trees([enter, DispatchExit(id_, outs)])[0][0]


def _gather(id_, keys):
    return _node(id_, "_C_cache_ops.cp_gather_indexer_k_quant_cache.default", GATHER,
                 [view(13, (8, 64, 132), "uint8"), view(14, (keys, 128), "float8_e4m3fn"), view(14, (keys, 4), "uint8"),
                  view(15, (1, 4), "int32"), view(16, (2,), "int32")])


def _empty(id_, shape, alloc, dtype="float32"):
    return _node(id_, "aten.empty.memory_format", EMPTY, [list(shape)], outs=view(alloc, shape, dtype))


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


def test_a_logits_row_reads_its_token_s_index_q_and_weights_and_its_request_s_keys_and_scales_from_the_index_cache():
    """a's decode row, then b's prefill of three rows in a [4, 512] allocation: b's last row is step row 3, the launch's third."""
    ctx = _ctx([("a", [19]), ("b", range(3))])
    ix = D._state(ctx)["modules"][LAYER]["index_cache"]
    D.DsaMqaGather().match(_gather(2, 3), ctx)
    r = D.DsaMqaAllocation().match(_empty(3, (4, 512), 40), ctx)
    assert isinstance(r, Group) and (r.consumed, r.hint, r.pattern) == ((3,), LAYER, "dsa.mqa_logits.invisible_deepgemm")
    assert [p.consumed for p in r.parts] == [(3,), (), ()], "the launch's first row carries the allocation's event"
    last, slots = r.parts[2], (64, 65, 66)
    assert isinstance(last, Compute) and (spec_label(last.spec), last.rows) == ("DsaMqaLogits_v3{N=3,H=32,D=128}", 1)
    assert last.operands == (PerRow(view(10, (1, 32, 128), "float8_e4m3fn", offset=3 * 4096)), PerRow(view(11, (1, 32), offset=3 * 32)),
                             GatherMany(ix.rows, (tuple(map(ix.row, slots)),)), GatherMany(ix.scales, (tuple(map(ix.scale, slots)),)))
    assert last.outputs == (PerRow(view(40, (1, 3), offset=2 * 512, strides=(512, 1))),)
    assert (last.hint, last.kernels) == (f"{LAYER}/mqa[t=2]", ("fp8_mqa_logits", "fp8_paged_mqa_logits"))
    assert (last.attribution, last.pattern) == ((("b", 2),), "dsa.mqa_logits.invisible_deepgemm")


NOT_NEXT = ("MQA logits allocation is not the next indexer launch's: a prefill chunk of whole requests over the keys its gather staged "
            "([align(rows, 4), width]), else the decodes ([rows, width])")


@pytest.mark.parametrize("keys,shape,request_keys,rows", [(11, (11, 512), 11, 11), (6, (12, 512), 11, 11), (None, (12, 512), 0, 0)],
                         ids=["unaligned-rows", "partial-request", "no-gather"])
def test_an_allocation_that_is_not_the_next_launch_is_refused(keys, shape, request_keys, rows):
    ctx = _ctx([("b", range(11)), ("c", range(4))])
    if keys is not None:
        D.DsaMqaGather().match(_gather(2, keys), ctx)
    off = D.DsaMqaAllocation().match(_empty(3, shape, 40), ctx)
    assert isinstance(off, Unsupported) and off.reason == NOT_NEXT
    assert {k: off.detail[k] for k in ("allocation", "gathered_keys", "request_keys", "rows")} == {
        "allocation": repr(view(40, shape)), "gathered_keys": keys, "request_keys": request_keys, "rows": rows}


def test_a_prefill_request_the_step_gives_no_length_counts_for_no_keys():
    ctx = _ctx([("b", range(3))])
    ctx.start = dataclasses.replace(ctx.start, seq_lens={})
    D.DsaMqaGather().match(_gather(2, 1), ctx)
    off = D.DsaMqaAllocation().match(_empty(3, (4, 512), 40), ctx)
    assert isinstance(off, Unsupported) and off.reason == NOT_NEXT
    assert (off.detail["gathered_keys"], off.detail["request_keys"], off.detail["rows"]) == (1, 0, 3)


def test_each_top_k_writes_the_rows_of_its_launch():
    ctx = _ctx([("a", [19]), ("b", range(11))])
    pre = _node(6, "_C.top_k_per_row_prefill.default", PREFILL_TOPK,
                [view(31015, (11, 11), strides=(512, 1)), view(1, (11,), "int32"), view(2, (11,), "int32"),
                 view(TOPK_ALLOC, (11, 2048), "int32", offset=2048), 11, 512, 1, 2048])
    r = D.DsaTopK().match(pre, ctx)
    assert isinstance(r, Group) and [p.attribution[0] for p in r.parts] == [("b", p) for p in range(11)]
    assert [p.outputs[0].view.offset for p in r.parts] == [(1 + p) * 2048 for p in range(11)]
    assert (r.consumed, r.hint, r.pattern) == ((6,), LAYER, "dsa.topk_short_row")
    assert [p.consumed for p in r.parts] == [(6,)] + [()] * 10, "the launch's first row carries the call's event"
    assert all(p.kernels == ("topKPerRowPrefill",) for p in r.parts)
    third = r.parts[2]
    assert isinstance(third, Compute) and (spec_label(third.spec), third.rows) == ("DsaTopKShortRow_v1{TOPK=2048}", 1)
    assert (third.operands, third.hint, third.pattern) == ((Const(32, 3, "len"),), f"{LAYER}/topk[t=2]", "dsa.topk_short_row")
    assert third.outputs == (PerRow(view(TOPK_ALLOC, (1, 2048), "int32", offset=3 * 2048)),)
    dec = _node(7, "_C.cooperative_topk.default", DECODE_TOPK,
                [view(31018, (1, 288), strides=(512, 1)), view(3, (1, 1), "int32"), view(TOPK_ALLOC, (1, 2048), "int32"),
                 view(4, (1048576,), "uint8"), 2048, 20])
    r = D.DsaTopK().match(dec, ctx)
    assert isinstance(r, Group) and [(p.attribution[0], p.outputs[0].view.offset) for p in r.parts] == [(("a", 19), 0)]
    assert (r.parts[0].operands, r.parts[0].kernels, r.parts[0].consumed) == ((Const(32, 20, "len"),), ("cooperative_topk",), (7,))
    off = _node(8, "_C.cooperative_topk.default", DECODE_TOPK,
                [view(31018, (1, 288), strides=(512, 1)), view(3, (1, 1), "int32"), view(TOPK_ALLOC, (1, 2048), "int32", offset=12 * 2048),
                 view(4, (1048576,), "uint8"), 2048, 20])
    assert isinstance(D.DsaTopK().match(off, ctx), Unsupported)


NORM_PINS = {"Q_DIM": 2048, "Q_BLOCK_SIZE": 2048, "KV_DIM": 512, "KPE_HALF_ROT_DIM": 32, "MLA_CACHE_BLOCK_SIZE": 64,
             "MLA_CACHE_FP8": False, "MLA_CACHE_DS_MLA": False, "MLA_NUM_TILES": 1, "MLA_TILE_DIM": 1, "TOPK": 2048,
             "TOPK_BLOCK_SIZE": 1024, "INDEX_ROPE_INTERLEAVE": True}
NORM_OFF_PIN = "fused norm/RoPE source or launch geometry differs from the recorded GLM-5.3 pin"
NORM_OFF_ROWS, NORM_OFF_SHAPE = "fused norm/RoPE rows or GLM-5.3 dimensions differ", "norm weights or GPT-J rotary table have the wrong shape"
INDEX_OFF_PIN, INDEX_OFF_SHAPE = "index-K dimensions differ from the recorded GLM-5.3 pin", "index-K inputs have the wrong shape"
CS, TWO = view(76, (4096, 64), "bfloat16"), [("a", [5]), ("b", [9])]
# what the index-K layer norm's epsilon may not be: absent, a word, a number as text, a bool, a list
OFF_EPS = [None, "small", "1e-4", True, [1e-4]]
# a top-k allocation off its width and strides together, then off one term of its layout alone: its dtype, its width, its strides
OFF_TOPKS = [view(TOPK_ALLOC, (16, 1024), "int32"), view(TOPK_ALLOC, (16, 2048), "int64"),
             view(TOPK_ALLOC, (16, 1024), "int32", strides=(2048, 1)), view(TOPK_ALLOC, (16, 2048), "int32", strides=(4096, 1))]


def _norm(rows=2, indexer=True, consts=None, warps=4, fn="vllm.models.glm." + D.NORM_KERNEL, **args):
    """The fused norm/RoPE launch over `rows` rows and caches of two 64-slot blocks, of an indexer layer (which also normalizes, rotates
    and quantizes its index-K into the index cache) or of an IndexShare layer; `args` replaces bound arguments by name."""
    bound = {"q_c_ptr": view(70, (rows, 2048), "bfloat16"), "q_c_out_ptr": view(71, (rows, 2048), "bfloat16"),
             "q_rms_norm_w_ptr": view(72, (2048,), "bfloat16"), "q_rms_eps": 1e-6, "kv_ptr": view(73, (rows, 512), "bfloat16"),
             "kv_rms_norm_w_ptr": view(74, (512,), "bfloat16"), "kv_rms_eps": 1e-5, "kpe_ptr": view(75, (rows, 64), "bfloat16"),
             "kpe_rope_cos_sin_cache_ptr": CS, "mla_cache_ptr": view(77, (2, 64, 576), "bfloat16"), "pos_ptr": view(63, (rows,), "int64"),
             "topk_indices_ptr": view(TOPK_ALLOC, (16, 2048), "int32")}
    dims = {"HAS_INDEXER": False, "INDEX_K_DIM": 1, "INDEX_K_BLOCK_SIZE": 1}
    if indexer:
        bound.update({"index_k_ptr": view(78, (rows, 128), "bfloat16"), "index_k_layer_norm_w_ptr": view(79, (128,)),
                      "index_k_layer_norm_bias_ptr": view(80, (128,)), "index_k_layer_norm_eps": 1e-4,
                      "index_k_rope_cos_sin_cache_ptr": CS, "indexer_cache_ptr": view(81, (2, 64, 132), "uint8"),
                      "indexer_cache_scale_ptr": view(81, (2, 64, 33)), "indexer_cache_block_size": 64, "indexer_cache_stride": 132})
        dims = {"HAS_INDEXER": True, "INDEX_K_DIM": 128, "INDEX_K_BLOCK_SIZE": 128, "INDEX_K_HALF_ROT_DIM": 32}
    bound.update(args)
    consts = {**NORM_PINS, **dims, **(consts or {})}
    launch = TritonLaunch(31, None, 0, "forward", fn, D.NORM_SRC_SHA, {**bound, **consts}, consts, [rows, 1, 1], warps, 3, 7, LAYER)
    return build_trees([launch])[0][0]


def test_a_fused_norm_rope_of_an_indexer_layer_is_two_norms_the_k_rotation_and_one_index_k_row_a_token():
    ctx = _ctx(TWO, slots=[5, 70])
    assert D.DsaNormRope().match(_norm(fn="vllm.models.glm.another_kernel"), ctx) is None, "another kernel is not this pattern's"
    r = D.DsaNormRope().match(_norm(), ctx)
    assert isinstance(r, Group) and (r.consumed, r.hint, r.pattern) == ((31,), LAYER, "dsa.fused_norm_rope")
    q_norm, kv_norm, k_pe, index_k = r.parts
    assert all(isinstance(p, Compute) and p.rows == 2 and p.pattern == "dsa.fused_norm_rope" for p in r.parts)
    assert [spec_label(p.spec) for p in r.parts] == [
        "DsaRMSNorm_v1{N=2048,EPS=1e-06}", "DsaRMSNorm_v1{N=512,EPS=1e-05}", "RoPEHeadGptJ_v2{D=64}",
        "DsaIndexerKRow_v1{D=128,ROPE=64,EPS=0.0001}"], "each epsilon is its own argument's"
    kv_cache, kpe_cache = view(77, (128, 512), "bfloat16", strides=(576, 1)), view(77, (128, 64), "bfloat16", offset=512, strides=(576, 1))
    assert q_norm.operands == (PerRow(view(70, (2, 2048), "bfloat16")), Shared(view(72, (2048,), "bfloat16")))
    assert q_norm.outputs == (PerRow(view(71, (2, 2048), "bfloat16")),)
    assert (q_norm.consumed, q_norm.hint, q_norm.kernels) == ((31,), f"{LAYER}/q_a_layernorm", (D.NORM_KERNEL,))
    assert q_norm.attribution == (("a", 5), ("b", 9))
    assert kv_norm.operands == (PerRow(view(73, (2, 512), "bfloat16")), Shared(view(74, (512,), "bfloat16")))
    assert kv_norm.outputs == (Gather(kv_cache, (5, 70)),) and (kv_norm.consumed, kv_norm.hint) == ((), f"{LAYER}/kv_a_layernorm")
    assert k_pe.operands == (PerRow(view(75, (2, 64), "bfloat16")), Gather(CS, (5, 9)))
    assert k_pe.outputs == (Gather(kpe_cache, (5, 70)),) and (k_pe.consumed, k_pe.hint) == ((), f"{LAYER}/k_pe")
    ix = D._index_cache(view(81, (2, 64, 132), "uint8"), view(81, (2, 64, 33)), 64, 132, 128)
    assert index_k.operands == (PerRow(view(78, (2, 128), "bfloat16")), Shared(view(79, (128,))), Shared(view(80, (128,))), Gather(CS, (5, 9)))
    assert index_k.outputs == (Gather(ix.rows, (ix.row(5), ix.row(70))), Gather(ix.scales, (ix.scale(5), ix.scale(70))))
    assert (index_k.consumed, index_k.hint) == ((), f"{LAYER}/indexer_k")
    assert [p.roots for p in r.parts] == [(None, None), (None, None), (None, None), (None, None, None, None)]
    state = D._state(ctx)["modules"][LAYER]
    assert (state["latent"], state["kv"], state["kpe"]) == (view(77, (128, 576), "bfloat16"), kv_cache, kpe_cache)
    assert (state["slots"], state["index_cache"]) == ((5, 70), ix)
    assert D._state(ctx)["topk"] == (TOPK_ALLOC, 0, 2048, (2048, 1))
    first = D.DsaNormRope().match(_norm(rows=1), _ctx([("a", [0])]))
    assert isinstance(first, Group) and first.parts[2].operands[1] == Gather(CS, (0,)), "position 0 is the rotary table's first row"


def test_a_fused_norm_rope_of_an_index_share_layer_is_the_two_norms_and_the_k_rotation_alone():
    ctx = _ctx(TWO, slots=[5, 70])
    before = D._state(ctx)["modules"][LAYER]["index_cache"]
    r = D.DsaNormRope().match(_norm(indexer=False), ctx)
    assert isinstance(r, Group) and [p.hint for p in r.parts] == [f"{LAYER}/q_a_layernorm", f"{LAYER}/kv_a_layernorm", f"{LAYER}/k_pe"]
    state = D._state(ctx)["modules"][LAYER]
    assert state["slots"] == (5, 70) and state["index_cache"] is before, "an IndexShare layer leaves the indexer's cache as it was"
    dims = D.DsaNormRope().match(_norm(indexer=False, consts={"INDEX_K_DIM": 128}), ctx)
    assert isinstance(dims, Unsupported) and dims.reason == "IndexShare layer has unexpected dummy index-K dimensions"
    assert (dims.detail["INDEX_K_DIM"], dims.detail["INDEX_K_BLOCK_SIZE"]) == (128, 1)


def test_a_step_s_first_fused_norm_rope_establishes_the_top_k_allocation_its_later_layers_are_held_to():
    ctx = _ctx(TWO, slots=[5, 70])
    D._state(ctx)["topk"] = None
    assert isinstance(D.DsaNormRope().match(_norm(topk_indices_ptr=view(99, (16, 2048), "int32", offset=4096)), ctx), Group)
    assert D._state(ctx)["topk"] == (99, 4096, 2048, (2048, 1))
    later = D.DsaNormRope().match(_norm(), ctx)
    assert isinstance(later, Unsupported) and later.reason == "IndexShare layer does not use the indexer layers' top-k allocation"
    assert (later.detail["observed"], later.detail["established"]) == ((TOPK_ALLOC, 0, 2048, (2048, 1)), (99, 4096, 2048, (2048, 1)))


NARROW_LATENT, SHORT_INDEX = view(77, (2, 64, 512), "bfloat16"), view(81, (1, 64, 132), "uint8")
OFF_STRIDE = (f"index-K cache {view(81, (2, 64, 132), 'uint8')!r} at block size 64 and stride 136 is not D=128 bytes and a 4-byte scale a "
              "slot in whole D-byte rows")


@pytest.mark.parametrize("kw,slots,reason,detail", [
    ({"warps": 8}, (5, 70), NORM_OFF_PIN, {"src_sha": D.NORM_SRC_SHA, "num_warps": 8, "num_stages": 3}),
    ({"consts": {"MLA_CACHE_FP8": True}}, (5, 70), "fused norm/RoPE constexprs differ from the recorded GLM-5.3 pin",
     {"mismatches": {"MLA_CACHE_FP8": (True, False)}}),
    ({"kv_ptr": None}, (5, 70), "fused norm/RoPE pointer arguments are not bound", {}),
    ({"pos_ptr": 3}, (5, 70), "positions argument is not a tensor view", {"positions": "3"}),
    ({"rows": 1}, (5, 70), NORM_OFF_ROWS, {"q": repr(view(70, (1, 2048), "bfloat16")), "step_rows": 2}),
    ({"q_c_ptr": view(70, (2, 1024), "bfloat16")}, (5, 70), NORM_OFF_ROWS, {"q": repr(view(70, (2, 1024), "bfloat16"))}),
    ({"q_c_out_ptr": view(71, (2, 1024), "bfloat16")}, (5, 70), NORM_OFF_ROWS, {"q_out": repr(view(71, (2, 1024), "bfloat16"))}),
    ({"kv_ptr": view(73, (2, 256), "bfloat16")}, (5, 70), NORM_OFF_ROWS, {"kv": repr(view(73, (2, 256), "bfloat16")), "step_rows": 2}),
    ({"kpe_ptr": view(75, (2, 64))}, (5, 70), "fused norm/RoPE is not entirely bf16", {}),
    ({"q_rms_norm_w_ptr": view(72, (1024,), "bfloat16")}, (5, 70), NORM_OFF_SHAPE, {"qw": repr(view(72, (1024,), "bfloat16"))}),
    ({"kv_rms_norm_w_ptr": view(74, (256,), "bfloat16")}, (5, 70), NORM_OFF_SHAPE, {"kvw": repr(view(74, (256,), "bfloat16"))}),
    ({"kpe_rope_cos_sin_cache_ptr": view(76, (4096, 64, 1), "bfloat16")}, (5, 70), NORM_OFF_SHAPE,
     {"cs": repr(view(76, (4096, 64, 1), "bfloat16"))}),
    ({"kpe_rope_cos_sin_cache_ptr": view(76, (4096, 32), "bfloat16")}, (5, 70), NORM_OFF_SHAPE,
     {"cs": repr(view(76, (4096, 32), "bfloat16"))}),
    ({"kpe_rope_cos_sin_cache_ptr": view(76, (4096, 64), "bfloat16", strides=(128, 1))}, (5, 70), NORM_OFF_SHAPE,
     {"cs": repr(view(76, (4096, 64), "bfloat16", strides=(128, 1)))}),
    ({"kpe_rope_cos_sin_cache_ptr": view(76, (8, 64), "bfloat16")}, (5, 70), "a position is outside the GPT-J rotary table",
     {"positions": (5, 9), "table_rows": 8}),
    ({"mla_cache_ptr": NARROW_LATENT}, (5, 70), f"cache is not [blocks, block, physical-width] row-major: {NARROW_LATENT!r}", {}),
    ({}, (5,), "StepStart slot_mapping does not cover the kernel rows", {"rows": 2, "slots": 1}),
    ({}, (5, 128), "a current-token slot is outside the cache", {"slots": (5, 128), "cache_slots": 128}),
    ({"q_rms_eps": None}, (5, 70), "norm epsilons are not static floats", {"q_eps": "None", "kv_eps": "1e-05"}),
    ({"kv_rms_eps": "small"}, (5, 70), "norm epsilons are not static floats", {"q_eps": "1e-06", "kv_eps": "'small'"}),
    ({"q_rms_eps": "1e-6"}, (5, 70), "norm epsilons are not static floats", {"q_eps": "'1e-6'", "kv_eps": "1e-05"}),
    ({"kv_rms_eps": True}, (5, 70), "norm epsilons are not static floats", {"q_eps": "1e-06", "kv_eps": "True"}),
    ({"index_k_layer_norm_bias_ptr": None}, (5, 70), "indexer-layer norm/RoPE pointers are not bound", {}),
    ({"consts": {"INDEX_K_DIM": 64}}, (5, 70), INDEX_OFF_PIN, {}),
    ({"consts": {"INDEX_K_BLOCK_SIZE": 64}}, (5, 70), INDEX_OFF_PIN, {}),
    ({"consts": {"INDEX_K_HALF_ROT_DIM": 16}}, (5, 70), INDEX_OFF_PIN, {}),
    ({"index_k_ptr": view(78, (2, 64), "bfloat16")}, (5, 70), INDEX_OFF_SHAPE, {"index_k": repr(view(78, (2, 64), "bfloat16"))}),
    ({"index_k_layer_norm_w_ptr": view(79, (64,))}, (5, 70), INDEX_OFF_SHAPE, {"weight": repr(view(79, (64,)))}),
    ({"index_k_layer_norm_bias_ptr": view(80, (64,))}, (5, 70), INDEX_OFF_SHAPE, {"bias": repr(view(80, (64,)))}),
    ({"index_k_rope_cos_sin_cache_ptr": view(76, (4096, 64, 1), "bfloat16")}, (5, 70), INDEX_OFF_SHAPE,
     {"cs": repr(view(76, (4096, 64, 1), "bfloat16"))}),
    ({"index_k_rope_cos_sin_cache_ptr": view(76, (4096, 32), "bfloat16")}, (5, 70), INDEX_OFF_SHAPE,
     {"cs": repr(view(76, (4096, 32), "bfloat16"))}),
    ({"indexer_cache_stride": 136}, (5, 70), OFF_STRIDE, {}),
    ({"indexer_cache_ptr": SHORT_INDEX, "indexer_cache_scale_ptr": view(81, (1, 64, 33))}, (5, 70),
     "a current-token slot is outside the index-K cache", {"slots": (5, 70), "cache_slots": 64}),
    *[({"index_k_layer_norm_eps": v}, (5, 70), "index-K layer-norm epsilon is not a static float", {"eps": repr(v)}) for v in OFF_EPS],
    *[({"topk_indices_ptr": v}, (5, 70), "topk_indices_ptr is not a contiguous i32 [capacity,2048] view", {"topk": repr(v)}) for v in OFF_TOPKS],
    ({"topk_indices_ptr": view(99, (16, 2048), "int32")}, (5, 70), "IndexShare layer does not use the indexer layers' top-k allocation",
     {"observed": (99, 0, 2048, (2048, 1)), "established": (TOPK_ALLOC, 0, 2048, (2048, 1))}),
], ids=["warps", "constexpr", "pointer", "positions", "rows", "q-width", "q-out-width", "kv-width", "dtype", "q-weight", "kv-weight",
        "table-rank", "table-width", "table-strides", "position-range", "latent-cache", "slots-short", "slot-range", "eps-absent", "eps-text",
        "eps-number-as-text", "eps-bool",
        "index-pointer", "index-dim", "index-block", "index-rot", "index-k-width", "index-weight", "index-bias", "index-table-rank",
        "index-table-width", "index-cache", "index-slot", "index-eps-absent", "index-eps-text", "index-eps-number-as-text", "index-eps-bool",
        "index-eps-list", "topk-layout", "topk-dtype", "topk-width", "topk-strides", "topk-allocation"])
def test_a_fused_norm_rope_off_its_pins_its_views_or_the_step_is_refused_with_what_differs(kw, slots, reason, detail):
    off = D.DsaNormRope().match(_norm(**kw), _ctx(TWO, slots=slots))
    assert isinstance(off, Unsupported) and off.reason == reason and off.consumed == (31,)
    assert {k: off.detail[k] for k in detail} == detail


def test_an_epsilon_recorded_as_an_int_is_read_as_its_float():
    r = D.DsaNormRope().match(_norm(q_rms_eps=1, kv_rms_eps=2, index_k_layer_norm_eps=3), _ctx(TWO, slots=[5, 70]))
    assert isinstance(r, Group) and [spec_label(p.spec) for p in r.parts] == [
        "DsaRMSNorm_v1{N=2048,EPS=1.0}", "DsaRMSNorm_v1{N=512,EPS=2.0}", "RoPEHeadGptJ_v2{D=64}", "DsaIndexerKRow_v1{D=128,ROPE=64,EPS=3.0}"]


def test_a_fused_norm_rope_s_refusal_names_its_pointer_arguments_and_the_first_eight_positions_or_slots():
    unbound = D.DsaNormRope().match(_norm(kv_ptr=None), _ctx(TWO, slots=[5, 70]))
    assert isinstance(unbound, Unsupported) and len(unbound.detail["args"]) == 16 and all(k.endswith("_ptr") for k in unbound.detail["args"])
    assert (unbound.detail["args"]["kv_ptr"], unbound.detail["args"]["q_c_ptr"]) == ("None", repr(view(70, (2, 2048), "bfloat16")))
    nine = [("a", range(9))]
    table = D.DsaNormRope().match(_norm(rows=9, kpe_rope_cos_sin_cache_ptr=view(76, (4, 64), "bfloat16")), _ctx(nine))
    assert isinstance(table, Unsupported) and (table.detail["positions"], table.detail["table_rows"]) == (tuple(range(8)), 4)
    latent = D.DsaNormRope().match(_norm(rows=9), _ctx(nine, slots=range(128, 137)))
    assert isinstance(latent, Unsupported) and (latent.detail["slots"], latent.detail["cache_slots"]) == (tuple(range(128, 136)), 128)
    short = {"indexer_cache_ptr": SHORT_INDEX, "indexer_cache_scale_ptr": view(81, (1, 64, 33))}
    index = D.DsaNormRope().match(_norm(rows=9, **short), _ctx(nine, slots=range(64, 73)))
    assert isinstance(index, Unsupported) and (index.detail["slots"], index.detail["cache_slots"]) == (tuple(range(64, 72)), 64)


Q_PINS = {"NUM_Q_HEADS": 8, "Q_PE_HALF_ROT_DIM": 32, "QL_NOPE_DIM": 512, "QL_NOPE_BLOCK": 512, "INDEX_ROPE_INTERLEAVE": True,
          "QUANTIZE_MQA": False}
Q_OFF_PIN = "fused-q source or launch geometry differs from the recorded GLM-5.3 pin"
# (argument, its name in the refusal, the view): each differs from the launch's in one term of the refusal's condition alone
OFF_Q_VIEWS = [("q_pe_ptr", "q", view(60, (2, 8, 32), "bfloat16")), ("q_pe_out_ptr", "q_out", view(61, (2, 8, 32), "bfloat16")),
               ("q_pe_ptr", "q", view(60, (2, 8, 64))), ("q_pe_out_ptr", "q_out", view(61, (2, 8, 64))),
               ("q_pe_cos_sin_ptr", "cs", view(62, (4096, 64))), ("q_pe_cos_sin_ptr", "cs", view(62, (4096, 64, 1), "bfloat16")),
               ("q_pe_cos_sin_ptr", "cs", view(62, (4096, 32), "bfloat16"))]
OFF_INDEX_Q = [("index_q_ptr", "q", view(65, (2, 32, 64), "bfloat16")), ("index_weights_ptr", "weights", view(67, (2, 16), "bfloat16")),
               ("index_q_fp8_ptr", "q_out", view(68, (2, 32, 64), "float8_e4m3fn")), ("index_weights_out_ptr", "weights_out", view(69, (2, 16)))]


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
    D._state(ctx)["modules"][LAYER]["mqa_launches"] = None
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
    assert (index_q.consumed, index_q.hint, index_q.roots) == ((), f"{LAYER}/indexer_q", (None, None, None))
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


def test_an_index_q_scale_in_the_last_bit_of_its_power_is_taken_and_one_a_part_in_1e14_off_is_refused():
    """The launch's two scales are floats vLLM computes; the fold takes one within 1e-15 of D^-0.5 and H^-0.5, relative."""
    ctx = _ctx([("a", [5]), ("b", [9])])
    near = _q(index_weights_softmax_scale=math.nextafter(128 ** -0.5, 1), index_weights_head_scale=math.nextafter(32 ** -0.5, 0))
    assert isinstance(D.DsaQ().match(near, ctx), Group)
    for name, scale in (("index_weights_softmax_scale", 128 ** -0.5), ("index_weights_head_scale", 32 ** -0.5)):
        far = D.DsaQ().match(_q(**{name: scale * (1 + 1e-14)}), ctx)
        assert isinstance(far, Unsupported) and far.reason == "index-Q scales differ from D^-0.5 and H^-0.5", name


# a scale that is no number: absent, a word, its pinned value as text, a list
OFF_SCALES = [("index_weights_softmax_scale", "softmax_scale", None), ("index_weights_head_scale", "head_scale", None),
              ("index_weights_softmax_scale", "softmax_scale", "small"), ("index_weights_softmax_scale", "softmax_scale", repr(128 ** -0.5)),
              ("index_weights_head_scale", "head_scale", [0.5])]


@pytest.mark.parametrize("kw,reason,detail", [
    ({"warps": 8}, Q_OFF_PIN, {"src_sha": D.Q_SRC_SHA, "num_warps": 8, "num_stages": 3}),
    ({"consts": {"QUANTIZE_MQA": True}}, "fused-q constexprs differ from the recorded GLM-5.3 pin", {"mismatches": {"QUANTIZE_MQA": (True, False)}}),
    ({"q_pe_out_ptr": None}, "fused-q RoPE pointers are not bound", {}),
    ({"pos_ptr": 3}, "positions argument is not a tensor view", {"positions": "3"}),
    *[({arg: v}, "fused-q RoPE views differ from bf16 [M,8,64]", {name: repr(v)}) for arg, name, v in OFF_Q_VIEWS],
    ({"q_pe_ptr": view(60, (2, 8, 64), "bfloat16", strides=(1024, 64, 1))}, "fused-q RoPE views do not flatten to [M*8, 64] head rows",
     {"q": repr(view(60, (2, 8, 64), "bfloat16", strides=(1024, 64, 1)))}),
    ({"index_weights_out_ptr": None}, "indexer-layer fused-q pointers are not bound", {}),
    ({"consts": {"INDEX_Q_HEAD_DIM": 64}}, "index-Q dimensions differ from the recorded GLM-5.3 pin", {}),
    *[({arg: v}, "index-Q views have the wrong shape", {name: repr(v)}) for arg, name, v in OFF_INDEX_Q],
    ({"index_weights_head_scale": 0.5}, "index-Q scales differ from D^-0.5 and H^-0.5", {"softmax_scale": 128 ** -0.5, "head_scale": 0.5}),
    *[({name: v}, "index-Q scales differ from D^-0.5 and H^-0.5", {key: v}) for name, key, v in OFF_SCALES],
], ids=["warps", "constexpr", "rope-pointer", "positions", "q-shape", "q-out-shape", "q-dtype", "q-out-dtype", "table-dtype", "table-rank",
        "rope-views", "head-rows", "index-pointer", "index-dims", "index-q-shape", "index-shapes", "index-q-out-shape", "index-weights-out-shape",
        "scales", "scale-absent", "head-scale-absent", "scale-text", "scale-number-as-text", "head-scale-list"])
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


SHARE_NAMES = {"req_id_ptr": "req", "block_table_ptr": "block_table", "token_indices_ptr": "source", "out_ptr": "out", "valid_count_ptr": "valid"}


@pytest.mark.parametrize("pointers", [
    {"req_id_ptr": view(50, (3,), "int32")}, {"req_id_ptr": view(50, (2,), "int64")},
    {"block_table_ptr": view(51, (2, 4), "int64")}, {"block_table_ptr": view(51, (8,), "int32")},
    {"out_ptr": view(52, (2, 1024), "int32")},
    {"token_indices_ptr": view(TOPK_ALLOC, (2, 1024), "int32"), "out_ptr": view(52, (2, 1024), "int32")},
    {"out_ptr": view(52, (2, 2048), "int64")},
    {"token_indices_ptr": view(TOPK_ALLOC, (2, 2048), "int64"), "out_ptr": view(52, (2, 2048), "int64")},
    {"valid_count_ptr": view(53, (3,), "int32")}, {"valid_count_ptr": view(53, (2,), "int64")},
    {"token_indices_ptr": view(TOPK_ALLOC, (2, 2048), "int32", strides=(4096, 1))}, {"out_ptr": view(52, (2, 2048), "int32", strides=(4096, 1))},
], ids=["req-shape", "req-dtype", "table-dtype", "table-rank", "out-shape", "source-shape", "out-dtype", "source-dtype", "valid-shape",
        "valid-dtype", "source-strides", "out-strides"])
def test_an_index_share_copy_off_one_term_of_its_pinned_layouts_is_refused_with_its_views(pointers):
    """Each case is off one term of the refusal's condition and no other (a source of another shape or dtype brings an output of
    the same, since the two are held to each other first)."""
    ctx = _ctx(TWO)
    off = D.DsaIndexShare().match(_share(**pointers), ctx)
    assert isinstance(off, Unsupported) and off.reason == "IndexShare conversion views differ from the pinned i32 layouts"
    assert {SHARE_NAMES[arg]: off.detail[SHARE_NAMES[arg]] for arg in pointers} == {SHARE_NAMES[arg]: repr(v) for arg, v in pointers.items()}
    assert "page_table" not in D._state(ctx)["modules"][LAYER]


def test_an_index_share_copy_of_the_shared_top_k_rows_is_staging_and_records_its_page_table():
    ctx = _ctx([("a", [5]), ("b", [9])])
    assert D.DsaIndexShare().match(_share(fn="pkg.another.kernel"), ctx) is None, "another kernel is not this pattern's"
    r = D.DsaIndexShare().match(_share(), ctx)
    assert isinstance(r, Transparent) and (r.effect, r.consumed, r.kernels) == ("noop", (41,), ("ConvertReqIndexToGlobalIndexKernel",))
    assert r.hint == f"{LAYER}/IndexShare top-k staging" and r.pattern == "dsa.index_share_copy"
    assert D._state(ctx)["modules"][LAYER]["page_table"] == (52, 0, 2048, (2048, 1))


MLA = "_vllm_fa3_C::fwd(Tensor q, Tensor k, Tensor v, Tensor? q_v) -> Tensor[]"


def _mla(rows=2, heads=8, out_width=512, dtype="bfloat16", q_v=True, outs=True, op="_vllm_fa3_C.fwd.default", k=None, v=None, **kwargs):
    """An FA3 sparse MLA call: `q` [rows, heads, 64], `q_v` and the output [rows, heads, 512], `k` and `v` of the paged cache; `kwargs`
    are its keyword arguments."""
    q, qv, out = view(20, (rows, heads, 64), dtype), view(21, (rows, heads, 512), "bfloat16"), view(22, (rows, heads, out_width), "bfloat16")
    k, v = view(23, (4, 64, 1, 64), "bfloat16") if k is None else k, view(24, (4, 64, 1, 512), "bfloat16") if v is None else v
    return _node(40, op, MLA, [q, k, v, qv if q_v else None], outs=[out] if outs else None, kwargs=kwargs), q, qv, out


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


MLA_DEFAULT = {"max_seqlen_q": 1, "max_seqlen_k": 2048, "softmax_scale": 0.0625, "is_causal": True, "window_size_left": -1,
               "window_size_right": -1, "softcap": 0.0, "is_rotary_interleaved": True, "sm_margin": 0, "cp_world_size": 1, "cp_rank": 0,
               "num_splits": 0, "page_table": view(52, (2, 2048), "int32")}
LATENT = view(77, (128, 576), "bfloat16")


def _halves(alloc=77, slots=128):
    """`k` and `v` of an FA3 call as the two halves of a latent cache's 576-word slot rows: `v` the first 512 words, `k` the 64 after."""
    return {"k": view(alloc, (slots, 1, 1, 64), "bfloat16", offset=512, strides=(576, 576, 64, 1)),
            "v": view(alloc, (slots, 1, 1, 512), "bfloat16", strides=(576, 576, 512, 1))}


def _before_its_mla(requests=TWO, slots=(5, 70), sms=132, **norm):
    """The step of `requests` (`TWO`'s when not given) on a record target of `sms` SMs, once its layer has folded the fused norm/RoPE
    launch (`norm` its arguments), which names the latent cache, and the request-index copy, which names the page table."""
    ctx = _ctx(requests, slots=slots)
    ctx.header = dataclasses.replace(ctx.header, profile={"gpu": {"multi_processor_count": sms}})
    assert isinstance(D.DsaNormRope().match(_norm(rows=len(slots), **norm), ctx), Group)
    assert isinstance(D.DsaIndexShare().match(_share(rows=len(slots)), ctx), Transparent)
    return ctx


def test_a_sparse_mla_on_the_default_path_is_one_split_attention_a_row_over_its_request_s_latent_rows():
    node, q, qv, out = _mla(**MLA_DEFAULT, **_halves())
    r = D.DsaSparseMla().match(node, _before_its_mla())
    assert isinstance(r, Group) and (r.consumed, r.hint, r.pattern) == ((40,), LAYER, "dsa.fa3_sparse_mla_split")
    assert [spec_label(p.spec) for p in r.parts] == [
        f"SparseMlaSplit_v1{{T={n},NS=1,NH=8,DR=64,DV=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,INV=Fa3InvSum_v1,SCALE=0.0625}}" for n in (6, 10)]
    for i, (part, (req, pos), slots) in enumerate(zip(r.parts, [("a", 5), ("b", 9)], [tuple(range(6)), tuple(range(64, 74))])):
        assert isinstance(part, Compute) and (part.rows, part.attribution, part.pattern) == (1, ((req, pos),), "dsa.fa3_sparse_mla_split")
        assert part.operands == (PerRow(view(21, (1, 8, 512), "bfloat16", offset=i * 4096)),
                                 PerRow(view(20, (1, 8, 64), "bfloat16", offset=i * 512)), GatherMany(LATENT, (slots,)))
        assert part.outputs == (PerRow(view(22, (1, 8, 512), "bfloat16", offset=i * 4096)),)
        assert (part.consumed, part.hint, part.kernels) == ((40,) if i == 0 else (), f"{LAYER}/sparse_mla[t={pos},splits=1]", ("FlashAttnFwdSm90",))


def test_a_sparse_mla_of_a_long_row_is_split_and_its_splits_are_combined():
    ctx = _before_its_mla([("a", [1999])], [1999], indexer=False, mla_cache_ptr=view(77, (32, 64, 576), "bfloat16"))
    r = D.DsaSparseMla().match(_mla(rows=1, **MLA_DEFAULT, **_halves(slots=2048))[0], ctx)
    assert isinstance(r, Group) and len(r.parts) == 1
    part = r.parts[0]
    assert spec_label(part.spec) == (
        "SparseMlaSplit_v1{T=2000,NS=28,NH=8,DR=64,DV=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,INV=Fa3InvSum_v1,SCALE=0.0625}")
    assert (part.hint, part.kernels) == (f"{LAYER}/sparse_mla[t=1999,splits=28]", ("FlashAttnFwdSm90", "FlashAttnFwdCombine"))
    assert part.operands[2] == GatherMany(view(77, (2048, 576), "bfloat16"), (tuple(range(2000)),))


@pytest.mark.parametrize("kw,reason,detail", [
    ({"cu_seqlens_k": 7}, "FA3 sparse MLA has unsupported argument cu_seqlens_k", {"value": "7"}),
    ({"k": view(23, (128, 1, 1, 64), "bfloat16")}, "FA3 k/v are not adjacent views of 576-word latent-cache rows",
     {"k": repr(view(23, (128, 1, 1, 64), "bfloat16")), "v": repr(_halves()["v"])}),
    (_halves(alloc=88), "FA3 does not read the latent cache fused_norm_rope wrote",
     {"observed": repr(view(88, (128, 576), "bfloat16")), "established": repr(LATENT)}),
    ({"page_table": view(53, (2, 2048), "int32")}, "FA3 page table is not the shared top-k allocation",
     {"page_table": repr(view(53, (2, 2048), "int32")), "established": (52, 0, 2048, (2048, 1))}),
    ({"page_table": view(52, (2048,), "int32")}, "FA3 page table is not the shared top-k allocation",
     {"page_table": repr(view(52, (2048,), "int32")), "established": (52, 0, 2048, (2048, 1))}),
    ({"page_table": None}, "FA3 page table is not the shared top-k allocation", {"page_table": "None", "established": (52, 0, 2048, (2048, 1))}),
    ({"num_splits": -1}, "FA3 num_splits is not a nonnegative integer", {"num_splits": -1}),
    (_halves(slots=64), "a selected latent-cache slot is outside the cache", {"T": 10, "cache_slots": 64}),
], ids=["argument", "halves", "latent-cache", "page-table", "page-table-rank", "page-table-absent", "splits", "slot-range"])
def test_a_sparse_mla_on_the_default_path_off_its_layer_s_caches_is_refused_with_what_differs(kw, reason, detail):
    off = D.DsaSparseMla().match(_mla(**{**MLA_DEFAULT, **_halves(), **kw})[0], _before_its_mla())
    assert isinstance(off, Unsupported) and off.reason == reason and off.consumed == (40,)
    assert {k: off.detail[k] for k in detail} == detail


@pytest.mark.parametrize("name", ["k_new", "v_new", "out", "scheduler_metadata", "pack_gqa", "s_aux", "cp_tot_seqused_k"])
def test_a_sparse_mla_that_carries_an_argument_the_default_path_leaves_out_is_refused_with_it(name):
    off = D.DsaSparseMla().match(_mla(**MLA_DEFAULT, **_halves(), **{name: 7})[0], _before_its_mla())
    assert isinstance(off, Unsupported) and off.reason == "FA3 sparse MLA arguments differ from the GLM-5.3 default path"
    assert off.detail["mismatches"] == {name: (7, None)}


@pytest.mark.parametrize("name", ["cu_seqlens_k", "cu_seqlens_k_new", "seqused_q", "kv_batch_idx", "leftpad_k", "rotary_cos", "rotary_sin",
                                  "seqlens_rotary", "q_descale", "k_descale", "v_descale"])
def test_a_sparse_mla_that_carries_an_argument_the_fold_does_not_read_is_refused_with_its_name_and_value(name):
    off = D.DsaSparseMla().match(_mla(**MLA_DEFAULT, **_halves(), **{name: 7})[0], _before_its_mla())
    assert isinstance(off, Unsupported) and off.reason == f"FA3 sparse MLA has unsupported argument {name}" and off.detail["value"] == "7"


def test_a_sparse_mla_of_a_row_one_key_past_a_block_is_two_splits_and_is_combined():
    ctx = _before_its_mla([("a", [64])], [64])
    r = D.DsaSparseMla().match(_mla(rows=1, **MLA_DEFAULT, **_halves())[0], ctx)
    assert isinstance(r, Group) and len(r.parts) == 1
    assert spec_label(r.parts[0].spec) == (
        "SparseMlaSplit_v1{T=65,NS=2,NH=8,DR=64,DV=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,INV=Fa3InvSum_v1,SCALE=0.0625}")
    assert (r.parts[0].hint, r.parts[0].kernels) == (f"{LAYER}/sparse_mla[t=64,splits=2]", ("FlashAttnFwdSm90", "FlashAttnFwdCombine"))


@pytest.mark.parametrize("half,off", [
    ("k", view(78, (128, 1, 1, 64), "bfloat16", offset=512, strides=(576, 576, 64, 1))),
    ("k", view(77, (128, 1, 1, 64), "bfloat16", strides=(576, 576, 64, 1))),
    ("k", view(77, (64, 1, 1, 64), "bfloat16", offset=512, strides=(576, 576, 64, 1))),
    ("k", view(77, (128, 1, 1, 64), "bfloat16", offset=512, strides=(1152, 576, 64, 1))),
    ("v", view(77, (128, 1, 1, 512), "bfloat16", strides=(1152, 576, 512, 1))),
], ids=["allocation", "offset", "slots", "k-row-stride", "v-row-stride"])
def test_a_sparse_mla_whose_k_or_v_is_off_one_term_of_a_latent_cache_s_halves_is_refused_with_both(half, off):
    kv = {**_halves(), half: off}
    r = D.DsaSparseMla().match(_mla(**MLA_DEFAULT, **kv)[0], _before_its_mla())
    assert isinstance(r, Unsupported) and r.reason == "FA3 k/v are not adjacent views of 576-word latent-cache rows"
    assert (r.detail["k"], r.detail["v"]) == (repr(kv["k"]), repr(kv["v"]))


def test_a_sparse_mla_is_held_to_its_layer_s_index_copy_the_record_target_s_132_sms_and_the_step_s_block_tables():
    node = _mla(**MLA_DEFAULT, **_halves())[0]
    uncopied = D.DsaSparseMla().match(node, _ctx(TWO))
    assert isinstance(uncopied, Unsupported) and uncopied.reason == "FA3 page table is not the shared top-k allocation"
    assert uncopied.detail["established"] is None
    sms = D.DsaSparseMla().match(node, _before_its_mla(sms=80))
    assert isinstance(sms, Unsupported) and sms.reason == "GLM-5.3 sparse split heuristic is pinned to the record target's 132-SM launch"
    assert sms.detail["num_sms"] == 80
    ctx = _before_its_mla()
    ctx.start = dataclasses.replace(ctx.start, block_tables={"a": [0]})
    untabled = D.DsaSparseMla().match(node, ctx)
    assert isinstance(untabled, Unsupported) and untabled.reason == "no block table for request 'b'"
    assert (untabled.detail["request"], untabled.detail["T"]) == ("b", 10)


def test_an_allocation_that_is_no_logits_buffer_is_left_and_one_the_indexer_cannot_fill_is_refused():
    ctx = _ctx([("a", [19])])
    assert D.DsaMqaAllocation().match(_empty(3, (12,), 40), ctx) is None, "a rank-1 allocation is not the indexer's"
    narrow = D.DsaMqaAllocation().match(_empty(4, (1, 16), 41), ctx)
    assert isinstance(narrow, Unsupported) and narrow.reason == "MQA row or cache does not cover its valid keys"
    assert {k: narrow.detail[k] for k in ("T", "first_key", "output_width", "cache_slots")} == {
        "T": 20, "first_key": 0, "output_width": 16, "cache_slots": 64}
    ctx = _ctx([("a", [19])])
    ctx.start = dataclasses.replace(ctx.start, block_tables={})
    untabled = D.DsaMqaAllocation().match(_empty(3, (1, 512), 40), ctx)
    assert isinstance(untabled, Unsupported) and untabled.reason == "no block table for request 'a'"
    assert (untabled.detail["request"], untabled.detail["T"]) == ("a", 20)
    ctx = _ctx([("a", [19])])
    del D._state(ctx)["modules"][LAYER]["index_q"]
    unfused = D.DsaMqaAllocation().match(_empty(3, (1, 512), 40), ctx)
    assert isinstance(unfused, Unsupported) and unfused.reason == "invisible MQA allocation lacks the fused-q or index-cache views"
    assert unfused.detail["module"] == LAYER


def test_a_gather_is_staging_and_the_rows_of_its_destination_are_the_keys_it_stages():
    ctx = _ctx([("b", range(11))])
    r = D.DsaMqaGather().match(_gather(2, 11), ctx)
    assert isinstance(r, Transparent) and (r.effect, r.consumed, r.hint) == ("noop", (2,), f"{LAYER}/indexer K-cache staging")
    assert (r.kernels, r.pattern) == ((D.MQA_STAGE,), "dsa.mqa_cache_gather") and D._state(ctx)["modules"][LAYER]["mqa_keys"] == 11
    unnamed = _node(4, "_C_cache_ops.cp_gather_indexer_k_quant_cache.default", "",
                    [view(13, (8, 64, 132), "uint8"), view(14, (7, 128), "float8_e4m3fn")])
    assert isinstance(D.DsaMqaGather().match(unnamed, ctx), Transparent), "a call recorded with no schema names its destination by position"
    assert D._state(ctx)["modules"][LAYER]["mqa_keys"] == 7


def test_a_gather_without_a_keys_destination_is_refused_and_stages_no_keys():
    ctx = _ctx([("b", range(11))])
    node = _node(2, "_C_cache_ops.cp_gather_indexer_k_quant_cache.default", GATHER, [view(13, (8, 64, 132), "uint8"), None])
    off = D.DsaMqaGather().match(node, ctx)
    assert isinstance(off, Unsupported) and off.reason == "indexer K-cache gather destination is not a [keys, D] view"
    assert off.detail["dst_k"] == "None" and "mqa_keys" not in D._state(ctx)["modules"][LAYER]


def _decode_topk(k=2048, out=None):
    """The decodes' top-k call of width `k`, writing `out` (the top-k allocation's first row when not given)."""
    return _node(7, "_C.cooperative_topk.default", DECODE_TOPK,
                 [view(31018, (1, 288), strides=(512, 1)), view(3, (1, 1), "int32"), view(TOPK_ALLOC, (1, 2048), "int32") if out is None else out,
                  view(4, (1048576,), "uint8"), k, 20])


def test_a_top_k_of_another_width_or_of_a_row_past_the_short_row_is_refused():
    narrow = D.DsaTopK().match(_decode_topk(1024), _ctx([("a", [19])]))
    assert isinstance(narrow, Unsupported) and narrow.reason == "top-k output is not contiguous i32 [rows,2048]"
    assert (narrow.detail["output"], narrow.detail["topk"]) == (repr(view(TOPK_ALLOC, (1, 2048), "int32")), 1024)
    long = D.DsaTopK().match(_decode_topk(), _ctx([("a", [2048])]))
    assert isinstance(long, Unsupported) and long.reason == "DSA top-k row exceeds the short-row shortcut"
    assert {k: long.detail[k] for k in ("position", "keys", "TOPK")} == {"position": 2048, "keys": 2049, "TOPK": 2048}


@pytest.mark.parametrize("out", [
    view(TOPK_ALLOC, (1, 2048), "int64"), view(TOPK_ALLOC, (1, 2048, 1), "int32"), view(TOPK_ALLOC, (1, 1024), "int32"),
    view(TOPK_ALLOC, (2, 2048), "int32", strides=(4096, 1)),
], ids=["dtype", "rank", "width", "strides"])
def test_a_top_k_whose_output_is_off_one_term_of_its_layout_is_refused_with_it(out):
    off = D.DsaTopK().match(_decode_topk(out=out), _ctx([("a", [19]), ("b", [20])]))
    assert isinstance(off, Unsupported) and off.reason == "top-k output is not contiguous i32 [rows,2048]"
    assert (off.detail["output"], off.detail["topk"]) == (repr(out), 2048)


def test_another_call_is_none_of_the_dispatch_patterns_whatever_its_output_or_its_library():
    ctx = _ctx([("a", [19])])
    zeros = _node(3, "aten.zeros.default", "aten::zeros(SymInt[] size) -> Tensor", [[1, 512]], outs=view(40, (1, 512)))
    for pattern in (D.DsaMqaAllocation(), D.DsaMqaGather(), D.DsaTopK(), D.DsaSparseMla()):
        assert pattern.match(zeros, ctx) is None, f"{pattern.name}: an f32 [1, 512] from another op is not its call"
    assert D.DsaMqaAllocation().match(_empty(3, (1, 512), 40, dtype="int32"), ctx) is None, "an i32 allocation is not the indexer's logits"
    other = _mla(op="_vllm_fa3_C.fwd_combine.default", **MLA_DEFAULT, **_halves())[0]
    assert D.DsaSparseMla().match(other, _before_its_mla()) is None, "another op of the FA3 library is not this pattern's"
    assert D._state(ctx)["modules"][LAYER]["mqa_launches"] == [[0], []], "and none of them takes the step's decode launch"


def test_a_row_outside_a_view_is_no_row_of_it():
    assert D._one_row(view(1, (2, 4)), 1) == view(1, (1, 4), offset=4)
    assert D._one_row(view(1, (4,)), 3) == view(1, (1,), offset=3), "a vector's row is one element of it"
    with pytest.raises(GeometryError, match="row 2 outside"):
        D._one_row(view(1, (2, 4)), 2)
    with pytest.raises(GeometryError, match="row 0 outside"):
        D._one_row(view(1, ()), 0)
