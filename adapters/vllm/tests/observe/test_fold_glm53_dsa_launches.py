"""GLM-5.3's DSA indexer launches in a multi-request step, folded as vLLM d9105ea80 issues them on cc 9.0 (`sparse_attn_indexer`):

* the leading one-row requests are decodes and every request after them a prefill (`split_decodes_and_prefills`, threshold 1);
* each prefill chunk gathers its requests' keys, then DeepGEMM fills an invisible ``[align(rows, 4), width]`` f32 allocation whose row of
  request j holds its logits from column ``cu_j``; the decodes' paged logits ``[rows, width]`` come after, from column 0;
* each top-k writes the run of the top-k allocation's rows its launch covers;
* every row is one `DsaMqaLogits_v3{N=T}` and one `DsaTopKShortRow_v1`, the Build's per row; a launch that is not the next one is refused.
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import pytest

from verity_vllm.observe.fold.profiles import profile_for_case
from verity_vllm.observe.events import DispatchEnter, DispatchExit, RunHeader, StepStart, ViewDesc
from verity_vllm.observe.fold.patterns import dsa as D
from verity_vllm.observe.fold.resolver import Group, StepContext, Transparent, Unsupported, spec_label
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
