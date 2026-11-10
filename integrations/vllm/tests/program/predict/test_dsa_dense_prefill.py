"""verity_vllm.program.predict.dsa's prefill route (vLLM's `use_dense_mha = prefill_max_seq_len <= index_topk`): a step's prefills of at
most INDEX_TOPK keys by dense MHA (kv_b_proj and FA3's prefill heads, and where some prefill of the step has context, each head's
lse, the context's partial and MergeAttnStates per token), and a step whose longest prefill reaches past INDEX_TOPK by the decode
kernels, the decode tokens of a dense step without the prefills' rows.

GLM-5.3's checkpoint is cut to its first 4 layers (three full indexers, then one that shares the third's top-k) and its
INDEX_TOPK raised to 16, and vLLM's decode
threshold at the rank's 8 query heads lowered to 2, so prefills of a few rows take either route."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from verity.definitions.registry.sparse_mla_split import launch_num_splits
from verity_vllm.program.frontend.rules import vllm_dsa
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.common import Unexpressible, _Launch
from verity_vllm.program.predict.phi import Phi
from tests.program.predict.test_dsa_long_rows import _canonical
from tests.program.predict.test_node import _first_layers
from tests.program.predict.test_predict import GLM53, _row_phi

TOPK, NH, LAYERS, INDEXERS = 16, 8, 4, 3


def _topk(phi: Phi, topk: int) -> Phi:
    return Phi.from_json({**phi.to_json(), "model": {**phi.model, "hf_config": {**phi.hf_config, "index_topk": topk}}})


@pytest.fixture
def phi(tmp_path, monkeypatch) -> Phi:
    monkeypatch.setitem(vllm_dsa.DECODE_THRESHOLD, NH, 2)
    p = _first_layers(_row_phi(GLM53, "GLM53", tmp_path), LAYERS)
    assert p.hf_config["num_attention_heads"] // 8 == NH and p.hf_config["indexer_types"] == ["full"] * INDEXERS + ["shared"]
    return _topk(p, TOPK)


def _calls(r, family: str) -> list:
    return [n for n in r.program.fn.body.nodes if n.fn.id.startswith(family + "{")]


def _families(r) -> Counter:
    return Counter(n.fn.id.split("{", 1)[0] for n in r.program.fn.body.nodes)


def test_a_steps_prefills_are_counted_as_the_metadata_builder_counts_them():
    """Every sequence from the first of more than the threshold's rows, in the reordered batch: the extends past the threshold,
    then every first chunk. Without such an extend, the first chunks past the threshold; a short first chunk beside a long one is
    refused, as the persistent batch's order decides it."""
    alone = _Launch()
    assert (alone.prefills((3, 0), 2), alone.prefills((3, 5), 2), alone.prefills((2, 0), 2)) == ([(3, 0)], [(3, 5)], [])
    mixed = _Launch([[[1, 5], [300, 10], [4, 0], [300, 0], [2, 7]]])
    assert mixed.prefills((1, 5), 256) == [(300, 10), (4, 0), (300, 0)]
    assert _Launch([[[1, 5], [300, 0], [301, 0]]]).prefills((1, 5), 256) == [(300, 0), (301, 0)]
    assert _Launch([[[1, 5], [2, 7]]]).prefills((2, 7), 256) == []
    assert mixed.keys([6], without=[(300, 10), (4, 0), (300, 0)]) == [6, 8, 9]
    with pytest.raises(Unexpressible, match=re.escape("schedules first chunks of [4] rows beside ones of [300] rows")):
        _Launch([[[4, 0], [300, 0]]]).prefills((300, 0), 256)
    with pytest.raises(Unexpressible, match=re.escape("do not hold this request's (3, 0) (rows, start)")):
        mixed.prefills((3, 0), 256)


def test_a_prefill_without_context_is_fa3s_causal_prefill_per_head(phi):
    """A step of one 4-row prefill: kv_b_proj on FlashInfer (4 rows), Attention_v4 per row and head over the row's own keys, no
    sparse MLA, merge or lse; the indexer's query and K slots are written, its logits not (no decode token in the step)."""
    r = build_program(phi, "step", M=4, pos0=0, with_desc=False)
    fam = _families(r)
    assert (fam["SparseMlaSplit_v1"], fam["MergeAttnStates_v1"], fam["AttentionHeadLse_v1"], fam["DsaMqaLogits_v3"]) == (0, 0, 0, 0)
    assert (fam["DsaIndexerQ_v1"], fam["DsaIndexerKRow_v1"]) == (4 * INDEXERS, 4 * INDEXERS)
    heads = _calls(r, "Attention_v4")
    assert [n.fn.bindings["T"] for n in heads] == [t for _ in range(LAYERS) for t in range(1, 5) for _ in range(NH)]
    assert {(n.fn.bindings["NH"], n.fn.bindings["KVH"], n.fn.bindings["D"], n.fn.bindings["BN"]) for n in heads} == {(1, 1, 256, 80)}
    kv_b = [n for n in _calls(r, "FlashInferFp8Linear_v1") if n.fn.bindings["N"] == NH * (192 + 256)]
    assert len(kv_b) == 4 * LAYERS and {n.fn.bindings["K"] for n in kv_b} == {512}


def test_a_chunk_with_context_merges_its_context_and_suffix(phi):
    """A 6-row prompt in two chunks of 3, each its step's prefill: the first by Attention_v4; the second, with 3 rows of context,
    per row and head the suffix's AttentionHeadLse over its own rows 0..i and the context's over the 3 history rows (kv_b_proj
    over them, non-causal: MASKED_FROM the last block), and MergeAttnStates per row."""
    r = build_program(phi, "request", LP=6, T=0, batch=[[[3, 0]], [[3, 3]]], chunks=(3, 3), with_desc=False)
    fam = _families(r)
    assert (fam["Attention_v4"], fam["MergeAttnStates_v1"], fam["SparseMlaSplit_v1"]) == (3 * NH * LAYERS, 3 * LAYERS, 0)
    lse = _calls(r, "AttentionHeadLse_v1")
    per_layer = [t for i in range(3) for t in [i + 1] * NH + [3] * NH]
    assert [n.fn.bindings["T"] for n in lse] == per_layer * LAYERS
    assert {n.fn.bindings["MASKED_FROM"] for n in lse if n.fn.bindings["T"] == 3} == {0}
    merge = _calls(r, "MergeAttnStates_v1")
    assert {(n.fn.bindings["NH"], n.fn.bindings["D"]) for n in merge} == {(NH, 256)}
    kv_b = [n for n in _calls(r, "FlashInferFp8Linear_v1") if n.fn.bindings["N"] == NH * (192 + 256)]
    assert len(kv_b) == (3 + 3 + 3) * LAYERS, "kv_b_proj over each chunk's rows, and over the second's 3 context rows"


def test_a_first_chunk_beside_a_chunk_with_context_merges_a_zero_partial(phi):
    """A 4-row first chunk in a step where another prefill has context: vLLM's suffix returns its lse for every prefill, and this
    one's context partial is the zero output with lse -inf, merged per token."""
    r = build_program(phi, "request", LP=4, T=0, batch=[[[3, 3], [4, 0]]], with_desc=False)
    fam = _families(r)
    assert (fam["Attention_v4"], fam["MergeAttnStates_v1"]) == (0, 4 * LAYERS)
    assert [n.fn.bindings["T"] for n in _calls(r, "AttentionHeadLse_v1")] == [t for _ in range(LAYERS) for t in range(1, 5)
                                                                             for _ in range(NH)]
    assert fam["DsaMqaLogits_v3"] == 0
    kv_b = [n for n in _calls(r, "FlashInferFp8Linear_v1") if n.fn.bindings["N"] == NH * (192 + 256)]
    assert len(kv_b) == 4 * LAYERS


def test_a_decode_beside_a_dense_prefill_runs_its_split_over_the_decodes(phi):
    """A 3-row prompt prefilled alone, then its decode token at position 3 beside another request's 4-row prefill: that prefill
    is dense MHA, so the sparse MLA launch holds the decode token only, and the indexer computes every token's logits (the step
    has a decode token), the prefill's alone none."""
    r = build_program(phi, "request", LP=3, T=1, batch=[[[3, 0]], [[1, 3], [4, 0]]], with_desc=False)
    (ns,) = launch_num_splits([4], int(phi.target_profile().num_sms), TOPK, 0)
    assert [n.fn.bindings["NS"] for n in _calls(r, "SparseMlaSplit_v1")] == [ns] * LAYERS
    assert (_families(r)["Attention_v4"], _families(r)["DsaMqaLogits_v3"]) == (3 * NH * LAYERS, INDEXERS)


def test_a_prefill_past_index_topk_takes_the_decode_kernels(phi):
    """INDEX_TOPK 4: a 6-row prefill reaches past it, so vLLM serves it by the sparse MQA kernels, the Program the decode path
    writes for its rows; with the threshold back at 256 the same rows are a decode, the same Program."""
    short = Phi.from_json({**_topk(phi, 4).to_json(), "engine": _canonical(phi.engine)})
    as_prefill = build_program(short, "step", M=6, pos0=0, with_desc=False)
    assert _families(as_prefill)["Attention_v4"] == 0 and len(_calls(as_prefill, "SparseMlaSplit_v1")) == 6 * LAYERS
    vllm_dsa.DECODE_THRESHOLD[NH] = 256
    assert build_program(short, "step", M=6, pos0=0, with_desc=False).digest == as_prefill.digest


def test_more_context_than_one_chunk_surely_holds_is_refused(phi):
    """17 extends of 13 context rows each: 221 rows, past min(8 x 16, 65536) = 128, where `plan_mla_context_chunks` may cut the
    context into several chunks, whose partials vLLM merges in turn."""
    with pytest.raises(Unexpressible, match=re.escape("a dense-MHA step's context of 221 rows, past the 128 rows")):
        build_program(phi, "request", LP=16, T=0, batch=[[[13, 0]], [[3, 13]] * 17], chunks=(13, 3), with_desc=False)
