"""verity_vllm.program.predict.template_counts: n_t(c), read off a request unit's key, is each template's instances in the Program
built for it, as a template partition counts them (`TemplatePartition` over every template the Program calls, a batch counting its
members), at GLM-5.3's class-G registration cut to its first layers: a request beside other sequences, in prefill chunks, after a
history, resumed, leaving at a window's edge, stochastic, under its length rule and in a launch past FlashInfer's fused all-reduce;
at INDEX_TOPK 16 under a decode threshold of 2, its dense-MHA prefills; at INDEX_TOPK 4, its rows past it, on a last layer that
shares the top-k and on one that computes it, and in several blocks a call, by the rows network and by table reads, whose long
rows' units (`TemplateCounts.units`) are their cut; GLM-5.3's at 131,072 keys; and the B8 window's planned units."""

from __future__ import annotations

from collections import Counter

import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.partition_object import TemplatePartition, calls, template_instances_query
from verity.ml.commitments.gates.embedding import read
from verity.ml.definitions.registry import dsa_topk_reads as R
from verity_vllm.engine.profiles.engine_classes import DSA_TOPK_ENV
from verity_vllm.program.frontend.rules import vllm_dsa
from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered, plan_steps, policy_of, request_program
from verity_vllm.program.predict.template_counts import TABLE_ROWS, TemplateCounts
from tests.program.predict.test_node import _first_layers
from tests.program.predict.test_steps import _b8_window, _class_g, _spec

GREEDY = {"kind": "greedy"}
EDGE = ("prefix", "prefix_ik", "prefix_iks")


def _phi(tmp_path, layers: int = 4, topk: int | None = None) -> Phi:
    """Class G's GLM-5.3 φ at its first `layers` layers; at `topk` its INDEX_TOPK, served under the canonical top-k row."""
    p = _first_layers(_class_g(tmp_path).phi, layers).to_json()
    if topk is None:
        return Phi.from_json(p)
    eng = {**p["engine"], "env": {**(p["engine"].get("env") or {}), **DSA_TOPK_ENV}}
    return Phi.from_json({**p, "model": {**p["model"], "hf_config": {**p["model"]["hf_config"], "index_topk": topk}}, "engine": eng})


def _instances(program) -> Counter:
    """Each template's instances in `program` as `TemplatePartition` numbers them, a root Call one and a batch its members, which
    the partition itself gives where it applies: a const called alone is a primitive node, which its query refuses."""
    nodes, out = program.root.body.nodes, Counter()
    for k, fn in calls(program):
        out[descriptor_id(fn)] += nodes[k].n if nodes[k].form == "batch" else 1
    if all(nodes[k].form in ("call", "batch") for k, _ in calls(program)):
        part, by = TemplatePartition(program, template_instances_query(sorted(out))), Counter()
        for _k, tid, _at, n in part.ranges:
            by[tid] += n
        assert by == out
    return out


def _cut(program, H: int) -> Counter:
    """`_instances` of `program` with its long rows in units, read off its Calls: each `DsaTopKIndex_v1{N,TOPK}` member the cut of
    its long row `DsaTopKReadRows_v1{N,TOPK}` (`dsa_topk_reads.CUTS`) but the reads, its counted mask's units with it; each
    `DsaHistoryReads_v1{V,TOPK,H}` member TOPK reads of the history table; and no counted mask of its own."""
    nodes, rd, out = program.root.body.nodes, read(TABLE_ROWS, H), Counter()
    for k, fn in calls(program):
        n, base = nodes[k].n if nodes[k].form == "batch" else 1, getattr(fn, "base_id", None)
        if base == "DsaTopKIndex_v1":
            row = bind(R.DsaTopKReadRows, N=fn.bindings["N"], TOPK=fn.bindings["TOPK"], BLOCK=R.BLOCK, READ=rd)
            out.update({descriptor_id(u): n * c for u, c in R.CUTS[row.base_id](row) if descriptor_id(u) != descriptor_id(rd)})
        elif base == "DsaHistoryReads_v1":
            out[descriptor_id(rd)] += n * fn.bindings["TOPK"]
        elif base != "DsaTopKMaskCounted_v1":
            out[descriptor_id(fn)] += n
    return out


def _check(phi: Phi, LP: int, T: int, policy: dict = GREEDY, table_reads: bool = False, **kw) -> None:
    """n_t of (LP, T) is the built Program's; under `table_reads` its units too (`_cut`)."""
    r = build_program(Phi.from_json({**phi.to_json(), "sampling": policy}), "request", LP=LP, T=T, dead_free=True, batched=True,
                      with_desc=False, table_reads=table_reads, **kw)
    tc = TemplateCounts(phi, table_reads=table_reads)
    want, got = _instances(r.program), tc.request(LP, T, policy, **kw)
    assert got == want, {t: (got[t], want[t]) for t in sorted(got.keys() | want.keys()) if got[t] != want[t]}
    if table_reads:
        want, got = _cut(r.program, int(tc.C["KV_LORA"]) + int(tc.C["QK_ROPE"])), tc.units(got)
        assert got == want, {t: (got[t], want[t]) for t in sorted(got.keys() | want.keys()) if got[t] != want[t]}


def _decodes(LP: int, T: int, first: list) -> list:
    return [first] + [[[1, LP + t]] for t in range(T)]


SERVED = {
    "beside-decodes": dict(LP=3, T=2, batch=[[[3, 0]], [[1, 3], [1, 9]], [[1, 4], [1, 10]]]),
    "chunks": dict(LP=6, T=1, chunks=(4, 2), batch=[[[4, 0], [1, 7]], [[2, 4]], [[1, 6]]]),
    "history": dict(LP=6, T=2, C=4, batch=[[[2, 4]], [[1, 6]], [[1, 7]]]),
    "edge-in-prompt": dict(LP=6, T=-2, edge=("prefix",), batch=[[[4, 0]]]),
    "resumed": dict(LP=6, T=3, C=7, batch=[[[1, 7]], [[1, 8]]]),
    "unfused": dict(LP=5, T=1, batch=[[[5, 0], [40, 0]], [[1, 5], [1, 40]]]),
    "stop": dict(LP=3, T=2, stop=_spec(3, 2, 3), batch=_decodes(3, 2, [[3, 0]])),
}
DENSE = {
    "alone": dict(LP=4, T=1, batch=_decodes(4, 1, [[4, 0]])),
    "chunk-with-context": dict(LP=6, T=0, chunks=(3, 3), batch=[[[3, 0]], [[3, 3]]]),
    "beside-a-decode": dict(LP=3, T=1, batch=_decodes(3, 1, [[3, 0], [1, 5]])),
    "beside-an-extend": dict(LP=3, T=1, batch=_decodes(3, 1, [[3, 0], [4, 3]])),
    "after-a-history": dict(LP=6, T=1, C=3, batch=_decodes(6, 1, [[3, 3]])),
    "then-past-topk": dict(LP=12, T=6, batch=_decodes(12, 6, [[12, 0]])),
}
LONG = {
    "prefill-and-decodes": dict(LP=6, T=2, batch=_decodes(6, 2, [[6, 0]])),
    "chunks-to-an-edge": dict(LP=9, T=-1, chunks=(5, 3), edge=EDGE, batch=[[[5, 0]], [[3, 5]]]),
    "unfused": dict(LP=6, T=1, batch=[[[6, 0], [40, 0]], [[1, 6], [1, 40]]]),
    "short-prompt": dict(LP=3, T=3, batch=_decodes(3, 3, [[3, 0]])),
    "after-a-history": dict(LP=8, T=1, C=4, batch=_decodes(8, 1, [[4, 4]])),
    "resumed-to-an-edge": dict(LP=4, T=4, C=6, edge=EDGE, batch=[[[1, 6]], [[1, 7]]]),
}


@pytest.mark.parametrize("case", SERVED)
def test_a_units_counts_are_its_programs_at_the_served_index_topk(tmp_path, case):
    _check(_phi(tmp_path), **SERVED[case])


@pytest.mark.slow
def test_a_stochastic_units_counts_hold_its_selects_consts(tmp_path):
    """The seeded top-p select at its registered split count, swapped to its dead-free version, and its per-row consts (top_p,
    the position, the temperature and S), which no greedy unit calls; the Program's select at GLM-5.3's vocabulary is most of
    its 2 minutes."""
    phi = _phi(tmp_path)
    policy, _ = policy_of(0, {"temperature": 0.8, "top_p": 0.95, "seed": 11}, phi.target_profile().num_sms, request_id="q1")
    _check(phi, 3, 2, policy, batch=_decodes(3, 2, [[3, 0]]))


@pytest.mark.parametrize("case", DENSE)
def test_a_dense_mha_prefills_counts_are_its_programs(tmp_path, monkeypatch, case):
    monkeypatch.setitem(vllm_dsa.DECODE_THRESHOLD, 8, 2)
    _check(_phi(tmp_path, topk=16), **DENSE[case])


@pytest.mark.parametrize("layers", [4, 3], ids=["shared-last", "indexer-last"])
@pytest.mark.parametrize("case", LONG)
def test_rows_past_index_topk_count_their_selection_and_the_slots_they_read(tmp_path, case, layers):
    _check(_phi(tmp_path, layers=layers, topk=4), **LONG[case])


@pytest.mark.parametrize("case", LONG)
def test_rows_past_index_topk_in_several_blocks_count_each_blocks_selection(tmp_path, monkeypatch, case):
    """At blocks of 2 keys (`vllm_dsa.BLOCK_KEYS`) a call's rows past INDEX_TOPK read several key counts: the chunk of rows of
    6, 7 and 8 keys has blocks of 6 and 8."""
    monkeypatch.setattr(vllm_dsa, "BLOCK_KEYS", 2)
    _check(_phi(tmp_path, topk=4), **LONG[case])


@pytest.mark.parametrize("block", [256, 2])
@pytest.mark.parametrize("layers", [4, 3], ids=["shared-last", "indexer-last"])
@pytest.mark.parametrize("case", ["prefill-and-decodes", "chunks-to-an-edge", "after-a-history"])
def test_rows_past_index_topk_by_table_reads_count_their_index_and_reads(tmp_path, monkeypatch, case, layers, block):
    """Under the registration's table reads (`Registered.dsa_table_reads`) a row past INDEX_TOPK counts its index in place of the
    settings and its history reads at its block's keys in place of the rows network, with no zero pad, in one block a call or
    in several; in units, each its long row's cut."""
    monkeypatch.setattr(vllm_dsa, "BLOCK_KEYS", block)
    _check(_phi(tmp_path, layers=layers, topk=4), table_reads=True, **LONG[case])


def test_glm53s_decode_at_131072_keys_reads_its_history_table_2048_times_a_layer(tmp_path):
    """GLM-5.3 whole (78 layers, 21 of them indexer layers) under table reads, one decode token after 131,071 positions: each
    layer's long row is 2,048 reads of its history table, 159,744 a token, and each indexer layer's its long row's cut besides
    (the counted mask's 5M + 4 units at M = 8 blocks of 16,384 keys, a `DsaMaskPositions_v1` a block and the merge).  Every
    other template is as n_t counts it, and units are the table form's alone."""
    phi, C = _phi(tmp_path, layers=78, topk=2048), 131071
    tc = TemplateCounts(phi, table_reads=True)
    n_t = tc.request(6, C - 6 + 1, C=C, batch=[[[1, C]]])
    units = tc.units(n_t)
    indexers, rd = len(tc.net.indexers), read(TABLE_ROWS, 576)
    row = bind(R.DsaTopKReadRows, N=131072, TOPK=2048, BLOCK=R.BLOCK, READ=rd)
    cut = Counter({descriptor_id(u): n * indexers for u, n in R.CUTS[row.base_id](row)})
    cut[descriptor_id(rd)] = 78 * 2048
    assert (indexers, units[descriptor_id(rd)]) == (21, 159744)
    assert units - n_t == cut and sum(cut.values()) - cut[descriptor_id(rd)] == 21 * (5 * 8 + 4 + 8 + 1)
    assert n_t - units == Counter({"DsaTopKMaskCounted_v1{N=131072,TOPK=2048}": 21, "DsaTopKIndex_v1{N=131072,TOPK=2048}": 21,
                                   "DsaHistoryReads_v1{V=131072,TOPK=2048,H=576}": 78})
    with pytest.raises(ValueError, match="table-read form's"):
        TemplateCounts(phi).units(n_t)


@pytest.mark.slow
def test_the_b8_windows_planned_units_count_their_request_programs(tmp_path):
    """Each unit the B8 window plans to in class G (launches of the window's co-scheduled rows, FULL graphs at its uniform
    decode steps), at 4 layers: n_t from the unit alone is its `request_program`'s template partition."""
    reg = Registered.of(_phi(tmp_path), default_seed=7)
    counts = TemplateCounts(reg.phi)
    for u in plan_steps(reg, *_b8_window()[:2]).requests:
        assert counts.unit(u) == _instances(request_program(reg, u).program), u.key()
