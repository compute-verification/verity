"""A request Program's KV rows from the circuit (`verity_vllm.commit.memory_map`) on a small Llama's request Programs: rows of
1 KiB (4 KV heads of 128 bf16); request r0 a 4-row prefill, then 5 decode steps, and r1 a 4-row prefill, then 1 decode step.
Each row is born where its Program computes it and lasts to its last structural reader, a gather that may select it
included.  A warm request (a prefix-cache hit) reads its first rows from its cached-prefix input.  GLM-5.3's MLA: the DSA
stand-in's rows are its latent rows, one plane a layer (a row past INDEX_TOPK reads them through its concentrator, or under
table reads through its reads of the history table, and a batched Program's members are read as its Calls).
"""
from __future__ import annotations

import copy

import pytest
import torch

from tests.program.predict.test_predict import TINY, _phi
from tests.program import test_dsa_derive as DSA
from tests.program.test_dsa_derive import served  # noqa: F401  (the DSA stand-in's fixture)
from verity.definitions.circuits.census import step_of
from verity.circuits.defs import Builder, CompositeDefinition, bind
from verity.circuits.program import Program, is_input
from verity.circuits.refs import Affine, Coll, concat
from verity.definitions.circuits.requirement import from_census
from verity.circuits.types import Array, RecordType, Value
from verity_vllm.commit import memory_map as MM
from verity_vllm.program import prune as P
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.export_compat import dsa_leaves
from verity_vllm.program.frontend.rules import vllm_bindings as VB, vllm_dsa
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.frontend.rules.vocab import PROFILES
from verity_vllm.program.predict.build import BF16, I32, build_program, decoder
from verity_vllm.program.predict.common import _Peers
from verity.definitions.replay import replayer

WIDE = TINY | {"hidden_size": 512, "num_key_value_heads": 4}


@pytest.fixture(scope="module")
def r0() -> Program:
    return build_program(_phi(WIDE), "request", LP=4, T=5).program


@pytest.fixture(scope="module")
def reqs(r0) -> dict[str, MM.Rows]:
    return {"r0": MM.rows(r0), "r1": MM.rows(build_program(_phi(WIDE), "request", LP=4, T=1).program)}


def selected_at(c: dict, values: set[int], step: int) -> dict:
    """Census `c` with each Call of `step` that reads one of `values` selecting it instead, a gather's candidate (the
    census's `selects`, a new entry's `gates` and `bits` a stand-in's, which the requirement doesn't read)."""
    c = copy.deepcopy(c)
    at, reads, sel = step_of(c), c["values"]["reads"], c["selects"]
    pools = {k: set(vs) for k, vs in zip(sel["call"], sel["values"])}
    for x in values:
        for k in [k for k in reads[x] if at(k) == step]:
            reads[x].remove(k)
            pools.setdefault(k, set()).add(x)
    old, ks = dict(zip(sel["call"], zip(sel["gates"], sel["bits"]))), sorted(pools)
    c["selects"] = {"call": ks, "gates": [old.get(k, (1, 0))[0] for k in ks], "bits": [old.get(k, (1, 0))[1] for k in ks],
                    "values": [sorted(pools[k]) for k in ks]}
    return c


def census_values(program: Program, c: dict, row: MM.Row) -> set[int]:
    """The census values a row's leaves are."""
    inputs, v = sum(1 for nd in program.root.body.nodes if is_input(nd.fn)), c["values"]
    return {x for _, j, lo, hi in row.group for x, k in enumerate(v["producer"])
            if k == j + inputs and v["lo"][x] < hi and v["hi"][x] > lo}


def test_rows_are_the_programs_kv(reqs):
    rs = reqs["r0"]
    assert (len(rs.layers), rs.positions, rs.row_bytes, rs.word_bytes, rs.steps) == (2, 9, 1024, 2, 6)
    # the prefill writes positions 0-3 at step 0, decode step t position 3 + t, and the last step reads every row
    for kv in rs.layers:
        for plane in kv:
            assert [x.born for x in plane] == [0, 0, 0, 0, 1, 2, 3, 4, 5] and {x.last for x in plane} == {5}
    groups = [x.group for kv in rs.layers for plane in kv for x in plane]
    assert len(set(groups)) == len(groups) == 2 * 2 * 9


def test_a_row_a_gather_may_select_lasts_to_the_last_step_that_may_select_it(r0, reqs):
    """A row's `last` is its last structural reader, a gather that may select it included: with r0's last step, 5, selecting
    its first four positions' rows instead of reading them (`selected_at`, a stand-in: no served Program gathers its KV),
    step 4 is their last use, but their `last` and the rows are as before, whichever rows step 5 selected."""
    c = MM.census_of(r0)
    first = {x for kv in reqs["r0"].layers for plane in kv for row in plane[:4] for x in census_values(r0, c, row)}
    picked = selected_at(c, first, 5)
    req = from_census(picked)
    assert first and first <= req.upper[5] and not first & req.uses[5] and first <= req.uses[4]
    assert {req.last[x] for x in first} == {5} and MM.rows(r0, picked) == reqs["r0"]


# -- a warm request: a prefix-cache hit on another request's rows -------------------------------------------------------------


def warm_program(LP: int, T: int, C: int) -> Program:
    """A warm request's Program as a Build derives one (`build --cached C`, `kv_prefix`): every layer's history opens with the C
    K and V rows of its `prefix` input (bf16 [L, 2, C, row], layer l the l-th to open), and its own rows follow from position C
    (rows C..LP-1 at its first step, then T decodes)."""
    net = decoder(_phi(WIDE), 1, LP + T)
    wrec, rl = RecordType(*net.leaves()), net.C.KVH * net.C.D
    params = (("prompt", Array(LP, I32)), ("prefix", Array(net.C.NL, Array(2, Array(C, Array(rl, BF16))))), ("weights", wrec))
    B = Builder(params, Value(1))
    model = net.model(B, Coll(wrec, Affine("p", 2, 0, wrec.leaves)), _Peers([], 1, 0))
    prefix = Affine("p", 1, 0, params[1][1].leaves)
    for layer, hist in enumerate(model.hist):
        for kv, rows in enumerate(hist):
            rows += [prefix.slice(((2 * layer + kv) * C + j) * rl, ((2 * layer + kv) * C + j + 1) * rl) for j in range(C)]
    prompt = Affine("p", 0, 0, LP)
    toks = [model.sample(model.forward([Coll(I32, prompt.slice(q, q + 1)) for q in range(C, LP)], C)[-1], 0, LP - 1)]
    for s in range(T):
        toks.append(model.sample(model.forward([Coll(I32, toks[-1].refs)], LP + s)[-1], s + 1, LP + s))
    out = Coll(Array(len(toks), I32), concat([t.refs for t in toks]))
    B.ret_type = out.type
    body = B.finish(out)
    defn = CompositeDefinition("Derived[warm]", 1, ("SRC",), lambda S, _p=params, _r=out.type: (_p, _r), replayer(body), register=False)
    sd = bind(defn, SRC={"class": "warm", "C": C})
    sd.set_body(body)
    return Program(sd)


#: the warm Program's cached prefix as its Build lays it out (``prefix_cache.json``): its `prefix` input, 4 rows of 512 bf16
PREFIX = {"input": 1, "C": 4, "row_leaves": 512, "layers": [f"model.layers.{i}.self_attn.attn" for i in range(2)]}


@pytest.fixture(scope="module")
def warm() -> Program:
    return warm_program(8, 4, 4)


@pytest.fixture(scope="module")
def session(warm) -> dict[str, MM.Rows]:
    return {"r0": MM.rows(build_program(_phi(WIDE), "request", LP=8, T=1).program), "r1": MM.rows(warm, prefix=PREFIX)}


def test_a_warm_programs_prefix_rows_are_its_cached_positions(warm, session):
    rs = session["r1"]
    assert (len(rs.layers), rs.positions, rs.cached, rs.row_bytes, rs.steps) == (2, 12, 4, 1024, 5)
    for layer, kv in enumerate(rs.layers):
        for plane, rows in enumerate(kv):
            # positions 0-3 are the prefix input's rows of this layer and plane, read from the first step to the last; the
            # Program computes 4-7 at its first step and 8-11 one a decode step
            assert [x.group for x in rows[:4]] == [((MM.CACHED, 1, o, o + 512),) for o in range((2 * layer + plane) * 4 * 512, (2 * layer + plane + 1) * 4 * 512, 512)]
            assert [x.born for x in rows] == [0] * 8 + [1, 2, 3, 4] and {x.last for x in rows} == {4}
            assert not any(s[0] == MM.CACHED for x in rows[4:] for s in x.group)
    assert MM.rows_of(MM.rows_doc(rs)) == rs and session["r0"].cached == 0
    with pytest.raises(ValueError, match="root node .*'s kc reads an input of the Program"):
        MM.rows(warm)
    with pytest.raises(ValueError, match="isn't row 0 of an input of the cached prefix's 2 x 2 x 2 x 512 leaves"):
        MM.rows(warm, prefix=PREFIX | {"C": 2})
    with pytest.raises(ValueError, match="isn't row 0 of an input"):
        MM.rows(build_program(_phi(WIDE), "request", LP=8, T=1).program, prefix=PREFIX)
    with pytest.raises(ValueError, match="a cached prefix of 4 rows of 256 leaves"):
        MM.rows(warm, prefix=PREFIX | {"row_leaves": 256})


# -- GLM-5.3's MLA: one plane of latent rows --------------------------------------------------------------------------------


class Sampled(DSA._Model):
    """The DSA stand-in (`tests/program/test_dsa_derive.py`) with each step closed, as an engine step is, by a greedy sampling event
    over its last token's output."""

    def forward(self, *xs):
        outs, k = [], 0
        for p0, n in self.steps:
            positions = torch.arange(p0, p0 + n, dtype=torch.int64)
            for layer in self.layers:
                outs += list(layer(positions, *xs[k:k + layer.n_inputs]))
                k += layer.n_inputs
            outs.append(torch.argmax(outs[-2][-1], dim=-1))
        return tuple(outs)


def dsa_request(steps=DSA.STEPS, topk: int = DSA.TOPK):
    """The DSA stand-in's request Program over ``steps``: an indexer layer and an IndexShare layer, each step a sampling event's."""
    model = Sampled(list(DSA._model(topk=topk, steps=steps).layers), steps)
    return DSA._derive(model, DSA._inputs(model)).program


@pytest.fixture
def dsa(served):  # noqa: F811
    program = dsa_request()
    return program, MM.rows(program)


def test_a_sparse_mla_layers_rows_are_its_latent_rows(dsa):
    program, rs = dsa
    # an indexer layer and an IndexShare layer, each its own history of one plane: positions 0 and 1 written at the prefill, 2 at
    # the decode, which reads every row
    assert (len(rs.layers), [len(kv) for kv in rs.layers], rs.positions, rs.row_bytes, rs.word_bytes, rs.steps) == (2, [1, 1], 3, 64, 2, 2)
    nodes = program.root.body.nodes
    inputs = sum(1 for nd in nodes if is_input(nd.fn))
    for (plane,) in rs.layers:
        assert [x.born for x in plane] == [0, 0, 1] and {x.last for x in plane} == {1}
        # a row is the normed latent, then the roped key, as vLLM's MLA cache row holds them; the indexer's keys are no row
        assert {tuple((nodes[inputs + j].fn.name, hi - lo) for _, j, lo, hi in x.group) for x in plane} == {
            (("RMSNormTriton", DSA.L), ("RoPEHeadGptJ", DSA.R))}
    assert not {x.group for x in rs.layers[0][0]} & {x.group for x in rs.layers[1][0]}


def _born_last(rs: MM.Rows) -> list:
    return [[(x.born, x.last) for x in plane] for kv in rs.layers for plane in kv]


def test_a_row_past_topk_reads_its_layers_rows_through_its_concentrator(dsa):
    """At TOPK 2 the decode's row of 3 keys is past TOPK: its sparse MLA reads the TOPK rows `DsaButterflyRowsPad` routes from its
    layer's history, and its layer's rows are that history's, as at TOPK 4, where it reads them directly."""
    program = dsa_request(topk=2)
    nodes = program.root.body.nodes
    inputs = sum(1 for nd in nodes if is_input(nd.fn))
    assert sum(getattr(nd.fn, "name", None) == "DsaButterflyRowsPad" for nd in nodes) == 2
    rs = MM.rows(program)
    assert (len(rs.layers), rs.positions, _born_last(rs)) == (2, 3, _born_last(dsa[1]))
    assert {tuple((nodes[inputs + j].fn.name, hi - lo) for _, j, lo, hi in x.group) for (plane,) in rs.layers for x in plane} == {
        (("RMSNormTriton", DSA.L), ("RoPEHeadGptJ", DSA.R))}


@pytest.mark.parametrize("routed", [("DsaButterflyRows", "DsaButterflyRowsPad"), ("DsaHistoryReads",)], ids=["rows-network", "table-reads"])
@pytest.mark.parametrize("block", [256, 2])
def test_a_batched_programs_rows_are_its_flat_programs(served, monkeypatch, block, routed):
    """Class G's form at TOPK 2 (`observe_batched_form`: latent rows, the long rows in blocks of `vllm_dsa.BLOCK_KEYS` keys): a
    prefill of two, then a call of three rows past TOPK.  Compiled batched (`prune.BATCHED`), each layer's latent rows are batches
    and the call's sparse MLA one batch of its three rows, each member reading its rows through its concentrator's batch, or under
    table reads (`observe_batched_form`'s ``table_reads``) through its reads of the history table, whose table is the block's history; `rows`
    reads each member, and every layer's rows are born and last read where the flat Program's are."""
    monkeypatch.setattr(vllm_dsa, "BLOCK_KEYS", block)
    monkeypatch.setitem(PROFILES, DSA.BATCHED.name, DSA.BATCHED)
    VB.observe_batched_form(True, table_reads=routed == ("DsaHistoryReads",))
    steps = ((0, 2), (2, 3))
    model = Sampled(list(DSA._model(topk=2, steps=steps).layers), steps)
    xs = DSA._inputs(model)

    def compiled(form):
        with dsa_leaves(DSA.SITE, model):
            rep = derive_report(model, xs, mode="profile", profile=DSA.BATCHED.name, name="dsa", rules=vllm_ruleset(), dead_free=form)
        assert rep.program is not None, rep.unsupported
        return rep.program

    flat, batched = compiled(True), compiled(P.BATCHED)
    nodes = batched.root.body.nodes
    assert [nd.n for nd in nodes if nd.form == "batch" and getattr(nd.fn, "name", None) == "SparseMla"] == [3, 3]
    assert all(nd.form == "call" for nd in flat.root.body.nodes if getattr(nd.fn, "name", None) == "SparseMla")
    assert set() < {getattr(nd.fn, "name", None) for nd in nodes if nd.form == "batch"} & set(MM.ROUTED) <= set(routed)
    a, b = MM.rows(flat), MM.rows(batched)
    assert (len(b.layers), b.positions, b.steps) == (2, 5, 2)
    assert _born_last(b) == _born_last(a) == [[(0, 1)] * 2 + [(1, 1)] * 3] * 2
