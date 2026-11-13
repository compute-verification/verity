"""verity_vllm.program.predict.node_window_embed: the vocabulary shard's embedding lowered into GLM-5.3's node window at TP8
rank 0, beside the frame's and the dense families, at a 32-id prompt, 2 outputs and one decoder layer: per column block of 192
words one `EmbeddingRead_v1{V=154880, H=192}` unit a member over that block's padded table, the all-reduce's own partial
read from the blocks' ``o0`` by wire. The tables are one width, declared and never committed by the window, and kept apart
from the plain weights, whose nonces they don't move.
"""
from __future__ import annotations

import re

import pytest
from verity_vllm.pipeline import expert_tables as XT
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_dense as ND
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.window_builder import Grain

from tests.program.predict.test_node_window import LP, glm_request
from verity.circuits.codec import descriptor_id
from verity.commitments.windows import interface as IF

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NE.FAMILIES}
WIDTH, V, BLOCKS = 6144, 154880, 6144 // NE.WIDTH
LABEL = "model.embed_tokens.weight"
READ = r"EmbeddingRead_v1\{V=154880,H=192\}"


@pytest.fixture(scope="module")
def program():
    return glm_request(LP, 1).program.fn


@pytest.fixture(scope="module")
def window(program):
    return NW.node_window(program, families=FAMILIES)


@pytest.fixture(scope="module")
def without(program):
    return NW.node_window(program, families={**NW.FAMILIES, **ND.FAMILIES})


def _tid(circuit: IF.Circuit, pattern: str) -> str:
    (tid,) = [t for t in circuit.units if re.fullmatch(pattern, t)]
    return tid


def test_units(window, program):
    """Each shard node is 32 read units a member (6,144 / 192): the prefill's 32 tokens and the decode's one, 32 x 33 in
    all, block b of member t at unit ``a + 32 b + t``; the read is the one template `unit_templates` adds to the row's."""
    circuit, w = window
    read = _tid(circuit, READ)
    assert [(n, fam) for _at, n, fam in w.allocs[read]] == [(BLOCKS * LP, "EmbeddingShard"), (BLOCKS * 1, "EmbeddingShard")]
    assert circuit.units[read] == BLOCKS * (LP + 1)
    (spec,) = {descriptor_id(n.fn) for n in program.body.nodes if NW._family(n) == "EmbeddingShard"}
    assert NE.unit_templates(spec)[-1] == (read, BLOCKS)
    for tid, fn in w.b.fns.items():
        IF.check_proof_block(fn, circuit.cuts.get(tid))


def test_wired_to_allreduce(window):
    """The all-reduce's column block b reads block b's ``o0`` as row 0 of ``parts`` by one wire pattern a step: the
    prefill's 32 x 32 units at steps (1, 32), the decode's 32 at step 1; its peers' rows stay free."""
    circuit, w = window
    read = _tid(circuit, READ)
    ar = {s: _tid(circuit, rf"AllReduceMnnvl_v1\{{WORLD=8,N=192,SHOT={s}\}}") for s in (1, 2)}
    got = {(x.reader, x.port, x.radix, x.base, x.step, x.cut) for x in circuit.wires if x.writer == read}
    assert got == {(ar[2], "parts", (BLOCKS, LP), 0, (1, BLOCKS), (0, 1)), (ar[1], "parts", (BLOCKS,), BLOCKS * LP, (1,), (0, 1))}
    own = [f for f in w.free if f.tid in ar.values() and f.rows[0] == 0]
    assert own == []
    assert {f.kind for f in w.free if f.tid in ar.values()} == {"peer"}


def test_census(window, without):
    """`EmbeddingShard` is gone from the unlowered families and its port from the free ones; its ``tok`` port is free
    under the prompt and the token select, which are not lowered."""
    circuit, w = window
    census, before = w.census(circuit), without[1].census(without[0])
    assert before["unlowered"]["EmbeddingShard"] == 2 and before["free"]["EmbeddingShard"]["words"] == (LP + 1) * WIDTH
    assert "EmbeddingShard" not in census["unlowered"] and "EmbeddingShard" not in census["free"]
    assert census["units"]["EmbeddingShard"] == BLOCKS * (LP + 1)
    read = _tid(circuit, READ)
    assert {(f.port, f.kind, f.family) for f in w.free if f.tid == read} == {("tok", "param", "prompt"),
                                                                            ("tok", "node", "TokenSelect")}


def test_tables_declared(window):
    """32 tables, one a block, each the whole vocabulary's 154,880 rows of 192 words: registered parameters of their own,
    each read by its block's units' root, domain and opening, and nothing committed."""
    circuit, w = window
    reg, read = circuit.registered, _tid(circuit, READ)
    params = [NE.shard_param(LABEL, NE.WIDTH, b) for b in range(BLOCKS)]
    assert sorted(t.param for t in reg.tables) == sorted(params)
    assert {(t.offset, t.words, t.rows) for t in reg.tables} == {(0, NE.WIDTH, V)}
    assert all(reg.params[p] == V * NE.WIDTH for p in params)
    for p in params:
        assert {(r.of, r.units) for r in reg.roots if r.table == p} == {("root", LP), ("domain", LP), ("root", 1), ("domain", 1)}
        assert {(o.reader, o.key, o.verdict, o.units) for o in reg.openings if o.table == p} == {(read, "tok", "o1", LP),
                                                                                              (read, "tok", "o1", 1)}


def test_split_keeps_weight_nonces(window, without):
    """The tables are kept apart from the plain weights: the plain weights' nonces are the same with and without the shard's
    tables present, handed every name they would not be, `window_weights` refuses a table's name, and the tables' own
    nonces are the tables' registration's, from `TABLES_NONCE`."""
    params, before = window[0].registered.params, without[0].registered.params
    plain, tables = NE.split(params)
    assert plain == WW.weight_params(before) and NE.split(before) == (plain, [])
    assert WW.weight_nonces(plain) == WW.weight_nonces(before)
    moved = WW.weight_nonces(params)
    assert any(moved[p] != n for p, n in WW.weight_nonces(before).items())
    assert len(tables) == BLOCKS
    with pytest.raises(ValueError, match="no weight parameter"):
        WW._param(tables[0])
    by = NE.tables_of(params)
    assert list(by) == [LABEL] and [b for b, *_ in by[LABEL]] == list(range(BLOCKS))
    nonces = XT.nonces(by, [label for label, _f, _n in window[1].b.weights], 0)
    assert sorted(nonces) == sorted(tables) and len(set(nonces.values())) == BLOCKS
    assert min(nonces.values()) >= XT.TABLES_NONCE > max(WW.weight_nonces(plain).values())


def test_second_width_refused(program):
    """One width a field: `shard_param` refuses any width but 192, a window whose norm grain cuts H at another width is
    refused, and so is a shard whose table was declared at another width before."""
    with pytest.raises(NW.Unexpressible, match="its one width is 192"):
        NE.shard_param(LABEL, 96, 0)
    with pytest.raises(NW.Unexpressible, match="in blocks of 96; a shard's tables are 192"):
        NW.node_window(program, families=NE.FAMILIES, grain=Grain(norm=96))
    w = NW.NodeWindow(program, NE.FAMILIES)
    w.b.params[f"weights/{LABEL}@96/b0/vocab"] = V * 96
    with pytest.raises(NW.Unexpressible, match=r"declared at \[96\] words a row and at 192"):
        w.lower()


def test_refusals():
    """A shard of no registered vocabulary is refused by name."""
    with pytest.raises(NW.Unexpressible, match="no registered vocabulary"):
        NE.shard_vocab("EmbeddingShard_v1{VS=16,H=2,START=16}")
