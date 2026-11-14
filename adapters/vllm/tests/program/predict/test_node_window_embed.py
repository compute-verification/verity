"""verity_vllm.program.predict.node_window_embed: the vocabulary shard's embedding lowered into GLM-5.3's node window at TP8
rank 0, beside the frame's and the dense families, at a 32-id prompt, 2 outputs and one decoder layer: per column block of 192
words one `EmbeddingRead_v1{V=154880, H=192}` unit a member over that block's padded table, the all-reduce's own partial
read from the blocks' ``o0`` by wire. The tables are one width, declared and never committed by the window, and kept apart
from the plain weights, whose nonces they don't move.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re

import numpy as np
import pytest
from verity_vllm.pipeline import expert_tables as XT
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_dense as ND
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import node_window_values as NV
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.window_builder import Grain
from verity_vllm.protocols.pouw.h1 import registered as H1R

from tests.program.predict.test_node_window import LP, glm_request
from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation.batch import evaluate_batch
from verity.commitments.gates.embedding import arguments, reference
from verity.commitments.gates.embedding import read as read_of
from verity.commitments.windows import interface as IF
from verity.protocols.proofs.spot.verifier.one_stage import registered as G

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
    assert NE.split(before) == (plain, []) == (WW.weight_params(before), [])
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


@pytest.mark.slow
def test_split_beside_expert_tables():
    """Over an MoE layer with the sparse families, whose expert blocks' tables are in the window too, the split keeps both
    kinds of table apart: the plain weights and their nonces are the same with and without the shard's tables, and handed
    unsplit all but the first would take another nonce."""
    from verity_vllm.program.predict import node_window_sparse as NS

    fn = glm_request(LP, 1, moe=True).program.fn
    base = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
    params, before = (NW.node_window(fn, families=f)[0].registered.params for f in ({**base, **NE.FAMILIES}, base))
    (plain, tables), (plain0, tables0) = NE.split(params), NE.split(before)
    assert plain == plain0 and WW.weight_nonces(plain) == WW.weight_nonces(plain0)
    assert set(tables) - set(tables0) == {NE.shard_param(LABEL, NE.WIDTH, b) for b in range(BLOCKS)}
    assert all(t.endswith("/table") for t in tables0) and tables0
    moved = WW.weight_nonces(params)
    assert sum(moved[p] != n for p, n in WW.weight_nonces(plain0).items()) == len(plain0) - 1


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


# ---- values: the units, fed from the circuit's declarations, against the shard's words

VS, KEY = 19360, hashlib.sha512(b"node-window embed test: a test committer's key").digest()[:32]
PROGRAM = hashlib.sha512(b"node-window embed test: a test program").hexdigest()
ZERO_SALT, ZERO_PATH = bytes(96), [bytes(64)] * 18


@pytest.fixture(scope="module")
def tables():
    """Made-up shard columns of blocks 0 and 1 (VS rows of 192 bf16 words each), padded and committed under a test key as the
    tables' registration commits them."""
    rng = np.random.default_rng(274)
    cols = {b: rng.integers(0, 0x7F80, (VS, NE.WIDTH), dtype=np.uint16) for b in (0, 1)}
    held = {NE.shard_param(LABEL, NE.WIDTH, b): H1R.commit(PROGRAM, NE.shard_param(LABEL, NE.WIDTH, b),
                                                           NE.table_words(c, 0, V), KEY, b, keep_rows=True, schema=G.SCHEMA_SEG)
            for b, c in cols.items()}
    return cols, held


def _declared(circuit: IF.Circuit, tid: str, u: int) -> str:
    """The one table unit ``u``'s root, domain id and opening are all declared of."""
    reg = circuit.registered
    got = {(x.of, x.table) for x in reg.roots if x.reader == tid and x.at <= u < x.at + x.units}
    got |= {("opening", o.table) for o in reg.openings if o.reader == tid and o.at <= u < o.at + o.units}
    (table,) = {t for _of, t in got}
    assert {of for of, _t in got} == {"root", "domain", "opening"}
    return table


def _inputs(held: dict, table: str, tok: int, row_of: int | None = None) -> tuple:
    """A unit's read inputs: its table's root and domain id, and the opening at ``tok`` (the row at ``row_of`` where given,
    a prover's wrong row); zeros past the table, which the read never opens."""
    tb, t = held[table], tok & 0xFFFFFFFF
    e = tb.entry()
    if t >= tb.leaves:
        return tok, bytes.fromhex(e["root"]), bytes.fromhex(e["domain"]), [0] * NE.WIDTH, ZERO_SALT, ZERO_PATH
    r = t if row_of is None else row_of
    path = bytes(tb.path(t))
    return (tok, bytes.fromhex(e["root"]), bytes.fromhex(e["domain"]), [int(x) for x in tb.rows[r]], bytes(tb.salts[r]),
            [path[k:k + 64] for k in range(0, len(path), 64)])


def _expected(cols: dict, block: int, tok: int) -> list[int]:
    """`EmbeddingShard_v1`'s words at block ``block``'s columns (its NumPy transcription, held to its reference evaluator
    by `node_window_values.native_selftest`): the shard's row where it owns the token, zeros elsewhere."""
    return [int(x) for x in cols[block][tok]] if 0 <= tok < VS else [0] * NE.WIDTH


def _units(window) -> list[tuple[int, int]]:
    """Each (unit, block) of the prefill's member 3 and the decode's member over blocks 0 and 1, by the frame's pieces."""
    circuit, w = window
    read = _tid(circuit, READ)
    out = []
    for j, m in ((0, 3), (20, 0)):
        (p,) = w.outs[j].pieces
        out += [(r.unit, b) for b, r in enumerate(p.rows(m, 0, 2 * NE.WIDTH, "test")) if r.tid == read]
    return out


@pytest.mark.parametrize("tok", [7, VS - 1, VS, 100000, V - 1], ids=["inside", "last-inside", "first-outside", "outside",
                                                                   "last-of-V"])
def test_values(window, tables, tok):
    """Each unit, fed from what the circuit declares for it (its table's root, domain id and the opening at its key), gives
    the shard's words at its block for a token the shard owns and zeros for one it doesn't, with verdict 1."""
    NV.native_selftest()
    circuit, _w = window
    cols, held = tables
    read = _tid(circuit, READ)
    for u, b in _units(window):
        h, ok = reference(V, *_inputs(held, _declared(circuit, read, u), tok))
        assert (h, ok) == (_expected(cols, b, tok), 1), (u, b)


def test_planted_errors_fail(window, tables):
    """A block declared over another block's table gives that block's words, and a wrong row at the key fails its opening:
    each differs from the shard's words or its verdict."""
    circuit, _w = window
    cols, held = tables
    read = _tid(circuit, READ)
    (u, b), = [(u, b) for u, b in _units(window) if b == 1][:1]
    other = NE.shard_param(LABEL, NE.WIDTH, 0)
    reg = circuit.registered
    swapped = dataclasses.replace(circuit, registered=dataclasses.replace(
        reg, roots=tuple(dataclasses.replace(x, table=other) if x.reader == read and x.at <= u < x.at + x.units else x
                         for x in reg.roots),
        openings=tuple(dataclasses.replace(o, table=other) if o.reader == read and o.at <= u < o.at + o.units else o
                       for o in reg.openings)))
    h, ok = reference(V, *_inputs(held, _declared(swapped, read, u), 7))
    assert ok == 1 and h == _expected(cols, 0, 7) != _expected(cols, b, 7)
    h, ok = reference(V, *_inputs(held, _declared(circuit, read, u), 7, row_of=8))
    assert ok == 0 and h != _expected(cols, b, 7)


@pytest.mark.slow
def test_values_boolean(window, tables):
    """The read's Boolean circuit itself on two units, a token inside the shard and one outside: the same words and verdicts
    as the reference the other value tests use. It catches a read that differs from its reference at this binding
    (V = 154,880, H = 192), which the cheaper tests take on trust; about 80 s for its 2 units."""
    circuit, _w = window
    cols, held = tables
    read = _tid(circuit, READ)
    fn = read_of(V, NE.WIDTH)
    (u, b), = _units(window)[:1]
    args = [arguments(V, *_inputs(held, _declared(circuit, read, u), tok)) for tok in (7, 100000)]
    got = np.asarray(evaluate_batch(fn, *(np.asarray([a[k] for a in args], np.uint8) for k in range(6)))).reshape(2, -1)
    for i, tok in enumerate((7, 100000)):
        words = [int(sum(int(got[i, 16 * k + j]) << j for j in range(16))) for k in range(NE.WIDTH)]
        assert (words, int(got[i, -1])) == (_expected(cols, b, tok), 1)


class _Weights:
    """The weights record's words where a test reads only the shard's field: its words at the field's first word."""

    def __init__(self, first: int, words: np.ndarray):
        self.first, self.words = first, words

    def __getitem__(self, s: slice) -> np.ndarray:
        assert (s.start, s.stop) == (self.first, self.first + len(self.words)), s
        return self.words


def test_values_tables(window, program):
    """The value check's `Tables` commits a column block's table from the made-up weights as the window declares it: the
    block's 192 columns of the shard's VS x 6,144 words, padded to V rows at START (`node_window_embed.table_words`)."""
    circuit, w = window
    first, words = {label: (f, n) for label, f, n in w.b.weights}[LABEL]
    assert words == VS * WIDTH
    stack = np.random.default_rng(274).integers(0, 0x7F80, words, dtype=np.uint16)
    tables = NV.Tables(circuit, NV.Values(program, {NW.WEIGHTS: _Weights(first, stack)}, w), w, PROGRAM)
    name = NE.shard_param(LABEL, NE.WIDTH, 1)
    tb = tables(name)
    want = NE.table_words(stack.reshape(VS, WIDTH)[:, NE.WIDTH:2 * NE.WIDTH], 0, V)
    assert tb.leaves == V and np.array_equal(np.asarray(tb.rows), want) and tables(name) is tb


def test_split_bnk_with_tables(window):
    """The absorption weights' entry (``weights/<label>@<K>/bnk``) lands with the tables, not among the plain weights: the
    plain weights and their nonces are the same with it and without it, and `tables_of` still gives the ``/vocab`` tables
    only."""
    params = list(window[0].registered.params)
    bnk = f"{NW.WEIGHTS}/model.layers.0.self_attn.kv_b_proj.weight@512/bnk"
    (plain, tables), (plain1, tables1) = NE.split(params), NE.split([*params, bnk])
    assert plain1 == plain and WW.weight_nonces(plain1) == WW.weight_nonces(plain)
    assert tables1 == sorted([*tables, bnk])
    assert NE.tables_of([*params, bnk]) == NE.tables_of(params)
