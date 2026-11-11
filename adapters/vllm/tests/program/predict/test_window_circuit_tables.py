"""verity_vllm.program.predict.window_circuit with H1's tables (`window_circuit.Tables`): the class circuit reads each round's
A table and input-ids table where the model makes them, each a read of a proof block's output row. Pinned on TINY's
exact-int8 checkpoint (`test_window_circuit_int8`) at `test_window_circuit`'s class (2 slots, 4 prompt ids, 2 outputs: 5
positions, H1's rounds 0 and 1) with H1's tables at K = 64, 8 rows a call and 6 calls a round (q/k/v, o and gate/up of
TINY's 2 layers).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, G2, WINDOW, _label, held, state_row
from tests.program.predict.test_window_circuit_int8 import GRAIN, _registered
from tests.program.predict.test_window_rows import KEY, PROGRAM, _weights
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import Tables, WindowClass
from verity_vllm.protocols.pouw.h1 import served as SV

from verity.protocols.pouw.verifier import credit
from verity.protocols.pouw.verifier.audit import hm96_salt
from verity.protocols.spot.verifier.one_stage import registered as CR
from verity.circuits.evaluation import evaluate_batch
from verity.commitments.windows import interface as IF

TABLES = Tables(64, 8, 6)
#: H1's leaf at K = 64: 16 u32 of codes, then the f32 scale, in u16 words
LEAF = 34
#: the fixed load at CLASS: two prompts of 4 at step 0, each decoded once at step 1
FIXED = ([(4, 2, G2), (4, 2, G2)], [[(0, 0, 4), (1, 0, 4)], [(0, 4, 1), (1, 4, 1)]])


def _lowered(tables: Tables | None = TABLES, cls: WindowClass = CLASS):
    return WC.lower(_registered(), cls, None, GRAIN, wired=False, tables=tables)


def _held_rows(circuit) -> dict[tuple[str, str, int], tuple[str, int]]:
    """Each output row a table read holds, ``(template, port, unit)``, as its table and row."""
    out = {}
    for r in circuit.registered.reads:
        if r.param.startswith(("row-table/", "input-ids/")):
            words = dict(circuit.layouts[r.reader][0].outs)[r.port]
            for i, leaf in enumerate(r.source(range(r.units))):
                assert leaf % words == 0, (r, leaf)
                out[(r.reader, r.port, r.at + i)] = (r.param, int(leaf) // words)
    return out


def test_the_tables_are_named_as_the_commit_record_registers_them():
    """A round's A table and its input-ids table are the Commit record's ``row-table/<s>/k<K>`` and ``input-ids/<s>``, the
    names PoUW's credit holds H1's tables to (`credit.commit_rows`, `credit.commit_ids`)."""
    assert [(TABLES.a(s), TABLES.ids(s)) for s in range(4)] == [(credit.commit_rows(s, 64), credit.commit_ids(s))
                                                                for s in range(4)]


def test_the_class_circuit_holds_each_quantizers_leaf_and_each_token_to_its_row_of_h1s_tables():
    """Each quantizer at K = 64 is `Int8RowQuant_v5{K, R, ROUND}` at its position's round (q/k/v's and gate/up's R = 1,
    o_proj's R = 4) and down_proj's at K = 128 stays `Int8RowQuant_v3`. The circuit registers each round's A table, 6 calls of 8 rows of H1's 34-word leaf,
    and its input-ids table, 8 rows of one u32. Slot k's position p is round 0's row 4k + p below the prompt and round 1's
    row k past it: each quantizer's leaf row ``o2`` is held to row 8c + i of its round's A table, c its index among its
    position's 6 in node order, and each position's token row (its lane's of the scan proof block at position 0, its
    `TokenFeed_v1`'s below the prompt, the previous select's past it) to row i of its round's input-ids table; no other
    output row is held to a table, and no two to one row. The reads fill no input, so the circuit's inputs by what fills
    them are the int8 circuit's without the tables. The circuit's program takes the tables as its values, round by round
    the A table then the input-ids table, ahead of its units' parameters."""
    low = _lowered()
    c = low.circuit
    assert {t: k for fn, k in c.templates if (t := _label(fn)).startswith("Int8RowQuant")} == {
        "Int8RowQuant_v5{K=64,R=1,ROUND=0}": 32, "Int8RowQuant_v5{K=64,R=4,ROUND=0}": 16,
        "Int8RowQuant_v5{K=64,R=1,ROUND=1}": 8, "Int8RowQuant_v5{K=64,R=4,ROUND=1}": 4, "Int8RowQuant_v3{K=128,R=1}": 20}
    assert {p: n for p, n in c.registered.params.items() if p.startswith(("row-table/", "input-ids/"))} == {
        "row-table/0/k64": 6 * 8 * LEAF, "row-table/1/k64": 6 * 8 * LEAF, "input-ids/0": 8 * 2, "input-ids/1": 8 * 2}
    assert c.registered.values == ("row-table/0/k64", "input-ids/0", "row-table/1/k64", "input-ids/1")
    assert [n for n, _ in c.program()[0].fn.signature()[0][:4]] == list(c.registered.values)
    pos, want = WR.positions(low.fn), {}
    for k, P in enumerate(low.programs):
        quants: dict[int, list[WC.Row]] = {}
        for j in sorted(P.packed):
            quants.setdefault(pos[j], []).append(P.packed[j])
        assert sorted(quants) == list(range(CLASS.positions)) and {len(q) for q in quants.values()} == {6}
        st, su = P.scan_unit
        for p in range(CLASS.positions):
            s, i = (0, 4 * k + p) if p < CLASS.prompt else (1, k)
            for c_, r in enumerate(quants[p]):
                want[(r.tid, "o2", r.unit)] = (f"row-table/{s}/k64", 8 * c_ + i)
            tok = ((st, f"o{4 * CLASS.lane(k)}", su) if p == 0 else (P.tokens[p][0], "o0", P.tokens[p][1]) if p < CLASS.prompt
                   else (P.collects[p - 1][0], "o0", P.collects[p - 1][1]))
            want[tok] = (f"input-ids/{s}", i)
    got = _held_rows(c)
    assert got == want and len(set(got.values())) == len(got) == 2 * 5 * 7
    doc = json.loads(json.dumps(c.description()))
    assert IF.input_count(doc, **held(c)) == {
        "units": 1568, "inputs": 10547104, "wired": 0, "registered": 9161856, "ingress": 0, "free": 1385248}


def test_each_tables_readers_are_templates_of_their_own():
    """A sample stratified by template samples a unit from every template, so each table has a sampled reader when its readers
    are templates of their own: each template that holds a table row holds one on each of its units, all of one table, and
    every table has one. Round s's quantizers bind ROUND = s; round 0's token rows are its `ScanSlots_v2` and `TokenFeed_v1`
    units' and round 1's its selects', `SelectCollect_v2{..., ROUND=1}`. Without the tables no template binds a round."""
    c = _lowered().circuit
    units, tables, held_units = {IF.descriptor_id(fn): n for fn, n in c.templates}, {}, {}
    for (tid, _port, u), (table, _row) in _held_rows(c).items():
        tables.setdefault(tid, set()).add(table)
        held_units.setdefault(tid, set()).add(u)
    assert all(len(ts) == 1 and held_units[t] == set(range(units[t])) for t, ts in tables.items())
    readers = sorted((t.split("{")[0], *ts, "ROUND=1" in t) for t, ts in tables.items())
    assert readers == [("Int8RowQuant_v5", "row-table/0/k64", False)] * 2 + [("Int8RowQuant_v5", "row-table/1/k64", True)] * 2 + [
        ("ScanSlots_v2", "input-ids/0", False), ("SelectCollect_v2", "input-ids/1", True), ("TokenFeed_v1", "input-ids/0", False)]
    assert {s for t, ts in tables.items() for s in ts} == {p for p in c.registered.params
                                                            if p.startswith(("row-table/", "input-ids/"))}
    assert not any("ROUND=" in IF.descriptor_id(fn) for fn, _ in _lowered(None).circuit.templates)


def test_a_fixed_load_window_is_the_class_circuit_with_the_tables_and_any_other_window_is_refused():
    """The fixed load's window (`Tables.schedule`: both prompts of 4 at step 0, each decoded at step 1) is the class circuit
    with H1's tables, wired; `test_window_circuit`'s WINDOW (a prompt of 3 chunked across two steps) is refused by name,
    since its rows aren't the public rows the tables are read at."""
    reg = _registered()
    fixed = WC.window_circuit(reg, S.compile_steps(reg, *S._session(*FIXED)), CLASS, GRAIN, provable=False, tables=TABLES)
    assert IF.circuit_bytes(fixed.description()) == IF.circuit_bytes(
        WC.class_circuit(reg, CLASS, None, GRAIN, provable=False, tables=TABLES).description())
    assert fixed.wires and {r.param for r in fixed.registered.reads} >= {"row-table/1/k64", "input-ids/1"}
    with pytest.raises(Unexpressible, match=r"not its class's fixed load \[\[\(0, 0, 4\), \(1, 0, 4\)\], \[\(0, 4, 1\), "
                                            r"\(1, 4, 1\)\]\], the rows H1's tables are read at"):
        WC.window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN, provable=False, tables=TABLES)


@pytest.fixture(scope="module")
def filled():
    """The class circuit with H1's tables filled from a window's values (`window_rows.Rows`: `test_window_rows`' random
    weights, a scan state whose row at bucket 0 carries two prompts of 4): its lowering, its slots and its rows."""
    reg = _registered()
    low = WC.lower(reg, CLASS, None, GRAIN, provable=False, tables=TABLES)
    fields = _weights(low.fn)
    lanes = [(1, 4, 2, [17, 3, 250, 9]), (1, 4, 2, [5, 77, 42, 8])]
    rows_ = [state_row(CLASS.scan, lanes)] + [state_row(CLASS.scan, [(1, 1, 1, [99, 0, 0, 0])] * 2)] * (CLASS.buckets - 1)
    state = CR.Registered.commit(PROGRAM, SS.SCAN_STATE, rows_, [hm96_salt(bytes(32), 0, j) for j in range(len(rows_))],
                                 schema=CR.SCHEMA_V2)
    values = {**WW.weight_values(reg, fields, KEY, GRAIN, keep_rows=True), SS.SCAN_STATE: state}
    slots = [WR.Slot(low.fn, CLASS, row, lambda label, lo, hi: fields[label][lo:hi]) for row, _ in WR.scan_slots(low, rows_)]
    return low, slots, WR.Rows(low, slots, values)


def test_a_quantizers_rows_are_its_definition_on_its_x_and_its_leaf_row_its_nodes_words_whole(filled):
    """Filled from a window's values (`filled`), each `Int8RowQuant_v5` proof block's leaf row ``o2`` is its node's words
    whole, its codes ``o0`` then its scale ``o1``, and its three rows are its Definition on the node's ``x``, as its proof
    holds them."""
    low, slots, rows = filled
    seen = 0
    for P, s in zip(low.programs, slots):
        for j, r in P.packed.items():
            o0, o1, o2 = (rows.outs[r.tid][k][r.unit] for k in range(3))
            want = np.asarray(evaluate_batch(rows.fns[r.tid], s.operand(low.fn.body.nodes[j].args[0])[None])).reshape(-1)
            assert np.array_equal(o2, np.concatenate([o0, o1])) and np.array_equal(o2, WR._words(s.values[j]))
            assert np.array_equal(np.concatenate([o0.view("<u1"), o1.view("<u4"), o2.view("<u4")]), want)
            seen += 1
    assert seen == 2 * 5 * 6


def _h1_tables(low, slots) -> dict[str, np.ndarray]:
    """H1's tables of the fixed load from the model's values: round s's A table's row 8c + i the leaf (codes, then scale) of
    the c-th quantizer at K = 64 of the position at round s's row i (`Tables.at`), every other row zero codes of scale 0;
    its input-ids table's row i that position's token, every other `EMPTY_ID` (`pouw_h1_served.ids_words`)."""
    pos, a, ids = WR.positions(low.fn), {s: np.zeros((6 * 8, LEAF), np.uint16) for s in (0, 1)}, {0: [None] * 8, 1: [None] * 8}
    for k, (P, sl) in enumerate(zip(low.programs, slots)):
        quants: dict[int, list[int]] = {}
        for j in sorted(P.packed):
            quants.setdefault(pos[j], []).append(j)
        for p in range(CLASS.positions):
            s, i = Tables.at(CLASS, k, p)
            for c, j in enumerate(quants[p]):
                a[s][8 * c + i] = WR._words(sl.values[j])
            ids[s][i] = sl.tokens[p]
    return {**{TABLES.a(s): a[s] for s in a}, **{TABLES.ids(s): SV.ids_words(ids[s], 8) for s in ids}}


def test_a_tables_rows_are_the_output_rows_its_proof_blocks_make(filled):
    """`Rows.value` gives each of H1's tables as the class circuit's proof blocks make it: round s's A table each
    quantizer's leaf row at row 8c + i and its input-ids table each position's token at row i, the fixed load's rows, and
    every row no proof block makes the empty row given. It refuses a table read off the rows given, or as another width."""
    low, slots, rows = filled
    for name, want in _h1_tables(low, slots).items():
        empty = np.zeros(LEAF, np.uint16) if name.startswith("row-table/") else SV.ids_words([], 1)[0]
        assert np.array_equal(rows.value(name, len(want), empty), want), name
    with pytest.raises(ValueError, match=r"reads 'row-table/1/k64' off its 8 rows of 34 words"):
        rows.value(TABLES.a(1), 8, np.zeros(LEAF, np.uint16))
    with pytest.raises(ValueError, match="reads 'input-ids/0' other than as an output row of 3 words"):
        rows.value(TABLES.ids(0), 8, [0, 0, 0])


def test_two_proof_blocks_making_one_row_two_ways_are_refused():
    """`Rows.value` takes a row two proof blocks make alike and refuses one they make two ways."""
    def rows(a, b):
        reads = [SimpleNamespace(param="t", reader=t, port="o", units=1, at=0, source=lambda u: 2 * u) for t in "ab"]
        return SimpleNamespace(circuit=SimpleNamespace(registered=SimpleNamespace(reads=reads)),
                               lays={t: SimpleNamespace(outs=[("o", 2)]) for t in "ab"},
                               outs={"a": [np.array([a], np.uint16)], "b": [np.array([b], np.uint16)]})
    assert np.array_equal(WR.Rows.value(rows([1, 2], [1, 2]), "t", 2, [0, 0]), [[1, 2], [0, 0]])
    with pytest.raises(ValueError, match="'t': two units make one of its rows two ways"):
        WR.Rows.value(rows([1, 2], [1, 3]), "t", 2, [0, 0])


@pytest.mark.parametrize("tables, cls, why", [
    (TABLES, WindowClass(2, 4, 2, lanes=1), "read at a fixed load's rows, every slot at the window's one boundary"),
    (Tables(64, 7, 6), CLASS, "round 0 its 2 slots' prompts of 4, at most 7 rows"),
    (Tables(64, 8, 5), CLASS, r"int8 linears of K = 64 by position \[6, 6, 6, 6, 6\], not H1's 5 calls a round"),
    (Tables(128, 8, 6), CLASS, r"int8 linears of K = 128 by position \[2, 2, 2, 2, 2\], not H1's 6 calls a round")])
def test_tables_the_class_cant_be_read_at_are_refused(tables, cls, why):
    """H1's tables are refused by name for a class of two boundaries (its slots aren't all admitted at round 0), a round 0 of
    more rows than a table's rows a call, and a position whose int8 linears of the tables' K aren't the tables' calls a
    round (TINY has 6 at K = 64, 2 at K = 128)."""
    with pytest.raises(Unexpressible, match=why):
        _lowered(tables, cls)


def test_tables_are_positive_integers():
    with pytest.raises(ValueError, match="positive integers"):
        Tables(64, 0, 6)
