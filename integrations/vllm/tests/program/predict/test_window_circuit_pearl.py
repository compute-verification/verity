"""verity_vllm.program.predict.window_circuit with Pearl+'s tables (`window_class.PearlTables`): the class circuit cut at its
linears, each linear but the LM head one `LinearARow_v1` unit held to its row of its round's A table of its K and
`LinearOut_v1` units taking its y from its round's y value of its K, and `window_rows` filling them. Pinned on TINY's bf16
checkpoint on the H200 (`test_window_rows`) at `test_window_circuit`'s class (2 slots, 4 prompt ids, 2 outputs: 5
positions, rounds 0 and 1) and grain, Pearl+'s y in rows of 16 words. A position's linears are q/k/v (N = 128), o (64) and
gate/up (256) at K = 64 and down (64) at K = 128, of each of TINY's 2 layers; its LM head (`Gemm_v2{64, 256}`, gate/up's
shape) stays computed.
"""
from __future__ import annotations

import re

import numpy as np
import pytest
from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import (CLASS, GRAIN, GROUPS, PROVABLE, PROVABLE_CLASS, PROVABLE_GRAIN, WINDOW,
                                                       _label, _source, held, state_row)
from tests.program.predict.test_window_circuit_int8 import GRAIN as INT8_GRAIN
from tests.program.predict.test_window_circuit_int8 import _registered as _int8
from tests.program.predict.test_window_circuit_tables import FIXED
from tests.program.predict.test_window_rows import KEY, LAW, PROGRAM, RUN, _read, _registered, _served, _weights
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_class import PearlTables, WindowClass

from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.pouw.schemes import SCHEMES
from verity.core.protocols.sampled_proofs.one_stage import registered as CR
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.windows import interface as IF

PEARL = PearlTables(words=16)
#: each position's linears' N by K, in node order: layer 0's q/k/v, o and gate/up, then layer 1's, and their down_projs
CALLS = {64: (128, 64, 256, 128, 64, 256), 128: (64, 64)}
#: a call's rows by round: round 0 both slots' prompts of 4, round 1 a row a slot
M = {0: 8, 1: 2}


@pytest.fixture(scope="module")
def low():
    return WC.lower(_registered(), CLASS, None, GRAIN, provable=False, tables=PEARL)


def _value_rows(circuit) -> dict[tuple[str, str, int], tuple[str, int]]:
    """Each row a read of Pearl+'s tables holds or fills, ``(template, port, unit)``, as its value and row."""
    out = {}
    for r in circuit.registered.reads:
        if r.param.startswith("pearl-"):
            lay = circuit.layouts[r.reader][0]
            words = dict(lay.rows + lay.outs)[r.port]
            for i, leaf in enumerate(r.source(range(r.units))):
                assert leaf % words == 0, (r, leaf)
                out[(r.reader, r.port, r.at + i)] = (r.param, int(leaf) // words)
    return out


def _calls(low) -> list[tuple[int, int, int, int, int]]:
    """Each slot's cut linears in node order: ``(slot, node, position, K, c)``, c its index among its position's of that K."""
    pos, nodes, out = WR.positions(low.fn), low.fn.body.nodes, []
    for k, P in enumerate(low.programs):
        seen: dict[tuple[int, int], int] = {}
        for j in sorted(P.arows):
            p, K = pos[j], nodes[j].fn.bindings["K"]
            seen[(p, K)] = seen.get((p, K), -1) + 1
            out.append((k, j, p, K, seen[(p, K)]))
    return out


def test_pearls_tables_are_named_by_round_and_k_and_a_calls_rows_are_its_rounds():
    """Round s's A table and y value of contraction K are ``pearl-a/<s>/k<K>`` and ``pearl-y/<s>/k<K>``, and a call's rows
    under the fixed load are round 0's every slot's prompt and a later round's a row a slot. Pearl+'s y rows are a positive
    number of words, and its scheme one of PoUW's."""
    assert (PEARL.a(1, 64), PEARL.y(0, 128)) == ("pearl-a/1/k64", "pearl-y/0/k128")
    assert [PEARL.rows(CLASS, s) for s in range(3)] == [8, 2, 2] and PEARL.scheme in SCHEMES
    with pytest.raises(ValueError, match="a positive number of words, not 0"):
        PearlTables(words=0)
    with pytest.raises(ValueError, match="'pearl-x' is no PoUW scheme"):
        PearlTables(scheme="pearl-x")


def test_the_cut_replaces_every_linear_but_the_lm_head(low):
    """Each linear is one `LinearARow_v1{K, R, ROUND}` unit, its x its writers' R rows (q/k/v's and gate/up's the norm's
    one, o's the attention's 4 heads', down's SiluMul's one), and N / 16 `LinearOut_v1{16, K, ROUND}` units, a template a
    round and K, and no `Gemm_v2` but the LM head's: its 16 slices a position, all of which the token selects' group units
    read, and its ``x0``, though gate/up is its shape. Every other template is the uncut circuit's at its count, and no
    input is free."""
    c = low.circuit
    n = {_label(fn): k for fn, k in c.templates}
    plain = {_label(fn): k for fn, k in WC.lower(_registered(), CLASS, None, GRAIN, provable=False).circuit.templates}
    want = {"Gemm_v2{K=64,N=16}": 16 * 10, "Gemm_v2{K=64,N=1}": 10}
    for s, m in M.items():
        want |= {f"LinearARow_v1{{K=64,R=1,ROUND={s}}}": 4 * m, f"LinearARow_v1{{K=64,R=4,ROUND={s}}}": 2 * m,
                 f"LinearARow_v1{{K=128,R=1,ROUND={s}}}": 2 * m, f"LinearOut_v1{{N=16,K=64,ROUND={s}}}": 56 * m,
                 f"LinearOut_v1{{N=16,K=128,ROUND={s}}}": 8 * m}
    assert {t: k for t, k in n.items() if t.startswith(("Linear", "Gemm"))} == want
    assert {t: k for t, k in n.items() if not t.startswith(("Linear", "Gemm"))} == {
        t: k for t, k in plain.items() if not t.startswith("Gemm")}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in c.templates}
    g, cut = ids[GROUPS], c.cuts.get(ids[GROUPS], {}).get("x", 1)
    heads = {_source(c, g, "x", u, r)[:2] for u in range(n[GROUPS]) for r in range(cut)}
    assert {t for t, _ in heads} == {ids["Gemm_v2{K=64,N=16}"]} and len(heads) == 16 * 10
    assert IF.input_count(c.description(), **held(c))["free"] == 0


def test_each_cut_linear_reads_its_a_row_and_its_y_at_the_fixed_loads_rows(low):
    """Slot k's position p is round 0's row 4k + p below the prompt and round 1's row k past it (`FixedLoad.at`). Each cut
    linear's A row (``o0``, x widened) is held to row c·m + i of its round's A table of its K, c its index among its
    position's linears of that K in node order, and its y units' ``y`` read the N / 16 rows from o·m + i·N / 16 of its
    round's y value of its K, o the rows of 16 its position's earlier linears of that K make: each call's m × N y
    row-major, call after call. Every row of every value is read by exactly one unit, and the circuit's program takes them
    as its values, round by round and K by K the A table then the y value, ahead of its units' parameters."""
    c, want = low.circuit, {}
    for k, j, p, K, ci in _calls(low):
        P, N = low.programs[k], low.fn.body.nodes[j].fn.bindings["N"]
        s, i = (0, 4 * k + p) if p < CLASS.prompt else (1, k)
        a, ((_lo, _hi, y),) = P.arows[j], P.outs[j]
        want[(a.tid, "o0", a.unit)] = (f"pearl-a/{s}/k{K}", ci * M[s] + i)
        o, q = sum(CALLS[K][:ci]) // 16, N // 16
        want |= {(y.tid, "y", y.unit + r): (f"pearl-y/{s}/k{K}", o * M[s] + i * q + r) for r in range(q)}
    got = _value_rows(c)
    assert got == want
    rows: dict[str, list[int]] = {}
    for v, r in got.values():
        rows.setdefault(v, []).append(r)
    words = {v: 2 * int(v.rsplit("k", 1)[1]) if v.startswith("pearl-a/") else 16 for v in rows}
    assert all(sorted(r) == list(range(c.registered.params[v] // words[v])) for v, r in rows.items())
    assert c.registered.values == tuple(v for s in M for K in CALLS for v in (f"pearl-a/{s}/k{K}", f"pearl-y/{s}/k{K}"))
    assert [n for n, _ in c.program()[0].fn.signature()[0][:8]] == list(c.registered.values)


def test_each_tables_readers_are_templates_of_their_own(low):
    """A sample stratified by template draws a unit from every template, so each value has a drawn reader: each template
    that reads one reads it on every unit and reads no other, and every value has readers."""
    c = low.circuit
    units, values, read = {IF.descriptor_id(fn): n for fn, n in c.templates}, {}, {}
    for (tid, _port, u), (v, _row) in _value_rows(c).items():
        values.setdefault(tid, set()).add(v)
        read.setdefault(tid, set()).add(u)
    assert all(len(vs) == 1 and read[t] == set(range(units[t])) for t, vs in values.items())
    assert {v for vs in values.values() for v in vs} == set(c.registered.values)


def test_the_census_counts_each_cut_linear_by_shape_and_ties_none(low):
    """`census` is a row a round's call: its round, its index among the round's calls of its K, K and N, the round's rows
    and its multiply-adds, its rows of the A table and of the y value, and PoUW's scheme; none is tied (`UNTIED`). Its
    multiply-adds are the circuit description's own, its `LinearOut_v1{W, K, ROUND}` units times W·K, and an uncut
    circuit has no census."""
    cen = WC.census(low)
    assert [(r["round"], r["call"], r["k"], r["n"], r["rows"]) for r in cen] == [
        (s, ci, K, N, M[s]) for s in M for K, ns in CALLS.items() for ci, N in enumerate(ns)]
    for r in cen:
        m, o = r["rows"], sum(CALLS[r["k"]][:r["call"]]) // 16
        assert r["macs"] == m * r["n"] * r["k"] and (r["scheme"], r["tied"], r["untied"]) == (PEARL.scheme, False, WC.UNTIED)
        assert r["a"] == {"value": PEARL.a(r["round"], r["k"]), "rows": [r["call"] * m, (r["call"] + 1) * m]}
        assert r["y"] == {"value": PEARL.y(r["round"], r["k"]), "rows": [o * m, (o + r["n"] // 16) * m]}
    assert sum(r["macs"] for r in cen) == sum(n * fn.bindings["N"] * fn.bindings["K"] for fn, n in low.circuit.templates
                                               if fn.name == "LinearOut") == 737280
    assert WC.census(WC.lower(_registered(), CLASS, None, GRAIN, wired=False)) == []


def test_a_fixed_load_window_is_the_cut_class_circuit_and_any_other_window_is_refused():
    """The fixed load's window (both prompts of 4 at step 0, each decoded at step 1) is the cut class circuit;
    `test_window_circuit`'s WINDOW (a prompt of 3 chunked across two steps) is refused by name, since its rows aren't the
    public rows Pearl+'s tables are read at."""
    reg = _registered()
    fixed = WC.window_circuit(reg, S.compile_steps(reg, *S._session(*FIXED)), CLASS, GRAIN, provable=False, tables=PEARL)
    assert IF.circuit_bytes(fixed.description()) == IF.circuit_bytes(
        WC.class_circuit(reg, CLASS, None, GRAIN, provable=False, tables=PEARL).description())
    with pytest.raises(Unexpressible, match=r"not its class's fixed load .*, the rows Pearl\+'s tables are read at"):
        WC.window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN, provable=False, tables=PEARL)


@pytest.mark.parametrize("registered, grain, tables, cls, why", [
    (_int8, INT8_GRAIN, PEARL, CLASS, r"Int8RowQuant_v1\{K=64[,}].*: an int8 linear, and Pearl\+'s tables cut BF16 linears"),
    (_registered, GRAIN, PEARL, WindowClass(2, 4, 2, lanes=1), "Pearl\\+'s tables are read at a fixed load's rows, every "
                                                               "slot at the window's one boundary"),
    (_registered, GRAIN, PearlTables(words=48), CLASS, r"linears \(K, N\) \[\(64, 64\), \(64, 128\), \(64, 256\), "
                                                       r"\(128, 64\)\], whose y isn't rows of Pearl\+'s 48 words")])
def test_a_circuit_pearls_tables_cant_cut_is_refused(registered, grain, tables, cls, why):
    """Refused by name: an int8 linear (TINY's exact-int8 checkpoint), a class of two boundaries (its slots aren't all
    admitted at round 0), and y rows of a width that doesn't divide every linear's N."""
    with pytest.raises(Unexpressible, match=why):
        WC.lower(registered(), cls, None, grain, wired=False, tables=tables)


def test_the_cut_at_the_smallest_provable_class_is_proof_blocks_c_flock_proves():
    """At `test_window_circuit`'s smallest provable model, class and grain, with y rows of 64 words (128 bytes, as at
    Llama 3.2 1B's head dimension), every template of the cut circuit is a proof block C-Flock's composer takes
    (`interface.check_proof_block`): the A rows (256 and 512 words) and the y rows are past 111 bytes."""
    c = WC.class_circuit(S._registered(PROVABLE), PROVABLE_CLASS, None, PROVABLE_GRAIN, tables=PearlTables(words=64))
    n = {_label(fn): k for fn, k in c.templates}
    P = PROVABLE_CLASS.positions * PROVABLE_CLASS.slots
    assert {t: k for t, k in n.items() if t.startswith(("Linear", "Gemm"))} == {
        "LinearARow_v1{K=128,R=1,ROUND=0}": 4 * P, "LinearARow_v1{K=128,R=2,ROUND=0}": 2 * P,
        "LinearARow_v1{K=256,R=1,ROUND=0}": 2 * P, "LinearOut_v1{N=64,K=128,ROUND=0}": 28 * P,
        "LinearOut_v1{N=64,K=256,ROUND=0}": 4 * P, "Gemm_v2{K=128,N=64}": 8 * P, "Gemm_v2{K=128,N=1}": P}
    assert all(IF.check_proof_block(fn, c.cuts.get(IF.descriptor_id(fn))) <= 27 for fn, _ in c.templates)


#: both slots' lanes of the scan state row at bucket 0: the fixed load's prompts of 4, each keeping 2 outputs
LANES = [(1, 4, 2, [17, 3, 250, 9]), (1, 4, 2, [5, 77, 42, 8])]
STATE = [state_row(CLASS.scan, LANES)] + [state_row(CLASS.scan, [(1, 1, 1, [99, 0, 0, 0])] * 2)] * (CLASS.buckets - 1)


@pytest.fixture(scope="module")
def filled(low):
    """The cut class circuit filled from a window's values (`window_rows.Rows`: `test_window_rows`' random weights, the fixed
    load's scan state, each slot's `Slot` taking the circuit's cut linears as registered): its weights, registered values,
    slots and rows."""
    fields = _weights(low.fn)
    state = CR.Registered.commit(PROGRAM, SS.SCAN_STATE, STATE, [hm96_salt(bytes(32), 0, j) for j in range(len(STATE))],
                                 schema=CR.SCHEMA_V2)
    values = {**WW.weight_values(_registered(), fields, KEY, GRAIN, keep_rows=True), SS.SCAN_STATE: state}

    def weights(label: str, lo: int, hi: int) -> np.ndarray:
        return fields[label][lo:hi]

    slots = [WR.Slot(low.fn, CLASS, row, weights, cut=low.cut) for row, _ in WR.scan_slots(low, STATE)]
    return weights, values, slots, WR.Rows(low, slots, values)


def _widened(x: np.ndarray) -> np.ndarray:
    """bf16 words ``x`` widened to FP32, as u16 words."""
    return (np.asarray(x, np.uint32) << 16).astype("<u4").view("<u2")


def test_a_cut_linears_a_row_is_its_x_widened_and_its_y_units_hold_its_value(low, filled):
    """Each cut linear's `LinearARow_v1` row is its Definition on the node's x, x's bf16 widened to FP32 exactly, and its
    `LinearOut_v1` units' output rows are the node's value, which their input rows are too (the rows the window commits by
    reference from the y value)."""
    _w, _v, slots, rows = filled
    for k, j, *_ in _calls(low):
        P, s = low.programs[k], slots[k]
        x, a, ((_lo, _hi, y),) = s.operand(low.fn.body.nodes[j].args[0]), P.arows[j], P.outs[j]
        got = rows.outs[a.tid][0][a.unit]
        assert np.array_equal(got, _widened(x))
        assert np.array_equal(got, np.asarray(evaluate_batch(rows.fns[a.tid], x[None])).astype("<u4").view("<u2").reshape(-1))
        q, w = len(WR._words(s.values[j])) // 16, WR._words(s.values[j])
        assert np.array_equal(rows.outs[y.tid][0][y.unit:y.unit + q].reshape(-1), w)
        assert np.array_equal(rows.given[y.tid]["y"][y.unit:y.unit + q].reshape(-1), w)
    assert len(_calls(low)) == 2 * 5 * 8 and len(low.cut) == 5 * 8


def _pearl_values(low, slots) -> dict[str, np.ndarray]:
    """Pearl+'s tables of the fixed load from the model's values: round s's A table's row c·m + i the c-th linear of K of
    the position at round s's row i (`FixedLoad.at`), its x widened, and its y value's N / 16 rows from o·m + i·N / 16 that
    linear's value."""
    out: dict[str, np.ndarray] = {}
    for k, j, p, K, ci in _calls(low):
        s, i = PEARL.at(CLASS, k, p)
        m, N, sl = M[s], low.fn.body.nodes[j].fn.bindings["N"], slots[k]
        a = out.setdefault(PEARL.a(s, K), np.zeros((len(CALLS[K]) * m, 2 * K), np.uint16))
        a[ci * m + i] = _widened(sl.operand(low.fn.body.nodes[j].args[0]))
        y = out.setdefault(PEARL.y(s, K), np.zeros((sum(CALLS[K]) // 16 * m, 16), np.uint16))
        o, q = sum(CALLS[K][:ci]) // 16, N // 16
        y[o * m + i * q:o * m + (i + 1) * q] = WR._words(sl.values[j]).reshape(q, 16)
    return out


def test_pearls_values_are_the_rows_the_cut_linears_make(low, filled):
    """`Rows.value` gives each of Pearl+'s values as the cut circuit's proof blocks make it: round s's A table each cut
    linear's A row, and its y value each one's y, at the fixed load's rows; every row is some unit's. It refuses a y value
    read as another width."""
    _w, _v, slots, rows = filled
    want = _pearl_values(low, slots)
    assert sorted(want) == sorted(low.circuit.registered.values)
    for name, v in want.items():
        assert np.array_equal(rows.value(name, len(v), np.full(v.shape[1], 0xFFFF, np.uint16)), v), name
    with pytest.raises(ValueError, match="reads 'pearl-y/0/k64' other than as an output row of 8 words or an input row a "
                                         "node's value gives"):
        rows.value(PEARL.y(0, 64), 896, np.zeros(8, np.uint16))


def test_a_cut_linear_served_under_other_statics_is_taken_as_served(low, filled):
    """Pearl+ serves each linear's y by its own product, not the BF16 GEMM's, so a cut linear's served value is its y
    whatever the statics it was served under: served under another ``DOT`` and off the GEMM's value, a slot of the cut
    circuit (`Lowered.cut`) takes it as served, and one of the uncut circuit refuses it by name, holding it to its class
    node's value."""
    weights, _v, slots, _rows = filled
    s0, j = slots[0], min(low.cut)
    rows, store = _served(low, s0, T=1)
    op = f"op{j}"
    rows = [{**d, "spec": re.sub(r'(?<=DOT=\{"fn":")\w+', "PearlPlus", d["spec"])} if d["op_path"] == op else d for d in rows]
    assert all(d["spec"] != descriptor_id(low.fn.body.nodes[j].fn) for d in rows if d["op_path"] == op)
    store[(0, op, "out")] = store[(0, op, "out")] ^ np.uint16(1)
    got = WR.Slot(low.fn, CLASS, s0.row, weights, WR.Served(rows, *_read(store)), cut=low.cut)
    assert got.source[j] == WR.SERVED and np.array_equal(got.values[j][0], s0.values[j][0] ^ np.uint16(1))
    with pytest.raises(ValueError, match=rf"node {j} .* at position 0: served as .*not its Definition's value on its "
                                         r"operands \(served as Gemm_v2\{.*PearlPlus"):
        WR.Slot(low.fn, CLASS, s0.row, weights, WR.Served(rows, *_read(store)))


@pytest.mark.slow
def test_a_windows_cut_units_are_their_definitions_on_the_inputs_it_commits(low, filled):
    """Recorded into a `Window` with Pearl+'s values registered (`Rows.value`'s rows under the committer's key), each
    `LinearARow_v1` and `LinearOut_v1` unit's output rows are its Definition on the inputs the window commits, x by its
    writers' wires and y by reference from the y value's rows, and the window closes over them. What no cheaper test
    reaches: the window resolving the cut's wires and committing a registered read of an input row by reference. Slow (a
    minute and more) for the window's members, which trace the embedding read's population Program."""
    _w, values, slots, rows = filled
    c = low.circuit
    for i, (name, v) in enumerate(_pearl_values(low, slots).items()):
        lay = {r.reader: IF.layout(c.template(r.reader)) for r in c.registered.reads if r.param == name}
        (schema,) = {dict(lay[r.reader].schemas)[r.port] for r in c.registered.reads if r.param == name}
        values = {**values, name: CR.Registered.commit(PROGRAM, name, v.tolist(),
                                                       [hm96_salt(KEY, (1 << 61) + i, j) for j in range(len(v))],
                                                       schema=schema)}
    win = IF.Window(c, RUN, LAW, 64, registered=values, forms=WR.FORMS)
    win.supply(CR)
    rows.record(win)
    bad = {}
    for fn, n in c.templates:
        if fn.name in ("LinearARow", "LinearOut"):
            tid = descriptor_id(fn)
            ins = [np.ascontiguousarray(x).view("<u2").reshape(n, -1) for x in win.input_words(tid, 0, n)]
            want = np.asarray(evaluate_batch(fn, *ins, kernel="reference")).reshape(n, -1)
            want = want.astype("<u4").view("<u2").reshape(n, -1) if fn.name == "LinearARow" else want.astype(np.uint16)
            if (u := np.flatnonzero((want != rows.outs[tid][0]).any(axis=1))).size:
                bad[tid] = u.tolist()
    assert bad == {}
    assert win.close(salt_key=bytes(32)) is not None
