"""verity_vllm.program.predict.window_circuit with Pearl+'s tables (`window_class.PearlTables`): the class circuit cut at its
linears, each linear but the LM head round s's call c, one `LinearARow_v1` unit held to its row of the call's A rows and one
`LinearOut_v1` unit reading its row of the call's y and handing it on in rows of W, and `window_rows` filling them with
the rows Pearl+'s call commits. Pinned on TINY's bf16 checkpoint on the H200 (`test_window_rows`) at `test_window_circuit`'s
class (2 slots, 4 prompt ids, 2 outputs: 5 positions, rounds 0 and 1) and grain, y handed on in rows of 16 words. A
position's cut linears are, of each of TINY's 2 layers, q/k/v (K = 64, N = 128), o (64, 64), gate/up (64, 256) and down
(128, 64), calls 0 to 7 in node order; its LM head (`Gemm_v2{64, 256}`, gate/up's shape) stays computed.
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
from verity_vllm.program.predict import window_cut as WK
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict import window_weights as WW
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_builder import Row
from verity_vllm.program.predict.window_class import PearlTables, WindowClass
from verity_vllm.protocols.pouw.pearl_c import ROW_ALIGN
from verity_vllm.protocols.pouw.pearl_c import served_y as SY

from verity.protocols.pouw.verifier.audit import hm96_salt
from verity.protocols.pouw.prover.schemes import SCHEMES
from verity.protocols.spot.verifier.one_stage import registered as CR
from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.types import Array, Value
from verity.commitments.windows import interface as IF

#: the scheme the remote demo's Pearl+ serves under, on the H200
SCHEME = "pearl-c-h100-v1-h3"
PEARL = PearlTables(SCHEME, words=16)
#: each position's cut linears' K and N, calls 0 to 7 in node order: each layer's q/k/v, o, gate/up and down
CALLS = ((64, 128), (64, 64), (64, 256), (128, 64)) * 2
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


def _calls(low) -> list[tuple[int, int, int, int]]:
    """Each slot's cut linears in node order: ``(slot, node, position, c)``, c its call, its index among its position's."""
    pos, out = WR.positions(low.fn), []
    for k, P in enumerate(low.programs):
        seen: dict[int, int] = {}
        for j in sorted(P.arows):
            seen[pos[j]] = seen.get(pos[j], -1) + 1
            out.append((k, j, pos[j], seen[pos[j]]))
    return out


def test_pearls_tables_are_named_by_round_and_call_at_the_rows_its_calls_commit():
    """Round s's call c's A rows and y are ``pearl-a/<s>/<c>`` and ``pearl-y/<s>/<c>``. A call's rows under the fixed load
    are round 0's every slot's prompt and a later round's a row a slot, and its A rows those then zero rows up to a multiple
    of Pearl+'s `ROW_ALIGN` (its m₁₆). Its y is handed on in rows of a positive number of words, and its scheme is one of
    PoUW's."""
    assert (PEARL.a(1, 3), PEARL.y(0, 5)) == ("pearl-a/1/3", "pearl-y/0/5")
    assert [PEARL.rows(CLASS, s) for s in range(3)] == [8, 2, 2] and PEARL.scheme in SCHEMES
    assert [PEARL.a_rows(cls, 0) for cls in (CLASS, WindowClass(4, 4, 2), WindowClass(2, 9, 2))] == [16, 16, 32] == [
        -(-m // ROW_ALIGN) * ROW_ALIGN for m in (8, 16, 18)]
    with pytest.raises(ValueError, match="a positive number of words, not 0"):
        PearlTables(SCHEME, words=0)
    with pytest.raises(ValueError, match="'pearl-x' is no PoUW scheme"):
        PearlTables(scheme="pearl-x")


def test_a_calls_y_is_handed_on_in_the_fewest_doublings_of_words_its_unit_fits(low):
    """Each call's W is the least of ``words`` doubled that divides its N and leaves its `LinearOut_v1` (its y row and
    N / W output rows) within the grain's rows: 16 for every call of TINY at 256 rows, gate/up's 256 words in rows of 32 at
    12 rows and of 128 at 3, and a call no W fits is refused by name."""
    assert WK.linears(low.fn, 16, 256, provable=False).calls == tuple((K, N, 16) for K, N in CALLS)
    assert WK.linears(low.fn, 16, 12, provable=False).calls == tuple((K, N, 32 if N == 256 else 16) for K, N in CALLS)
    assert WK.linears(low.fn, 16, 3, provable=False).calls == tuple((K, N, N // 2) for K, N in CALLS)
    with pytest.raises(Unexpressible, match=r"linears \(K, N\) \[\(64, 64\), \(64, 128\), \(64, 256\), \(128, 64\)\], whose "
                                            r"y no rows of Pearl\+'s 16 words doubled hand on in one unit within 1 rows$"):
        WK.linears(low.fn, 16, 1, provable=False)


def test_the_cut_replaces_every_linear_but_the_lm_head(low):
    """Each linear is one `LinearARow_v1{K, R, ROUND, CALL}` unit, its x its writers' R rows (q/k/v's and gate/up's the
    norm's one, o's the attention's 4 heads', down's SiluMul's one), and one `LinearOut_v1{N, 16, ROUND, CALL}` unit, a
    template each a round's call, and no `Gemm_v2` but the LM head's: its 16 slices a position, all of which the token
    selects' group units read, and its ``x0``, though gate/up is its shape. Every other template is the uncut circuit's at
    its count, and no input is free."""
    c = low.circuit
    n = {_label(fn): k for fn, k in c.templates}
    plain = {_label(fn): k for fn, k in WC.lower(_registered(), CLASS, None, GRAIN, provable=False).circuit.templates}
    want = {"Gemm_v2{K=64,N=16}": 16 * 10, "Gemm_v2{K=64,N=1}": 10}
    for s, m in M.items():
        for ci, (K, N) in enumerate(CALLS):
            want[f"LinearARow_v1{{K={K},R={4 if ci % 4 == 1 else 1},ROUND={s},CALL={ci}}}"] = m
            want[f"LinearOut_v1{{N={N},W=16,ROUND={s},CALL={ci}}}"] = m
    assert {t: k for t, k in n.items() if t.startswith(("Linear", "Gemm"))} == want
    assert {t: k for t, k in n.items() if not t.startswith(("Linear", "Gemm"))} == {
        t: k for t, k in plain.items() if not t.startswith("Gemm")}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in c.templates}
    g, cut = ids[GROUPS], c.cuts.get(ids[GROUPS], {}).get("x", 1)
    heads = {_source(c, g, "x", u, r)[:2] for u in range(n[GROUPS]) for r in range(cut)}
    assert {t for t, _ in heads} == {ids["Gemm_v2{K=64,N=16}"]} and len(heads) == 16 * 10
    assert IF.input_count(c.description(), **held(c))["free"] == 0


def test_each_cut_linear_reads_its_row_of_its_calls_a_rows_and_y(low):
    """Slot k's position p is round 0's row 4k + p below the prompt and round 1's row k past it (`FixedLoad.at`). Each cut
    linear, round s's call c, holds its A row (``o0``, x widened) to row i of the call's A rows and reads its ``y`` as row
    i of the call's y, i its position's row, and its readers read y as its unit's N / 16 output rows in order. Every real
    row of every value is read by exactly one unit, the A rows' filler rows by none, and the circuit's program takes them
    as its values, round by round and call by call the A rows then the y, ahead of its units' parameters. Each is FP32
    words in ``row-seg/v1`` rows as Pearl+'s trees commit them (`ROW_FORMAT`): its program parameter 32-bit leaves, and
    the ``o0`` and ``y`` ports reading it ``row-seg/v1`` rows in their templates' statements."""
    c, want = low.circuit, {}
    for k, j, p, ci in _calls(low):
        P, (K, N) = low.programs[k], CALLS[ci]
        s, i = (0, 4 * k + p) if p < CLASS.prompt else (1, k)
        a, y = P.arows[j], P.ys[j]
        want[(a.tid, "o0", a.unit)], want[(y.tid, "y", y.unit)] = (PEARL.a(s, ci), i), (PEARL.y(s, ci), i)
        assert (a.words, y.words) == (2 * K, 2 * N)
        assert P.outs[j] == [(16 * o, 16 * (o + 1), Row(y.tid, y.unit, f"o{o}", 16)) for o in range(N // 16)]
    got = _value_rows(c)
    assert got == want
    rows: dict[str, list[int]] = {}
    for v, r in got.values():
        rows.setdefault(v, []).append(r)
    for s, m in M.items():
        for ci, (K, N) in enumerate(CALLS):
            assert sorted(rows[PEARL.a(s, ci)]) == sorted(rows[PEARL.y(s, ci)]) == list(range(m))
            assert (c.registered.params[PEARL.a(s, ci)], c.registered.params[PEARL.y(s, ci)]) == (16 * 2 * K, m * 2 * N)
    assert c.registered.values == tuple(v for s in M for ci in range(len(CALLS)) for v in (PEARL.a(s, ci), PEARL.y(s, ci)))
    assert [n for n, _ in c.program()[0].fn.signature()[0][:32]] == list(c.registered.values)
    assert c.registered.formats == {v: WK.ROW_FORMAT for v in c.registered.values} == {v: {"row_schema": IF.SEG,
                                                                                            "word_bits": 32}
                                                                                        for v in c.registered.values}
    sig = dict(c.program()[0].fn.signature()[0][:32])
    for s, m in M.items():
        for ci, (K, N) in enumerate(CALLS):
            assert (sig[PEARL.a(s, ci)], sig[PEARL.y(s, ci)]) == (Array(16 * K, Value(32)), Array(m * N, Value(32)))
    seg, fns = IF.seg_ports(c.registered), {IF.descriptor_id(fn): fn for fn, _ in c.templates}
    for tid, port in {(t, p) for t, p, _u in want}:
        ins, outs = IF.unit_rows(fns[tid], c.cuts.get(tid), seg[tid])
        rows = {r[0]: r[1] for r in ins} | {p: s for (s, _j), (p, _w) in zip(outs, IF.layout(fns[tid]).outs)}
        assert seg[tid] == {port} and {p for p, s in rows.items() if s == IF.SEG} == {port}


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
    """`census` is a row a round's call: its round and call, K, N and W, the round's rows and its multiply-adds, its A rows'
    value and rows (m₁₆) and its y's (m), and PoUW's scheme; none is tied (`UNTIED`). Its multiply-adds are the circuit
    description's own, each call's `LinearARow_v1` units times its K and its `LinearOut_v1`'s N, and an uncut circuit has
    no census."""
    cen = WK.census(low)
    assert [(r["round"], r["call"], r["k"], r["n"], r["w"], r["rows"]) for r in cen] == [
        (s, ci, K, N, 16, M[s]) for s in M for ci, (K, N) in enumerate(CALLS)]
    for r in cen:
        s, ci, m = r["round"], r["call"], r["rows"]
        assert r["macs"] == m * r["n"] * r["k"] and (r["scheme"], r["tied"], r["untied"]) == (PEARL.scheme, False, WK.UNTIED)
        assert (r["a"], r["y"]) == ({"value": PEARL.a(s, ci), "rows": 16}, {"value": PEARL.y(s, ci), "rows": m})
    fns = {(fn.name, fn.bindings["ROUND"], fn.bindings["CALL"]): (fn, n) for fn, n in low.circuit.templates
           if fn.name in ("LinearARow", "LinearOut")}
    assert sum(r["macs"] for r in cen) == sum(n * fn.bindings["K"] * fns[("LinearOut", s, ci)][0].bindings["N"]
                                              for (name, s, ci), (fn, n) in fns.items() if name == "LinearARow") == 737280
    assert WK.census(WC.lower(_registered(), CLASS, None, GRAIN, wired=False)) == []


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
    (_registered, GRAIN, PearlTables(SCHEME, words=48), CLASS, r"linears \(K, N\) \[\(64, 64\), \(64, 128\), \(64, 256\), "
                                                       r"\(128, 64\)\], whose y no rows of Pearl\+'s 48 words doubled hand "
                                                       r"on in one unit within 256 rows that C-Flock proves")])
def test_a_circuit_pearls_tables_cant_cut_is_refused(registered, grain, tables, cls, why):
    """Refused by name: an int8 linear (TINY's exact-int8 checkpoint), a class of two boundaries (its slots aren't all
    admitted at round 0), and y rows of a width no doubling of which divides a linear's N."""
    with pytest.raises(Unexpressible, match=why):
        WC.lower(registered(), cls, None, grain, wired=False, tables=tables)


def test_the_cut_at_the_smallest_provable_class_is_proof_blocks_c_flock_proves():
    """At `test_window_circuit`'s smallest provable model (2 heads of 64 and a KV head, so q/k/v's N is 256), class and
    grain, with y handed on in rows of 64 words (128 bytes, as at Llama 3.2 1B's head dimension), every template of the cut
    circuit is a proof block C-Flock's composer takes (`interface.check_proof_block`): the A rows (256 and 512 words) and
    the y rows (2N words) are past 111 bytes."""
    c = WC.class_circuit(S._registered(PROVABLE), PROVABLE_CLASS, None, PROVABLE_GRAIN, tables=PearlTables(SCHEME, words=64))
    n = {_label(fn): k for fn, k in c.templates}
    P = PROVABLE_CLASS.positions * PROVABLE_CLASS.slots
    want = {"Gemm_v2{K=128,N=64}": 8 * P, "Gemm_v2{K=128,N=1}": P}
    for ci, (K, N, R) in enumerate(((128, 256, 1), (128, 128, 2), (128, 512, 1), (256, 128, 1)) * 2):
        want |= {f"LinearARow_v1{{K={K},R={R},ROUND=0,CALL={ci}}}": P, f"LinearOut_v1{{N={N},W=64,ROUND=0,CALL={ci}}}": P}
    assert {t: k for t, k in n.items() if t.startswith(("Linear", "Gemm"))} == want
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


def test_a_cut_linears_a_row_is_its_x_widened_and_its_y_unit_holds_its_value(low, filled):
    """Each cut linear's `LinearARow_v1` row is its Definition on the node's x, x's bf16 widened to FP32 exactly; its
    `LinearOut_v1` unit's output rows are the node's value in rows of 16, and its input row the value widened (the row the
    window commits by reference from the call's y)."""
    _w, _v, slots, rows = filled
    for k, j, *_ in _calls(low):
        P, s = low.programs[k], slots[k]
        x, a, y = s.operand(low.fn.body.nodes[j].args[0]), P.arows[j], P.ys[j]
        got = rows.outs[a.tid][0][a.unit]
        assert np.array_equal(got, _widened(x))
        assert np.array_equal(got, np.asarray(evaluate_batch(rows.fns[a.tid], x[None])).astype("<u4").view("<u2").reshape(-1))
        w = WR._words(s.values[j])
        assert np.array_equal(np.concatenate([rows.outs[y.tid][o][y.unit] for o in range(len(w) // 16)]), w)
        assert np.array_equal(rows.given[y.tid]["y"][y.unit], _widened(w))
    assert len(_calls(low)) == 2 * 5 * 8 and len(low.cut) == 5 * 8


def _pearl_values(low, slots) -> dict[str, np.ndarray]:
    """Pearl+'s tables of the fixed load as each call commits them, from the model's values: round s's call c's A rows each
    of its m rows the x of the position at round s's row i (`FixedLoad.at`) widened to FP32, then zero rows up to its
    m₁₆, and its y each row that call's value so widened (`served_y.f32_rows`, the rows Pearl+'s verifier opens)."""
    out: dict[str, np.ndarray] = {}
    for k, j, p, ci in _calls(low):
        (K, N), (s, i), sl = CALLS[ci], PEARL.at(CLASS, k, p), slots[k]
        a = out.setdefault(PEARL.a(s, ci), np.zeros((PEARL.a_rows(CLASS, s), 2 * K), np.uint16))
        y = out.setdefault(PEARL.y(s, ci), np.zeros((M[s], 2 * N), np.uint16))
        for v, at, n in ((a, sl.operand(low.fn.body.nodes[j].args[0]), K), (y, WR._words(sl.values[j]), N)):
            (row,) = SY.f32_rows(np.asarray(at, "<u2").tobytes(), n)
            v[i] = np.frombuffer(row, "<u2")
    return out


def test_pearls_values_are_the_rows_the_cut_linears_make(low, filled):
    """`Rows.value` gives each of Pearl+'s values as the cut circuit's proof blocks make it, which is as Pearl+'s calls
    commit them: a call's A rows each its cut linear's A row at the fixed load's rows, then the filler rows no unit makes
    (``empty``: Pearl+'s are zero), and its y each one's y widened, every row some unit's. It refuses a y read as another
    width."""
    _w, _v, slots, rows = filled
    want = _pearl_values(low, slots)
    assert sorted(want) == sorted(low.circuit.registered.values)
    for name, v in want.items():
        assert np.array_equal(rows.value(name, len(v), np.zeros(v.shape[1], np.uint16)), v), name
        made = (rows.value(name, len(v), np.full(v.shape[1], 0xFFFF, np.uint16)) != 0xFFFF).any(axis=1)
        assert made.sum() == M[int(name.split("/")[1])], name
    with pytest.raises(ValueError, match="reads 'pearl-y/0/1' other than as an output row of 8 words or an input row a "
                                         "node's value gives"):
        rows.value(PEARL.y(0, 1), 8, np.zeros(8, np.uint16))


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


def test_a_cut_linear_prefill_prunes_is_read_at_every_prompt_position(low, filled):
    """Prefill prunes the last layer from its query on at the prompt rows before the last, but the engine computes it there,
    the record binds it a row a prompt position and Pearl+ commits each call's y at them all: a cut linear there, which the
    circuit never computes, is read from those rows (off the GEMM's value, a slot of the cut circuit takes it as served),
    where the uncut circuit computes it and the other pruned nodes stay computed. Bound at its last row alone, it is
    refused by name."""
    weights, _v, slots, _rows = filled
    s0, nodes, lp = slots[0], low.fn.body.nodes, slots[0].row[1]
    K = len(nodes) // (max(WR.positions(low.fn)) + 1)
    query = [k for k in range(K) if nodes[k].fn.name == "RoPE"][-2]
    pruned = tuple(k for k in range(query, K - 3) if nodes[k].fn.name != "RoPE" or k == query)
    cut = [k for k in pruned if k in low.cut]
    assert [nodes[k].fn.name for k in cut] == ["Gemm"] * 3
    rows, store = _served(low, s0, T=1, also=pruned, whole=pruned)
    got = WR.Slot(low.fn, CLASS, s0.row, weights, WR.Served(rows, *_read(store)), cut=low.cut)
    assert got.source == [WR.COMPUTED if p < lp - 1 and (k in pruned and k not in cut or k >= K - 3) else WR.SERVED
                          for p in range(5) for k in range(K)]
    assert all(np.array_equal(a, b) for x, y in zip(got.values, s0.values) for a, b in zip(x, y))

    o, N = cut[0], WR._fields(nodes[cut[0]].fn)[0].leaves
    y = store[(0, f"op{o}", "out")]
    off = {**store, (0, f"op{o}", "out"): np.concatenate([y[:N] ^ np.uint16(1), y[N:]])}
    taken = WR.Slot(low.fn, CLASS, s0.row, weights, WR.Served(rows, *_read(off)), cut=low.cut)
    assert taken.source[o] == WR.SERVED and np.array_equal(taken.values[o][0], s0.values[o][0] ^ np.uint16(1))
    computed = WR.Slot(low.fn, CLASS, s0.row, weights, WR.Served(rows, *_read(off)))
    assert computed.source[o] == WR.COMPUTED and np.array_equal(computed.values[o][0], s0.values[o][0])

    rows, store = _served(low, s0, T=1, also=pruned)
    with pytest.raises(ValueError, match=rf"op{o}: a node the circuit never computes, pruned at the prompt rows before the "
                                         rf"last, its members bound at its last one alone, not a row a prompt position "
                                         rf"\({lp}\)"):
        WR.Slot(low.fn, CLASS, s0.row, weights, WR.Served(rows, *_read(store)), cut=low.cut)
    WR.Slot(low.fn, CLASS, s0.row, weights, WR.Served(rows, *_read(store)))


@pytest.mark.slow
def test_a_windows_cut_units_are_their_definitions_on_the_inputs_it_commits(low, filled):
    """Recorded into a `Window` with Pearl+'s values registered (`Rows.value`'s rows under the committer's key, FP32 words
    in row-seg rows: `ROW_FORMAT`), each
    `LinearARow_v1` and `LinearOut_v1` unit's output rows are its Definition on the inputs the window commits, x by its
    writers' wires and y by reference from the call's y rows, and the window closes over them. What no cheaper test
    reaches: the window resolving the cut's wires and committing a registered read of an input row by reference. Slow (a
    minute and more) for the window's members, which trace the embedding read's population Program."""
    _w, values, slots, rows = filled
    c = low.circuit
    for i, (name, v) in enumerate(_pearl_values(low, slots).items()):
        schema, bits = IF.value_format(c.registered, name)
        values = {**values, name: CR.Registered.commit(PROGRAM, name, v.view(f"<u{bits // 8}").tolist(),
                                                       [hm96_salt(KEY, (1 << 61) + i, j) for j in range(len(v))],
                                                       word_bits=bits, schema=schema)}
    win = IF.Window(c, RUN, LAW, 64, registered=values, forms=WR.FORMS)
    win.supply(CR)
    rows.record(win)
    bad = {}
    for fn, n in c.templates:
        if fn.name in ("LinearARow", "LinearOut"):
            tid, widths = descriptor_id(fn), [ty.homogeneous_width() for _, ty in fn.signature()[0]]
            ins = [np.ascontiguousarray(x).view("<u2").reshape(n, -1).view(f"<u{w // 8}")
                   for x, w in zip(win.input_words(tid, 0, n), widths)]
            want = np.asarray(evaluate_batch(fn, *ins, kernel="reference")).reshape(n, -1)
            want = want.astype("<u4").view("<u2").reshape(n, -1) if fn.name == "LinearARow" else want.astype(np.uint16)
            if (u := np.flatnonzero((want != np.concatenate(rows.outs[tid], axis=1)).any(axis=1))).size:
                bad[tid] = u.tolist()
    assert bad == {}
    assert win.close(salt_key=bytes(32)) is not None
