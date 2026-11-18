"""`Q_fit` v1 (`verity.circuits.fit`): the builder's packing, `partition_object.verify`'s refusals by name, and `restrict` on a
piece computing what C computes there."""
from __future__ import annotations

import functools
import importlib.util
import random
from pathlib import Path

import pytest

from verity.circuits import boolean as BL
from verity.circuits import fit as FT
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.reference import evaluate_region
from verity.circuits.ops import words as W
from verity.circuits.program import Program, input_id
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple

_SPEC = importlib.util.spec_from_file_location("fit_chain", Path(__file__).parent / "ops/test_boolean_block_circuit.py")
_BB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BB)

#: each root input a row of 125 bytes, past `fit.LAST_BLOCK_BYTES`, so a block reading one holds the pin
WIDE, A2, A7 = Array(1000, BL.BIT), Array(2, BL.BIT), Array(7, BL.BIT)
XS = tuple((f"x{j}", WIDE) for j in range(6))


def _u_body(B, S, *xs):
    ys = [B.call(BL.XOR, BL.operands(x[0], x[1])) for x in xs]
    return array_of(ys + [B.call(BL.AND, BL.operands(ys[0], xs[5][1]))])


U = bind(CompositeDefinition("FitToyU", 1, (), lambda S: (tuple((f"x{j}", A2) for j in range(6)), A7), _u_body,
                             register=False))


def _root_body(B, S, *xs):
    xs = [x[0:2] for x in xs]
    return tuple_of(B.call(U, *xs), B.call(U, *xs))


@functools.cache
def toy() -> tuple[Program, bytes]:
    """Two units of U over six root inputs, each its own row: a unit has 7 rows (six inputs and its output row), and a piece
    of its nodes one row an input it reads and one a node it holds."""
    program = Program(CompositeDefinition("FitToy", 1, (), lambda S: (XS, Tuple(A7, A7)), _root_body, register=False))
    return program, b"".join(descriptor_chunks(program))


@functools.cache
def chain() -> Program:
    return _BB._chain()


def codes(program, q, blocks, **kw) -> set[str]:
    return set(PO.verify(PO.build(program, q, blocks), program, **kw).codes)


def test_the_chain_packs_as_its_cap_says_and_verifies():
    """At its window's grain, two units, the template query's cut; at the default cap a unit a block; uncapped, all eight."""
    program = chain()
    for cap, blocks in ((2, [[0, 2], [2, 4], [4, 6], [6, 8]]), (1, [[u, u + 1] for u in range(8)]), (None, [[0, 8]])):
        assert FT.build(program, cap=cap) == blocks
        assert PO.verify(PO.build(program, FT.query(cap=cap), blocks), program).ok
    assert FT.build(program) == FT.build(program, cap=1)


def test_a_block_past_its_cap_is_refused_and_the_cap_is_under_the_digest():
    program = chain()
    assert codes(program, FT.query(cap=1), [[0, 2], [2, 4], [4, 6], [6, 8]]) == {"fit-cap"}
    a, b = (PO.build(program, FT.query(cap=cap), [[0, 8]]) for cap in (8, None))
    assert PO.digest(a) != PO.digest(b) and PO.verify(a, program).ok and PO.verify(b, program).ok


def _units(program) -> list[int]:
    return [k for k, n in enumerate(program.root.body.nodes) if getattr(n.fn, "id", None) == U.id]


def test_the_toy_fits_whole_at_seven_rows_and_is_cut_into_its_nodes_at_six():
    """Two whole units write two rows (a writer row a unit), so they fit together at eight; a piece of three nodes reads three
    rows and writes three, one a node."""
    program, _ = toy()
    assert FT.build(program, rows=7) == [[0, 1], [1, 2]]
    assert FT.build(program, rows=8, cap=None) == [[0, 2]]
    assert FT.build(program, rows=7, cap=None) == [[0, 1], [1, 2]]
    want = [{"path": [[k, 0]], "nodes": r} for k in _units(program) for r in ([0, 3], [3, 6], [6, 7])]
    assert FT.build(program, rows=6) == want
    assert PO.verify(PO.build(program, FT.query(rows=6), want), program).ok
    fp, fit = FT.FitPartition(program, FT.query(rows=6)), FT.Fit(program)
    m = fit.measure(FT.part_of(program, fp, want[0]))
    assert m.rows == 6 and m.writers == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]


def test_a_block_past_its_rows_or_its_bits_is_refused_by_name():
    program, _ = toy()
    assert codes(program, FT.query(rows=7, cap=None), [[0, 2]]) == {"fit-rows"}
    assert codes(program, FT.query(bits=1 << 19), [[0, 1], [1, 2]]) == {"fit-bits"}


@pytest.mark.parametrize("blocks,code", [([[0, 1]], "blocks-gap"), ([[1, 2], [0, 1]], "blocks-unsorted"),
                                         ([[0, 3]], "fit-entry"), ([{"path": [[0, 0]], "nodes": [0, 1]}], "fit-entry")])
def test_a_cover_that_is_not_the_calls_each_once_in_order_is_refused(blocks, code):
    program, _ = toy()
    assert code in codes(program, FT.query(), blocks)


def test_a_block_holding_units_of_two_inference_blocks_is_refused():
    program, _ = toy()
    q = FT.query(cap=None)
    assert codes(program, q, [[0, 2]], inference=lambda u: u) == {"fit-straddles"}
    assert PO.verify(PO.build(program, q, [[0, 2]]), program, inference=lambda u: 0).ok
    assert PO.verify(PO.build(program, q, [[0, 1], [1, 2]]), program, inference=lambda u: u).ok
    pieces = FT.build(program, rows=6)
    assert PO.verify(PO.build(program, FT.query(rows=6), pieces), program, inference=lambda u: u).ok


def test_a_fit_object_without_its_blocks_is_refused():
    program, _ = toy()
    obj = PO.build(program, FT.query(), [[0, 2]])
    del obj["blocks"]
    assert "blocks-missing" in PO.validate(obj).codes


def test_a_gate_that_alone_does_not_fit_is_refused_at_registration():
    program, _ = toy()
    with pytest.raises(FT.Unfit) as e:
        FT.build(program, rows=2)
    assert e.value.code == "fit-gate"


def _bit(t, bits, x):
    g = bits.gate_of(x)
    return (t[g] >> bits.out[g].index(x)) & 1


@pytest.mark.parametrize("rows,cap", [(6, 1), (7, 1), (8, None)])
def test_a_block_restricted_computes_what_C_computes_there(rows, cap):
    program, c = toy()
    blocks = FT.build(program, rows=rows, cap=cap)
    obj = PO.build(program, FT.query(rows=rows, cap=cap), blocks)
    bits = R.Bits(R.program(c))
    C, rng, t = program.circuit, random.Random(7), {}
    for g in C.gates():
        if input_id(g.prim.id) is not None:
            t[g.index] = rng.getrandbits(1)
    evaluate_region(C, C[0:C.size], t)
    seen = []
    for e in blocks:
        b = R.restrict(c, obj, e, bits)
        assert R.evaluate(b.listing(), [_bit(t, bits, x) for x in b.input_bits]) == [_bit(t, bits, x) for x in b.output_bits]
        inside = {g for lo, hi in b.words for g in range(lo, hi)}
        assert list(b.inputs) == sorted({o for g in inside for o in C.gate(g).operands} - inside)
        seen += sorted(inside)
    assert len(seen) == len(set(seen))


def test_a_piece_not_among_the_blocks_is_refused():
    program, c = toy()
    obj = PO.build(program, FT.query(rows=6), FT.build(program, rows=6))
    with pytest.raises(R.Refused, match="E_UNITS"):
        R.restrict(c, obj, {"path": [[obj["blocks"][0]["path"][0][0], 0]], "nodes": [0, 2]})


U32 = W.U32


def _p_body(B, S, *xs):
    return B.call(W.PouwXor32, B.call(W.PouwXor32, xs[0][0], xs[1][0]), xs[2][0])


def _q_body(B, S, p):
    return tuple_of(B.call(W.PouwXor32, p, p), array_of([B.call(W.PouwNoiseByte, p, p) for _ in range(16)]))


def _s_body(B, S, q, d):
    return B.call(W.PouwDot16U8S8, q, d, d)


ROW32 = Array(32, U32)
PU = bind(CompositeDefinition("FitPinP", 1, (), lambda S: (tuple((f"x{j}", ROW32) for j in range(3)), U32), _p_body,
                              register=False))
QU = bind(CompositeDefinition("FitPinQ", 1, (), lambda S: ((("p", U32),), Tuple(U32, Array(16, W.U8))), _q_body,
                              register=False))
SU = bind(CompositeDefinition("FitPinS", 1, (), lambda S: ((("q", U32), ("d", Array(16, W.U8))), U32), _s_body,
                              register=False))


@functools.cache
def joined() -> Program:
    """u0 reads three 128-byte rows and writes p; u1 reads p and writes a word and 16 bytes, two rows; u2 reads both and
    writes s. At four rows u0 and u1 together have five, so the builder stops after u0; u1 and u2 have one 4-byte row in and
    one out, no pin, and joined to u0 they have four rows and the pin."""
    def body(B, S, *xs):
        p = B.call(PU, *xs)
        q = B.call(QU, p)
        return B.call(SU, q[0], q[1])
    return Program(CompositeDefinition("FitPinJoin", 1, (), lambda S: (tuple((f"x{j}", ROW32) for j in range(3)), U32), body,
                                       register=False))


def test_a_block_with_no_pin_joins_the_block_before_it_where_rows_and_bits_allow():
    program = joined()
    fp, fit = FT.FitPartition(program, FT.query(rows=4)), FT.Fit(program)
    m0 = fit.measure(FT.part_of(program, fp, [0, 1]))
    assert (m0.rows, m0.pinned) == (4, True) and fit.measure(FT.part_of(program, fp, [0, 2])).rows == 5
    fit.finish(FT.part_of(program, fp, [0, 1]), m0)
    tail = fit.measure(FT.part_of(program, fp, [1, 3]))
    assert tail.rows <= 4 and not tail.pinned
    assert FT.build(program, rows=4, cap=None) == [[0, 3]]
    assert PO.verify(PO.build(program, FT.query(rows=4, cap=None), [[0, 3]]), program).ok


def test_a_block_with_no_pin_is_refused_in_verify_and_where_it_joins_no_neighbour_at_build():
    """At a unit a block, the joined toy's u1 and u2 may not join u0."""
    program = joined()
    assert codes(program, FT.query(rows=4, cap=None), [[0, 1], [1, 3]]) == {"fit-pin"}
    assert codes(program, FT.query(rows=4), [[0, 1], [1, 2], [2, 3]]) == {"fit-pin"}
    with pytest.raises(FT.Unfit) as e:
        FT.build(program, rows=4)
    assert e.value.code == "fit-pin"


def test_a_root_input_nodes_registered_rows_are_its_rows():
    """The joined toy's inputs registered eight words a row: u0 reads one 32-byte row of each, no pin, and five rows in all
    with u1's two."""
    program = joined()
    nodes = [k for k, n in enumerate(program.root.body.nodes) if n.form != "call"]
    fp = FT.FitPartition(program, FT.query(rows=4, cap=None))
    whole, rows8 = FT.Fit(program), FT.Fit(program, {k: 8 for k in nodes})
    part = FT.part_of(program, fp, [0, 1])
    assert whole.measure(part).pinned and not rows8.measure(part).pinned
    g = program.root.body.offsets[nodes[0]]
    assert whole.row(g)[:2] == (g, 128) and rows8.row(g)[:2] == (g, 32) and rows8.row(g + 8)[:2] == (g + 8, 32)
    assert codes(program, FT.query(rows=4, cap=None), [[0, 3]], registered={k: 8 for k in nodes}) == {"fit-pin"}
    with pytest.raises(FT.Unfit):
        FT.Fit(program, {len(program.root.body.nodes) - 1: 8})


def test_a_partitions_writer_rows_are_measured_once_for_all_its_blocks(monkeypatch):
    program, c = toy()
    obj = PO.build(program, FT.query(rows=6), FT.build(program, rows=6))
    bits = R.Bits(R.program(c))
    R._WRITERS.clear()
    passes = []
    finish = FT.Fit.finish
    monkeypatch.setattr(FT.Fit, "finish", lambda self, *a: (passes.append(1), finish(self, *a))[1])
    got = [R.restrict(c, obj, e, bits).descriptor for e in obj["blocks"]]
    assert len(passes) == len(obj["blocks"]) and len(R._WRITERS) == 1
    R._WRITERS.clear()
    assert [R.restrict(c, obj, e, bits).descriptor for e in reversed(obj["blocks"])] == got[::-1]
