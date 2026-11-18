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
from verity.circuits.program import Program, input_id
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple

_SPEC = importlib.util.spec_from_file_location("fit_chain", Path(__file__).parent / "ops/test_boolean_block_circuit.py")
_BB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BB)

A2, A7 = Array(2, BL.BIT), Array(7, BL.BIT)
XS = tuple((f"x{j}", A2) for j in range(6))


def _u_body(B, S, *xs):
    ys = [B.call(BL.XOR, BL.operands(x[0], x[1])) for x in xs]
    return array_of(ys + [B.call(BL.AND, BL.operands(ys[0], ys[5]))])


U = bind(CompositeDefinition("FitToyU", 1, (), lambda S: (XS, A7), _u_body, register=False))


def _root_body(B, S, *xs):
    return tuple_of(B.call(U, *xs), B.call(U, *xs))


@functools.cache
def toy() -> tuple[Program, bytes]:
    """Two units of U over six root inputs, each its own row: a unit has 7 rows (six inputs and its output run)."""
    program = Program(CompositeDefinition("FitToy", 1, (), lambda S: (XS, Tuple(A7, A7)), _root_body, register=False))
    return program, b"".join(descriptor_chunks(program))


@functools.cache
def chain() -> Program:
    return _BB._chain()


def codes(program, q, blocks, **kw) -> set[str]:
    return set(PO.verify(PO.build(program, q, blocks), program, **kw).codes)


def test_the_chain_packs_as_the_template_query_cuts_it_and_verifies():
    program = chain()
    assert FT.build(program, cap=2) == [[0, 2], [2, 4], [4, 6], [6, 8]]
    assert FT.build(program) == [[0, 8]]
    for blocks in ([[0, 2], [2, 4], [4, 6], [6, 8]], [[0, 8]], [[u, u + 1] for u in range(8)]):
        assert PO.verify(PO.build(program, FT.query(), blocks), program).ok


def test_the_toy_fits_whole_at_seven_rows_and_is_cut_into_its_nodes_at_four():
    program, _ = toy()
    assert FT.build(program, rows=7) == [[0, 2]]
    root = program.root.body
    k0, k1 = [k for k, n in enumerate(root.nodes) if n.fn is U or getattr(n.fn, "id", None) == U.id]
    want = [{"path": [[k, 0]], "nodes": r} for k in (k0, k1) for r in ([0, 3], [3, 6], [6, 7])]
    assert FT.build(program, rows=4) == want
    assert PO.verify(PO.build(program, FT.query(rows=4), want), program).ok


def test_a_block_past_its_rows_or_its_bits_is_refused_by_name():
    program, _ = toy()
    assert codes(program, FT.query(rows=4), [[0, 2]]) == {"fit-rows"}
    assert codes(program, FT.query(bits=1 << 19), [[0, 1], [1, 2]]) == {"fit-bits"}


@pytest.mark.parametrize("blocks,code", [([[0, 1]], "blocks-gap"), ([[1, 2], [0, 1]], "blocks-unsorted"),
                                         ([[0, 3]], "fit-entry"), ([{"path": [[0, 0]], "nodes": [0, 1]}], "fit-entry")])
def test_a_cover_that_is_not_the_calls_each_once_in_order_is_refused(blocks, code):
    program, _ = toy()
    assert code in codes(program, FT.query(), blocks)


def test_a_block_holding_units_of_two_inference_blocks_is_refused():
    program, _ = toy()
    assert codes(program, FT.query(), [[0, 2]], inference=lambda u: u) == {"fit-straddles"}
    assert PO.verify(PO.build(program, FT.query(), [[0, 2]]), program, inference=lambda u: 0).ok
    assert PO.verify(PO.build(program, FT.query(), [[0, 1], [1, 2]]), program, inference=lambda u: u).ok
    pieces = FT.build(program, rows=4)
    assert PO.verify(PO.build(program, FT.query(rows=4), pieces), program, inference=lambda u: u).ok


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


@pytest.mark.parametrize("rows", [4, 7])
def test_a_block_restricted_computes_what_C_computes_there(rows):
    program, c = toy()
    blocks = FT.build(program, rows=rows)
    obj = PO.build(program, FT.query(rows=rows), blocks)
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
    obj = PO.build(program, FT.query(rows=4), FT.build(program, rows=4))
    with pytest.raises(R.Refused, match="E_UNITS"):
        R.restrict(c, obj, {"path": [[obj["blocks"][0]["path"][0][0], 0]], "nodes": [0, 2]})
