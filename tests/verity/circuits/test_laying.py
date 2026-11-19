from __future__ import annotations

import functools
import random

import pytest

from verity.circuits import boolean as BL
from verity.circuits import laying as LY
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.reference import evaluate_region
from verity.circuits.program import Program, input_id
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple

A4 = Array(4, BL.BIT)
PAIR = Tuple(A4, A4)


def _composite(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


T = _composite("LayingToyT", (("a", A4), ("k", A4)), A4, lambda B, S, a, k: array_of(
    [B.call(BL.XOR, BL.operands(a[i], B.call(BL.AND, BL.operands(k[i], a[(i + 1) % 4])))) for i in range(4)]))


def _u(B, S, a, k):
    y = B.call(T, a, k)
    ys = B.batch(T, array_of([y, a]), k, axes=(0, None))
    z = B.call(BL.XOR, BL.operands(ys[0][0], ys[1][3]))
    return tuple_of(B.call(T, ys[1], k), array_of([z, z, y[2], a[1]]))


U = _composite("LayingToyU", (("a", A4), ("k", A4)), PAIR, _u)
V = _composite("LayingToyV", (("a", A4), ("k", A4)), PAIR, lambda B, S, a, k: (lambda o: tuple_of(B.call(T, o[0], k), o[1]))(
    B.call(U, a, k)))


def _program(body, ret) -> Program:
    params = (("x", Array(2, A4)), ("k", A4))
    return Program(CompositeDefinition("LayingToy", 1, (), lambda S: (params, ret), body, register=False), registered=["k"])


@functools.cache
def toy() -> Program:
    def body(B, S, x, k):
        vs = B.batch(V, x, k, axes=(0, None))
        return tuple_of(vs, B.call(T, vs[1][0], k))
    return _program(body, Tuple(Array(2, PAIR), A4))


def ands(limit):
    return lambda ins, outs, n: n <= limit


@functools.cache
def laid(limit: int) -> Program:
    return LY.lay(toy(), ands(limit))


@pytest.mark.parametrize("limit", [3, 8])
def test_laying_keeps_every_gate_in_place_and_every_unit_fits(limit):
    P0, P1 = toy(), laid(limit)
    C0, C1 = P0.circuit, P1.circuit
    assert len(P1.root.body.nodes) > len(P0.root.body.nodes) and C0.size == C1.size
    assert [(g.prim.id, list(g.operands)) for g in C0.gates()] == [(g.prim.id, list(g.operands)) for g in C1.gates()]
    assert list(C0.root_outputs()) == list(C1.root_outputs())
    m = LY.Measure(P1, LY.query(P1))
    assert all(ands(limit)(*m.block(u, u + 1)) for u in range(len(m.units)))
    rng = random.Random(limit)
    t = {g.index: rng.getrandbits(1) for g in C0.gates() if input_id(g.prim.id)}
    v0, v1 = evaluate_region(C0, C0[0:C0.size], dict(t)), evaluate_region(C1, C1[0:C1.size], dict(t))
    assert [v0[g] for g in C0.root_outputs()] == [v1[g] for g in C1.root_outputs()]


def test_a_unit_that_fits_is_left_as_it_is():
    assert LY.lay(toy(), ands(10**6)) is toy()


@pytest.mark.parametrize("program", [toy, lambda: laid(3)])
def test_a_blocks_measure_is_the_block_restrict_cuts(program):
    P = program()
    q = LY.query(P)
    c = b"".join(descriptor_chunks(P))
    m = LY.Measure(P, q)
    n = len(m.units)
    for a, b in [(u, u + 1) for u in range(n)] + [(0, n), (1, n), (0, n // 2), (n // 3, n - 1)]:
        blk = R.restrict(c, {"query": q}, [(a, b)])
        g = R._Gates(blk.D)
        ins = tuple(sum(g.width(x) for x in r) for r in R._runs(list(blk.inputs), g.of))
        outs = tuple(sum(g.width(x) for x in r) for r in R._runs(list(blk.outputs), g.width))
        assert m.block(a, b)[:2] == (ins, outs), (a, b)


def test_packing_closes_a_block_where_the_next_unit_would_not_fit_and_the_object_takes_it():
    P, fits = laid(3), (lambda ins, outs, n: n <= 12 and len(ins) + len(outs) <= 4)
    blocks = LY.pack(P, fits)
    m = LY.Measure(P, LY.query(P))
    assert blocks[0][0] == 0 and blocks[-1][1] == len(m.units) and all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))
    assert all(fits(*m.block(a, b)) for a, b in blocks)
    assert all(not fits(*m.block(a, b + 1)) for a, b in blocks[:-1])
    obj = PO.build(P, LY.query(P), blocks)
    assert PO.validate(obj).ok and PO.verify(obj, P).ok


def test_a_wire_inside_a_block_is_no_port():
    P = laid(3)
    m = LY.Measure(P, LY.query(P))
    whole = m.block(0, len(m.units))
    assert sum(whole[1]) == 16 and sum(whole[1]) < sum(sum(m.block(u, u + 1)[1]) for u in range(len(m.units)))


def test_a_primitive_past_the_limit_and_a_unit_past_it_unlaid_are_refused():
    with pytest.raises(LY.Unlaid) as e:
        LY.lay(toy(), ands(0))
    assert e.value.code == "lay-primitive"
    with pytest.raises(LY.Unlaid) as e:
        LY.pack(toy(), ands(8))
    assert e.value.code == "pack-unit"


def test_a_body_with_a_scan_is_refused():
    step = _composite("LayingToyStep", (("c", BL.BIT), ("x", BL.BIT)), Tuple(BL.BIT, BL.BIT),
                      lambda B, S, c, x: (lambda y: tuple_of(y, y))(B.call(BL.XOR, BL.operands(c, x))))
    scanned = _composite("LayingToyScan", (("a", A4), ("k", A4)), A4,
                         lambda B, S, a, k: B.scan(step, a[0], [a])[1])

    def body(B, S, x, k):
        return B.call(scanned, x[0], k)
    with pytest.raises(LY.Unlaid) as e:
        LY.lay(_program(body, A4), lambda *_a: False)
    assert e.value.code == "lay-scan"
