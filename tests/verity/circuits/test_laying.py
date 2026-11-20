from __future__ import annotations

import functools
import random

import pytest

from verity.circuits import boolean as BL
from verity.circuits import laying as LY
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.reference import evaluate_region
from verity.circuits.population import wired_population_program
from verity.circuits.program import Program, input_id
from verity.circuits.refs import Affine, Strided, array_of, tuple_of
from verity.circuits.types import Array, Tuple
from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC

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


MARK = {"supplied": "LayingToySalt", "args": [], "layout": "bits", "bits": 4}


@functools.cache
def with_s(binding: str) -> Program:
    params = (("x", Array(2, A4)), ("k", A4), ("s", A4))

    def body(B, S, x, k, s):
        vs = B.batch(V, x, k, axes=(0, None))
        return tuple_of(vs, B.call(T, vs[1][0], s))
    fn = CompositeDefinition(f"LayingToyS{binding}", 1, (), lambda S: (params, Tuple(Array(2, PAIR), A4)), body, register=False)
    return Program(fn, registered=["k"] + (["s"] if binding == "registration" else []),
                   supplied={"s": MARK} if binding == "supplied" else {})


@functools.cache
def recut(n: int) -> Program:
    tid = descriptor_id(U)
    rec = {"templates": [[tid, n]],
           "registered": {"params": {"w": 2}, "formats": {"w": {"word_bits": 4}}, "tables": [], "roots": [], "openings": [],
                          "reads": [{"reader": tid, "port": "k", "param": "w", "at": 0, "radix": [n], "base": 1, "step": [0]}]}}
    w = wired_population_program([(U, n)], rec)[0]
    return Program(w.fn, registered=("w",), annotations=w.annotations)


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


def test_laying_keeps_a_supplied_input_supplied_with_its_mark():
    P0 = with_s("supplied")
    P1 = LY.lay(P0, ands(3))
    assert P1 is not P0 and P1.bindings == P0.bindings == ("run", "registration", "supplied")
    assert P1.supplied == {"s": MARK}
    ids = [nd.fn.id for nd in P1.root.body.nodes if input_id(nd.fn.id)]
    assert ids == [nd.fn.id for nd in P0.root.body.nodes if input_id(nd.fn.id)] and ids[-1] == "SuppliedInput1_v1"
    assert [input_id(i)[0] for i in ids].count("supplied") == 1
    assert [g.prim.id for g in P0.circuit.gates()] == [g.prim.id for g in P1.circuit.gates()]


@pytest.mark.parametrize("n", [1, 4])
def test_a_unit_reading_a_registered_input_in_narrower_leaves_is_laid_and_measured_in_the_inputs_leaves(n):
    P0, P1 = recut(n), LY.lay(recut(n), ands(4))
    assert len(P1.root.body.nodes) > len(P0.root.body.nodes)
    parts = [x.refs for nd in P1.root.body.nodes for x in nd.args]
    assert {r.base for r in parts if isinstance(r, (Affine, Strided)) and (r.space, r.idx) == ("n", 0)} == {16}
    m = LY.Measure(P1, LY.query(P1))
    assert {r for rs in m.reads for r in rs if r[3] == "registration"} == {(4, 5, 4, "registration")}


def test_the_measure_reads_a_supplied_input_as_a_port_of_its_own_binding_as_it_reads_a_registered_one():
    ms = [LY.Measure(P, LY.query(P)) for P in (LY.lay(with_s(b), ands(3)) for b in ("supplied", "registration"))]
    n = len(ms[0].units)
    assert n == len(ms[1].units)
    split = 0
    for a in range(n):
        for b in range(a + 1, n + 1):
            (i0, o0, n0), (i1, o1, n1) = ms[0].block(a, b), ms[1].block(a, b)
            assert (sum(i0), o0, n0) == (sum(i1), o1, n1) and len(i1) <= len(i0) <= len(i1) + 1
            split += len(i0) > len(i1)
    assert split


@pytest.mark.parametrize("program", [toy, lambda: laid(3), lambda: LY.lay(with_s("supplied"), ands(3))])
def test_a_blocks_measure_is_the_block_restrict_cuts(program):
    P = program()
    q = LY.query(P)
    c = b"".join(descriptor_chunks(P))
    m = LY.Measure(P, q)
    n = len(m.units)
    for a, b in sorted({(u, u + 1) for u in range(n)} | {(0, u + 1) for u in range(n)} | {(u, n) for u in range(n)}):
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
    assert all(not fits(*m.block(a, b + 1)) or (m.reads_unit[b] and m.reads_unit[b][-1] >= a) for a, b in blocks[:-1])
    obj = PO.build(P, LY.query(P), [blocks])
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


class _Line:
    def __init__(self, ands):
        self.w, self.units = list(ands), list(range(len(ands)))
        self.reads_unit = [[] for _ in ands]

    def block(self, a, b):
        return (), (), sum(self.w[a:b])

    def grow(self, a):
        for b in range(a + 1, len(self.w) + 1):
            yield b, self.block(a, b)

    def inside_read(self, lo, hi):
        return None


def _packed(ands_, ends=()):
    return LY.pack(None, ands(12), m=_Line(ands_), enough=lambda _i, _o, n, _u: n >= 4, ends=ends)


@pytest.mark.parametrize("units, blocks", [
    ([5, 5, 1, 12], [(0, 3), (3, 4)]),
    ([5, 6, 6, 5, 1], [(0, 2), (2, 5)]),
    ([2, 1, 3, 12, 4], [(0, 3), (3, 4), (4, 5)]),
])
def test_a_block_under_the_floor_takes_the_units_after_it_or_joins_the_block_before(units, blocks):
    assert _packed(units) == blocks


@pytest.mark.parametrize("units, ends", [([12, 1, 12], ()), ([1, 1], ()), ([5, 5, 1, 12], (2,))])
def test_a_block_under_the_floor_that_nothing_takes_is_refused(units, ends):
    with pytest.raises(LY.Unlaid) as e:
        _packed(units, ends)
    assert e.value.code == "pack-floor"


def test_a_floor_every_block_meets_leaves_the_packing_and_a_trailing_block_under_it_is_refused():
    P = _wide(7)
    assert LY.pack(P, ands(10), enough=lambda _i, _o, n, _u: n >= 4) == LY.pack(P, ands(10))
    with pytest.raises(LY.Unlaid) as e:
        LY.pack(P, ands(10), enough=lambda _i, _o, n, _u: n >= 5)
    assert e.value.code == "pack-floor"


def test_a_level_packed_within_a_coarser_levels_ends_refines_it():
    P = _wide(12)
    coarse = LY.pack(P, ands(12))
    fine = LY.pack(P, ands(5), ends=[b for _a, b in coarse])
    assert {b for _a, b in coarse} <= {b for _a, b in fine} and len(fine) > len(coarse)
    assert PO.validate(PO.build(P, LY.query(P), [coarse, fine])).ok
    with pytest.raises(LY.Unlaid) as e:
        LY.pack(P, ands(5), ends=[0])
    assert e.value.code == "pack-ends"


def test_the_provers_floor_is_a_floor_pack_takes():
    floor = lambda _i, _o, _n, u: u >= LY.zk_min_units(21, 1)
    assert LY.zk_min_units(21, 1) == 9 and LY.zk_min_units(21, 2) == 17 and LY.zk_min_units(22, 64) == 1
    assert LY.pack(None, ands(12), m=_Line([1] * 21), enough=floor) == [(0, 12), (12, 21)]
    with pytest.raises(LY.Unlaid) as e:
        LY.pack(None, ands(12), m=_Line([1] * 20), enough=floor)
    assert e.value.code == "pack-floor"


def _chain(length: int = 3) -> Program:
    """`length` root units of `T`, each reading the one before; the first reads the root's input."""
    def body(B, S, x, k):
        y = x
        for _ in range(length):
            y = B.call(T, y, k)
        return y
    params = (("x", A4), ("k", A4))
    return Program(CompositeDefinition("LayingChain", 1, (), lambda S: (params, A4), body, register=False), registered=["k"])


def _wide(n: int) -> Program:
    """`n` units of `T`, a batch whose members read only the root's input."""
    params = (("x", Array(n, A4)), ("k", A4))
    return Program(CompositeDefinition("LayingWide", 1, (), lambda S: (params, Array(n, A4)),
                                       lambda B, S, x, k: B.batch(T, x, k, axes=(0, None)), register=False), registered=["k"])


def _independent() -> Program:
    def body(B, S, x, k):
        return array_of([B.call(T, x, k)[0], B.call(T, k, k)[0], B.call(T, x, x)[0], B.call(T, k, x)[0]])
    params = (("x", A4), ("k", A4))
    return Program(CompositeDefinition("LayingIndependent", 1, (), lambda S: (params, Array(4, BL.BIT)), body, register=False),
                   registered=["k"])


def test_a_unit_reading_a_unit_ends_the_block_before_it_and_the_door_admits_what_pack_returns():
    P = _chain()
    m = LY.Measure(P, LY.query(P))
    assert m.reads_unit == [[], [0], [1]]
    assert LY.pack(P, lambda *_a: True) == [(0, 1), (1, 2), (2, 3)]
    c = b"".join(descriptor_chunks(P))
    part, key = {"query": LY.query(P)}, descriptor_id(T)
    for a, b in LY.pack(P, lambda *_a: True):
        OC.admitted_block(R.program(c), part, [(a, b)], key)
    with pytest.raises(OC.Refused, match="operands-outside"):
        OC.admitted_block(R.program(c), part, [(0, 3)], key)


def test_a_floor_that_would_join_a_unit_to_the_unit_it_reads_is_refused_by_name():
    P = _chain()
    with pytest.raises(LY.Unlaid) as e:
        LY.pack(P, lambda *_a: True, enough=lambda _i, _o, _n, units: units >= 2)
    assert e.value.code == "pack-inside-read"
    assert str(e.value) == ("pack-inside-read: units [0, 2) would be one proof block to meet the floor, and unit 1 reads unit 0 of "
                            "it: a proof block holds no unit that reads a unit of its own")


def test_a_floor_joining_a_block_to_the_block_before_it_is_refused_where_a_unit_reads_one_of_it():
    P = _chain(4)
    m = LY.Measure(P, LY.query(P))
    with pytest.raises(LY.Unlaid) as e:
        LY.pack(P, lambda *_a: True, enough=lambda _i, _o, _n, units: units >= 2, m=m)
    assert e.value.code == "pack-inside-read"


def test_units_that_read_no_unit_of_their_block_still_join():
    P = _independent()
    m = LY.Measure(P, LY.query(P))
    assert m.reads_unit == [[], [], [], []]
    assert LY.pack(P, lambda *_a: True) == [(0, 4)]
    assert LY.pack(P, lambda *_a: True, enough=lambda _i, _o, _n, units: units >= 2) == [(0, 4)]


@pytest.mark.parametrize("limit", [3, 8, 12])
def test_every_block_pack_returns_on_a_mixed_program_holds_no_unit_that_reads_a_unit_of_it(limit):
    P = laid(3)
    tp = PO.TemplatePartition(P, LY.query(P))
    for a, b in LY.pack(P, lambda ins, outs, n: n <= limit and len(ins) + len(outs) <= 64):
        gates = [tp.gates(u) for u in range(a, b)]
        for u in range(a, b):
            assert not [g for g in tp.inputs(u) if any(lo <= g < hi for lo, hi in gates)], (a, b, u)
