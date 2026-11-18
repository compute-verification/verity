"""`population.unwired_population_program`: a wired population's cut into the unwired one and its wires' positions."""
from __future__ import annotations

import pytest

from verity.circuits.codec import descriptor_id, program_digest
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.ops import gemm as G
from verity.circuits.ops.boolean import cast as BC, gemm as BG
from verity.circuits.population import unwired_population_program
from verity.circuits.program import Program
from verity.circuits.refs import Coll, coll_concat, concat
from verity.circuits.types import Array, Tuple, Value
from verity.commitments.periods import interface as I
from verity.commitments.periods.population import population_program

RUN, SAMPLING_DESIGN = "ab" * 64, {"law": "stratified", "k": 2}
GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
GID = descriptor_id(GEMM)


def rounded(n: int = 6) -> I.Circuit:
    """``n`` f32 words rounded to bf16 and widened back, each widening's ``a`` wired from its rounding's output."""
    first, then = bind(BG.F2fpBf16), bind(BC.Bf16ToF32)
    wire = I.Wire(descriptor_id(then), "a", descriptor_id(first), "o0", at=0, radix=(n,), base=0, step=(1,))
    return I.Circuit("bits", ((first, n), (then, n)), wires=(wire,))


def chain(requests: int = 2, steps: int = 2, layers: int = 2) -> I.Circuit:
    """One ``Gemm_v2`` template reading itself: unit u >= ``requests`` reads unit u - ``requests``, in ranges of an order."""
    n = layers * steps * requests
    wires = (I.Wire(GID, "x", GID, "o0", at=requests, radix=(n - requests,), base=0, step=(1,)),)
    return I.Circuit("chain", ((GEMM, n),), external=(f"{GID}/x",), wires=wires,
                     order=I.ordered({GID: n}, wires, {GID: range(0, n, requests)}))


def _values(circuit: I.Circuit) -> list[str]:
    return [v for v, _n, _w in I.program_values(circuit.registered)]


@pytest.mark.parametrize("make", [rounded, chain])
def test_the_cut_is_the_unwired_population_and_its_positions_are_the_members_reads(make):
    """The cut of a circuit's wired Program is the periods' unwired population of its templates, digest and query, and
    each member's wired port reads, unit by unit, the writer units the cut's positions give (`rows.Binding.reads`)."""
    circuit = make()
    wired, _ = circuit.program()
    unwired, query, positions = unwired_population_program(wired, circuit.cuts, _values(circuit))
    want, want_query = population_program(list(circuit.templates), values=I.program_values(circuit.registered))
    assert (program_digest(unwired), query) == (program_digest(want), want_query)
    w = I.Window(circuit, RUN, SAMPLING_DESIGN, 2).window
    reads = {}
    for m in w.members:
        for x in m.wired:
            got = positions[(m.template, x.port, 0)][w.tree(m, x.port)]
            by = dict(zip(got[0].tolist(), got[1].tolist()))
            lo = w.owns[m.key]
            assert [by[u] for u in range(lo, lo + m.n)] == w.reads(m, x.port).tolist()
            reads.update((u, by[u]) for u in range(lo, lo + m.n))
    assert sum(len(a) for trees in positions.values() for a, _ in trees.values()) == len(reads) > 0


@pytest.mark.parametrize("held", [(), ("w",)])
def test_registered_values_stay_and_other_reads_become_the_readers_own_rows(held):
    """Four ``Gemm_v2`` units each reading registered ``w`` whole: held, ``w`` is the unwired population's value; not
    held, its root parameter is a read the cut drops, each reader's row its own. The record's values out of the root's
    order are refused (cut-root)."""
    reads = (I.Read(GID, "w", "w", 0, (4,), 0, (0,)),)
    circuit = I.Circuit("reads", ((GEMM, 4),), registered=I.Registered({"w": 256}, reads=reads, values=held))
    wired, _ = circuit.program()
    assert "w" in dict(wired.fn.body.params)
    unwired, query, positions = unwired_population_program(wired, circuit.cuts, _values(circuit))
    want, want_query = population_program(list(circuit.templates), values=I.program_values(circuit.registered))
    assert (program_digest(unwired), query, positions) == (program_digest(want), want_query, {})
    with pytest.raises(ValueError, match="cut-root"):
        unwired_population_program(wired, circuit.cuts, ("v", *held))


def test_an_unwired_population_is_refused_as_no_wired_one():
    """An unwired Program states no wires to cut, and a cut of it would let it stand where a registration names the wired
    one, so it is refused, not returned."""
    circuit = rounded()
    unwired, _ = unwired_population_program(circuit.program()[0], circuit.cuts, ())[:2]
    with pytest.raises(ValueError, match="cut-input"):
        unwired_population_program(unwired, {}, ())


V16 = Value(16)
COPY = PrimitiveDefinition("CutTestCopy16", 1, (("a", V16),), V16, lambda a: a, conformance="protocol-defined")
PAIR_RET = Tuple(Array(2, V16), Array(2, V16))
PAIR = bind(CompositeDefinition("CutTestPair", 1, (), lambda S: ((("x", Array(2, V16)),), PAIR_RET),
                                lambda B, S, x: coll_concat(PAIR_RET, [B.call(COPY, x[i]) for i in (0, 1, 0, 1)]),
                                register=False))


def reader(leaves: int) -> object:
    return bind(CompositeDefinition("CutTestRead", leaves, (), lambda S: ((("x", Array(leaves, V16)),), V16),
                                    lambda B, S, x: B.call(COPY, x[0]), register=False))


def wired_by_hand(leaves: int, offsets: list[int], writers: int = 2) -> tuple[object, str]:
    """A ``WiredPopulations_v1`` root: ``writers`` `PAIR` units (two fields of two leaves each), then a reader of
    ``leaves`` leaves a unit, unit i reading the writers' output leaves from ``offsets[i]``."""
    rd = reader(leaves)
    ids, counts = (descriptor_id(PAIR), descriptor_id(rd)), (writers, len(offsets))
    rets = Tuple(Array(writers, PAIR_RET), Array(len(offsets), Value(16)))

    def body(B, S, x):
        out = B.batch(PAIR, x, axes=(0,))
        arg = Coll(Array(len(offsets), Array(leaves, V16)), concat([out.refs.slice(o, o + leaves) for o in offsets]))
        return coll_concat(rets, [out, B.batch(rd, arg, axes=(0,))])

    defn = CompositeDefinition("WiredPopulations", 1, ("templates", "record", "counts"),
                               lambda S: ((("0.x", Array(writers, Array(2, V16))),), rets), body, register=False)
    return Program(bind(defn, templates=ids, record="0" * 128, counts=counts)), ids[1]


def _listed(positions) -> dict:
    return {k: {t: (a.tolist(), b.tolist()) for t, (a, b) in v.items()} for k, v in positions.items()}


def test_each_wired_row_reads_one_whole_field_of_one_writer_unit():
    """Uncut, reader unit 0 reads writer unit 1's field o1 and unit 1 reads unit 0's o0; cut in two rows, a reader unit's
    rows read writer unit 0's o1 and unit 1's o0, contiguous in the writers' outputs and still a field a row."""
    pid = descriptor_id(PAIR)
    prog, rid = wired_by_hand(2, [6, 0])
    assert _listed(unwired_population_program(prog, {}, ())[2]) == {(rid, "x", 0): {f"{pid}/o1": ([0], [1]),
                                                                              f"{pid}/o0": ([1], [0])}}
    prog, rid = wired_by_hand(4, [2])
    assert _listed(unwired_population_program(prog, {rid: {"x": 2}}, ())[2]) == {(rid, "x", 0): {f"{pid}/o1": ([0], [0])},
                                                                            (rid, "x", 1): {f"{pid}/o0": ([0], [1])}}


@pytest.mark.parametrize("leaves, offsets, cuts, refusal", [
    (2, [1], 1, "cut-offset"),
    (4, [0], 1, "cut-fields"),
    (4, [2], 1, "cut-units"),
    (4, [1], 2, "cut-offset"),
])
def test_a_wired_row_that_is_no_whole_field_of_one_writer_unit_is_refused_by_name(leaves, offsets, cuts, refusal):
    """A row read off its field's first leaf (cut-offset), over a unit's two fields (cut-fields) or across two writer
    units (cut-units) is refused, by name."""
    prog, rid = wired_by_hand(leaves, offsets)
    with pytest.raises(ValueError, match=refusal):
        unwired_population_program(prog, {rid: {"x": cuts}}, ())


def test_a_wired_row_at_another_width_is_no_root():
    """A row of 8-bit leaves reading a 16-bit field is refused when its root is built (E_REINTERPRET), so no wired root
    the cut reads holds one."""
    rd = bind(CompositeDefinition("CutTestRead8", 1, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                  lambda B, S, x: x[0], register=False))
    ids, rets = (descriptor_id(PAIR), descriptor_id(rd)), Tuple(Array(1, PAIR_RET), Array(1, Value(8)))

    def body(B, S, x):
        out = B.batch(PAIR, x, axes=(0,))
        return coll_concat(rets, [out, B.batch(rd, Coll(Array(1, Array(2, Value(8))), out.refs.slice(0, 2)), axes=(0,))])

    defn = CompositeDefinition("WiredPopulations", 1, ("templates", "record", "counts"),
                               lambda S: ((("0.x", Array(1, Array(2, V16))),), rets), body, register=False)
    with pytest.raises(TypeError, match="E_REINTERPRET"):
        Program(bind(defn, templates=ids, record="0" * 128, counts=(1, 1)))
