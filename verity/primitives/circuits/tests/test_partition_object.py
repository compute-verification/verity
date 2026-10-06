"""`verity/partition/v1` names the partition as a query, `Q_word` v1 (`verity.primitives.circuits.partition_object`), evaluated by the canonical evaluator
in `verity.primitives.circuits.cut`: the object's form and digest, the evaluation (owners, units, the derived committed set, `locate`), `verify` against
a Program, and the pinned vector -- descriptor bytes in, each Call's graph and evaluated cut out (`partition_vectors.json`)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from verity.primitives.circuits import codec, cut, partition_object as PO
from verity.primitives.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.primitives.circuits.program import Program
from verity.primitives.circuits.refs import tuple_of
from verity.primitives.circuits.types import Array, Tuple, Value
from verity_catalog.definitions.prims import const
from verity_catalog.definitions.scalar import F32Fabs, F32Fmaxf, I32Add

F32 = I32 = Value(32)
VECTORS = json.loads((Path(__file__).parent / "partition_vectors.json").read_text())


def _c(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


#: fmaxf(fmaxf(x0, x1), |x2|): |x2| is wiring
Max3 = _c("VMax3", (("x", Array(3, F32)),), F32, lambda B, S, x: B.call(F32Fmaxf, B.call(F32Fmaxf, x[0], x[1]), B.call(F32Fabs, x[2])))
#: (s, s + x0), s = x0 + x1: two 32-bit outputs, two units
Pair = _c("VPair", (("x", Array(2, I32)),), Tuple(I32, I32),
          lambda B, S, x: (lambda s: tuple_of(s, B.call(I32Add, s, x[0])))(B.call(I32Add, x[0], x[1])))
#: x0 + x1 computed twice inside the one unit: a redundant gate, not a recompute
Twice = _c("VTwice", (("x", Array(2, I32)),), I32,
           lambda B, S, x: B.call(I32Add, B.call(I32Add, x[0], x[1]), B.call(I32Add, x[0], x[1])))
#: a separable body: each row's Max3 its own cut
Rows = _c("VRows", (("x", Array(2, Array(3, F32))),), Array(2, F32), lambda B, S, x: B.batch(Max3, x, axes=(0,)))
#: x0 + (1 + 2): the sum of constants is structure
AddOne = _c("VAddOne", (("x", Array(1, I32)),), I32,
            lambda B, S, x: B.call(I32Add, x[0], B.call(I32Add, B.call(const(32, 1)), B.call(const(32, 2)))))


def _fan(B, S, x):
    a = B.call(I32Add, x[0], x[1])
    B.call(I32Add, x[0], x[2])                                         # dead: read by nothing, returned by nothing
    return tuple_of(B.call(I32Add, a, x[2]), B.call(I32Add, a, x[3]))


#: a committed interior value (read by two output units) and a dead gate
Fan = _c("VFan", (("x", Array(4, I32)),), Tuple(I32, I32), _fan)
CALLS = (Max3, Pair, Twice, Rows, AddOne, Fan)
Top = _c("VTop", tuple((n, fn.signature()[0][0][1]) for n, fn in zip("abcdef", CALLS)), Tuple(*[fn.signature()[1] for fn in CALLS]),
         lambda B, S, *a: tuple_of(*(B.call(fn, v) for fn, v in zip(CALLS, a))))


@pytest.fixture(scope="module")
def program():
    return Program(Top)


def test_the_width_rule_counts_bits():
    assert cut.fits(16, 1) and cut.fits(16, 2) and cut.fits(32, 1) and not cut.fits(32, 2) and not cut.fits(33, 1) and not cut.fits(17, 2)
    p = cut.CallGraph(Pair)
    assert cut.boundary_widths(p, [0, 1]) == ([32, 32], [1, 1]) and cut.boundary_widths(p, [0, 0]) == ([64], [2])
    v = cut.check_cut(p, [0, 0])
    assert v.codes == ("unit-too-wide",) and v.detail["unit-too-wide"]["gates"] == [[0, 64, 2]]
    assert cut.check_cut(p, [0, 0], limits=None).ok


def test_the_object_names_the_query(program):
    obj = PO.build(program)
    assert obj == {"format": "verity/partition/v1", "program": codec.program_digest(program),
                   "query": {"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}}}
    assert PO.digest(obj) == hashlib.sha512(b"verity/partition/v1\x00" + codec.canonical_json(obj)).hexdigest()
    assert PO.validate(obj).ok and PO.verify(obj, program).ok


@pytest.mark.parametrize("edit,code", [
    (lambda o: o.update(owners=[]), "keys"),                                    # nothing derived is stored
    (lambda o: o.update(format="verity/partition/v0"), "format"),
    (lambda o: o.update(program="ab"), "program-digest"),
    (lambda o: o["query"].update(version=3), "query-unknown"),
    (lambda o: o["query"].update(name="Q_module_body"), "query-unknown"),
    (lambda o: o["query"]["params"].pop("W"), "query-params"),
    (lambda o: o["query"]["params"].update(EXTRAS=2), "query-params"),
    (lambda o: o["query"]["params"].update(X=0), "query-params"),
    (lambda o: o["query"].update(cut="x"), "query-keys"),
])
def test_the_object_form_is_checked(program, edit, code):
    obj = copy.deepcopy(PO.build(program))
    edit(obj)
    v = PO.validate(obj)
    assert not v.ok and code in v.codes, v.codes


def test_the_evaluation(program):
    part = PO.evaluate(program)
    got = {c.fn.id.split("_v")[0]: (c.units, c.owner()) for _k, c in part.calls}
    assert got == {"VMax3": (1, [0, 0]), "VPair": (2, [0, 1]), "VTwice": (1, [0, 0, 0]), "VRows": (2, [0, 0, 1, 1]),
                   "VAddOne": (1, [0]), "VFan": (4, [2, 3, 0, 1])}
    assert part.population == 11 and [k for k, _c in part.calls] == [6, 7, 8, 9, 10, 11]      # the six Input nodes come first
    fan = [p for p, (_k, c) in enumerate(part.calls) if c.fn is Fan][0]
    assert part.calls[fan][1].kinds == ["output", "output", "committed", "dead"] and part.committed(fan) == [0, 2, 3]
    twice = [p for p, (_k, c) in enumerate(part.calls) if c.fn is Twice][0]
    assert part.graph(twice).recomputed == [(1, 0)]                          # inside one unit: redundant
    rows = [p for p, (_k, c) in enumerate(part.calls) if c.fn is Rows][0]
    assert part.calls[rows][1].parts is not None and part.locate(5) == (rows, 9, 1, [(0, 1, 0)])
    assert part.locate(0)[:3] == (0, 6, 0) and part.locate(10)[:3] == (5, 11, 3)
    with pytest.raises(IndexError):
        part.locate(11)


def test_verify_holds_serving_to_the_derived_committed_set(program):
    obj = PO.build(program)
    fan = 11
    part = PO.evaluate(program)
    full = {k: part.committed(p) for p, (k, _c) in enumerate(part.calls)}
    assert full[fan] == [0, 2, 3] and PO.verify(obj, program, served=full).ok
    (f,) = PO.verify(obj, program, served=full | {fan: [2, 3]}).detail["cut"]["gates"]  # the interior value read by two units, not served
    assert f["call"] == fan and "read-uncommitted" in f["codes"]
    (f,) = PO.verify(obj, program, served=full | {fan: [0, 1, 2, 3]}).detail["cut"]["gates"]   # the dead gate served
    assert f["codes"] == ["committed-unread"]
    v = PO.verify(obj, program, served={fan: [0, 2, 3]})                       # every Call that commits must be given
    assert v.codes == ("served-missing",) and v.detail["served-missing"]["gates"] == sorted(set(full) - {fan})
    assert PO.verify(obj, program, served=full | {999: [0]}).codes == ("served-unknown",)
    other = copy.deepcopy(obj)
    other["program"] = "f" * 128
    assert PO.verify(other, program).codes == ("program",)
    narrow = PO.build(program, PO.query(X=16, W=16))                          # Pair's and Fan's 32-bit outputs are too wide at W = 16
    assert {f["call"] for f in PO.verify(narrow, program).detail["cut"]["gates"]} >= {7, 11}


def test_a_value_two_members_compute_fails_verify():
    twice = _c("VTwiceSep", (("x", Array(2, I32)),), Tuple(I32, I32), lambda B, S, x: tuple_of(B.call(I32Add, x[0], x[1]), B.call(I32Add, x[0], x[1])))
    top = _c("VTop2", (("a", Array(2, I32)),), Tuple(I32, I32), lambda B, S, a: B.call(twice, a))
    P = Program(top)
    assert cut.separable(twice) is not None
    (f,) = PO.verify(PO.build(P), P).detail["cut"]["gates"]
    assert "gate-recomputed" in f["codes"]


def test_q_word_v2_reports_a_value_two_members_compute():
    """v2 accepts what v1 refuses only for `gate-recomputed`, with v1's units, and reports the pair; `check_cut` and
    `validate_unit_cut` refuse by default (v1's rule), and a recompute inside one unit stays a redundant gate at both."""
    twice = _c("VTwiceSep", (("x", Array(2, I32)),), Tuple(I32, I32), lambda B, S, x: tuple_of(B.call(I32Add, x[0], x[1]), B.call(I32Add, x[0], x[1])))
    P = Program(_c("VTop2", (("a", Array(2, I32)),), Tuple(I32, I32), lambda B, S, a: B.call(twice, a)))
    one, two = PO.build(P), PO.build(P, PO.query(version=2))
    assert two["query"] == {"name": "Q_word", "version": 2, "params": {"X": 16, "W": 32}} and PO.digest(two) != PO.digest(one)
    assert PO.owners_digest(PO.evaluate(P, two["query"])) == PO.owners_digest(PO.evaluate(P, one["query"]))
    v = PO.verify(two, P)
    assert v.ok and v.detail["recomputed_across"] == {"n": 1, "gates": [{"call": 1, "n": 1, "gates": [[1, 0]]}]}
    assert "recomputed_across" not in PO.verify(one, P).detail
    g = cut.CallGraph(twice)
    own = PO.evaluate(P).owner(0)
    assert own == [0, 1] and cut.check_cut(g, own).codes == ("gate-recomputed",)
    r = cut.check_cut(g, own, recompute="report")
    assert r.ok and r.detail["recomputed_across"] == {"n": 1, "gates": [[1, 0]]} and r.detail["redundant_gates"] == 0
    r = cut.check_cut(cut.CallGraph(Twice), [0, 0, 0], recompute="report")
    assert r.ok and r.detail["redundant_gates"] == 1 and r.detail["recomputed_across"]["n"] == 0
    with pytest.raises(ValueError):
        cut.check_cut(g, own, recompute="ignore")
    with pytest.raises(ValueError):
        PO.query(version=3)


def test_the_pinned_vector_bytes_in_graph_and_partition_out(program):
    desc = codec.encode_program(program)
    assert codec.canonical_json({k: v for k, v in desc.items() if k != "annotations"}).decode() == VECTORS["descriptor"]
    D = codec.decode_program(json.loads(VECTORS["descriptor"]), REGISTRY)           # the verifier's side: from the bytes alone
    assert PO.order_violation(D) is None                                             # topological: every query applies
    assert codec.program_digest(D.desc) == VECTORS["program_digest"] == codec.program_digest(program)
    obj = PO.build(D)
    assert obj == VECTORS["object"] and PO.digest(obj) == VECTORS["object_digest"]
    part = PO.evaluate(D)
    assert part.population == VECTORS["population"] and PO.owners_digest(part) == VECTORS["owners_digest"]
    for p, ((k, c), want) in enumerate(zip(part.calls, VECTORS["calls"])):
        g = part.graph(p)
        assert (k, c.fn.id) == (want["call"], want["definition"])
        assert {"total": g.total, "structure": g.free, "n": g.n, "keep": g.keep.tolist(), "width": g.width.tolist(),
                "prim": [g.prim_names[x] for x in g.prim.tolist()], "src": g.src.tolist(), "dst": g.dst.tolist(),
                "input_reads": [list(x) for x in g.input_reads], "outputs": list(g.outputs), "returned_inputs": list(g.returned_inputs),
                "recomputed": [list(x) for x in g.recomputed]} == want["graph"]
        assert (c.owner(), c.units, part.committed(p)) == (want["owner"], want["units"], want["committed"])
    assert [[i, list(part.locate(i)[:3]), [list(x) for x in part.locate(i)[3]]] for i in range(part.population)] == VECTORS["locate"]
    assert PO.verify(obj, D).ok
