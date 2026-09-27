"""`verity/partition/v1` (`verity.ir.partition_object`) and the references it is checked on (`verity.ir.cut`): the build from a cut, the
canonical bytes and the SHA-512 digest (a pinned vector), the object's own form, and the object against a Program -- the invariant on
graphs derived from the Program and the width rule in bits (<= 16, or exactly one value of <= 32)."""
from __future__ import annotations

import copy
import hashlib

import pytest

from verity.ir import cut, partition_object as PO
from verity.ir.codec import canonical_json
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.program import Program
from verity.ir.refs import tuple_of
from verity.ir.types import Array, Tuple, Value
from verity.ml.scalar import F32Fabs, F32Fmaxf, I32Add

F32, I32 = Value(32), Value(32)

#: Max3(x) = fmaxf(fmaxf(x0, x1), |x2|): two computing gates, |x2| is wiring
Max3 = bind(CompositeDefinition("TMax3", 1, (), lambda S: ((("x", Array(3, F32)),), F32),
                           lambda B, S, x: B.call(F32Fmaxf, B.call(F32Fmaxf, x[0], x[1]), B.call(F32Fabs, x[2])), register=False))
#: Pair(x) = (s, s + x0) with s = x0 + x1: two 32-bit outputs, so two units, s committed and read by the second
Pair = bind(CompositeDefinition("TPair", 1, (), lambda S: ((("x", Array(2, I32)),), Tuple(I32, I32)),
                           lambda B, S, x: (lambda s: tuple_of(s, B.call(I32Add, s, x[0])))(B.call(I32Add, x[0], x[1])), register=False))
Top = CompositeDefinition("TTop", 1, (), lambda S: ((("a", Array(3, F32)), ("b", Array(2, I32))), Tuple(F32, Tuple(I32, I32))),
                          lambda B, S, a, b: tuple_of(B.call(Max3, a), B.call(Pair, b)), register=False)


@pytest.fixture(scope="module")
def program():
    return Program(bind(Top))


def _good(program):
    return PO.build(program, [(0, [0, 0], [1]), (1, [0, 1], [0, 1])])


def test_the_call_graph_removes_structure_and_keeps_the_returned_gates():
    g = cut.CallGraph(Max3)
    assert (g.n, g.free, g.total) == (2, 1, 3) and g.free_by_prim == {"F32Fabs_v1": 1}
    assert list(g.src) == [0] and list(g.dst) == [1] and g.outputs == [1] and list(g.width) == [32, 32]
    assert sorted(g.input_reads) == [(0, 0), (1, 0), (2, 1)]                   # |x2| is read through: the gate reads x2 itself
    p = cut.CallGraph(Pair)
    assert p.n == 2 and p.outputs == [0, 1] and list(zip(p.src, p.dst)) == [(0, 1)] and p.recomputed == []


def test_the_width_rule_counts_bits():
    assert cut.fits(16, 1) and cut.fits(16, 2) and cut.fits(32, 1) and not cut.fits(32, 2) and not cut.fits(33, 1) and not cut.fits(17, 2)
    assert cut.fits(18, 2, extras=2) and cut.WIDTH_RULE == (16, 32, 0)
    p = cut.CallGraph(Pair)
    assert cut.boundary_widths(p, [0, 1]) == ([32, 32], [1, 1])
    assert cut.boundary_widths(p, [0, 0]) == ([64], [2])
    v = cut.check_cut(p, [0, 0], [0, 1])
    assert v.codes == ("unit-too-wide",) and v.detail["unit-too-wide"]["gates"] == [[0, 64, 2]]
    assert cut.check_cut(p, [0, 0], [0, 1], limits=None).ok                  # the invariant alone holds


def test_build_digest_and_canonical_bytes(program):
    obj = _good(program)
    assert obj["format"] == "verity/partition/v1" and len(obj["program"]) == 128
    assert [(u["call"], u["unit"]) for u in obj["units"]] == [(0, 0), (1, 0), (1, 1)]
    assert PO.canonical(obj) == canonical_json(obj)
    assert PO.digest(obj) == hashlib.sha512(b"verity/partition/v1\x00" + canonical_json(obj)).hexdigest()
    assert PO.program_sha512(program) == PO.program_sha512(program) and PO.validate(obj).ok and PO.verify(obj, program).ok
    assert PO.build(program, [(1, [0, 1], [1, 0]), (0, [0, 0], [1])]) == obj  # Calls and committed gates in canonical order


def test_a_pinned_digest_vector():
    obj = {"format": "verity/partition/v1", "program": "0" * 128,
           "calls": [{"call": 0, "owner": [0, 0, 1], "committed": [1, 2]}],
           "units": [{"call": 0, "unit": 0, "class": "a" * 128}, {"call": 0, "unit": 1, "class": "b" * 128}]}
    assert PO.validate(obj).ok
    assert PO.canonical(obj) == (b'{"calls":[{"call":0,"committed":[1,2],"owner":[0,0,1]}],"format":"verity/partition/v1","program":"'
                                 + b"0" * 128 + b'","units":[{"call":0,"class":"' + b"a" * 128 + b'","unit":0},{"call":0,"class":"'
                                 + b"b" * 128 + b'","unit":1}]}')
    assert PO.digest(obj) == hashlib.sha512(b"verity/partition/v1\x00" + PO.canonical(obj)).hexdigest()


@pytest.mark.parametrize("edit,code", [
    (lambda o: o.update(reads=[]), "keys"),                                     # a derived property is never stored
    (lambda o: o.update(format="verity/partition/v0"), "format"),
    (lambda o: o.update(program="ab"), "program-digest"),
    (lambda o: o["calls"].reverse(), "calls-order"),
    (lambda o: o["calls"][1].update(owner=[0, 2]), "owner-values"),
    (lambda o: o["calls"][1].update(committed=[1, 0]), "committed"),
    (lambda o: o["calls"][0].update(committed=[2]), "committed"),
    (lambda o: o["units"].reverse(), "units"),
    (lambda o: o["units"].pop(), "units"),
    (lambda o: o["units"][0].update(**{"class": "x"}), "unit-class"),
    (lambda o: o["calls"][0].update(widths=[32]), "call-keys"),
])
def test_the_object_form_is_checked(program, edit, code):
    obj = copy.deepcopy(_good(program))
    edit(obj)
    v = PO.validate(obj)
    assert not v.ok and code in v.codes, v.codes


def test_the_object_is_checked_against_the_program(program):
    obj = _good(program)
    other = copy.deepcopy(obj)
    other["program"] = "f" * 128
    assert PO.verify(other, program).codes == ("program",)
    wide = PO.build(program, [(0, [0, 0], [1]), (1, [0, 0], [0, 1])])        # Pair as one unit: 64 bits over two gates
    (f,) = PO.verify(wide, program).detail["cut"]["gates"]
    assert f["call"] == 1 and f["codes"] == ["unit-too-wide"]
    assert PO.verify(wide, program, limits=None).ok
    uncommitted = PO.build(program, [(0, [0, 0], [1]), (1, [0, 1], [1])])     # s is read by the other unit and returned, not committed
    (f,) = PO.verify(uncommitted, program).detail["cut"]["gates"]
    assert {"read-uncommitted", "output-uncommitted"} <= set(f["codes"])
    short = PO.build(program, [(0, [0], [0])])
    assert PO.verify(short, program).detail["cut"]["gates"][0]["codes"] == ["owner-length"]
    stray = copy.deepcopy(obj)
    stray["calls"][1]["call"] = 7
    assert "call-not-a-call" in PO.verify(stray, program).codes


def test_classes_come_from_the_backend_when_given(program):
    obj = PO.build(program, [(0, [0, 0], [1]), (1, [0, 1], [0, 1])], classes=lambda call, unit: hashlib.sha512(f"{call}/{unit}".encode()).hexdigest())
    assert obj["units"][2]["class"] == hashlib.sha512(b"1/1").hexdigest() and PO.verify(obj, program).ok
    assert _good(program)["units"][0]["class"] == PO.unit_class("TMax3_v1", 0)
