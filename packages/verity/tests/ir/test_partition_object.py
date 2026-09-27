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
    return PO.build(program, [(0, [0, 0]), (1, [0, 1])])


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
    v = cut.check_cut(p, [0, 0])
    assert v.codes == ("unit-too-wide",) and v.detail["unit-too-wide"]["gates"] == [[0, 64, 2]]
    assert cut.check_cut(p, [0, 0], limits=None).ok                          # the invariant alone holds


def test_build_digest_and_canonical_bytes(program):
    obj = _good(program)
    assert obj["format"] == "verity/partition/v1" and len(obj["program"]) == 128 and set(obj) == {"format", "program", "cuts", "calls"}
    assert [c["call"] for c in obj["calls"]] == [0, 1] and len(obj["cuts"]) == 2
    for key, cut in obj["cuts"].items():
        assert key == hashlib.sha512(canonical_json(cut)).hexdigest() == PO.cut_key(cut)
    assert PO.canonical(obj) == canonical_json(obj)
    assert PO.digest(obj) == hashlib.sha512(b"verity/partition/v1\x00" + canonical_json(obj)).hexdigest()
    assert PO.validate(obj).ok and PO.verify(obj, program).ok
    assert PO.build(program, [(1, [0, 1]), (0, [0, 0])]) == obj              # Calls in canonical order


def test_every_activation_of_a_specialization_shares_one_cut():
    top = bind(CompositeDefinition("TTwice", 1, (), lambda S: ((("a", Array(3, F32)), ("b", Array(3, F32))), Tuple(F32, F32)),
                                   lambda B, S, a, b: tuple_of(B.call(Max3, a), B.call(Max3, b)), register=False))
    P = Program(top)
    obj = PO.build(P, [(0, [0, 0]), (1, [0, 0])])
    assert len(obj["cuts"]) == 1 and obj["calls"][0]["cut"] == obj["calls"][1]["cut"] and PO.verify(obj, P).ok
    U = PO.Units(obj)
    assert U.population == 2 and U.locate(1)[:3] == (1, 1, 0) and U.locate(0)[3] == U.locate(1)[3]


def test_units_are_derived_by_call_then_owner(program):
    obj = _good(program)
    U = PO.Units(obj)
    assert U.population == 3
    assert [U.locate(i)[1:3] for i in range(3)] == [(0, 0), (1, 0), (1, 1)]
    assert U.locate(2)[3] == obj["cuts"][obj["calls"][1]["cut"]]["classes"][1] == PO.unit_class("TPair_v1", 1)
    with pytest.raises(IndexError):
        U.locate(3)


def test_a_pinned_digest_vector():
    cut = {"owner": [0, 0, 1], "classes": ["a" * 128, "b" * 128]}
    key = PO.cut_key(cut)
    obj = {"format": "verity/partition/v1", "program": "0" * 128, "cuts": {key: cut}, "calls": [{"call": 0, "cut": key}, {"call": 3, "cut": key}]}
    assert PO.validate(obj).ok and PO.Units(obj).population == 4
    assert key == ("83cbe8589bf05a4d3321ba7501291785ef171796fe1150bd14594f6b85f34ec1"
                   "a80d16775d1050f6d09b3672cb772031383c065fe75bdc7551c89a612ce786db")
    assert PO.digest(obj) == ("7273d670c5b9e73fe5439a2ad7c68f0159ba5b0de1a00af499ee364830874ebc"
                              "53202037d99679b569c6551e39f51a3ac3e9be57a408c9c9e7dce233d3f41b4b")
    assert PO.digest(obj) == hashlib.sha512(b"verity/partition/v1\x00" + PO.canonical(obj)).hexdigest()


def _cut(o):
    return next(iter(o["cuts"].values()))


@pytest.mark.parametrize("edit,code", [
    (lambda o: o.update(units=[]), "keys"),                                     # a derived property is never stored
    (lambda o: o.update(format="verity/partition/v0"), "format"),
    (lambda o: o.update(program="ab"), "program-digest"),
    (lambda o: o["calls"].reverse(), "calls-order"),
    (lambda o: _cut(o).update(owner=[0, 2]), "owner-values"),
    (lambda o: _cut(o).update(owner=[1, 1]), "cut-key"),                        # an edited cut no longer matches its key
    (lambda o: o["calls"][0].update(cut="0" * 128), "call-cut"),
    (lambda o: o["cuts"].update({"f" * 128: {"owner": [0], "classes": ["c" * 128]}}), "cut-unused"),
    (lambda o: _cut(o).update(committed=[1]), "cut-keys"),                      # the committed set is derived, never stored
    (lambda o: o["calls"][0].update(widths=[32]), "call-keys"),
    (lambda o: _cut(o).update(reads=[]), "cut-keys"),
])
def test_the_object_form_is_checked(program, edit, code):
    obj = copy.deepcopy(_good(program))
    edit(obj)
    v = PO.validate(obj)
    assert not v.ok and code in v.codes, v.codes


@pytest.mark.parametrize("cut,code", [({"owner": [0, 1], "classes": ["a" * 128]}, "classes"),
                                      ({"owner": [0, 0], "classes": ["x"]}, "classes"),
                                      ({"owner": [1, 1], "classes": ["a" * 128]}, "owner-values")])
def test_a_cuts_own_form_is_checked(cut, code):
    key = PO.cut_key(cut)
    v = PO.validate({"format": "verity/partition/v1", "program": "0" * 128, "cuts": {key: cut}, "calls": [{"call": 0, "cut": key}]})
    assert not v.ok and code in v.codes, v.codes


def test_the_object_is_checked_against_the_program(program):
    obj = _good(program)
    other = copy.deepcopy(obj)
    other["program"] = "f" * 128
    assert PO.verify(other, program).codes == ("program",)
    wide = PO.build(program, [(0, [0, 0]), (1, [0, 0])])                    # Pair as one unit: 64 bits over two gates
    (f,) = PO.verify(wide, program).detail["cut"]["gates"]
    assert f["call"] == 1 and f["codes"] == ["unit-too-wide"]
    assert PO.verify(wide, program, limits=None).ok
    assert cut.derived_committed(cut.CallGraph(Pair), [0, 1]) == [0, 1] and cut.derived_committed(cut.CallGraph(Max3), [0, 0]) == [1]
    assert PO.verify(obj, program, served={1: [0, 1]}).ok                     # serving commits exactly the derived set
    (f,) = PO.verify(obj, program, served={1: [1]}).detail["cut"]["gates"]  # s is read by the other unit and returned, not served
    assert f["call"] == 1 and {"read-uncommitted", "output-uncommitted"} <= set(f["codes"])
    (f,) = PO.verify(obj, program, served={0: [0, 1]}).detail["cut"]["gates"]  # a value no other unit reads, served anyway
    assert f["call"] == 0 and f["codes"] == ["committed-unread"]
    short = PO.build(program, [(0, [0])])
    assert PO.verify(short, program).detail["cut"]["gates"][0]["codes"] == ["owner-length"]
    stray = copy.deepcopy(obj)
    stray["calls"][1]["call"] = 7
    assert "call-not-a-call" in PO.verify(stray, program).codes


def test_classes_come_from_the_backend_when_given(program):
    obj = PO.build(program, [(0, [0, 0]), (1, [0, 1])], classes=lambda call, unit: hashlib.sha512(f"{call}/{unit}".encode()).hexdigest())
    assert PO.Units(obj).locate(2)[3] == hashlib.sha512(b"1/1").hexdigest() and PO.verify(obj, program).ok
