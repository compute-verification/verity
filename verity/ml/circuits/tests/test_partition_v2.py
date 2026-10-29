"""`verity/partition/v2` (`verity.ml.circuits.partition_v2`): the advice of a private partition, its rows, the statement, the
global predicates G1-G7 and their refusals, the local predicates L1-L4 on each sampled proof block (from the opened rows alone, against
the advice root of `verity.ml.commitments.advice`), v1's numbering in query mode, and the pinned vector
`partition_v2_vectors.json`.  `python verity/ml/circuits/tests/test_partition_v2.py --write` rewrites it.

The program is v1's pinned program (`partition_vectors.json`) on 16-bit values: the same Calls on BF16 instead of I32 and F32,
since the width limit of 16 bits out refuses every unit of v1's that returns a 32-bit value."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

import pytest

import test_partition_object as T1
import test_template_instance as TT

from verity.ml.circuits import codec, cut, partition_object as PO, partition_v2 as V
from verity.ml.circuits.defs import REGISTRY
from verity.ml.circuits.evaluation import evaluate_region
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.commitments import advice as A
from verity.ml.commitments.frame_v3.vectors import expand
from verity.ml.circuits.ops.fp32 import Bf16Add
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import Bf16GtStrict, SelectBf16

PATH = Path(__file__).parent / "partition_v2_vectors.json"
V1 = json.loads((Path(__file__).parent / "partition_vectors.json").read_text())
BF16 = Value(16)


def _max(B, a, b):
    return B.call(SelectBf16, B.call(Bf16GtStrict, a, b), a, b)


def _fan(B, S, x):
    a = B.call(Bf16Add, x[0], x[1])
    B.call(Bf16Add, x[0], x[2])                                        # dead: read by nothing, returned by nothing
    return tuple_of(B.call(Bf16Add, a, x[2]), B.call(Bf16Add, a, x[3]))


Max3 = T1._c("HMax3", (("x", Array(3, BF16)),), BF16, lambda B, S, x: _max(B, _max(B, x[0], x[1]), x[2]))
CALLS = (
    Max3,
    T1._c("HPair", (("x", Array(2, BF16)),), Tuple(BF16, BF16),
          lambda B, S, x: (lambda s: tuple_of(s, B.call(Bf16Add, s, x[0])))(B.call(Bf16Add, x[0], x[1]))),
    T1._c("HTwice", (("x", Array(2, BF16)),), BF16,
          lambda B, S, x: B.call(Bf16Add, B.call(Bf16Add, x[0], x[1]), B.call(Bf16Add, x[0], x[1]))),
    T1._c("HRows", (("x", Array(2, Array(3, BF16))),), Array(2, BF16), lambda B, S, x: B.batch(Max3, x, axes=(0,))),
    T1._c("HAddOne", (("x", Array(1, BF16)),), BF16,
          lambda B, S, x: B.call(Bf16Add, x[0], B.call(Bf16Add, B.call(const(16, 0x3F80)), B.call(const(16, 0x4000))))),
    T1._c("HFan", (("x", Array(4, BF16)),), Tuple(BF16, BF16), _fan),
)
Top = T1._c("HTop", tuple((n, fn.signature()[0][0][1]) for n, fn in zip("abcdef", CALLS)), Tuple(*[fn.signature()[1] for fn in CALLS]),
            lambda B, S, *a: tuple_of(*(B.call(fn, v) for fn, v in zip(CALLS, a))))


def _program() -> Program:
    return Program(Top)


def _descriptor(P: Program) -> str:
    return codec.canonical_json({k: v for k, v in codec.encode_program(P).items() if k != "annotations"}).decode()


def _v1_program() -> Program:
    """v1's pinned program, from its descriptor bytes alone: every value 32 bits."""
    return codec.decode_program(json.loads(V1["descriptor"]), REGISTRY)


@pytest.fixture(scope="module")
def program():
    return _program()


def _salts(label: str, n: int) -> list[bytes]:
    return [expand(f"verity/partition/v2/vectors/{label}/salt/{i}", A.INST.salt_bytes) for i in range(n)]


def _registered(program, q=None, preds=None, family="test", grid_bits=V.GRID_BITS, label="vtop", unchecked=False):
    adv = (V._build_unchecked if unchecked else V.build)(program, q)
    if preds is None:
        preds = V.from_v1(PO.build(program, q))
    preds = dict(preds)
    preds["grid_bits"] = grid_bits
    rs = V.pad(V.rows(adv), grid_bits)
    reg = A.register(rs, V.ROW_BITS, V.BINDING, _salts(label, len(rs)))
    return adv, rs, reg, V.statement(family, preds, adv, reg.root)


def _transcript(program, seed: int = 1) -> dict[int, int]:
    C, rnd, t = program.circuit, random.Random(seed), {}
    for i in range(C.size):
        g = C.gate(i)
        if getattr(g.prim, "family", None) == "Input":
            t[i] = rnd.getrandbits(g.prim.ret.w)
    return evaluate_region(C, C[0:C.size], t)


def test_the_grid():
    assert [V.grid(x) for x in (0, 1, 15, 16, 17, 31, 32, 33, 2049)] == [0, 1, 15, 16, 18, 32, 32, 36, 2304]
    assert V.grid(11, 2) == 12 and V.grid(2304) == 2304
    for x in range(1, 5000):
        y = V.grid(x)
        assert x <= y < x * (1 + 2 ** (1 - V.GRID_BITS)) and V.grid(y) == y
    with pytest.raises(ValueError):
        V.grid(-1)


def test_the_advice_round_trips(program):
    adv = V.build(program)
    assert adv.units == 11 and [c.node for c in adv.calls] == [6, 7, 8, 9, 10, 11]
    assert [(e.kind, e.units) for e in adv.entries] == [(V.FLAT, 1), (V.FLAT, 2), (V.FLAT, 1), (V.PARTS, 2), (V.FLAT, 1), (V.FLAT, 4)]
    assert adv.entries[3].parts == ((0, 2),)                                   # HRows' members are HMax3, entry 0, cut once
    rs = V.rows(adv)
    assert all(len(r) == V.ROW_BYTES for r in rs) and V.decode(rs) == adv == V.decode(V.pad(rs))
    for bad in (rs[:-1], [rs[0][:-1] + b"\1", *rs[1:]], [*rs, b"\1" + bytes(V.ROW_BYTES - 1)], [bytes(V.ROW_BYTES), *rs[1:]]):
        with pytest.raises(V.Malformed):
            V.decode(bad)
    assert V._payload_rows(adv.entries[0]) == 1                                  # one unit: its shape only, the owners implied
    with pytest.raises(ValueError):
        V.rows(_mutate(adv, entry=(0, {"owner": (0, 1)})))


def test_query_mode_numbers_units_as_v1(program):
    adv, rs, _reg, st = _registered(program)
    part = PO.evaluate(program)
    for i in range(adv.units):
        k, node, member, path, _j, u, _en = V.locate(st, V.Reader(rs), i)
        pk, pnode, pu, ppath = part.locate(i)
        assert (k, node, member) == (pk, pnode, 0) and [(a, b) for a, b, _c in ppath] == path
        assert u == (ppath[-1][2] if ppath else pu)


@pytest.mark.parametrize("label,name,q", TT.CASES)
def test_template_mode_numbers_units_as_v1(label, name, q):
    """Built unchecked, since the program of two templates has units of two size classes and only the numbering is at issue."""
    P = TT.PROGRAMS[name]
    adv, rs, reg, st = _registered(P, q, label=name, unchecked=True)
    assert V.check(st, rs, P).ok
    tp = PO.TemplatePartition(P, q)
    assert [V.locate(st, V.Reader(rs), i)[1:3] for i in range(adv.units)] == [tuple(tp.locate(i)) for i in range(adv.units)]
    t = _transcript(P)
    assert all(V.check_unit(st, V.Reader(rs), i, P, t).ok for i in range(st["units"]))


def test_the_global_predicates_hold(program):
    adv, rs, _reg, st = _registered(program)
    v = V.check(st, rs, program)
    assert v.ok and v.detail == {"units": 11, "calls": 6, "entries": 6, "rows": len(rs)}
    assert V.validate(st).ok and len(V.digest(st)) == 128
    assert V.check(st, rs[:-1], program).codes == ("rows-count",)


def _mutate(adv: V.Advice, call=None, entry=None, **kw) -> V.Advice:
    if call is not None:
        k, fields = call
        calls = list(adv.calls)
        calls[k] = dataclasses.replace(calls[k], **fields)
        adv = dataclasses.replace(adv, calls=tuple(calls))
    if entry is not None:
        j, fields = entry
        entries = list(adv.entries)
        entries[j] = dataclasses.replace(entries[j], **fields)
        adv = dataclasses.replace(adv, entries=tuple(entries))
    return dataclasses.replace(adv, **kw)


def _restart(adv: V.Advice) -> V.Advice:
    calls, start = [], 0
    for c in adv.calls:
        calls.append(dataclasses.replace(c, start=start))
        start += c.members * adv.entries[c.entry].units
    return dataclasses.replace(adv, calls=tuple(calls))


def _classes(adv, j, program, owner):
    from verity.ml.circuits.units import Activation
    fn = [fn for _k, fn in PO.calls(program) if V.definition_digest(fn) == adv.entries[j].definition][0]
    shape, classes = V._classes(uc.shape for uc in Activation(fn).circuits(owner))
    return {"owner": tuple(owner), "units": max(owner) + 1, "shape": shape, "classes": classes}


#: (label, edit of the advice (program, advice) -> advice, predicates edit, the refusal code)
REFUSALS = [
    ("another program", lambda P, a: _mutate(a, program="f" * 128), None, "header-program"),
    ("a Call dropped", lambda P, a: dataclasses.replace(a, calls=a.calls[:-1]), None, "call-count"),
    ("a Call's node", lambda P, a: _mutate(a, call=(0, {"node": 5})), None, "call-node"),
    ("a Call's members", lambda P, a: _mutate(a, call=(0, {"members": 2})), None, "call-members"),
    ("a Call's start", lambda P, a: _mutate(a, call=(1, {"start": 2})), None, "call-start"),
    ("a Call's entry past the table", lambda P, a: _mutate(a, call=(0, {"entry": 9})), None, "call-definition"),
    ("two Calls of one Definition at two entries", lambda P, a: _mutate(a, call=(1, {"entry": 0})), None, "call-definition"),
    ("an entry's Definition", lambda P, a: _mutate(a, entry=(2, {"definition": "0" * 128})), None, "entry-definition"),
    ("an entry no Call reaches", lambda P, a: dataclasses.replace(a, entries=a.entries + a.entries[:1]), None, "entry-unreached"),
    ("two entries of one Definition", lambda P, a: dataclasses.replace(a, entries=a.entries + a.entries[:1]), None, "entry-duplicate"),
    ("entries out of first-reach order",
     lambda P, a: _mutate(dataclasses.replace(a, entries=(a.entries[1], a.entries[0], *a.entries[2:]), calls=(
         dataclasses.replace(a.calls[0], entry=1), dataclasses.replace(a.calls[1], entry=0), *a.calls[2:])),
         entry=(3, {"parts": ((1, 2),)})), None, "entry-order"),
    ("a parts entry for a body cut whole", lambda P, a: _mutate(a, entry=(0, {"kind": V.PARTS, "owner": (), "shape": (), "classes": (),
                                                                                "parts": ((None, 1),)})), None, "entry-kind"),
    ("a parts entry's members", lambda P, a: _mutate(a, entry=(3, {"parts": ((0, 3),)})), None, "parts-shape"),
    ("a parts entry's units", lambda P, a: _mutate(a, entry=(3, {"units": 3})), None, "entry-units"),
    ("a flat entry's gate count", lambda P, a: _mutate(a, entry=(2, {"gates": 2, "owner": (0, 0)})), None, "entry-gates"),
    ("a unit no gate is in", lambda P, a: _restart(_mutate(a, entry=(1, {"owner": (0, 2), "units": 3, "shape": (0, 1, 1),
                                                                           "classes": a.entries[1].classes}))), None, "owner-dense"),
    ("HPair's two outputs in one unit", lambda P, a: _restart(_mutate(a, entry=(1, _classes(a, 1, P, [0, 0])))), None, "unit-too-wide"),
    ("HFan's units recut and the shapes left", lambda P, a: _mutate(a, entry=(5, {"shape": (0, 0, 0, 0),
                                                                                   "classes": a.entries[5].classes[:1]})), None, "shape"),
    ("more units than the canonical query's", lambda P, a: _restart(_mutate(a, entry=(2, _classes(a, 2, P, [0, 1, 2])))),
     {"width": None}, "units-canonical"),
    ("a unit over the gate cap", lambda P, a: a, {"max_gates": 2}, "unit-too-large"),
    ("a unit over the input cap", lambda P, a: a, {"max_in_bits": 32}, "unit-too-large"),
    ("HTwice's recompute across its units", lambda P, a: _restart(_mutate(a, entry=(2, _classes(a, 2, P, [0, 1, 0])))),
     {"canonical": None}, "cut"),
]


@pytest.mark.parametrize("label,edit,pedit,code", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_each_global_predicate_refuses(program, label, edit, pedit, code):
    adv = edit(program, V.build(program))
    preds = V.from_v1(PO.build(program))
    preds.update(pedit or {})
    rs = V.pad(V.rows(adv))
    st = V.statement("test", preds, adv, "0" * 128)
    v = V.check(st, rs, program)
    assert not v.ok and code in v.codes, (v.codes, v.detail)


def test_the_width_limit_refuses_every_unit_returning_a_32_bit_value():
    """At most 16 bits out, with no exception for one wider value: on v1's program, whose values are all 32 bits, every unit
    with an output is refused, though `Q_word{16, 32}` admits each as one value of at most W bits and v1 verifies it.  Its
    units without an output are of the 16-bit class and the others of the 32-bit one, so `build` refuses them and they are
    built unchecked."""
    P = _v1_program()
    assert PO.verify(PO.build(P), P).ok
    with pytest.raises(V.MixedClasses):
        V.build(P)
    for preds in (V.predicates(), V.from_v1(PO.build(P))):
        assert preds["width"] == V.WIDTH == 16
        adv, rs, _reg, st = _registered(P, preds=preds, unchecked=True)
        v = V.check(st, rs, P)
        assert v.codes == ("unit-too-wide",)
        wide = {g["definition"].split("_v")[0]: [u for u, _ob, _og in g["units"]] for g in v.detail["unit-too-wide"]["gates"]}
        assert wide == {"VMax3": [0], "VPair": [0, 1], "VTwice": [0], "VAddOne": [0], "VFan": [0, 1, 2]}
        assert all([ob, og] == [32, 1] for g in v.detail["unit-too-wide"]["gates"] for _u, ob, og in g["units"])
    _adv, rs, _reg, st = _registered(P, preds=V.predicates(32), unchecked=True)
    assert V.check(st, rs, P).ok


def test_the_size_classes():
    assert V.UNIT_CLASSES[0] == V.WIDTH and V.UNIT_CLASSES[-1] == V.UNIT_CLASS_MAX == 262_144 and len(V.UNIT_CLASSES) == 15
    assert [V.unit_class(b) for b in (0, 1, 16, 17, 32, 33, 1024, 262_144, 262_145)] == [16, 16, 16, 32, 32, 64, 1024, 262_144, None]


def test_the_builder_refuses_units_of_two_size_classes(program):
    """One unit per Call puts HPair's two 16-bit returns in one unit of 32 bits out, beside HMax3's unit of 16: refused by
    name, with the first unit of each class.  `build` takes no argument that skips the refusal; only tests reach
    `_build_unchecked` (`tests/test_boundaries.py`)."""
    assert list(inspect.signature(V.build).parameters) == ["program", "q", "owners"]
    with pytest.raises(V.MixedClasses, match=r"span 2 size classes \(16 bits: HMax3_v1 unit 0; 32 bits: HPair_v1 unit 0\)") as e:
        V.build(program, owners=lambda fn, g: [0] * g.n)
    assert e.value.spans == {16: ("HMax3_v1", 0), 32: ("HPair_v1", 0)}
    with pytest.raises(V.MixedClasses, match=r"32 bits: TDot_v1 unit 0; 64 bits: TScale_v1\{N=2,C=1.5\} unit 0"):
        V.build(*next((TT.PROGRAMS[n], q) for _l, n, q in TT.CASES if n == "mixed"))


def test_the_builder_takes_units_of_one_size_class(program):
    """`Q_word{16, 32}`'s cut of the BF16 program is all of the 16-bit class, and the population program's instances, 64 bits
    out each, all of the 64-bit one: each is built, and it holds its class's width but not the class below."""
    _adv, rs, _reg, st = _registered(program)
    assert V.check(st, rs, program).ok
    P, q = next((TT.PROGRAMS[n], q) for _l, n, q in TT.CASES if n == "population")
    _adv, rs, _reg, st = _registered(P, q, preds=V.predicates(64), label="population")
    assert V.check(st, rs, P).ok
    _adv, rs, _reg, st = _registered(P, q, preds=V.predicates(32), label="population")
    assert V.check(st, rs, P).codes == ("unit-too-wide",)


def test_the_unit_count_is_on_the_grid(program):
    adv, rs, _reg, st = _registered(program)
    off = dict(st, units=12)
    assert V.check(off, rs, program).codes == ("units-bucket",)
    assert V.validate(dict(st, units=17)).codes == ("units",)               # 17 is not on the 4-bit grid


@pytest.mark.parametrize("edit,code", [
    (lambda s: s.update(extra=1), "keys"),
    (lambda s: s.update(format="verity/partition/v1"), "format"),
    (lambda s: s.update(family=""), "family"),
    (lambda s: s["predicates"].pop("grid_bits"), "predicate-keys"),
    (lambda s: s["predicates"].update(width=[16, 32]), "predicates"),
    (lambda s: s["predicates"].update(width=0), "predicates"),
    (lambda s: s["predicates"].update(recompute="ignore"), "predicates"),
    (lambda s: s["predicates"].update(max_gates=0), "predicates"),
    (lambda s: s["predicates"].update(canonical={"name": "Q_word", "version": 9, "params": {"X": 16, "W": 32}}), "predicates"),
    (lambda s: s["predicates"].update(grid_bits=0), "predicates"),
    (lambda s: s["advice"].update(root="ab"), "advice"),
    (lambda s: s["advice"].update(rows=33), "advice"),
])
def test_the_statement_form_is_checked(program, edit, code):
    st = json.loads(json.dumps(_registered(program)[3]))
    edit(st)
    v = V.validate(st)
    assert not v.ok and code in v.codes, v.codes
    with pytest.raises(ValueError):
        V.digest(st)


def test_the_local_predicates_from_the_opened_rows_alone(program):
    """Each unit's check reads only the rows `openings` names; each opens against the advice root, and the check from those
    rows alone agrees with the check from all of them."""
    adv, rs, reg, st = _registered(program)
    t = _transcript(program)
    root, count = bytes.fromhex(st["advice"]["root"]), st["advice"]["rows"]
    for i in range(st["units"]):
        opened = {}
        for r in V.openings(st, rs, i):
            o = reg.open(r)
            assert A.verify(root, count, V.ROW_BITS, V.BINDING, o)
            opened[r] = o.row
        v = V.check_unit(st, V.Reader(opened), i, program, t)
        assert v.ok and v == V.check_unit(st, V.Reader(rs), i, program, t), (i, v)
        assert len(opened) <= 9                                                # the header, a bisection, a directory and payload rows


def test_a_wrong_value_fails_its_unit(program):
    _adv, rs, _reg, st = _registered(program)
    t = _transcript(program)
    v = V.check_unit(st, V.Reader(rs), 8, program, t)
    assert v.detail["definition"] == "HFan_v1" and v.detail["outputs"] == [42] and v.detail["inputs"] == [39, 17]
    bad = dict(t)
    bad[42] ^= 1
    w = V.check_unit(st, V.Reader(rs), 8, program, bad)
    assert w.codes == ("unit-incorrect",) and w.detail["unit-incorrect"]["gates"] == [42]
    assert all(V.check_unit(st, V.Reader(rs), i, program, bad).ok for i in range(11) if i != 8)   # only the unit computing it
    del bad[39]
    assert V.check_unit(st, V.Reader(rs), 8, program, bad).codes == ("unit-incorrect",)


def test_local_refusals(program):
    adv, rs, _reg, st = _registered(program)
    t = _transcript(program)
    assert V.check_unit(st, V.Reader(rs), 11, program, t).codes == ("unit-outside",)
    assert V.check_unit(st, V.Reader([bytes(V.ROW_BYTES)] + rs[1:]), 0, program, t).codes == ("unit-locate",)
    other = V.rows(_mutate(adv, entry=(5, {"classes": ("0" * 128,) + adv.entries[5].classes[1:]})))
    assert V.check_unit(st, V.Reader(V.pad(other)), 7, program, t).codes == ("unit-shape",)
    other = V.rows(_mutate(adv, entry=(5, {"definition": "0" * 128})))
    assert V.check_unit(st, V.Reader(V.pad(other)), 7, program, t).codes == ("unit-definition",)
    cyc = V.rows(_mutate(adv, entry=(3, {"parts": ((3, 2),)})))                 # a parts entry its own child: refused, not a loop
    assert V.check_unit(st, V.Reader(V.pad(cyc)), 4, program, t).codes == ("unit-locate",)


def test_a_filler_unit_is_empty(program):
    adv, rs, _reg, st = _registered(program, grid_bits=2)
    assert (adv.units, st["units"]) == (11, 12) and V.check(st, rs, program).ok
    assert V.locate(st, V.Reader(rs), 11) is None and V.check_unit(st, V.Reader(rs), 11, program, {}).detail == {"filler": True}
    assert V.openings(st, rs, 11) == [0]


def test_a_value_two_members_compute_is_v1s_verdict():
    """A separable body's members are cut apart, so the recompute rule is checked on the whole body: refused under v1's rule,
    accepted under `Q_word` v2's, as `partition_object.verify` rules."""
    P = _twice_top()
    adv = V.build(P)
    rs = V.pad(V.rows(adv))
    for version, ok in ((1, False), (2, True)):
        st = V.statement("t", V.from_v1(PO.build(P, PO.query(version=version))), adv, "0" * 128)
        v = V.check(st, rs, P)
        assert v.ok == ok == PO.verify(PO.build(P, PO.query(version=version)), P).ok
        assert ok or v.detail["cut"]["gates"][0]["codes"] == ["gate-recomputed"]


def _definitions(fn, out):
    if fn.id not in out:
        out[fn.id] = fn
        for node in getattr(getattr(fn, "body", None), "nodes", ()):
            if getattr(node, "fn", None) is not None:
                _definitions(node.fn, out)
    return out


def test_the_one_pass_recompute_pairs_are_the_graphs():
    """`_recomputed`, which a parts entry's recompute rule uses instead of the body's `cut.CallGraph`, gives the graph's pairs and
    computing-gate count on every Definition of the pinned programs: wiring, structure, wide fan-in, scans, batches, recomputes."""
    import test_qword_vectors as TQ
    defs: dict = {}
    for P in [*TQ.PROGRAMS.values(), *TT.PROGRAMS.values(), Program(T1.Top), _program(), _twice_top()]:
        for _k, fn in PO.calls(P):
            _definitions(fn, defs)
    checked = 0
    for fn in defs.values():
        try:
            g = cut.CallGraph(fn)
        except cut.QueryInapplicable:
            continue
        assert V._recomputed(fn) == ([tuple(p) for p in g.recomputed], g.n), fn.id
        checked += 1
    assert checked >= 30 and any(V._recomputed(fn)[0] for fn in defs.values())


def _twice_top() -> Program:
    twice = T1._c("HTwiceSep", (("x", Array(2, BF16)),), Tuple(BF16, BF16),
                  lambda B, S, x: tuple_of(B.call(Bf16Add, x[0], x[1]), B.call(Bf16Add, x[0], x[1])))
    return Program(T1._c("HTop2", (("a", Array(2, BF16)),), Tuple(BF16, BF16), lambda B, S, a: B.call(twice, a)))


def test_owner_function_mode(program):
    """Any flat cut per Definition is advice: one unit per Call here, checked without a width rule or canonical bound.  Its
    units are of the 16-bit and the 32-bit classes, so they are built unchecked."""
    adv = V._build_unchecked(program, owners=lambda fn, g: [0] * g.n)
    assert adv.units == 6 and all(e.kind == V.FLAT for e in adv.entries)
    st = V.statement("t", V.predicates(None), adv, "0" * 128)
    assert V.check(st, V.pad(V.rows(adv)), program).ok


def build() -> dict:
    P = _program()
    adv, rs, reg, st = _registered(P)
    t = _transcript(P)
    refusals = []
    for label, edit, pedit, code in REFUSALS:
        a = edit(P, V.build(P))
        preds = V.from_v1(PO.build(P))
        preds.update(pedit or {})
        rows = V.pad(V.rows(a))
        refusals.append([label, hashlib.sha512(b"".join(rows)).hexdigest(), pedit or {},
                         list(V.check(V.statement("test", preds, a, "0" * 128), rows, P).codes)])
    units = []
    for i in range(st["units"]):
        v = V.check_unit(st, V.Reader(rs), i, P, t)
        units.append([i, list(V.locate(st, V.Reader(rs), i)[:6]), V.openings(st, rs, i), v.detail["outputs"], v.detail["inputs"]])
    return {
        "about": "verity/partition/v2 on its descriptor, v1's pinned program (partition_vectors.json) on BF16 values, under "
                 "the width limit of 16 bits out: the advice rows, their "
                 "hm96-sha512/row/v2 frame-v3-sha512 registration under partition_v2.BINDING with salt i = "
                 "expand('verity/partition/v2/vectors/vtop/salt/<i>', 72) (frame_v3.vectors.expand), the statement and its "
                 "digest, each unit's [index, [call, node, member, path, entry, unit], rows opened, output gates, input gates], "
                 "and the global refusals: [label, SHA-512 of the edited advice's padded rows, predicate edits, codes].  Written by "
                 "verity/ml/circuits/tests/test_partition_v2.py --write.",
        "descriptor": _descriptor(P),
        "program_digest": codec.program_digest(P),
        "rows": [r.hex() for r in rs],
        "leaves": [x.hex() for x in reg.leaves],
        "root": reg.root.hex(),
        "statement": st,
        "statement_digest": V.digest(st),
        "units": units,
        "refusals": refusals,
    }


def test_the_pinned_vector():
    want = json.loads(PATH.read_text())
    got = json.loads(json.dumps(build()))
    for k in want:
        assert got[k] == want[k], k
    assert set(got) == set(want)
    rs = [bytes.fromhex(r) for r in want["rows"]]
    P = codec.decode_program(json.loads(want["descriptor"]), REGISTRY)           # the verifier's side: from the bytes alone
    assert codec.program_digest(P) == want["program_digest"] == codec.program_digest(_program())
    assert V.check(want["statement"], rs, P).ok and V.decode(rs) == V.build(P)
    assert A.register(rs, V.ROW_BITS, V.BINDING, _salts("vtop", len(rs))).root.hex() == want["root"] == want["statement"]["advice"]["root"]


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
