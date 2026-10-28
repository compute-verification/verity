"""`circuit_check.by_calls`: a Program root a binding file lists in ``_by_calls_roots`` is checked by its calls and never built.
Its body is calls only, its descriptor entry decodes and round-trips, and each distinct callee relies on a target of its
family's sidecar that agrees with it in every switch.  The report joins those targets' verdicts to the root's.  The catalog,
`targets.select`, the catalog records (`units`) and the key (`cache.descriptor`) never build a callee's body.

The stand-in callee ``Leaf`` refuses to build above ``N=4``, so a root bound at ``N=1000`` fails any test that builds a body
it calls."""
from __future__ import annotations

import types

import pytest

from verity.ml.circuits import Array, bind
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.ops.scalar import BIT, F32, F32Eq
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Tuple

from circuit_check import by_calls as BC
from circuit_check import cache as CC
from circuit_check import checks as CK
from circuit_check import sidecars as SC
from circuit_check import targets as TG
from circuit_check import units as UN

BUILT_UP_TO = 4
LARGE = 1000


def _leaf_body(B, S, a, b):
    if S.N > BUILT_UP_TO:
        raise AssertionError(f"a callee's body was built at N={S.N}")
    return B.batch(F32Eq, a, b, axes=(0, 0))


def _pair(S):
    return ("x", Array(S.N, F32)), ("y", Array(S.N, F32))


Leaf = CompositeDefinition("ByCallsLeaf", 1, ("N", "MODE"), lambda S: (_pair(S), Array(S.N, BIT)), _leaf_body, register=False)
Same = CompositeDefinition("ByCallsSame", 1, (), lambda S: ((("a", F32), ("b", F32)), BIT),
                           lambda B, S, a, b: B.call(F32Eq, a, b), register=False)
Root = CompositeDefinition(
    "ByCallsRoot", 1, ("N",), lambda S: (_pair(S), Tuple(Array(S.N, BIT), Array(S.N, BIT), BIT)),
    lambda B, S, x, y: tuple_of(B.call(bind(Leaf, N=S.N, MODE="a"), x, y), B.call(bind(Leaf, N=S.N, MODE="b"), y, x),
                                B.call(Same, x[0], y[0])),
    register=False)
Flat = CompositeDefinition("ByCallsFlat", 1, ("N",), lambda S: (_pair(S), Array(S.N, BIT)),
                           lambda B, S, x, y: B.batch(F32Eq, x, y, axes=(0, 0)), register=False)
Miswired = CompositeDefinition("ByCallsMiswired", 1, ("N",), lambda S: (_pair(S), Array(S.N + 1, BIT)),
                               lambda B, S, x, y: B.call(bind(Leaf, N=S.N + 1, MODE="a"), x, y), register=False)

SIDECAR = "byc/leaf.circuit-check.py"


@pytest.fixture
def declare(monkeypatch):
    def go(*roots):
        monkeypatch.setattr(TG, "BY_CALLS", {r.id for r in roots})
    return go


@pytest.fixture
def sidecar(monkeypatch):
    """Stand-in targets of ``Leaf``'s sidecar (`by_calls._family_targets`), and what it binds that select drops."""
    def go(*fns, dropped=()):
        monkeypatch.setattr(BC, "_family_targets", lambda c: (SIDECAR, {f.id: f for f in fns}))
        monkeypatch.setattr(BC, "_family_dropped", lambda c: {f.id: f for f in dropped})
    return go


def _small():
    return bind(Leaf, N=2, MODE="a"), bind(Leaf, N=3, MODE="a"), bind(Leaf, N=2, MODE="b")


def test_a_root_is_checked_by_its_calls_naming_the_binding_each_relies_on(declare, sidecar):
    root = bind(Root, N=LARGE)
    declare(root)
    a2, _a3, b2 = _small()
    sidecar(*_small())
    e = BC.check(root)
    assert e["mode"] == "by-calls" and e["gates"] is None and e["failures"] == [], e["failures"]
    record = e["by_calls"]
    assert record["nodes"] == 3 and len(record["entry_sha512"]) == 128 and "not the whole Program" in record["entry_of"]
    rows = {r["callee"]: r for r in record["callees"]}
    assert rows[bind(Leaf, N=LARGE, MODE="a").id] == {"callee": bind(Leaf, N=LARGE, MODE="a").id, "family": Leaf.id, "nodes": 1,
                                                      "binding": a2.id, "sidecar": SIDECAR, "sizes": ["N"]}
    assert rows[bind(Leaf, N=LARGE, MODE="b").id]["binding"] == b2.id
    assert rows[bind(Same).id]["binding"] == bind(Same).id and rows[bind(Same).id]["sidecar"] is None
    assert CK.check_definition(root)["mode"] == "by-calls"
    assert [t.id for t in BC.relied_targets(root)] == [a2.id, b2.id, bind(Same).id]
    with pytest.raises(AssertionError, match=f"N={LARGE}"):
        _ = bind(Root, N=LARGE).gates


def test_a_root_whose_body_calls_a_primitive_is_refused(declare):
    flat = bind(Flat, N=2)
    declare(flat)
    e = BC.check(flat)
    assert [f["code"] for f in e["failures"]] == ["primitive-node"] and F32Eq.id in e["failures"][0]["name"]


def test_a_root_whose_wiring_doesnt_type_is_refused(declare, sidecar):
    bad = bind(Miswired, N=2)
    declare(bad)
    sidecar(*_small())
    assert [f["code"] for f in BC.check(bad)["failures"]] == ["wiring"]


def test_a_callee_with_no_target_agreeing_in_its_switches_is_unbound(declare, sidecar):
    root = bind(Root, N=LARGE)
    declare(root)
    a2 = bind(Leaf, N=2, MODE="a")
    sidecar(a2)
    e = BC.check(root)
    (f,) = e["failures"]
    assert f["code"] == "unbound" and f["callee"] == bind(Leaf, N=LARGE, MODE="b").id
    assert "agrees with it in MODE" in f["name"] and a2.id in f["name"] and "select" not in f["name"]


def test_an_unbound_callee_names_the_agreeing_binding_select_drops(declare, sidecar):
    """A sidecar that binds an agreeing target over MAX_GATES, beside a smaller one of the family that doesn't agree, has it
    dropped (`targets.select`): the failure says so, not only that none is bound."""
    root = bind(Root, N=LARGE)
    declare(root)
    sidecar(bind(Leaf, N=2, MODE="a"), dropped=[bind(Leaf, N=3, MODE="b"), bind(Leaf, N=3, MODE="a")])
    (f,) = BC.check(root)["failures"]
    assert f["code"] == "unbound" and bind(Leaf, N=3, MODE="b").id in f["name"] and "`targets.select` drops" in f["name"]
    assert bind(Leaf, N=3, MODE="a").id not in f["name"]


def test_statics_differ_by_switch_or_by_size():
    assert BC.differs({"N": 2, "M": "a", "R": {"SPLIT_K": 4, "K": "x"}}, {"N": 9, "M": "a", "R": {"SPLIT_K": 8, "K": "y"}}) \
        == (["R.K"], ["N", "R.SPLIT_K"])
    assert BC.differs({"S": (1, 2)}, {"S": (3, 4)}) == ([], ["S"])
    assert BC.differs({"B": True}, {"B": 1}) == (["B"], [])
    assert BC.differs({"N": 2}, {}) == (["N"], [])
    assert BC.differs({"E": 1e-5}, {"E": 1e-5}) == ([], [])


def test_the_report_joins_each_relied_on_targets_verdict(declare, sidecar):
    root = bind(Root, N=LARGE)
    declare(root)
    a2, _a3, b2 = _small()
    sidecar(*_small())
    e = BC.check(root)
    F = CK.Findings()
    F.fail("boolean", "x", b2.id)
    entries = [e, {"definition": a2.id, "failures": []}, {"definition": b2.id, "failures": F.failures}]
    out = BC.joined(entries)
    assert e["failures"] == [] and out[1:] == entries[1:]
    verdicts = {r["callee"]: r["verdict"] for r in out[0]["by_calls"]["callees"]}
    assert verdicts == {bind(Leaf, N=LARGE, MODE="a").id: "passed", bind(Leaf, N=LARGE, MODE="b").id: "failed",
                        bind(Same).id: "unchecked"}
    assert [f["code"] for f in out[0]["failures"]] == ["callee-failed", "callee-unchecked"]
    assert CK.summarize(out)["by_calls"] == {root.id: sorted([a2.id, b2.id, bind(Same).id])}


def test_the_catalog_its_records_and_its_key_never_build_a_by_calls_root(declare):
    root = bind(Root, N=LARGE)
    declare(root)
    found = TG.reachable([root])
    assert found == {root.id: root} and TG.select(found) == {root.id: root}
    d = CC.descriptor(root)
    assert not d.startswith("error:") and d == CC.descriptor(root)
    nodes, objs = UN._reach([root])
    assert objs == {root.id: root} and nodes[root.id]["bc"] == 1 and nodes[root.id]["d"] == d and "c" not in nodes[root.id]
    assert UN._select(nodes) == {root.id}


def test_a_binding_files_by_calls_roots_are_declared_and_each_must_be_complete(monkeypatch):
    def binding_file(**ns):
        m = types.ModuleType("byc.circuit-check")
        m.__file__ = "byc.circuit-check.py"
        vars(m).update(ns)
        monkeypatch.setattr(SC, "binding_modules", lambda: [("byc", m)])

    monkeypatch.setattr(TG, "load_registries", lambda: None)
    monkeypatch.setattr(TG, "BY_CALLS", set())
    root = bind(Root, N=LARGE)
    binding_file(_by_calls_roots=lambda: [root])
    TG.declare_by_calls()
    assert TG.BY_CALLS == {root.id} and TG.is_by_calls(root) and not TG.is_by_calls(bind(Leaf, N=2, MODE="a"))
    assert "_by_calls_roots" in TG.root_functions()
    binding_file(_by_calls_roots=lambda: [bind(Leaf, N=2)])
    with pytest.raises(TypeError, match="byc.circuit-check#_by_calls_roots"):
        TG.declare_by_calls()
