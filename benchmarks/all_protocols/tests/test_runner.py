"""The runner's decisions on one period with fake legs, and the table every lane plugs into:

* a leg whose protocol rejects rejects the composed verdict, whatever the others answer (onsite's reject dominance);
* nothing composes to accept while onsite's components are stubs or a leg is missing, even when every leg accepts;
* a leg that raises, or answers outside the core answer's shapes, is that row's error and never a verdict, and the other
  legs still run;
* once every component is a port, onsite's own executable decides on the legs' cases;
* `LEGS` names every onsite component once, and onsite's components are read from `CoreOnsite.lean` as written;
* the horizon's union bound over periods, vacuous at the sum of δ reaching 1;
* the loop's period keeps only the window's public files, and its circuit is the one registered before it ran.
"""
from __future__ import annotations

import json
import stat
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from verity_all_protocols import compose as C, runner as RN
from verity_all_protocols.legs import LEGS, Entry, Leg, Missing, Period

PROFILE = {"covered": [], "delta": "1/2", "drawn": {"unit": 1},
           "levels": [{"draw": {"subset": 1}, "name": "unit", "query": "p", "size": 4}], "linked": False,
           "programs": ["prog"], "prescribed": []}


def answer(verdict: str, protocol: str = "verity/fake/v0") -> dict:
    if verdict == "accept":
        return {"commitment": "ab" * 64, "profile": PROFILE, "protocol": protocol, "public": {"n": 1}, "reason": None,
                "verdict": "accept"}
    return {"commitment": "ab" * 64, "profile": None, "protocol": protocol, "public": None, "reason": f"{verdict} by fake",
            "verdict": verdict}


def fake(verdict: str, quantity: dict | None = None, case: dict | None = None):
    def leg(period: Period) -> Leg:
        (period.out / "ran").write_text(str(period.index))
        return Leg(answer(verdict, f"verity/fake-{period.out.name}/v0"), quantity, case)
    return leg


def period(tmp_path: Path) -> Period:
    return Period(index=0, node="test-node", records=tmp_path / "records", out=tmp_path / "out", circuit="cd" * 64,
                  run_root="ef" * 64, law="subset:2", cap=8)


def table(**legs) -> tuple[Entry, ...]:
    return tuple(Entry(e.name, e.component, e.owner, legs.get(e.name, e.leg)) for e in LEGS)


QUANTITY = {"wrong_units_bound": 3, "units": 4, "delta": "1/2", "epsilon": "3/4"}


def test_a_rejecting_leg_rejects_the_composition(tmp_path):
    r = RN.decide(period(tmp_path), table(integrity=fake("accept", QUANTITY), space=fake("reject")))
    assert r["protocols"]["space"]["verdict"] == "reject"
    assert r["protocols"]["space"]["quantity"] is None
    assert r["protocols"]["work"] == {"verdict": "missing", "owner": "compute-accounting"}
    assert r["composed"]["verdict"] == "reject"
    assert "space" in r["composed"]["reason"]
    assert r["horizon"]["composed"] == "reject"


def test_a_rejecting_bandwidth_audit_rejects_though_it_is_no_onsite_component(tmp_path):
    r = RN.decide(period(tmp_path), table(integrity=fake("accept", QUANTITY), bandwidth=fake("reject")))
    assert r["composed"]["verdict"] == "reject"
    assert r["composed"]["outside_onsite"] == ["bandwidth"]


def test_nothing_composes_to_accept_while_onsite_has_stubs(tmp_path):
    every = {e.name: fake("accept", QUANTITY, {"commitment": "00", "evidence": "00", "params": {}}) for e in LEGS}
    r = RN.decide(period(tmp_path), table(**every), exe="/nonexistent/onsite")
    assert all(x["verdict"] == "accept" for x in r["protocols"].values())
    assert r["composed"]["verdict"] == "pending-ports"
    assert "onsite's stubs" in r["composed"]["reason"]


def test_missing_legs_are_never_verdicts(tmp_path):
    r = RN.decide(period(tmp_path), table(integrity=fake("accept", QUANTITY)))
    missing = {n: x for n, x in r["protocols"].items() if n != "integrity"}
    assert {x["verdict"] for x in missing.values()} == {"missing"}
    assert {n: x["owner"] for n, x in missing.items()} == {
        "work": "compute-accounting", "space": "memory-accounting", "bandwidth": "memory-accounting",
        "communication": "network-accounting"}
    assert r["composed"]["verdict"] == "pending-ports"
    assert r["protocols"]["integrity"]["quantity"] == QUANTITY
    assert r["protocols"]["integrity"]["delta"] == "1/2"


def test_a_failing_leg_is_an_error_and_the_others_still_run(tmp_path):
    def boom(p: Period) -> Leg:
        raise RuntimeError("the prover's build is missing")

    def malformed(p: Period) -> Leg:
        return Leg({**answer("accept"), "reason": "accepts with a reason"})

    r = RN.decide(period(tmp_path), table(integrity=boom, work=malformed, space=fake("accept", QUANTITY)))
    assert r["protocols"]["integrity"]["verdict"] == "error"
    assert "the prover's build is missing" in r["protocols"]["integrity"]["reason"]
    assert (tmp_path / "out" / "integrity" / "error.txt").exists()
    assert r["protocols"]["work"]["verdict"] == "error"
    assert "not a core answer" in r["protocols"]["work"]["reason"]
    assert r["protocols"]["space"]["verdict"] == "accept"
    assert (tmp_path / "out" / "space" / "ran").read_text() == "0"
    assert r["composed"]["verdict"] == "pending-ports"


def test_onsite_decides_once_every_component_is_a_port(tmp_path):
    seen = tmp_path / "seen.json"
    exe = tmp_path / "onsite"
    exe.write_text(f"#!{sys.executable}\n" + f'''import json, sys
p, c, e = (open(f, "rb").read() for f in sys.argv[2:5])
json.dump({{"argv1": sys.argv[1], "params": json.loads(p), "commitment": c.hex(), "evidence": json.loads(e)}},
          open({str(seen)!r}, "w"))
print(json.dumps({{"commitment": c.hex(), "profile": {{}}, "protocol": "verity/onsite/v0", "public": {{}}, "reason": None,
                  "verdict": "accept"}}))
''')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    case = {"commitment": "01", "evidence": "02", "params": {"k": 1}}
    every = {e.name: fake("accept", QUANTITY, case) for e in LEGS}
    ports = ["Ports.certifier", "Ports.pous", "Ports.pouw", "Ports.sampledProofs"]
    r = RN.decide(period(tmp_path), table(**every), onsite=ports, exe=str(exe))
    assert r["composed"]["verdict"] == "accept"
    got = json.loads(seen.read_text())
    assert got["argv1"] == "check"
    ids = [x["id"] for x in got["evidence"]["components"]]
    assert ids == sorted(ids) and len(ids) == 4 and "verity/fake-bandwidth/v0" not in ids
    assert got["commitment"] == C.bind(C.PROTOCOL, [(i, "ab" * 64) for i in ids]).hex()
    assert got["params"] == {"components": [{"id": i, "params": {"k": 1}} for i in ids]}


def test_onsite_accepting_while_bandwidth_is_undecided_is_undecided(tmp_path):
    exe = tmp_path / "onsite"
    exe.write_text(f"#!{sys.executable}\nimport json\n" + 'print(json.dumps({"commitment": None, "profile": {}, '
                   '"protocol": "verity/onsite/v0", "public": {}, "reason": None, "verdict": "accept"}))\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    case = {"commitment": "01", "evidence": "02", "params": {}}
    every = {e.name: fake("accept", QUANTITY, case) for e in LEGS}
    every["bandwidth"] = fake("undecided")
    r = RN.decide(period(tmp_path), table(**every), onsite=["Ports.a"], exe=str(exe))
    assert r["composed"]["verdict"] == "undecided"


def test_the_table_names_every_onsite_component_once():
    onsite = C.onsite_components()
    assert onsite == ["Stub.certifier", "Stub.pous", "Stub.pouw", "Stub.sampledProofs"], \
        "onsite's components changed: a port replaced a stub, so the runner may now decide through `onsite check`"
    named = sorted(e.component for e in LEGS if e.component is not None)
    assert named == sorted(C.stub_component(x) for x in onsite)
    assert len({e.name for e in LEGS}) == len(LEGS) == 5
    assert all(isinstance(e.leg, Missing) or callable(e.leg) for e in LEGS)


def test_onsite_components_are_read_as_written():
    src = "def components : List Protocol := [Ports.certifier, Stub.pous,\n  Pouw.Core.protocol, Stub.sampledProofs]"
    assert C.onsite_components(src) == ["Ports.certifier", "Stub.pous", "Pouw.Core.protocol", "Stub.sampledProofs"]
    assert [C.stub_component(x) for x in C.onsite_components(src)] == [None, "pous", None, "sampled-proofs"]
    with pytest.raises(ValueError, match="no `def components"):
        C.onsite_components("def other := []")


def _report(verdict: str, composed: str = "pending-ports") -> dict:
    q = {"wrong_units_bound": 3, "units": 1024, "delta": "1/2", "epsilon": "3/1024"}
    return {"protocols": {"integrity": {"verdict": verdict, "quantity": q if verdict == "accept" else None}},
            "composed": {"verdict": composed}}


def test_the_horizon_is_a_union_bound_over_periods():
    h = C.horizon([_report("accept")], period_s=3600)
    assert h["integrity"] == {"periods": 1, "accepted": 1, "refused": 0, "wrong_units_bound": 3, "units": 1024,
                              "delta": "1/2", "vacuous": False}
    assert h["month"] == {"periods": 720, "period_s": 3600, "delta": "1", "vacuous": True, "wrong_units_bound": 2160,
                          "units": 737280}
    h = C.horizon([_report("accept"), _report("reject", "reject"), _report("accept")], period_s=0)
    assert h["integrity"]["accepted"] == 2 and h["integrity"]["refused"] == 1
    assert h["integrity"]["delta"] == "1" and h["integrity"]["vacuous"]
    assert h["month"] == {"periods": None, "period_s": 0}
    assert h["composed"] == "reject"


def test_the_loop_period_keeps_only_public_files(tmp_path):
    a = Namespace(law="subset:2", cap=8, requests="3", steps=1, tamper={0}, node="test-node", tools={})
    p, loop = RN.loop_period(0, "test", a, tmp_path / "scratch", tmp_path / "out")
    kept = sorted(f.name for f in (tmp_path / "out" / "window").iterdir())
    assert "registration.json" in kept and "circuit.json" in kept
    assert not [f for f in kept if "private" in f]
    assert any((p.records / f).name.endswith("-private.bin") for f in (x.name for x in p.records.iterdir()))
    reg = json.loads((p.records / "registration.json").read_text())
    assert reg["window"]["circuit"] == p.circuit == loop["circuit_before"]
    assert p.run_root == loop["run"]
    assert p.prover_env == {"FC_ZK_SELF_CHECK": "skip"}
    assert loop["mismatched"]
