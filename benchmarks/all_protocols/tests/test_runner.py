"""The runner's decisions on one period with fake legs, and the table every lane plugs into:

* a leg whose protocol rejects rejects the composed verdict, whatever the others answer (onsite's reject dominance);
* nothing composes to accept while onsite's components are stubs or a leg is missing, even when every leg accepts;
* a leg that raises, or answers outside the core answer's shapes, is that row's error and never a verdict, and the other
  legs still run;
* once every component is a port, onsite's own executable decides on the legs' cases;
* given its executable, onsite checks every period: each stub on a case with no statement, each port with no case on one
  it refuses, and the report says where each case came from and whether onsite rejected on each rejecting leg's own case;
* each leg reads the answers of the legs before it, and its port's case reaches onsite;
* `TABLE` names every onsite component once, a leg is its file in `legs/`, and onsite's components are read from
  `CoreOnsite.lean` as written;
* the horizon's union bound over periods, vacuous at the sum of δ reaching 1;
* the loop's period keeps only the window's public files, and its circuit is the one registered before it ran (slow).
"""
from __future__ import annotations

import dataclasses
import json
import stat
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from verity_all_protocols import compose as C, runner as RN, table as T

PROFILE = {"covered": [], "delta": "1/2", "drawn": {"unit": 1},
           "levels": [{"draw": {"subset": 1}, "name": "unit", "query": "p", "size": 4}], "linked": False,
           "programs": ["prog"], "prescribed": []}


def answer(verdict: str, protocol: str = "verity/fake/v0") -> dict:
    if verdict == "accept":
        return {"commitment": "ab" * 64, "profile": PROFILE, "protocol": protocol, "public": {"bound": 1}, "reason": None,
                "verdict": "accept"}
    return {"commitment": "ab" * 64, "profile": None, "protocol": protocol, "public": None, "reason": f"{verdict} by fake",
            "verdict": verdict}


def fake(verdict: str, case: dict | None = None):
    def leg(period_dir: Path, node: str, out: Path) -> dict:
        (out / "ran").write_text(json.dumps({"node": node, "answers": sorted(f.name for f in (period_dir / "answers").iterdir())}))
        if case is not None:
            (out / f"{COMPONENT[out.name]}.case.json").write_text(json.dumps(case))
        return answer(verdict, f"verity/fake-{out.name}/v0")
    return leg


COMPONENT = {e.name: e.component for e in T.TABLE}


def period(tmp_path: Path, tampered: bool = False) -> Path:
    d = tmp_path / "p0"
    d.mkdir()
    (d / "period.json").write_text(json.dumps({"period": 0, "node": "test-node", "circuit": "cd" * 64,
                                               "run_root": "ef" * 64, "law": "subset:2", "cap": 8, "tools": {},
                                               "prover_env": {"FC_ZK_SELF_CHECK": "skip"} if tampered else {}}))
    return d


def decide(tmp_path: Path, **legs) -> dict:
    """`decide` on one period with the legs given by name and every other protocol missing."""
    kw = {k: legs.pop(k) for k in ("onsite", "exe", "table") if k in legs}
    by_component = {COMPONENT[n]: f for n, f in legs.items()}
    return RN.decide(period(tmp_path), "test-node", tmp_path / "out", legs=by_component.get, **kw)


def test_a_rejecting_leg_rejects_the_composition(tmp_path):
    r = decide(tmp_path, integrity=fake("accept"), space=fake("reject"))
    assert r["protocols"]["space"]["verdict"] == "reject"
    assert r["protocols"]["space"]["quantity"] is None
    assert r["protocols"]["work"] == {"verdict": "missing", "owner": "compute-accounting"}
    assert r["composed"]["verdict"] == "reject"
    assert "space" in r["composed"]["reason"]
    assert r["horizon"]["composed"] == "reject"


OUTSIDE = tuple(dataclasses.replace(e, onsite=False) if e.name == "bandwidth" else e for e in T.TABLE)


def test_a_rejecting_audit_rejects_though_it_is_no_onsite_component(tmp_path):
    r = decide(tmp_path, integrity=fake("accept"), bandwidth=fake("reject"), table=OUTSIDE)
    assert r["composed"]["verdict"] == "reject"
    assert r["composed"]["outside_onsite"] == ["bandwidth"]


def onsite_exe(tmp_path: Path, verdict: str = "accept", reason: str | None = None) -> Path:
    """A stand-in `onsite` that records its inputs in ``seen.json`` and answers ``verdict``."""
    exe = tmp_path / "onsite"
    exe.write_text(f"#!{sys.executable}\n" + f'''import json, sys
p, c, e = (open(f, "rb").read() for f in sys.argv[2:5])
json.dump({{"argv1": sys.argv[1], "params": json.loads(p), "commitment": c.hex(), "evidence": json.loads(e)}},
          open({str(tmp_path / "seen.json")!r}, "w"))
print(json.dumps({{"commitment": c.hex(), "profile": None if {verdict!r} != "accept" else {{}}, "protocol": "verity/onsite/v0",
                  "public": None if {verdict!r} != "accept" else {{}}, "reason": {reason!r}, "verdict": {verdict!r}}}))
''')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_nothing_composes_to_accept_while_onsite_has_stubs(tmp_path):
    every = {e.name: fake("accept", {"commitment": "00", "evidence": "00", "params": {}}) for e in T.TABLE}
    r = decide(tmp_path, **every, onsite=[*EVERY_PORT[:-1], "Stub.sampledProofs"], exe=str(onsite_exe(tmp_path)))
    assert all(x["verdict"] == "accept" for x in r["protocols"].values())
    assert r["composed"]["verdict"] == "pending-ports"
    assert "onsite's stubs" in r["composed"]["reason"]


def test_missing_legs_are_never_verdicts(tmp_path):
    r = decide(tmp_path, integrity=fake("accept"))
    missing = {n: x for n, x in r["protocols"].items() if n != "integrity"}
    assert {x["verdict"] for x in missing.values()} == {"missing"}
    assert {n: x["owner"] for n, x in missing.items()} == {
        "work": "compute-accounting", "space": "memory-accounting", "bandwidth": "memory-accounting",
        "communication": "network-accounting"}
    assert r["composed"]["verdict"] == "pending-ports"
    assert r["protocols"]["integrity"]["quantity"] == {"bound": 1}
    assert r["protocols"]["integrity"]["delta"] == "1/2"
    assert r["protocols"]["integrity"]["epsilon"] == "1/4"
    assert r["horizon"]["integrity"]["wrong_units_bound"] == 1 and r["horizon"]["integrity"]["units"] == 4
    assert json.loads((tmp_path / "out" / "answers" / "sampled-proofs.json").read_text())["verdict"] == "accept"


def test_a_failing_leg_is_an_error_and_the_others_still_run(tmp_path):
    def boom(period_dir: Path, node: str, out: Path) -> dict:
        raise RuntimeError("the prover's build is missing")

    def malformed(period_dir: Path, node: str, out: Path) -> dict:
        return {**answer("accept"), "reason": "accepts with a reason"}

    r = decide(tmp_path, integrity=boom, work=malformed, space=fake("accept"))
    assert r["protocols"]["integrity"]["verdict"] == "error"
    assert "the prover's build is missing" in r["protocols"]["integrity"]["reason"]
    assert (tmp_path / "out" / "integrity" / "error.txt").exists()
    assert r["protocols"]["work"]["verdict"] == "error"
    assert "not a core answer" in r["protocols"]["work"]["reason"]
    assert r["protocols"]["space"]["verdict"] == "accept"
    assert json.loads((tmp_path / "out" / "space" / "ran").read_text()) == {"node": "test-node", "answers": []}
    assert r["composed"]["verdict"] == "pending-ports"


PORTS = sorted(C.PORTS, key=lambda x: C.PORTS[x][1])
EVERY_PORT = [x for x in PORTS if x != "NetworkCheck.Check.protocol"]


def test_onsite_decides_once_every_component_is_a_port(tmp_path):
    exe = onsite_exe(tmp_path)
    case = {"commitment": "01", "evidence": "02", "params": {"k": 1}}
    every = {e.name: fake("accept", case) for e in T.TABLE}
    r = decide(tmp_path, **every, onsite=EVERY_PORT, exe=str(exe))
    assert r["composed"]["verdict"] == "accept"
    got = json.loads((tmp_path / "seen.json").read_text())
    assert got["argv1"] == "check"
    ids = [x["id"] for x in got["evidence"]["components"]]
    assert ids == sorted(C.PORTS[x][1] for x in EVERY_PORT)
    assert got["commitment"] == C.bind(C.PROTOCOL, [(i, "ab" * 64) for i in ids]).hex()
    assert got["params"] == {"components": [{"id": i, "params": {"k": 1}} for i in ids]}
    assert set(r["composed"]["inputs"].values()) == {"leg"}


def test_onsite_checks_every_period_beside_stubs_and_missing_cases(tmp_path):
    """Work rejects on its own case beside an undecided stub and a port with no case: onsite runs on all three, the stub
    on a case with no statement, the port on one it refuses, and it rejects on work's case."""
    exe = onsite_exe(tmp_path, "reject", "verity/pouw/v1 reject: params: no template")
    onsite = ["PouwCheck.protocol", "SampledProofs.OneStage.protocol", "Stub.pous"]
    r = decide(tmp_path, integrity=fake("accept"), work=fake("reject", {"commitment": "01", "evidence": "02", "params": {}}),
               onsite=onsite, exe=str(exe))
    c = r["composed"]
    assert c["verdict"] == "reject" and c["answer"]["verdict"] == "reject" and c["agree"]
    assert c["inputs"] == {"verity/pouw/v1": "leg", "verity/sampled-proofs/one-stage/v1": "no case (integrity accept)",
                           "verity/stub/pous/v0": "stub"}
    got = {x["id"]: x for x in json.loads((tmp_path / "seen.json").read_text())["evidence"]["components"]}
    assert got["verity/stub/pous/v0"]["evidence"] == C.STUB_CASE["evidence"]
    assert got["verity/sampled-proofs/one-stage/v1"]["commitment"] == ""
    again = tmp_path / "again"
    again.mkdir()
    exe = onsite_exe(again, "reject", "verity/sampled-proofs/one-stage/v1 reject: params: missing")
    r = decide(again, integrity=fake("accept"), work=fake("reject", {"commitment": "01", "evidence": "02", "params": {}}),
               onsite=onsite, exe=str(exe))
    assert r["composed"]["verdict"] == "reject" and not r["composed"]["agree"]


def test_a_component_compose_does_not_know_is_refused():
    with pytest.raises(ValueError, match="compose.PORTS"):
        C.component_of("Ports.a")
    assert C.component_of("Stub.sampledProofs") == ("sampled-proofs", "verity/stub/sampled-proofs/v0", True)


def test_onsite_accepting_while_an_outside_audit_is_undecided_is_undecided(tmp_path):
    exe = onsite_exe(tmp_path)
    case = {"commitment": "01", "evidence": "02", "params": {}}
    every = {e.name: fake("accept", case) for e in T.TABLE}
    every["bandwidth"] = fake("undecided")
    r = decide(tmp_path, **every, onsite=["SampledProofs.OneStage.protocol"], exe=str(exe), table=OUTSIDE)
    assert r["composed"]["verdict"] == "undecided"


def test_the_table_names_every_onsite_component_once():
    onsite = C.onsite_components()
    assert onsite == ["NetworkCheck.Period.protocol", "PousCheck.RunProtocol.protocol", "PouwCheck.protocol",
                      "RemoteAudit.Protocol.protocol", "SampledProofs.OneStage.protocol"], "onsite's components changed"
    named = sorted(e.component for e in T.TABLE if e.onsite)
    assert named == sorted(C.component_of(x)[0] for x in onsite)
    assert len({e.name for e in T.TABLE}) == len({e.component for e in T.TABLE}) == len(T.TABLE) == 5
    assert T.TABLE[0].name == "integrity", "integrity runs first: PoUW's evidence is its answer"


def test_a_leg_is_its_file(tmp_path):
    assert T.leg_of("pous", tmp_path) is None
    (tmp_path / "sampled_proofs.py").write_text("def leg(period_dir, node, out):\n    return {'node': node}\n")
    assert T.leg_of("sampled-proofs", tmp_path)(None, "n", None) == {"node": "n"}
    assert T.leg_of("sampled-proofs").__module__ == "all_protocols_leg_sampled_proofs"


def test_each_leg_reads_the_answers_before_it(tmp_path):
    decide(tmp_path, integrity=fake("reject"), work=fake("accept"))
    assert json.loads((tmp_path / "out" / "work" / "ran").read_text())["answers"] == ["sampled-proofs.json"]
    assert json.loads((tmp_path / "out" / "integrity" / "ran").read_text())["answers"] == []


def test_onsite_components_are_read_as_written():
    src = "def components : List Protocol := [Ports.certifier, Stub.pous,\n  Pouw.Core.protocol, Stub.sampledProofs]"
    assert C.onsite_components(src) == ["Ports.certifier", "Stub.pous", "Pouw.Core.protocol", "Stub.sampledProofs"]
    assert [C.stub_component(x) for x in C.onsite_components(src)] == [None, "pous", None, "sampled-proofs"]
    with pytest.raises(ValueError, match="no `def components"):
        C.onsite_components("def other := []")


def _report(verdict: str, composed: str = "pending-ports") -> dict:
    a = {"profile": {"levels": [{"size": 1024}]}}
    row = {"verdict": verdict, "quantity": {"bound": 3}, "delta": "1/2", "answer": a} if verdict == "accept" else \
        {"verdict": verdict, "quantity": None}
    return {"protocols": {"integrity": row}, "composed": {"verdict": composed}}


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


@pytest.mark.slow
def test_the_loop_period_keeps_only_public_files(tmp_path):
    a = Namespace(law="subset:2", cap=8, requests="3", steps=1, tamper={0}, node="test-node", tools={"x": "y"})
    d, loop = RN.loop_period(0, "test", a, tmp_path / "scratch", tmp_path / "out")
    kept = sorted(f.name for f in (tmp_path / "out" / "window").iterdir())
    assert "registration.json" in kept and "circuit.json" in kept and "period.json" in kept
    assert not [f for f in kept if "private" in f]
    assert [f for f in (x.name for x in d.iterdir()) if f.endswith("-private.bin")]
    p = T.period_of(d)
    reg = json.loads((d / "registration.json").read_text())
    assert reg["window"]["circuit"] == p["circuit"] == loop["circuit_before"]
    assert p["run_root"] == loop["run"] and p["tools"] == {"x": "y"} and p["node"] == "test-node"
    assert p["prover_env"] == {"FC_ZK_SELF_CHECK": "skip"}
    assert loop["mismatched"]
