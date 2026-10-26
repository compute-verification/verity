"""The runner's decisions on one period with fake legs, and the table every lane plugs into:

* a leg whose protocol rejects rejects the composed verdict, whatever the others answer (onsite's reject dominance);
* nothing composes to accept while onsite's components are stubs or a leg is missing, even when every leg accepts;
* a leg that raises, or answers outside the core answer's shapes, is that row's error and never a verdict, and the other
  legs still run;
* once every component is a port, onsite's own executable decides on the legs' cases;
* each leg reads the answers of the legs before it, and its port's case reaches onsite;
* `TABLE` names every onsite component once, a leg is its file in `legs/`, and onsite's components are read from
  `CoreOnsite.lean` as written;
* the horizon's union bound over periods, vacuous at the sum of δ reaching 1, and the network certifier's K over a
  week, a month and a year with the faulty units paid for once (`HorizonLeakAccepted`), each period's K beside it;
* the loop's period keeps only the window's public files, and its circuit is the one registered before it ran (slow).
"""
from __future__ import annotations

import json
import stat
import sys
from argparse import Namespace
from fractions import Fraction
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
    kw = {k: legs.pop(k) for k in ("onsite", "exe") if k in legs}
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


def test_a_rejecting_bandwidth_audit_rejects_though_it_is_no_onsite_component(tmp_path):
    r = decide(tmp_path, integrity=fake("accept"), bandwidth=fake("reject"))
    assert r["composed"]["verdict"] == "reject"
    assert r["composed"]["outside_onsite"] == ["bandwidth"]


def test_nothing_composes_to_accept_while_onsite_has_stubs(tmp_path):
    every = {e.name: fake("accept", {"commitment": "00", "evidence": "00", "params": {}}) for e in T.TABLE}
    r = decide(tmp_path, **every, exe="/nonexistent/onsite")
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
    every = {e.name: fake("accept", case) for e in T.TABLE}
    ports = ["Ports.certifier", "Ports.pous", "Ports.pouw", "Ports.sampledProofs"]
    r = decide(tmp_path, **every, onsite=ports, exe=str(exe))
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
    every = {e.name: fake("accept", case) for e in T.TABLE}
    every["bandwidth"] = fake("undecided")
    r = decide(tmp_path, **every, onsite=["Ports.a"], exe=str(exe))
    assert r["composed"]["verdict"] == "undecided"


def test_the_table_names_every_onsite_component_once():
    onsite = C.onsite_components()
    assert onsite == ["Stub.certifier", "Stub.pous", "Stub.pouw", "Stub.sampledProofs"], \
        "onsite's components changed: a port replaced a stub, so the runner may now decide through `onsite check`"
    named = sorted(e.component for e in T.TABLE if e.onsite)
    assert named == sorted(C.stub_component(x) for x in onsite)
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


def _period(units: int, drawn: int, K: int = 24, schedules: int = 4, cells: int = 3, free: int = 1,
            jitter: int = 2) -> dict:
    return {"K": K, "eps": "1/100", "windows": 1, "schedules": schedules, "cells": cells,
            "audits": [{"units": units, "drawn": drawn, "free": free, "jitter": jitter}]}


def _communication(verdict: str, quantity: dict | None = None) -> dict:
    row = {"verdict": verdict, "protocol": C.CERTIFIER, "quantity": quantity if verdict == "accept" else None}
    return {"protocols": {"integrity": {"verdict": "missing"}, "communication": row}, "composed": {"verdict": "reject"}}


def test_the_horizon_pays_for_the_faulty_units_once():
    def fmax(units: int, drawn: int, delta: Fraction) -> int:
        return C.horizon_leak([_period(units, drawn)], delta)["Fmax"]
    assert fmax(1024, 64, Fraction(1, 100)) == 74
    assert fmax(1024, 2, Fraction(1, 100)) == 2358
    mixed = {**_period(3, 3), "audits": [*_period(3, 3)["audits"], *_period(2, 1)["audits"]]}
    assert C.horizon_leak([mixed], Fraction(1, 100))["q"] == "1/2"
    assert C.horizon_leak([mixed], Fraction(1, 100))["Fmax"] == 10


def test_the_network_certifiers_k_is_pooled_once_per_span_with_the_per_window_figure_beside_it():
    p = _period(3, 3)
    reports = [_communication("accept", p), _communication("reject"), _communication("accept", p)]
    c = C.communication(reports, period_s=60, spans={"five": 300}, machines=10)
    assert {k: c[k] for k in ("periods", "accepted", "refused")} == {"periods": 3, "accepted": 2, "refused": 1}
    assert c["per_window"] == [{"K": 24, "eps": "1/100", "windows": 1, "share": 0.5}] * 2
    so_far = c["pooled"]["1/100"]
    assert (so_far["units"], so_far["Fmax"], so_far["K"]) == (6, 5, 8 + 6 + 80 + 15 + 2 + 4)
    five = c["five"]["pooled"]["1/100"]
    assert c["five"]["periods"] == 5 and five["windows"] == 5 and five["q"] == "1"
    assert five["terms"] == {"schedules": 20, "cells": 15, "faulty_bits": 80, "faulty_sets": 20, "free": 5,
                             "jitter": 10}
    assert (five["units"], five["Fmax"], five["K"], five["share"]) == (15, 5, 150, 0.625)
    assert five["e_for_eps_2delta"] == "1/500" and five["eps"].startswith("1/100 + ") and five["conditional"]
    assert c["five"]["pooled"]["1/1000000"]["Fmax"] == 14 and c["five"]["pooled"]["1/1000000"]["K"] == 330
    many = c["five"]["fleet"]["1/100"]
    assert (many["machines"], many["delta_each"], many["Fmax_each"], many["K"], many["units"]) == (
        10, "1/1000", 7, 1900, 150)
    none = C.horizon([_report("accept")], period_s=0)["communication"]
    assert none["accepted"] == 0 and none["pooled"] is None and none["week"] == {"periods": None}


@pytest.mark.slow
def test_the_loop_period_keeps_only_public_files(tmp_path):
    a = Namespace(law="subset:2", cap=8, requests="3", steps=1, tamper={0}, node="test-node", tools={"x": "y"},
                  periods=1, weights=None)
    d, loop = RN.loop_period(0, "test", a, tmp_path / "scratch", tmp_path / "out")
    kept = sorted(f.name for f in (tmp_path / "out" / "window").iterdir())
    assert "registration.json" in kept and "circuit.json" in kept and "period.json" in kept
    assert not [f for f in kept if "private" in f]
    assert [f for f in (x.name for x in d.iterdir()) if f.endswith("-private.bin")]
    p = T.period_of(d)
    reg = json.loads((d / "registration.json").read_text())
    assert reg["window"]["circuit"] == p["circuit"] == loop["circuit_before"]
    assert p["run_root"] == loop["run"] and p["tools"] == {"x": "y"} and p["node"] == "test-node"
    assert p["periods"] == 1
    assert p["prover_env"] == {"FC_ZK_SELF_CHECK": "skip"}
    assert loop["mismatched"]
