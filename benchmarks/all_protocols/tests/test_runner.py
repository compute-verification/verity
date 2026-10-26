"""The runner's decisions on one window with fake legs, and the table every lane plugs into:

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
* windows audited at once each get their report, and the monitoring-period report over all of them leads the line;
* under ``--pouw h1`` PoUW's audit of the H1 window runs before the legs, on the window's own PoUW circuit;
* the network certifier's K over a week, a month and a year with the faulty units paid for once
  (`HorizonLeakAccepted`), each window's K beside it;
* the loop's window keeps only its public files, and its circuit is the one registered before it ran (slow).
"""
from __future__ import annotations

import dataclasses
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
    d = tmp_path / "w0"
    d.mkdir()
    (d / "period.json").write_text(json.dumps({"window": 0, "period": 0, "node": "test-node", "circuit": "cd" * 64,
                                               "run_root": "ef" * 64, "law": "subset:2", "cap": 8, "tools": {},
                                               "prover_env": {"FC_ZK_SELF_CHECK": "skip"} if tampered else {}}))
    return d


def decide(tmp_path: Path, **legs) -> dict:
    """`decide` on one window with the legs given by name and every other protocol missing."""
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


OUTSIDE = tuple(dataclasses.replace(e, onsite=False) if e.name == "bandwidth" else e for e in T.TABLE)


def test_a_rejecting_audit_rejects_though_it_is_no_onsite_component(tmp_path):
    r = decide(tmp_path, integrity=fake("accept"), bandwidth=fake("reject"), table=OUTSIDE)
    assert r["composed"]["verdict"] == "reject"
    assert r["composed"]["outside_onsite"] == ["bandwidth", "wipe"]


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
        "communication": "network-accounting", "intensity": "network-accounting", "wipe": "memory-accounting"}
    assert r["composed"]["verdict"] == "pending-ports"
    assert r["protocols"]["integrity"]["quantity"] == {"bound": 1}
    assert r["protocols"]["integrity"]["delta"] == "1/2"
    assert r["protocols"]["integrity"]["epsilon"] == "1/4"
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
    assert onsite == ["NetworkCheck.Period.protocol", "NciCheck.Protocol.protocol", "PousCheck.RunProtocol.protocol",
                      "PouwCheck.protocol", "RemoteAudit.Protocol.protocol", "SampledProofs.OneStage.protocol"], \
        "onsite's components changed"
    named = sorted(e.component for e in T.TABLE if e.onsite)
    assert named == sorted(C.component_of(x)[0] for x in onsite)
    assert len({e.name for e in T.TABLE}) == len({e.component for e in T.TABLE}) == len(T.TABLE) == 7
    assert len(named) == 6
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


def test_windows_audited_one_after_another_each_report_and_the_monitoring_period_leads(tmp_path, monkeypatch, capsys):
    def loop(index, run, a, scratch, out):
        d = scratch / f"w{index}"
        d.mkdir(parents=True)
        (d / "period.json").write_text(json.dumps({"window": index, "circuit": "cd" * 64, "run_root": "ef" * 64,
                                                   "law": a.law, "cap": 8, "delta": a.delta, "prover_env": {}}))
        return d, {"population": 4}

    monkeypatch.setattr(RN, "loop_window", loop)
    decide_, auditing, overlapped = RN.decide, [], []

    def decide(d, node, out, exe=None):
        overlapped.append(bool(auditing))
        auditing.append(d)
        try:
            return decide_(d, node, out, legs={"sampled-proofs": fake("accept")}.get)
        finally:
            auditing.remove(d)

    monkeypatch.setattr(RN, "decide", decide)
    tools = tmp_path / "tools.json"
    tools.write_text("{}")
    rc = RN.main(["--out", str(tmp_path / "out"), "--tools", str(tools), "--windows", "3", "--start", "5"])
    assert rc == 0
    assert overlapped == [False, False, False]
    reports = [json.loads(x) for x in (tmp_path / "out" / "reports.jsonl").read_text().splitlines()]
    assert [r["window"] for r in reports] == [5, 6, 7]
    assert all((tmp_path / "out" / f"w{w}" / "report.json").exists() for w in (5, 6, 7))
    m = json.loads((tmp_path / "out" / "monitoring-period.json").read_text())
    assert m["integrity"]["observed"]["windows"] == 3 and m["integrity"]["observed"]["verdict"] == "accept"
    lines = [json.loads(x.split(" ", 1)[1]) for x in capsys.readouterr().out.splitlines() if x.startswith("ALLP ")]
    assert len(lines) == 3 and list(lines[-1]) == ["monitoring_period", "window"]
    assert set(lines[-1]["monitoring_period"]) >= {"week", "month", "year", "observed"}


def test_pouw_h1_audits_before_the_legs(tmp_path, monkeypatch):
    calls = []

    def python(module, argv, log=None):
        calls.append((module, list(argv)))
        log.with_suffix(".log").write_text('POUW {"verdict": "accept", "W": 262144}\n')
        tree = Path(argv[argv.index("--out") + 1])
        tree.mkdir(parents=True, exist_ok=True)
        (tree / "pouw.json").write_text('{"verdict": "accept"}')
        (tree / "rows.bin").write_bytes(b"prover's")
        return Namespace(returncode=0)

    d = period(tmp_path)
    p = json.loads((d / "period.json").read_text())
    (d / "period.json").write_text(json.dumps({**p, "pouw": {"mode": "h1", "circuit": "77" * 64},
                                               "tools": {"m0-pythonpath": "m", "flock-circuit": "b",
                                                         "flock-verify": "l", "flock-firewall": "f"}}))
    monkeypatch.setattr(RN, "_python", python)
    order = []
    monkeypatch.setattr(RN, "decide", lambda *x, **k: order.append(len(calls)) or {
        "loop": {}, "seconds": {}, "protocols": {}, "composed": {}})
    a = Namespace(node="n", pouw="h1", onsite=None)
    r = RN.window(0, "run", a, tmp_path, tmp_path / "out", (d, {}))
    assert order == [1] and calls[0][0] == "verity_torch_loop.pouw_audit"
    argv = calls[0][1]
    assert argv[argv.index("--circuit") + 1] == "77" * 64 and argv[argv.index("--window") + 1] == str(d / "pouw")
    assert "--forged" not in argv
    assert r["pouw_audit"] == {"rc": 0, "line": {"verdict": "accept", "W": 262144}}
    kept = sorted(f.name for f in (tmp_path / "out").rglob("*") if f.is_file())
    assert "pouw.json" in kept and "rows.bin" not in kept


@pytest.mark.parametrize("flag", ["--jobs", "--session-jobs"])
def test_jobs_and_session_jobs_take_only_one_naming_the_rule(flag, tmp_path, capsys):
    assert RN.one_window("1") == RN.one_session("1") == 1
    with pytest.raises(SystemExit):
        RN.main(["--out", str(tmp_path), "--tools", str(tmp_path / "tools.json"), flag, "2"])
    err = capsys.readouterr().err
    assert flag in err and "through the firewall one at a time" in err and "whichever window" in err


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
    c = C.communication(reports, period_s=60, spans={"five": 300}, deltas=(Fraction(1, 100), Fraction(1, 10**6)),
                        machines=10)
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
    none = C.communication([_communication("reject")], period_s=0, spans={"week": 7 * 86400}, deltas=(Fraction(1, 100),),
                           machines=10)
    assert none["accepted"] == 0 and none["pooled"] is None and none["week"] == {"periods": None}


@pytest.mark.slow
def test_the_loop_window_keeps_only_public_files(tmp_path):
    a = Namespace(law="subset:2", cap=8, requests="3", steps=1, tamper={0}, node="test-node", tools={"x": "y"},
                  pouw=None, delta="1/100", start=0, windows=1, weights=None)
    d, loop = RN.loop_window(0, "test", a, tmp_path / "scratch", tmp_path / "out")
    kept = sorted(f.name for f in (tmp_path / "out" / "window").iterdir())
    assert "registration.json" in kept and "circuit.json" in kept and "period.json" in kept
    assert not [f for f in kept if "private" in f]
    assert [f for f in (x.name for x in d.iterdir()) if f.endswith("-private.bin")]
    p = T.period_of(d)
    reg = json.loads((d / "registration.json").read_text())
    assert reg["window"]["circuit"] == p["circuit"] == loop["circuit_before"]
    assert p["run_root"] == loop["run"] and p["tools"] == {"x": "y"} and p["node"] == "test-node"
    assert p["prover_env"] == {"FC_ZK_SELF_CHECK": "skip"} and p["window"] == 0 and p["delta"] == "1/100"
    assert p["periods"] == 1 and "registered" not in p
    assert loop["mismatched"]
