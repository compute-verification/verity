"""The runner's decisions on one window with fake legs, and the table every lane plugs into:

* a leg whose protocol rejects rejects the composed verdict, whatever the others answer (onsite's reject dominance);
* nothing composes to accept while onsite's components are stubs or a leg is missing, even when every leg accepts;
* a leg that raises, or answers outside the core answer's shapes, is that row's error and never a verdict, and the other
  legs still run;
* once every component is a port, onsite's own executable decides on the legs' cases;
* each leg reads the answers of the legs before it, and its port's case reaches onsite;
* `TABLE` names every onsite component once, a leg is its file in `legs/`, and onsite's components are read from
  `CoreOnsite.lean` as written;
* windows audited at once each get their report, and the monitoring-period report over all of them leads the line;
* under ``--pouw h1`` PoUW's audit of the H1 window runs before the legs, on the window's own PoUW circuit;
* the loop's window keeps only its public files, and its circuit is the one registered before it ran (slow).
"""
from __future__ import annotations

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
    d = tmp_path / "w0"
    d.mkdir()
    (d / "period.json").write_text(json.dumps({"window": 0, "period": 0, "node": "test-node", "circuit": "cd" * 64,
                                               "run_root": "ef" * 64, "law": "subset:2", "cap": 8, "tools": {},
                                               "prover_env": {"FC_ZK_SELF_CHECK": "skip"} if tampered else {}}))
    return d


def decide(tmp_path: Path, **legs) -> dict:
    """`decide` on one window with the legs given by name and every other protocol missing."""
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


def test_windows_audited_at_once_each_report_and_the_monitoring_period_leads(tmp_path, monkeypatch, capsys):
    def loop(index, run, a, scratch, out):
        d = scratch / f"w{index}"
        d.mkdir(parents=True)
        (d / "period.json").write_text(json.dumps({"window": index, "circuit": "cd" * 64, "run_root": "ef" * 64,
                                                   "law": a.law, "cap": 8, "delta": a.delta, "prover_env": {}}))
        return d, {"population": 4}

    monkeypatch.setattr(RN, "loop_window", loop)
    decide_ = RN.decide
    monkeypatch.setattr(RN, "decide", lambda d, node, out, exe=None: decide_(d, node, out,
                                                                             legs={"sampled-proofs": fake("accept")}.get))
    tools = tmp_path / "tools.json"
    tools.write_text("{}")
    rc = RN.main(["--out", str(tmp_path / "out"), "--tools", str(tools), "--windows", "3", "--start", "5", "--jobs", "2"])
    assert rc == 0
    reports = [json.loads(x) for x in (tmp_path / "out" / "reports.jsonl").read_text().splitlines()]
    assert sorted(r["window"] for r in reports) == [5, 6, 7]
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
    assert order == [1] and calls[0][0] == "verity_toy.pouw_audit"
    argv = calls[0][1]
    assert argv[argv.index("--circuit") + 1] == "77" * 64 and argv[argv.index("--window") + 1] == str(d / "pouw")
    assert "--forged" not in argv
    assert r["pouw_audit"] == {"rc": 0, "line": {"verdict": "accept", "W": 262144}}


@pytest.mark.slow
def test_the_loop_window_keeps_only_public_files(tmp_path):
    a = Namespace(law="subset:2", cap=8, requests="3", steps=1, tamper={0}, node="test-node", tools={"x": "y"},
                  pouw=None, delta="1/100")
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
    assert loop["mismatched"]
