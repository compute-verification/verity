"""chain_record's steps that read no prover: custody holds each gate run's files to its end.json; the report needs the endpoint's step on the advice profile and never accepts a stand-in, merges N2's members proved in
several runs, and counts a negative that escaped a sampled sample within the sample's bound but never one whose incorrect
window was sampled; the sample's bound at δ is the integrity profile's count bound, which counts a miss of exactly δ, as
Lean's does; leak counts the scan's widths and terms on a network attester run's files, and refuses a sample and a
network attester the scan's class doesn't serve; the job's copy of the
gates' records into its output keeps no gate run's witness/. On the attester-only path the report needs no scan,
names what it stubs, and counts the rehearsal's refused runs and the I/O tie's verdict on each tampered run the check
let through as negatives; d1's K is all_protocols' monitoring-period count over the compute audits' samples and free
inputs and the network's own terms. Given no N2 summary the report is the scan's alone, its verdict `scan accepted`
(exit 0) when the honest round is accepted, every other refused and no witness piece reached the verifier (`scan
accepted on the period check` when the honest round's registration let a tied value through), and `scan refused`
(exit 1) on a wrong verdict, a leak or a missing record; the full report holds the scan to its negatives too. On
certifier/v0's period answer, a report writes no block record's certifier entry: an answer the check accepts on an
accepted chain gets the rows form's reason, and another case, a missing case or answer, and a refused or scan-only
chain each their own."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "network_traces")
PERIOD_CASE = HERE.parents[1] / "tests" / "verity" / "protocols" / "resource_accounting" / "network" / "verifier" / "network" / "period_case.py"
BLOCK_SCHEMA = Path(__file__).resolve().parents[3] / "benchmarks/one_stage/interfaces/verity_e2e_demo/schemas/block-record.v2.schema.json"


def _load(name: str):
    sys.path.insert(0, str(HERE))
    sys.path.append(str(HERE.parents[1]))  # the root, for benchmarks.network_traces
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load("chain_record")


def test_custody(tmp_path):
    run = tmp_path / "c20261025T000000Z-0000"
    (run / "out" / "honest").mkdir(parents=True)
    (run / "out" / "honest" / "a.json").write_text("{}")
    digest = hashlib.sha512(b"{}").hexdigest()
    (run / "end.json").write_text(json.dumps({"ended": "ended", "files": {"honest/a.json": digest}}))
    (run / "request.json").write_text(json.dumps({"peer": "a-node", "by": "a-lane", "port": 8011, "requests": 4}))
    got = cr.custody(tmp_path)
    assert got["verdict"] == "accepted" and got["gate_runs"][0]["request"] == {"port": 8011, "requests": 4}
    assert "ended_utc" not in got["gate_runs"][0]
    (run / "out" / "honest" / "a.json").write_text("{ }")
    got = cr.custody(tmp_path)
    assert got["verdict"] == "refused" and got["gate_runs"][0]["mismatches"] == {"honest/a.json": "sha512 differs"}
    assert cr.custody(tmp_path / "none")["verdict"] == "refused"


def test_job_keeps_no_gate_runs_witness(tmp_path):
    """chain_record.sh's copy of the gates' records into $OUT, which custody uploads, leaves every witness/ behind."""
    script = Path(__file__).resolve().parents[3] / "benchmarks/network_traces/chain_record.sh"
    lines = script.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith('(cd "$G" && find'))
    copy = "\n".join(lines[i:i + 2])
    run = tmp_path / "G" / "g2" / "c20261026T000000Z-0000"
    for name in ("end.json", "out/deployment.json", "witness/check.json", "witness/cells.jsonl"):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text("{}")
    (tmp_path / "OUT" / "gates").mkdir(parents=True)
    subprocess.run(["bash", "-c", copy], check=True, env={"G": str(tmp_path / "G"), "OUT": str(tmp_path / "OUT"),
                                                          "PATH": os.environ["PATH"]})
    kept = sorted(p.relative_to(tmp_path / "OUT" / "gates" / "g2" / run.name).as_posix()
                  for p in (tmp_path / "OUT").rglob("*") if p.is_file())
    assert kept == ["end.json", "out/deployment.json"]


def _steps(tmp_path, job: dict, endpoint: bool = True) -> SimpleNamespace:
    def put(name, doc):
        (tmp_path / name).write_text(json.dumps(doc))
        return str(tmp_path / name)

    window = {"accepted": True, "proof": {"proved": True}, "as_expected": True}
    return SimpleNamespace(
        custody=put("custody.json", {"verdict": "accepted"}),
        certs=put("certs.json", {"verdict": "accepted", "windows": 1}),
        endpoint=put("endpoint.json", {"chain": {
            "bits": {"bound": True, "rebuilt": True, "commit_windows": [{"bound": True, "rebuilt": True}]},
            "tie": {"ran": True, "runs": {"honest": {"verdict": "accepted"}}}}}) if endpoint else None,
        n2=put("n2.json", {"ok": True, "members": {
            "honest": {"windows": [window], "network": {"accepted": True}, "ok": True},
            "tampered": {"windows": [window | {"accepted": False}], "network": {"accepted": False}, "ok": True}}}),
        scan=put("scan.json", {"ok": True, "T": 1, "rounds": {"honest-0": {"audit": {"window_accepted": True}}}}),
        job=put("job.json", job), network_certifier_run="g1/network_attester")


def test_report_holds_the_advice_profile_to_the_endpoint_and_refuses_stand_ins(tmp_path):
    ok = {"profile": "advice", "rc": {"n2": "0", "scan": "0"}}
    got = cr.report(_steps(tmp_path, ok))
    assert got["verdict"] == "accepted" and got["negatives_refused"] == {"n2/tampered": True}
    assert got["d1"]["acc"] == [True]
    assert cr.report(_steps(tmp_path, ok, endpoint=False))["verdict"] == "refused"
    assert cr.report(_steps(tmp_path, ok | {"profile": "planned"}, endpoint=False))["verdict"] == "accepted"
    assert cr.report(_steps(tmp_path, ok | {"stand_in": {"n2": "the simulated link"}}))["verdict"] == "refused"
    steps = _steps(tmp_path, ok)
    steps.network_certifier_run = None
    assert cr.report(steps)["verdict"] == "refused"
    failed = cr.report(_steps(tmp_path, {"profile": "advice", "rc": {"n2": "1", "scan": "0"},
                                         "why": {"n2": "ValueError: no profile"}}))
    assert failed["verdict"] == "refused"
    assert (failed["steps"]["n2"]["verdict"], failed["steps"]["n2"]["why"]) == ("failed", "ValueError: no profile")
    steps = _steps(tmp_path, {"profile": "advice", "rc": {"n2": "1", "scan": "0"}, "why": {"n2": "done: FAILED"}})
    n2 = json.loads(Path(steps.n2).read_text())
    n2["ok"] = False
    n2["members"]["honest"]["received"] = {"control": {"holds_all": False, "blobs_found": 57, "of": 337,
                                                       "file": "prover-instances.bin"}}
    Path(steps.n2).write_text(json.dumps(n2))
    got = cr.report(steps)["steps"]["n2"]
    assert got["verdict"] == "failed" and got["members"]["honest"]["accepted_windows"] == 1
    assert got["why"].startswith("honest: the control (the prover's own prover-instances.bin) doesn't hold all 337")


def test_report_merges_runs_and_counts_a_negative_that_escaped_the_sample(tmp_path):
    ok = {"profile": "advice", "rc": {"n2": "0", "scan": "0"}}
    steps = _steps(tmp_path, ok)
    sample = {"selection": "sampled", "law": "subset:2", "k": 2, "population": 4}
    window = {"accepted": True, "proof": {"proved": True}, "as_expected": True}
    reference = [{"window": w, "expect": int(w != 0)} for w in range(4)]

    def run(name, member, units):
        doc = {"draw": sample, "ok": name == "honest", "members": {name: member},
               "sampled": {name: {"draw": {"units": units}, "why": []}}}
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
        return str(tmp_path / f"{name}.json")

    structured = {"leaked": [], "control": {"holds_all": True}}
    steps.n2 = [run("honest", {"windows": [window] * 2, "network": {"accepted": True}, "received": structured}, [1, 2]),
                run("moved", {"windows": [window] * 2, "network": {"accepted": True}, "reference": reference}, [1, 2])]
    got = cr.report(steps)
    n2s = got["steps"]["n2"]
    assert got["verdict"] == "accepted" and got["negatives_refused"] == {"n2/moved": "escaped the sample"}
    assert not {"structured", "prove_s", "verify_s"} & set(n2s["members"]["honest"])
    assert n2s["members"]["moved"]["escaped_the_draw"] == {"incorrect_windows": [0], "drawn": [1, 2]}
    assert n2s["eps_at_delta"]["law"] == "subset:2" and (n2s["eps_at_delta"]["units"], n2s["eps_at_delta"]["drawn"]) == (4, 2)
    assert n2s["eps_at_delta"]["incorrect_at_delta"]["0.5"] == 1
    steps.n2[1] = run("moved", {"windows": [window] * 2, "network": {"accepted": True}, "reference": reference}, [0, 1])
    got = cr.report(steps)["steps"]["n2"]
    assert got["verdict"] == "failed" and got["why"] == "moved: network accepted"
    steps.n2[0] = run("honest", {"windows": [window] * 2, "network": {"accepted": True},
                                 "received": {**structured, "leaked": ["head egress 45 @320"]}}, [1, 2])
    got = cr.report(steps)["steps"]["n2"]
    assert got["why"].startswith("honest: witness pieces leaked to the verifier: ['head egress 45 @320']")


@pytest.fixture(scope="module")
def period():
    spec = importlib.util.spec_from_file_location("period_case", PERIOD_CASE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["period_case"] = mod
    spec.loader.exec_module(mod)
    return mod


def _period(d: Path, case: dict, answer: dict) -> str:
    """all_protocols' runner's pair, ``certifier.answer.json`` with ``certifier.case.json`` beside it."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "certifier.case.json").write_text(json.dumps(case))
    (d / "certifier.answer.json").write_text(json.dumps(answer))
    return str(d / "certifier.answer.json")


def _chain_of(tmp_path) -> SimpleNamespace:
    """An accepted chain with no compute audits."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    steps = _steps(tmp_path, {"profile": "advice", "rc": {"n2": "0", "scan": "0"}, "run": "r-chain"})
    Path(steps.custody).write_text(json.dumps({"verdict": "accepted", "gate_runs": [{"gate_run": "c1"}]}))
    steps.audit = []
    return steps


def test_no_block_entry_is_written_for_an_answer_the_period_check_accepts(tmp_path, period):
    """A certifier/v0 case from its Python mirror (`period_case`, held to `NetworkCheck.Period` by its difftest),
    accepted, on an accepted chain."""
    case = period.case("accept", 1)
    answer = period.reference(case)
    assert answer["verdict"] == "accept"
    steps = _chain_of(tmp_path)
    honest = steps.period_answer = _period(tmp_path / "period", case, answer)
    got = cr.report(steps)
    assert got["verdict"] == "accepted" and got["block"] == {"written": False, "why": cr.ROWS_FORM}
    assert f"no certifier entry: {cr.ROWS_FORM}" in cr.markdown(got)
    out = tmp_path / "out"
    assert cr.main(["report", "--custody", steps.custody, "--certs", steps.certs, "--endpoint", steps.endpoint,
                    "--n2", steps.n2, "--scan", steps.scan, "--job", steps.job, "--warden-run", steps.network_certifier_run,
                    "--audit", *steps.audit, "--period-answer", steps.period_answer, "--out", str(out)]) == 0
    assert not (out / "block.json").exists()

    def why(**change):
        for k, v in change.items():
            setattr(steps, k, v)
        block = cr.report(steps)["block"]
        assert block["written"] is False
        return block["why"]

    other = period.case("accept", 0)
    assert why(period_answer=_period(tmp_path / "mixed", other, answer)) == (
        "the case isn't the answer's: its commitment's digest is another")
    steps.period_answer = honest
    assert why(period_case=str(tmp_path / "none.json")).endswith("none.json (--period-case): what the answer commits "
                                                                 "to can't be held to the chain")
    steps.period_case = None
    assert why(period_answer=str(tmp_path / "none.json")).endswith("none.json: not a file")
    assert why(period_answer=None) == cr.NO_PERIOD_ANSWER
    refused = cr.report(SimpleNamespace(**{**vars(steps), "period_answer": honest, "network_certifier_run": None}))
    assert refused["verdict"] == "refused" and refused["block"]["why"].startswith("the chain is refused")
    assert "no certifier entry: the chain is refused" in cr.markdown(refused)
    scan_only = cr.report(SimpleNamespace(**{**vars(steps), "n2": None, "period_answer": honest}))
    assert scan_only["verdict"] == "scan accepted" and scan_only["block"]["why"] == (
        "the chain is scan accepted: only an accepted chain carries a K")


def test_a_scan_only_report_exits_0_on_the_expected_verdicts_and_1_on_a_wrong_one_a_leak_or_a_missing_record(tmp_path):
    steps = _steps(tmp_path, {"profile": "advice", "rc": {"n2": None, "scan": "0"}, "scan_only": True})
    n2_summary, steps.n2 = steps.n2, None
    rnd = lambda accepted, found=(): {"audit": {"window_accepted": accepted, "received": {"verifier_found": list(found)}}}
    put = lambda honest, forged, ok=True: Path(steps.scan).write_text(json.dumps(
        {"ok": ok, "T": 32, "rounds": {"honest-0": honest, "chain-0": forged}}))
    args = ["report", "--custody", steps.custody, "--certs", steps.certs, "--endpoint", steps.endpoint, "--scan",
            steps.scan, "--job", steps.job, "--warden-run", steps.network_certifier_run]
    run = lambda name: (cr.main(args + ["--out", str(tmp_path / name)]),
                        json.loads((tmp_path / name / "report.json").read_text()))

    put(rnd(True), rnd(False))
    rc, got = run("pass")
    assert rc == 0 and got["verdict"] == "scan accepted" and got["scope"] == cr.SCAN_ONLY == "scan only, N2 not given"
    assert got["steps"]["n2"]["verdict"] == "not given" and got["negatives_refused"] == {"scan/chain-0": True}
    assert "Verdict: **scan accepted**, scan only, N2 not given." in (tmp_path / "pass" / "report.md").read_text()

    put({"audit": {**rnd(True)["audit"], "rests_on_period_check": ["prog"]}}, rnd(False))
    rc, got = run("tied")
    scan = got["steps"]["scan"]
    assert rc == 0 and got["verdict"] == f"scan {cr.RESTS}" == "scan accepted on the period check"
    assert scan["verdict"] == cr.RESTS and scan["rests_on_period_check"] == ["prog"] and "refuses" in scan["why"]
    assert scan["rounds"]["honest-0"]["rests_on_period_check"] == ["prog"] and got["why"]["scan"].startswith(cr.RESTS)
    md = (tmp_path / "tied" / "report.md").read_text()
    assert f"Verdict: **scan {cr.RESTS}**" in md and f"| scan | {cr.RESTS} |" in md and "| scan | accepted |" not in md

    put(rnd(True), rnd(True))
    rc, got = run("forged-accepted")
    assert rc == 1 and got["scope"] == cr.SCAN_ONLY and got["verdict"] == "scan refused"
    md = (tmp_path / "forged-accepted" / "report.md").read_text()
    assert "Verdict: **scan refused**, scan only, N2 not given." in md
    assert (got["steps"]["scan"]["verdict"], got["steps"]["scan"]["why"]) == ("failed", "chain-0: window accepted")
    put(rnd(False), rnd(False))
    rc, got = run("honest-refused")
    assert rc == 1 and got["verdict"] == "scan refused"
    put(rnd(True, ["head egress 3 @64"]), rnd(False))
    rc, got = run("leak")
    assert rc == 1 and got["verdict"] == "scan refused"
    assert got["steps"]["scan"]["why"] == "honest-0: witness pieces found at the verifier: ['head egress 3 @64']"
    put(rnd(True), rnd(False), ok=False)
    assert run("not-ok")[0] == 1
    Path(steps.scan).unlink()
    rc, got = run("no-scan")
    assert rc == 1 and got["verdict"] == "scan refused" and got["steps"]["scan"]["verdict"] == "failed"
    put(rnd(True), rnd(False))
    Path(steps.custody).unlink()
    rc, got = run("no-custody")
    assert rc == 1 and got["verdict"] == "scan refused"

    steps.n2 = n2_summary
    Path(steps.custody).write_text(json.dumps({"verdict": "accepted"}))
    put(rnd(True), rnd(False))
    got = cr.report(steps)
    assert got["scope"] is None and got["verdict"] == "accepted"
    assert "Verdict: **accepted**." in cr.markdown(got)
    put(rnd(True), rnd(True))
    got = cr.report(steps)
    assert got["scope"] is None and got["verdict"] == "refused" and got["negatives_refused"]["scan/chain-0"] is False


def test_the_scans_block_round_is_accepted_like_its_honest_round_and_is_no_negative(tmp_path):
    def rnd(accepted):
        return {"audit": {"window_accepted": accepted}}

    def scan(block):
        return {"ok": True, "T": 1, "rounds": {"block-0": rnd(block), "honest-0": rnd(True), "chain-0": rnd(False),
                                               "block-1": rnd(True), "honest-1": rnd(True), "stale-2": rnd(False)}}

    got = cr.scan_step(scan(True), "g1/network_attester")
    assert got["verdict"] == "accepted" and got["negatives"] == {"chain-0": True, "stale-2": True}, got
    got = cr.scan_step(scan(False), "g1/network_attester")
    assert (got["verdict"], got["why"]) == ("failed", "block-0: window refused")
    second = scan(True)
    second["rounds"]["honest-1"] = rnd(False)
    assert cr.scan_step(second, "g1/network_attester")["verdict"] == "refused"


def test_leak_counts_the_scans_terms_on_the_network_certifier_run(tmp_path):
    import math
    from dataclasses import asdict
    from verity.protocols.resource_accounting.network.prover import scan_zk as sz
    run = tmp_path / "c1"
    (run / "out").mkdir(parents=True)
    (run / "out/deployment.json").write_text(json.dumps({"params": asdict(sz.PARAMS)}))
    scan = {"draw": {"honest": 65}, "T": 32}
    (run / "end.json").write_text(json.dumps({"run": "c1", "verdict": {"accepted": True, "windows": 75}}))
    for request, why in ((None, "asks for no number of requests"), ({"client_port": 8012, "windows": 75}, "relay session"),
                         ({"requests": True}, "asks for no number"), ({"requests": 0}, "asks for no number")):
        if request is not None:
            (run / "request.json").write_text(json.dumps(request))
        with pytest.raises(SystemExit, match=why):
            cr.leak(run, scan)
    (run / "request.json").write_text(json.dumps({"device": "warden", "requests": 100}))
    for verdict in ({"accepted": False, "windows": 75}, {"windows": 75, "requests": 100}):
        (run / "end.json").write_text(json.dumps({"run": "c1", "verdict": verdict}))
        with pytest.raises(SystemExit, match="isn't accepted"):
            cr.leak(run, scan)
    (run / "end.json").write_text(json.dumps({"run": "c1", "verdict": {"accepted": True, "windows": 75}}))
    doc = cr.leak(run, scan)
    assert doc["run"]["requests"] == 100 and doc["run"]["requests_per_link_window"] == 100 / 75
    assert doc["widths"] == {"NS": 149_648, "NE": 1_136, "entries": 8, "NV": 4_832, "egress_row": 65_536}
    assert doc["terms"]["b"] == 154_480 and doc["terms"]["fmax"] == {"0.01": 5, "1e-06": 14}
    assert doc["terms"]["schedules_per_request"] == doc["terms"]["cells_per_request"] == 7
    assert (doc["terms"]["jitter_bound"], doc["terms"]["jitter_per_block"]) == (2, 32)
    assert doc["links"] == {"link": 1, "datacenter": 12_500, "world": 46_870_000}
    assert doc["weights"]["label"] == "GLM-5.3 in FP4" and "GLM-5.3 in FP4" in cr.leak_markdown(doc)
    assert any("leak_bounds.at_link_rate" in c and "not the scope's K" in c for c in doc["conditions"])
    json.dumps(doc)
    rows = {(r["period"], r["scope"], r["prog"]): r for r in doc["rows"]}
    week = {p: rows["week", "link", p] for p in ("registered", "anchors")}
    lw, rate, cap = 604_800 / 3.2, 100 / 75, 2 * 7
    assert week["registered"]["per_period"] == pytest.approx(lw * (8 * 1_136 + 32 + rate * cap))
    assert week["anchors"]["per_link_window"] == pytest.approx(rate * (1_121 + cap) + 32)
    assert week["anchors"]["pooled"]["0.01"] - week["anchors"]["per_period"] == pytest.approx(
        5 * (154_480 + math.log2(lw * 65 + 1)))
    assert Fraction(week["anchors"]["eps"]) == Fraction(189_000, 2 ** 20)
    for scope, n in doc["links"].items():
        assert rows["week", scope, "registered"]["per_period"] == pytest.approx(n * week["registered"]["per_period"])
        assert Fraction(rows["week", scope, "registered"]["eps"]) == n * Fraction(189_000, 2 ** 20)
    with pytest.raises(SystemExit, match="sampled 16 of 65"):
        cr.leak(run, {"draw": {"honest": 16}})
    (run / "out/deployment.json").write_text(json.dumps({"params": {**asdict(sz.PARAMS), "lanes": 8}}))
    with pytest.raises(ValueError, match="lanes = 4"):
        cr.leak(run)


def test_subset_bound():
    assert cr.subset_bound(26, 26, 2.0 ** -20) == 0
    assert cr.subset_bound(256, 64, 0.5) == 2
    assert cr.subset_bound(256, 2, 0.5) == 74
    assert cr.subset_bound(0, 0, 0.5) is None
    # one incorrect proof block of 4 escapes a sample of 2 with C(3, 2)/C(4, 2) = 1/2 exactly: within δ = 1/2
    assert cr.subset_bound(4, 2, 0.5) == 1


def test_the_certifier_only_path_needs_no_scan_names_its_stubs_and_counts_the_tampers(tmp_path):
    job = {"profile": "advice", "rc": {"n2": "0", "scan": "2"}, "path": cr.CERTIFIER_ONLY}
    steps = _steps(tmp_path, job)
    steps.scan = steps.network_certifier_run = None
    Path(steps.custody).write_text(json.dumps({"verdict": "accepted", "gate_runs": [
        {"gate_run": "c1", "request": {"where": "off-host: a pool slot", "engine": "seeded:phase1"},
         "strategies": {"honest": "accepted", "late": "refused", "extra": "refused", "token": "accepted"}}]}))
    ep = json.loads(Path(steps.endpoint).read_text())
    ep["chain"]["tie"]["runs"] |= {"token": {"verdict": "refused", "ok": True}, "extra": {"verdict": "n/a"},
                                   "late": {"verdict": "refused", "ok": None}}
    ep["negatives"] = {"late": {"own_code": "audit.status:coverage-never-established"}, "token": {"own_code": None}}
    Path(steps.endpoint).write_text(json.dumps(ep))
    got = cr.report(steps)
    assert got["verdict"] == "accepted" and got["steps"]["scan"]["verdict"] == "not run"
    assert got["stubbed"][:3] == list(cr.STUBBED) and len(got["stubbed"]) == 5
    assert got["negatives_refused"] == {"rehearsal/c1/late": True, "rehearsal/c1/extra": True, "tie/token": True,
                                        "n2/tampered": True}
    assert "on the certifier-only path" in cr.markdown(got) and "Stubbed:" in cr.markdown(got)
    ep["chain"]["tie"]["runs"]["token"] = {"verdict": "accepted", "ok": False}
    Path(steps.endpoint).write_text(json.dumps(ep))
    assert cr.report(steps)["negatives_refused"]["tie/token"] is False
    Path(steps.job).write_text(json.dumps(job | {"path": None}))
    off = cr.report(steps)
    assert off["verdict"] == "refused" and off["stubbed"] == []


def test_a_refused_report_names_each_failing_step_and_its_why_and_a_scan_that_exited_its_finds(tmp_path):
    ok = {"profile": "advice", "rc": {"n2": "0", "scan": "0"}}
    assert cr.report(_steps(tmp_path, ok))["why"] is None
    steps = _steps(tmp_path, ok | {"rc": {"n2": "0", "scan": "1"}})

    def found(*k):
        return {"audit": {"window_accepted": True, "received": {"verifier_found": list(k)}}}

    Path(steps.scan).write_text(json.dumps({"ok": False, "T": 1, "rounds": {
        "honest-0": found("start o0 0 @0"), "chain-0": found("start o0 0 @0"), "tampered-0": found()}}))
    got = cr.report(steps)
    why = ("scan_zk exited 1; honest-0: witness pieces found at the verifier: ['start o0 0 @0']; chain-0: witness "
           "pieces found at the verifier: ['start o0 0 @0']; scan_zk's ok is not true")
    assert (got["verdict"], got["steps"]["scan"]["verdict"], got["steps"]["scan"]["why"]) == ("refused", "failed", why)
    assert got["why"] == {"scan": f"failed: {why}"}
    Path(steps.job).write_text(json.dumps(ok | {"rc": {"n2": "0", "scan": "1"}, "why": {"scan": "done: FAILED"}}))
    assert cr.report(steps)["why"] == {"scan": f"failed: done: FAILED; {why}"}
    Path(steps.scan).unlink()
    assert cr.report(steps)["why"] == {"scan": "failed: done: FAILED; scan_zk exited 1"}
    stand_in = cr.report(_steps(tmp_path, ok | {"stand_in": {"n2": "the simulated link"}}))
    assert stand_in["why"] == {"n2": "stand-in: the simulated link"}


def test_k_counts_the_audits_sample_and_free_inputs_and_the_networks_terms_once_a_period(tmp_path):
    h = cr.pooled_k([{"units": 6, "drawn": 6, "free": 2, "jitter": 4}], Fraction(1, 100),
                     {"advice": 8, "syncs": 0, "cells": 6})
    assert (h["Fmax"], h["terms"]["faulty_sets"], h["K"]) == (5, 15, 8 + 6 + 80 + 15 + 2 + 4)
    assert h["conditional"] == cr.POOLED_CONDITION
    circuit = {"templates": [["T", 4]], "external": ["T/y"], "wires": [],
               "statements": {"T": {"in_rows": [["x", 0, 8, ["x"]], ["y", 1, 16, ["y"]]]}}}
    assert cr.free_inputs(circuit) == 4 * (8 + 16)
    (tmp_path / "circuit.json").write_text(json.dumps(circuit))
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "subset", "k": 2}}))
    strata = [{"k": 1, "template": "s0", "units": [[0, 1]]}, {"k": 1, "template": "s1", "units": [[1, 4]]}]
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "stratified", "k": 2,
                                                                                     "strata": strata}}))
    assert (cr.audit_of(tmp_path)["drawn"], cr.audit_of(tmp_path)["q"]) == (2, "1/3")
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "stratified", "k": 1,
                                                                                     "strata": [{**strata[0], "k": 0}, strata[1]]}}))
    with pytest.raises(SystemExit, match="each sampled"):
        cr.audit_of(tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "subset", "k": 2}}))
    audit = cr.audit_of(tmp_path)
    assert (audit["units"], audit["drawn"], audit["free"]) == (4, 2, 96) and "jitter" not in audit
    cert = {"sync_bits": 7.0, "verdict": "accepted", "windows": 3, "window_buckets": 60}
    assert cr.network_terms(cert, {"advice_bits": {"derived": 40, "v0_total": 90}}) == {"advice": 40, "syncs": 21,
                                                                                        "cells": 0}
    assert cr.network_terms(cert, {"advice_bits": {"derived": None, "v0_total": 90}})["advice"] == 90
    d1 = cr.d1_terms(cert, {}, {}, {}, {"advice_bits": {"derived": 40}}, [audit])
    k = d1["K"][f"{2.0 ** -20:.3g}"]
    assert k["Fmax"] == math.ceil(20 * math.log(2) * 2) and k["K"] == sum(k["terms"].values())
    import leak_bounds as lb
    assert (d1["K"]["audits"][0]["steps"], k["terms"]["jitter"], k["jitter_term"]) == (180, 180, lb.JITTER)
    assert "open" in cr.d1_terms(cert, {}, {}, {}, {"advice_bits": {"derived": 40}})["K"]
    unbucketed = {k: v for k, v in cert.items() if k != "window_buckets"}
    open_ = cr.d1_terms(unbucketed, {}, {}, {}, {"advice_bits": {"derived": 40}}, [audit])["K"]["open"]
    assert "window_buckets" in open_
    assert cr.free_inputs(circuit, PIN, PIN) == 4 * (8 + 16)


#: the verifier's pin of ``weights``, and a window whose registration records the same root
PIN = {"weights": "ab" * 64}


def _gemm_window(external: bool = True) -> dict:
    """A window circuit record of four ``Gemm_v2{K=16, N=16}`` proof blocks (``x`` 256 bits, ``w`` 4,096 a unit), every
    unit's ``w`` a registered read of the 256 leaves of ``weights``: with ``external``, every ``x`` the system's I/O, else
    units 2 and 3 read units 0 and 1's ``o0`` through a wire and units 0 and 1's ``x`` are free."""
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import bind
    from verity.circuits.ops import gemm as G
    from verity.commitments.periods import interface as I

    gemm = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
    gid = descriptor_id(gemm)
    reg = I.Registered({"weights": 256}, reads=(I.Read(gid, "w", "weights", at=0, radix=(4,), base=0, step=(0,)),))
    c = I.Circuit("gemms", ((gemm, 4),), external=(f"{gid}/x",) if external else (), registered=reg,
                  wires=() if external else (I.Wire(gid, "x", gid, "o0", at=2, radix=(2,), base=0, step=(1,)),))
    return _record(c)


def _record(c) -> dict:
    """``c``'s stated record as `chain_record` reads a window's ``circuit.json``: its description, external ports and
    each template's input rows, a cut port's rows ``p#j``."""
    from verity.commitments.periods import interface as I

    return {**c.description(), "external": list(c.external),
            "statements": {tid: {"in_rows": [list(r) for r in I.unit_rows(fn, c.cuts.get(tid))[0]]}
                           for (fn, _n), tid in zip(c.templates, c.ids)}}


def test_free_inputs_take_out_a_cut_ports_wired_rows_as_the_period_counts_them():
    """A wire into a port cut ``k`` ways fills one ``p#j`` row a reader index, its port's words over ``k``
    (`Spot.Wires`' ``wiredBits``), which `free_inputs` takes out as `interface.input_count` does: three
    ``Gemm_v2{K=16, N=16}`` units reading their ``x`` cut in two from six ``N=8`` slices leave none of it free. Matching
    a cut row's element ports ``x[r]`` against the wire's ``x`` took out none. An external cut port stays free, and two
    wires filling one row are refused."""
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import bind
    from verity.circuits.ops import gemm as G
    from verity.commitments.periods import interface as I

    gemm, g8 = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE), bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
    gid, g8id = descriptor_id(gemm), descriptor_id(g8)
    wire = I.Wire(gid, "x", g8id, "o0", at=0, radix=(3, 2), base=0, step=(2, 1), cut=(0, 2))
    c = I.Circuit("sliced", ((g8, 6), (gemm, 3)), wires=(wire,), cuts={gid: {"x": 2}})
    doc = _record(c)
    assert [r[0] for r in doc["statements"][gid]["in_rows"]][:2] == ["x#0", "x#1"]
    terms, lean = cr.input_terms(doc), I.input_count(doc)
    assert {k: terms[k] for k in ("inputs", "wired", "registered", "free")} == {
        k: lean[k] for k in ("inputs", "wired", "registered", "free")}
    assert terms["wired"] == 3 * 2 * 128 and terms["free"] == 6 * (256 + 2048) + 3 * 4096
    ext = _record(I.Circuit("sliced", ((g8, 6), (gemm, 3)), cuts={gid: {"x": 2}}, external=(f"{gid}/x",)))
    lean = I.input_count(ext)
    assert cr.input_terms(ext) == {k: lean[k] for k in ("inputs", "wired", "registered", "free")}
    assert cr.free_inputs(ext) == terms["free"] + 3 * 256
    again = I.Wire(gid, "x", g8id, "o0", at=1, radix=(1,), base=0, step=(0,), cut=(1, 1))
    with pytest.raises(ValueError, match="both fill rows of units from 1"):
        cr.free_inputs({**doc, "wires": [*doc["wires"], again.doc()]})


def test_free_inputs_hold_to_the_period_count_on_wires_and_registered_reads():
    """On the uncut window too, `input_terms` is `interface.input_count`, pinned or not, an external port free."""
    from verity.commitments.periods import interface as I

    for doc in (_gemm_window(external=False), _gemm_window()):
        for pins in (None, PIN):
            terms, lean = cr.input_terms(doc, pins, PIN), I.input_count(doc, pins=pins, roots=PIN)
            assert terms == {k: lean[k] for k in ("inputs", "wired", "registered", "free")}


def test_the_faulty_sets_bits_are_the_powers_without_computing_it():
    """`power_bits` is ``(m^f - 1).bit_length()``: computed on small cases, and read off the logarithm (forced with
    ``exact_below=0``) it agrees with the power everywhere; at keep-591e's sample (``m`` 6,963,297, ``f`` 96,531,780) it
    finishes, within one bit of ``f·log2 m``."""
    for m in [*range(1, 40), 1 << 20, 6_963_297, 10 ** 9 + 7]:
        for f in [*range(0, 12), 97, 1000, 4097]:
            exact = (m ** f - 1).bit_length()
            assert cr.power_bits(m, f) == exact == cr.power_bits(m, f, exact_below=0), (m, f)
    m, f = 6_963_297, 96_531_780
    assert 0 < cr.power_bits(m, f) - f * math.log2(m) <= 1
    with pytest.raises(ValueError, match="m ≥ 1"):
        cr.power_bits(0, 3)


def test_free_inputs_leave_out_the_registered_reads_of_a_parameter_held_at_the_verifiers_pin():
    """`free_inputs` subtracts what `NetworkCheck.Circuit.countsOf` holds (`interface.input_count`'s ``registered``): the
    weights' registered reads leave the count where the verifier's pin of ``weights`` is the root the window's
    registration records, and stay free under no pins, another root, no root recorded, or a pin of another root; a wire
    takes out its rows, and an external port stays free, as ``countsOf`` leaves it. A pin of the scan's state is
    refused."""
    ext, wired = _gemm_window(), _gemm_window(external=False)
    assert (cr.free_inputs(ext), cr.free_inputs(wired)) == (4 * (256 + 4096), 2 * 256 + 4 * 4096)
    assert (cr.free_inputs(ext, PIN, PIN), cr.free_inputs(wired, PIN, PIN)) == (4 * 256, 2 * 256)
    assert cr.registered_bits(ext, PIN, {"weights": PIN["weights"].upper()}) == 4 * 4096
    for pins, roots in [(None, PIN), (PIN, None), (PIN, {"weights": "cd" * 64}), ({"weights": "cd" * 64}, PIN),
                        ({"acts": PIN["weights"]}, {"acts": PIN["weights"]}), ("weights", PIN)]:
        assert (cr.free_inputs(ext, pins, roots), cr.registered_bits(ext, pins, roots)) == (4 * (256 + 4096), 0), (pins, roots)
    with pytest.raises(ValueError, match="scan's state is no value the verifier fixes"):
        cr.free_inputs(ext, {**PIN, cr.SCAN_STATE: "00" * 64}, PIN)
    gid = ext["templates"][0][0]
    assert cr.input_terms({**ext, "external": [f"{gid}/x", f"{gid}/w"]}, PIN, PIN) == cr.input_terms(ext, PIN, PIN)
    with pytest.raises(ValueError, match=r"\{param: root\}"):
        cr.free_inputs(ext, {"weights"}, PIN)


def test_an_audit_leaves_out_the_registered_rows_its_registration_records_at_the_verifiers_pin(tmp_path):
    """`audit_of` reads the window's registered roots off its registration's window and holds them to the verifier's
    pins: the weights leave its free inputs only where the window binds ``circuit.json`` by its SHA-512, and a window
    binding another circuit with a held pin is refused; with no pins, or a pin the window doesn't record, K counts them."""
    body = json.dumps(_gemm_window()).encode()
    window = {"circuit": hashlib.sha512(body).hexdigest(), "registered_roots": PIN}
    (tmp_path / "circuit.json").write_bytes(body)
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "subset", "k": 2},
                                                            "window": window}))
    held = cr.audit_of(tmp_path, PIN)
    assert (held["free"], held["inputs"]["registered"], held["held"]) == (4 * 256, 4 * 4096, ["weights"])
    for pins in (None, {"weights": "cd" * 64}):
        got = cr.audit_of(tmp_path, pins)
        assert (got["free"], got["inputs"]["registered"], got["held"]) == (4 * (256 + 4096), 0, [])
        assert got["drawn"] == held["drawn"]
    network = {"advice": 8, "syncs": 0, "cells": 0}
    before, after = (cr.pooled_k([a | {"jitter": 0}], Fraction(1, 100), network) for a in (cr.audit_of(tmp_path), held))
    assert before["K"] - after["K"] == 4 * 4096 and after["terms"]["free"] == 4 * 256
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "subset", "k": 2},
                                                            "window": {**window, "circuit": "00" * 64}}))
    with pytest.raises(SystemExit, match="binds another circuit record than circuit.json"):
        cr.audit_of(tmp_path, PIN)
    assert cr.audit_of(tmp_path)["free"] == 4 * (256 + 4096)
