"""chain_record's steps that read no prover: custody holds each gate run's files to its end.json; certs accepts a virtual
certifier-serve run's records, clock certificates and the client's kept copies (`n2_zk.virtual`, fresh keys, each
window's declared sync signed), refuses the window of a record whose signature is changed, every window from a clock
certificate's, a kept certificate whose signature is changed, and a key under `lms.DEPLOYED`'s height with --deployed;
the report needs the endpoint's step on the advice profile and never accepts a stand-in, merges N2's members proved in
several runs, and counts a negative that escaped a sampled draw within the draw's bound but never one whose faulty
window was drawn; the draw's bound at δ is the integrity profile's count bound, which counts a miss of exactly δ, as
Lean's does; leak counts the scan's widths and terms on a warden run's files, and refuses a sampled draw and a warden
the scan's class doesn't serve; the job's copy of the
gates' records into its output keeps no gate run's witness/. On the certifier-only path the report needs no scan,
names what it stubs, and counts the rehearsal's refused runs and the I/O tie's verdict on each tampered run the check
let through as negatives; d1's K is all_protocols' monitoring-period count over the compute audits' draws and free
inputs and the network's own terms. Given no N2 summary the report is the scan's alone, its verdict `scan accepted`
(exit 0) when the honest round is accepted, every other refused and no witness piece reached the verifier, and `scan refused` (exit 1)
on a wrong verdict, a leak or a missing record; the full report holds the scan to its negatives too. On certifier/v0's
period answer, an accepted report writes the block record's certifier entry by `block_certifier.block_leg`, in the
block record's schema, only when the answer is this chain's (its case's commitment the answer's, the chain's link,
session and key, link-windows the chain certified under N2's own registrations, the report's compute audits) and the
verifier's own on its case, and none for another session's period, another case, the rows form, an answer the check
doesn't give, other audits, another registration, a link-window N2 didn't register, a missing case, no verifier, a
refused or scan-only chain or no server."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
PERIOD_CASE = HERE.parents[1] / "verity" / "core" / "protocols" / "network" / "tests" / "period_case.py"
BLOCK_SCHEMA = HERE.parents[1] / "examples" / "h200" / "interfaces" / "verity_e2e_demo" / "schemas" / "block-record.v1.schema.json"


def _load(name: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr, n2 = _load("chain_record"), _load("n2_zk")


class _Clocked(n2.cs.CertifyingPublisher):
    """`CertifyingPublisher` handed each egress record with its window's declared sync, as `ServedProxy` hands it (the
    virtual run's proxy hands none), so the run keeps `clocks.jsonl`."""

    def __init__(self, out, *args, **kwargs):
        super().__init__(out, *args, **kwargs)
        plan = json.loads((out / "plan.json").read_text())
        self.syncs = [n2.cs.ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]

    def __call__(self, record, params, sync=None):
        super().__call__(record, params, self.syncs[record.window])


@pytest.fixture(scope="module")
def honest(tmp_path_factory):
    out = tmp_path_factory.mktemp("chain")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(n2.cs, "CertifyingPublisher", _Clocked)
        n2.virtual(out, requests=8, tokens=4, strategies=("honest",))
    _keep(out / "honest", out / "honest" / "client")
    return out / "honest"


def _keep(run: Path, client: Path) -> Path:
    """The client's kept files as `keep_certificates` writes them: each certificate's public part with its session."""
    session = json.loads((run / "deployment.json").read_text())["session"]
    certified, _ = n2.cs.read_certified(run / "certified.jsonl")
    line = lambda c: json.dumps({"session": session} | n2.cs.certificate_json(c)) + "\n"
    client.mkdir(exist_ok=True)
    (client / "certificates.jsonl").write_text("".join(line(r.certificate) for r in certified.values()))
    (client / "clocks.jsonl").write_text("".join(line(c.certificate)
                                                 for c in n2.cs.read_clocks(run / "clocks.jsonl")))
    return client


def _corrupt(src: Path, dst: Path, name: str, line: int) -> None:
    shutil.copytree(src, dst)
    rows = (dst / name).read_text().splitlines()
    doc = json.loads(rows[line])
    cert = doc.get("certificate", doc)
    cert["signature"] = cert["signature"][:-2] + ("00" if cert["signature"][-2:] != "00" else "11")
    rows[line] = json.dumps(doc)
    (dst / name).write_text("\n".join(rows) + "\n")


def test_certs_accepts_the_kept_files(honest):
    got = cr.certs(honest, honest / "client")
    assert got["verdict"] == "accepted", got.get("why")
    assert got["certified_windows"] == got["windows"] == got["clocks_signed"]
    assert got["client_check"]["records_are_the_clients"] and got["client_check"]["clocks_are_the_clients"]
    assert got["signed_are_declared"] and not got["key"]["deployed"]
    assert cr.certs(honest, honest / "client", deployed=True)["verdict"] == "refused"


def test_certs_refuses_a_changed_signature(honest, tmp_path):
    _corrupt(honest, tmp_path / "rec", "certified.jsonl", 0)
    got = cr.certs(tmp_path / "rec", tmp_path / "rec" / "client")
    assert got["verdict"] == "refused" and got["certified_windows"] == got["windows"] - 1
    assert not got["client_check"]["records_are_the_clients"]
    _corrupt(honest, tmp_path / "clk", "clocks.jsonl", 0)
    got = cr.certs(tmp_path / "clk", tmp_path / "clk" / "client")
    assert got["verdict"] == "refused" and got["clocks_signed"] == 0 and got["certified_windows"] == 0
    _corrupt(honest, tmp_path / "kept", "client/certificates.jsonl", 0)
    got = cr.certs(tmp_path / "kept", tmp_path / "kept" / "client")
    assert got["verdict"] == "refused" and got["certified_windows"] == got["windows"]
    assert "its signature doesn't verify" in got["client_check"]["refused"][0]


def test_certs_refuses_a_client_without_clocks(honest, tmp_path):
    (tmp_path / "c").mkdir()
    shutil.copy(honest / "client" / "certificates.jsonl", tmp_path / "c")
    assert cr.certs(honest, tmp_path / "c")["verdict"] == "refused"


def test_custody(tmp_path):
    run = tmp_path / "c20261025T000000Z-0000"
    (run / "out" / "honest").mkdir(parents=True)
    (run / "out" / "honest" / "a.json").write_text("{}")
    digest = hashlib.sha512(b"{}").hexdigest()
    (run / "end.json").write_text(json.dumps({"ended": "the rehearsal ended", "files": {"honest/a.json": digest}}))
    assert cr.custody(tmp_path)["verdict"] == "accepted"
    (run / "out" / "honest" / "a.json").write_text("{ }")
    got = cr.custody(tmp_path)
    assert got["verdict"] == "refused" and got["gate_runs"][0]["mismatches"] == {"honest/a.json": "sha512 differs"}
    assert cr.custody(tmp_path / "none")["verdict"] == "refused"


def test_job_keeps_no_gate_runs_witness(tmp_path):
    """chain_record.sh's copy of the gates' records into $OUT, which custody uploads, leaves every witness/ behind."""
    lines = (HERE / "chain_record.sh").read_text().splitlines()
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
        certs=put("certs.json", {"verdict": "accepted", "rows": [{"window": 0, "certified": True}]}),
        endpoint=put("endpoint.json", {"chain": {
            "bits": {"bound": True, "rebuilt": True, "commit_windows": [{"bound": True, "rebuilt": True}]},
            "tie": {"ran": True, "runs": {"honest": {"verdict": "accepted"}}}}}) if endpoint else None,
        n2=put("n2.json", {"ok": True, "members": {
            "honest": {"windows": [window], "network": {"accepted": True}, "ok": True},
            "tampered": {"windows": [window | {"accepted": False}], "network": {"accepted": False}, "ok": True}}}),
        scan=put("scan.json", {"ok": True, "T": 1, "rounds": {"honest": {"audit": {"window_accepted": True}}}}),
        job=put("job.json", job), warden_run="g1/warden")


def test_report_holds_the_advice_profile_to_the_endpoint_and_refuses_stand_ins(tmp_path):
    ok = {"profile": "advice", "rc": {"n2": "0", "scan": "0"}}
    got = cr.report(_steps(tmp_path, ok))
    assert got["verdict"] == "accepted" and got["negatives_refused"] == {"n2/tampered": True}
    assert got["d1"]["acc"] == [True]
    assert cr.report(_steps(tmp_path, ok, endpoint=False))["verdict"] == "refused"
    assert cr.report(_steps(tmp_path, ok | {"profile": "planned"}, endpoint=False))["verdict"] == "accepted"
    assert cr.report(_steps(tmp_path, ok | {"stand_in": {"n2": "the simulated link"}}))["verdict"] == "refused"
    steps = _steps(tmp_path, ok)
    steps.warden_run = None
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
    assert got["why"].startswith("honest: the control (the prover's own prover-instances.bin) holds 57 of 337")


def test_report_merges_runs_and_counts_a_negative_that_escaped_the_draw(tmp_path):
    ok = {"profile": "advice", "rc": {"n2": "0", "scan": "0"}}
    steps = _steps(tmp_path, ok)
    draw = {"selection": "sampled", "law": "subset:2", "k": 2, "population": 4}
    window = {"accepted": True, "proof": {"proved": True}, "as_expected": True}
    reference = [{"window": w, "expect": int(w != 0)} for w in range(4)]

    def run(name, member, units):
        doc = {"draw": draw, "ok": name == "honest", "members": {name: member},
               "sampled": {name: {"draw": {"units": units}, "why": []}}}
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
        return str(tmp_path / f"{name}.json")

    structured = {"leaked": [], "verifier_found": ["head egress 45 @320"], "structured": {"head egress 45 @320": "across"},
                  "control": {"holds_all": True}}
    steps.n2 = [run("honest", {"windows": [window] * 2, "network": {"accepted": True}, "received": structured}, [1, 2]),
                run("moved", {"windows": [window] * 2, "network": {"accepted": True}, "reference": reference}, [1, 2])]
    got = cr.report(steps)
    n2s = got["steps"]["n2"]
    assert got["verdict"] == "accepted" and got["negatives_refused"] == {"n2/moved": "escaped the draw"}
    assert n2s["members"]["honest"]["structured"] == {"head egress 45 @320": "across"}
    assert n2s["members"]["moved"]["escaped_the_draw"] == {"faulty_windows": [0], "drawn": [1, 2]}
    assert n2s["eps_at_delta"]["law"] == "subset:2" and (n2s["eps_at_delta"]["units"], n2s["eps_at_delta"]["drawn"]) == (4, 2)
    assert n2s["eps_at_delta"]["wrong_units_at_delta"]["0.5"] == 1
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


def _chain_of(tmp_path, case: dict) -> SimpleNamespace:
    """An accepted chain whose certifier session, link-windows, N2 registrations and compute audits are ``case``'s."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R
    tmp_path.mkdir(parents=True, exist_ok=True)
    steps = _steps(tmp_path, {"profile": "advice", "rc": {"n2": "0", "scan": "0"}, "run": "r-chain"})
    Path(steps.custody).write_text(json.dumps({"verdict": "accepted", "gate_runs": [{"gate_run": "c1"}]}))
    p, c = case["params"], json.loads(bytes.fromhex(case["commitment"]))
    regs = {r["window"]["window"]: R.digest(r) for r in c["windows"]}
    Path(steps.certs).write_text(json.dumps({"verdict": "accepted", "link": p["link"], "session": p["session"],
                                             "key": {"public": p["key"]},
                                             "rows": [{"window": w, "certified": True} for w in regs]}))
    n2 = json.loads(Path(steps.n2).read_text())
    n2["members"]["honest"]["windows"] = [{"window": w, "registration": d, "accepted": True, "proof": {"proved": True},
                                           "as_expected": True} for w, d in regs.items()]
    Path(steps.n2).write_text(json.dumps(n2))
    circuit = {"templates": [["T", 4]], "external": ["T/y"], "wires": [],
               "statements": {"T": {"in_rows": [["x", 0, 8, ["x"]], ["y", 1, 16, ["y"]]]}}}
    steps.audit = []
    for i, reg in enumerate(c["audits"]):
        d = tmp_path / f"audit{i}"
        d.mkdir()
        (d / "registration.json").write_text(json.dumps(reg))
        (d / "circuit.json").write_text(json.dumps(circuit))
        steps.audit.append(str(d))
    steps.server = "node-1"
    return steps


def test_the_block_entry_is_written_only_for_this_chains_period_and_the_verifiers_own_answer(tmp_path, period,
                                                                                             monkeypatch):
    """certifier/v0 cases from its Python mirror (`period_case`, held to `NetworkCheck.Period` by its difftest), which
    also re-decides them here in the Lean executable's place."""
    import jsonschema
    from block_certifier import block_leg
    monkeypatch.setattr(cr, "decide_period", period.reference)
    case = period.case("accept", 1, "clock")
    answer = period.reference(case)
    steps = _chain_of(tmp_path, case)
    honest = steps.period_answer = _period(tmp_path / "period", case, answer)
    got = cr.report(steps)
    assert got["verdict"] == "accepted" and got["block"]["written"] is True, got["block"]
    entry = got["block"]["entry"]
    assert entry == block_leg(answer, server="node-1", runs=["r-chain", "c1"])
    schema = json.loads(BLOCK_SCHEMA.read_text())
    jsonschema.Draft202012Validator({"$schema": schema["$schema"], "$defs": schema["$defs"],
                                     "$ref": "#/$defs/leg"}).validate(entry)
    K, eps = answer["public"]["K"], answer["public"]["eps"]
    at = f"K = {K}" if Fraction(eps) <= Fraction(1, 100) else "not established"
    assert f"K = {K} bits over 1 link-windows, error {eps} (quantity.eps); at δ = 1/100: {at}" in cr.markdown(got)
    out = tmp_path / "out"
    assert cr.main(["report", "--custody", steps.custody, "--certs", steps.certs, "--endpoint", steps.endpoint,
                    "--n2", steps.n2, "--scan", steps.scan, "--job", steps.job, "--warden-run", steps.warden_run,
                    "--audit", *steps.audit, "--period-answer", steps.period_answer, "--server", steps.server,
                    "--out", str(out)]) == 0
    assert json.loads((out / "block.json").read_text()) == entry

    def why(**change):
        for k, v in change.items():
            setattr(steps, k, v)
        block = cr.report(steps)["block"]
        assert block["written"] is False
        return block["why"]

    foreign = period.case("accept", 0, "clock")
    assert why(period_answer=_period(tmp_path / "foreign", foreign, period.reference(foreign))) == (
        "the case's link, session: not the chain's, so another certifier session's period")
    assert why(period_answer=_period(tmp_path / "mixed", foreign, answer)) == (
        "the case isn't the answer's: its commitment's digest is another")
    rows = period.build("rows-honest", 0, small=True)
    assert why(period_answer=_period(tmp_path / "rows", rows, period.reference(rows))).startswith("a rows-form answer")
    assert why(period_answer=_period(tmp_path / "k", case, answer | {"public": answer["public"] | {"K": 1}})) == (
        f"network-period check on the case answers accept with K {answer['public']['K']}, not the given answer: it "
        "isn't the verifier's on this case")
    audits, steps.period_answer = steps.audit, honest
    assert why(audit=[]) == "the answer's 1 compute audits aren't this report's 0 --audit registrations"
    steps.audit = audits
    n2 = json.loads(Path(steps.n2).read_text())
    n2["members"]["honest"]["windows"][0]["registration"] = "0" * 128
    Path(steps.n2).write_text(json.dumps(n2))
    assert why() == "link-windows [0]: the answer's registrations aren't N2's on this chain (another period's or run's)"
    n2["members"]["honest"]["windows"][0]["window"] = 5
    Path(steps.n2).write_text(json.dumps(n2))
    assert why() == "the answer covers link-windows [0] the chain doesn't have: it certified [0] and N2 registered [5]"
    steps = _chain_of(tmp_path / "again", case)
    steps.period_answer = honest
    assert why(period_case=str(tmp_path / "none.json")).endswith("none.json (--period-case): what the answer commits "
                                                                 "to can't be held to the chain")
    steps.period_case = None
    monkeypatch.undo()
    monkeypatch.delenv("VERITY_NETWORK_PERIOD_EXE", raising=False)
    assert why().startswith("network-period check on the case: no VERITY_NETWORK_PERIOD_EXE")
    assert why(period_answer=str(tmp_path / "none.json")).endswith("none.json: not a file")
    assert why(period_answer=honest, server=None).startswith("no --server")
    assert why(period_answer=None) == cr.NO_PERIOD_ANSWER
    refused = cr.report(SimpleNamespace(**{**vars(steps), "period_answer": honest, "server": "node-1",
                                           "warden_run": None}))
    assert refused["verdict"] == "refused" and refused["block"]["why"].startswith("the chain is refused")
    assert "no certifier entry: the chain is refused" in cr.markdown(refused)
    scan_only = cr.report(SimpleNamespace(**{**vars(steps), "n2": None, "period_answer": honest, "server": "node-1"}))
    assert scan_only["verdict"] == "scan accepted" and scan_only["block"]["why"] == (
        "the chain is scan accepted: only an accepted chain carries a K")


def test_a_scan_only_report_exits_0_on_the_expected_verdicts_and_1_on_a_wrong_one_a_leak_or_a_missing_record(tmp_path):
    steps = _steps(tmp_path, {"profile": "advice", "rc": {"n2": None, "scan": "0"}, "scan_only": True})
    n2_summary, steps.n2 = steps.n2, None
    rnd = lambda accepted, found=(): {"audit": {"window_accepted": accepted, "received": {"verifier_found": list(found)}}}
    put = lambda honest, forged, ok=True: Path(steps.scan).write_text(json.dumps(
        {"ok": ok, "T": 32, "rounds": {"honest": honest, "forged": forged}}))
    args = ["report", "--custody", steps.custody, "--certs", steps.certs, "--endpoint", steps.endpoint, "--scan",
            steps.scan, "--job", steps.job, "--warden-run", steps.warden_run]
    run = lambda name: (cr.main(args + ["--out", str(tmp_path / name)]),
                        json.loads((tmp_path / name / "report.json").read_text()))

    put(rnd(True), rnd(False))
    rc, got = run("pass")
    assert rc == 0 and got["verdict"] == "scan accepted" and got["scope"] == cr.SCAN_ONLY == "scan only, N2 not given"
    assert got["steps"]["n2"]["verdict"] == "not given" and got["negatives_refused"] == {"scan/forged": True}
    assert "Verdict: **scan accepted**, scan only, N2 not given." in (tmp_path / "pass" / "report.md").read_text()

    put(rnd(True), rnd(True))
    rc, got = run("forged-accepted")
    assert rc == 1 and got["scope"] == cr.SCAN_ONLY and got["verdict"] == "scan refused"
    md = (tmp_path / "forged-accepted" / "report.md").read_text()
    assert "Verdict: **scan refused**, scan only, N2 not given." in md
    assert (got["steps"]["scan"]["verdict"], got["steps"]["scan"]["why"]) == ("failed", "forged: window accepted")
    put(rnd(False), rnd(False))
    rc, got = run("honest-refused")
    assert rc == 1 and got["verdict"] == "scan refused"
    put(rnd(True, ["head egress 3 @64"]), rnd(False))
    rc, got = run("leak")
    assert rc == 1 and got["verdict"] == "scan refused"
    assert got["steps"]["scan"]["why"] == "honest: witness pieces found at the verifier: ['head egress 3 @64']"
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
    assert got["scope"] is None and got["verdict"] == "refused" and got["negatives_refused"]["scan/forged"] is False


def test_the_scans_block_round_is_accepted_like_its_honest_round_and_is_no_negative(tmp_path):
    def rnd(accepted):
        return {"audit": {"window_accepted": accepted}}

    def scan(block):
        return {"ok": True, "T": 1, "rounds": {"block": rnd(block), "honest": rnd(True), "chain": rnd(False)}}

    got = cr.scan_step(scan(True), "g1/warden")
    assert got["verdict"] == "accepted" and got["negatives"] == {"chain": True}, got
    got = cr.scan_step(scan(False), "g1/warden")
    assert (got["verdict"], got["why"]) == ("failed", "block: window refused")


def test_leak_counts_the_scans_terms_on_the_warden_run(tmp_path):
    import math
    from dataclasses import asdict
    sz = sys.modules.get("scan_zk") or _load("scan_zk")
    sys.modules["scan_zk"] = sz
    run = tmp_path / "c1"
    (run / "out").mkdir(parents=True)
    (run / "out/deployment.json").write_text(json.dumps({"params": asdict(sz.PARAMS)}))
    (run / "end.json").write_text(json.dumps({"run": "c1", "verdict": {"windows": 75, "requests": 100}}))
    doc = cr.leak(run, {"draw": {"honest": 64}, "T": 32})
    assert doc["widths"] == {"NS": 149_648, "NP": 4_496, "NV": 4_832, "egress_row": 65_536}
    assert doc["terms"]["b"] == 154_480 and doc["terms"]["fmax"] == {"0.01": 5, "1e-06": 14}
    week = {r["prog"]: r for r in doc["rows"] if r["period"] == "week" and r["machines"] == 1}
    lw, rate, cap = 604_800 / 3.2, 100 / 75, math.log2(66)
    assert week["registered"]["per_period"] == pytest.approx(lw * (32 * 4_496 + rate * cap))
    assert week["anchors"]["per_link_window"] == pytest.approx(rate * (1_121 + cap))
    assert week["anchors"]["pooled"]["0.01"] - week["anchors"]["per_period"] == pytest.approx(
        5 * (154_480 + math.log2(lw * 64 + 1)))
    with pytest.raises(SystemExit, match="drew 16 of 64"):
        cr.leak(run, {"draw": {"honest": 16}})
    (run / "out/deployment.json").write_text(json.dumps({"params": {**asdict(sz.PARAMS), "lanes": 8}}))
    with pytest.raises(ValueError, match="lanes = 4"):
        cr.leak(run)


def test_subset_bound():
    assert cr.subset_bound(26, 26, 2.0 ** -20) == 0
    assert cr.subset_bound(256, 64, 0.5) == 2
    assert cr.subset_bound(256, 2, 0.5) == 74
    assert cr.subset_bound(0, 0, 0.5) is None
    # one wrong unit of 4 escapes a sample of 2 with C(3, 2)/C(4, 2) = 1/2 exactly: within δ = 1/2
    assert cr.subset_bound(4, 2, 0.5) == 1


def test_the_certifier_only_path_needs_no_scan_names_its_stubs_and_counts_the_tampers(tmp_path):
    job = {"profile": "advice", "rc": {"n2": "0", "scan": "2"}, "path": cr.CERTIFIER_ONLY}
    steps = _steps(tmp_path, job)
    steps.scan = steps.warden_run = None
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


def test_k_counts_the_audits_draw_and_free_inputs_and_the_networks_terms_once_a_period(tmp_path):
    h = cr.horizon_k([{"units": 6, "drawn": 6, "free": 2, "jitter": 4}], Fraction(1, 100),
                     {"advice": 8, "syncs": 0, "cells": 6})
    assert (h["Fmax"], h["terms"]["faulty_sets"], h["K"]) == (5, 15, 8 + 6 + 80 + 15 + 2 + 4)
    assert h["conditional"] == cr.HORIZON_CONDITION
    circuit = {"templates": [["T", 4]], "external": ["T/y"], "wires": [],
               "statements": {"T": {"in_rows": [["x", 0, 8, ["x"]], ["y", 1, 16, ["y"]]]}}}
    assert cr.free_inputs(circuit) == 4 * 8
    (tmp_path / "circuit.json").write_text(json.dumps(circuit))
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "subset", "k": 2}}))
    strata = [{"k": 1, "template": "s0", "units": [[0, 1]]}, {"k": 1, "template": "s1", "units": [[1, 4]]}]
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "stratified", "k": 2,
                                                                                     "strata": strata}}))
    assert (cr.audit_of(tmp_path)["drawn"], cr.audit_of(tmp_path)["q"]) == (2, "1/3")
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "stratified", "k": 1,
                                                                                     "strata": [{**strata[0], "k": 0}, strata[1]]}}))
    with pytest.raises(SystemExit, match="each drawn"):
        cr.audit_of(tmp_path)
    (tmp_path / "registration.json").write_text(json.dumps({"population": 4, "law": {"law": "subset", "k": 2}}))
    audit = cr.audit_of(tmp_path)
    assert (audit["units"], audit["drawn"], audit["free"], audit["jitter"]) == (4, 2, 32, 0)
    cert = {"sync_bits": 7.0, "rows": [{"window": w, "certified": True} for w in range(3)]}
    assert cr.network_terms(cert, {"advice_bits": {"derived": 40, "v0_total": 90}}) == {"advice": 40, "syncs": 21,
                                                                                        "cells": 0}
    assert cr.network_terms(cert, {"advice_bits": {"derived": None, "v0_total": 90}})["advice"] == 90
    d1 = cr.d1_terms(cert, {}, {}, {}, {"advice_bits": {"derived": 40}}, [audit])
    k = d1["K"][f"{2.0 ** -20:.3g}"]
    assert k["Fmax"] == math.ceil(20 * math.log(2) * 2) and k["K"] == sum(k["terms"].values())
    assert "open" in cr.d1_terms(cert, {}, {}, {}, {"advice_bits": {"derived": 40}})["K"]
