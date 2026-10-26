"""chain_record's steps that read no prover: custody holds each gate run's files to its end.json; certs accepts a virtual
certifier-serve run's records, clock certificates and the client's kept copies (`n2_zk.virtual`, fresh keys, each
window's declared sync signed), refuses the window of a record whose signature is changed, every window from a clock
certificate's, a kept certificate whose signature is changed, and a key under `lms.DEPLOYED`'s height with --deployed;
the report needs the endpoint's step on the advice profile and never accepts a stand-in, merges N2's members proved in
several runs, and counts a negative that escaped a sampled draw within the draw's bound but never one whose faulty
window was drawn; the draw's bound at δ is C(n − b − 1, k) / C(n, k)'s; leak counts the scan's widths and terms on a
warden run's files, and refuses a sampled draw and a warden the scan's class doesn't serve. On the certifier-only path
the report needs no scan, names what it stubs, and counts the rehearsal's refused runs and the I/O tie's verdict on
each tampered run the check let through as negatives; d1's K is #4404's horizon count over the compute audits' draws
and free inputs and the network's own terms."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]


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
    steps.n2[1] = run("moved", {"windows": [window] * 2, "network": {"accepted": True}, "reference": reference}, [0, 1])
    got = cr.report(steps)["steps"]["n2"]
    assert got["verdict"] == "failed" and got["why"] == "moved: network accepted"
    steps.n2[0] = run("honest", {"windows": [window] * 2, "network": {"accepted": True},
                                 "received": {**structured, "leaked": ["head egress 45 @320"]}}, [1, 2])
    got = cr.report(steps)["steps"]["n2"]
    assert got["why"].startswith("honest: witness pieces leaked to the verifier: ['head egress 45 @320']")


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
    assert doc["widths"] == {"NS": 149_520, "NP": 4_480, "NV": 4_832, "egress_row": 65_536}
    assert doc["terms"]["b"] == 154_352 and doc["terms"]["fmax"] == {"0.01": 5, "1e-06": 14}
    week = {r["prog"]: r for r in doc["rows"] if r["period"] == "week" and r["machines"] == 1}
    lw, rate, cap = 604_800 / 3.2, 100 / 75, math.log2(66)
    assert week["registered"]["per_period"] == pytest.approx(lw * (32 * 4_480 + rate * cap))
    assert week["anchors"]["per_link_window"] == pytest.approx(rate * (1_120 + cap))
    assert week["anchors"]["pooled"]["0.01"] - week["anchors"]["per_period"] == pytest.approx(
        5 * (154_352 + math.log2(lw * 64 + 1)))
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
