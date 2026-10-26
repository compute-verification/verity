"""chain_record's steps that read no prover: custody holds each gate run's files to its end.json; certs accepts a virtual
certifier-serve run's kept certificates and clock certificates (`n2_zk.virtual`, fresh keys, each window's declared sync
signed), refuses the window of a certificate whose signature is changed and every window from a clock certificate's, and
refuses a key under `lms.DEPLOYED`'s height with --deployed; the report needs the endpoint's step on the advice profile
and never accepts a stand-in; the draw's bound at δ is C(n − b − 1, k) / C(n, k)'s."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
    return out / "honest"


def _corrupt(src: Path, dst: Path, name: str, line: int) -> None:
    dst.mkdir()
    for f in ("certified.jsonl", "clocks.jsonl"):
        shutil.copy(src / f, dst / f)
    rows = (dst / name).read_text().splitlines()
    doc = json.loads(rows[line])
    sig = doc["certificate"]["signature"]
    doc["certificate"]["signature"] = sig[:-2] + ("00" if sig[-2:] != "00" else "11")
    rows[line] = json.dumps(doc)
    (dst / name).write_text("\n".join(rows) + "\n")


def test_certs_accepts_the_kept_files(honest):
    got = cr.certs(honest, honest)
    assert got["verdict"] == "accepted", got.get("why")
    assert got["certified_windows"] == got["windows"] == got["clocks_signed"]
    assert got["signed_are_declared"] and not got["key"]["deployed"]
    assert cr.certs(honest, honest, deployed=True)["verdict"] == "refused"


def test_certs_refuses_a_changed_signature(honest, tmp_path):
    _corrupt(honest, tmp_path / "rec", "certified.jsonl", 0)
    got = cr.certs(honest, tmp_path / "rec")
    assert got["verdict"] == "refused" and got["certified_windows"] == got["windows"] - 1
    _corrupt(honest, tmp_path / "clk", "clocks.jsonl", 0)
    got = cr.certs(honest, tmp_path / "clk")
    assert got["verdict"] == "refused" and got["clocks_signed"] == 0 and got["certified_windows"] == 0


def test_certs_refuses_a_client_without_clocks(honest, tmp_path):
    (tmp_path / "c").mkdir()
    shutil.copy(honest / "certified.jsonl", tmp_path / "c")
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


def test_subset_bound():
    assert cr.subset_bound(26, 26, 2.0 ** -20) == 0
    assert cr.subset_bound(256, 64, 0.5) == 2
    assert cr.subset_bound(256, 2, 0.5) == 74
    assert cr.subset_bound(0, 0, 0.5) is None
