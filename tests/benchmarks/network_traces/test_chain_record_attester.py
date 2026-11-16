"""`chain_record.py`'s network attester path, on the chain's synthetic gate run (`attester_runs.gate_run`, beside the
chain's own tests in the network attester's prover): the gate publishes a registration less its wiring, and the
record's leak guard holds the wiring; the certificates step releases only the keys its list declares, keeps the
traffic's in the witness, and refuses a certifier rehearsal's run by name (`network_attester_chain.PROXY`); the report
on the network attester path requires the certificates, the endpoint, the tie, N2 and the scan, names its stubs, and
refuses a step whose input is a certifier rehearsal's or another network attester run's."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.network_traces import leak_guard as LG
from tests.verity.protocols.resource_accounting.network.prover.attester_runs import certifier_rehearsal, gate_run
from tests.verity.protocols.resource_accounting.network.prover.test_scan_zk import sz
from verity.protocols.resource_accounting.network.prover import network_attester_chain as WC

HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "network_traces")


def _load(name: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"network_traces_cra_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load("chain_record")


def test_the_gate_publishes_a_registration_less_its_wiring_and_the_guard_holds_the_wiring(tmp_path):
    """Of each registration of an execution phase's cells, the gate's ``out/`` holds the certificate, with the link,
    direction and window the registration names, and the line as `network_attester_serve.Publisher` writes it less the wiring
    and the cost: the witness's own bytes but for those two, the root and the domain among them, and no wiring. The
    line whole is in ``witness/``, where the chain reads it. A record's certificate is written in one key order, so
    its text in the witness's line is its text in the public one. So the record's leak guard, which reads a job's files
    against the gate run's witness, holds nothing of a record whose files name a registration's public keys as a
    verifier's files do (the published line, the registration less its wiring indented, a port with its root and
    domain, the certificates). The control: the published line with the wiring written into it is held, and so are
    a registration whole and the wiring alone, each moved out of the record."""
    run = gate_run(tmp_path / "g")
    witness = (run / "witness" / "cells.jsonl").read_text().splitlines()
    lines = [json.loads(x) for x in witness]
    regs = [x["registration"] for x in lines if "registration" in x]
    public = [json.loads(x) for x in (run / "out" / "cells-certificates.jsonl").read_text().splitlines()]
    assert len(regs) == 2 * sz.W63.windows and len(public) == len(lines)
    assert [x["certificate"] for x in public] == [
        {"key_id": r["certificate"]["key"], "link": r["link"], "direction": r["direction"], "window": r["window"],
         "digest": r["certificate"]["digest"], "signature": r["certificate"]["signature"]} for r in regs]
    assert not (run / "out" / "cells.jsonl").exists()
    published = (run / "out" / "cells-registrations.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in published] == [
        {"registration": {k: v for k, v in x["registration"].items() if k != "wiring"}} for x in lines]
    assert published == [re.sub(r', "cost": \{[^}]*\}', "", x.replace(f'"wiring": "{r["wiring"]}", ', ""))
                         for x, r in zip(witness, regs)]
    assert all(len(p) + len(f'"wiring": "{r["wiring"]}", ') < len(x) for p, x, r in zip(published, witness, regs))
    out = b"".join(p.read_bytes() for p in sorted((run / "out").rglob("*")) if p.is_file())
    assert not [r["value"] for r in regs if r["wiring"].encode() in out] and b"wiring" not in out
    assert b"commit_ms" not in out and b'"cost"' not in out
    assert all(r[k].encode() in out for r in regs for k in ("root", "domain"))
    assert WC.RC.check(WC.witness(run))["summary"]["cell_rows"]["registered"] == len(regs)
    certified = (run / "witness" / "certified.jsonl").read_text().splitlines()
    certificates = (run / "out" / "certificates.jsonl").read_text().splitlines()
    assert len(certified) == len(certificates) == 3 * sz.W63.windows
    assert all(x.endswith(", " + c[1:]) for x, c in zip(certified, certificates))

    record = tmp_path / "record"
    shutil.copytree(run / "out", record / "gates" / "out")
    less = [{k: v for k, v in r.items() if k != "wiring"} for r in regs]
    port = ("value", "root", "rows", "words", "word_bits", "domain", "row_schema")
    released = {
        "certificates.json": json.dumps({"rows": [x["certificate"] for x in public]}, indent=1, sort_keys=True),
        "published.jsonl": "".join(x + "\n" for x in published),
        "registration.json": json.dumps(less[0], indent=1, sort_keys=True),
        "registered.json": json.dumps({"ports": {f"o{i}": {k: r[k] for k in port} for i, r in enumerate(regs)}},
                                      sort_keys=True)}
    control = published[0].replace('"root": ', f'"wiring": "{regs[0]["wiring"]}", "root": ')
    assert control != published[0] and control == re.sub(r', "cost": \{[^}]*\}', "", witness[0])
    held = {"control.jsonl": control + "\n", "whole.json": json.dumps(regs[1], indent=1, sort_keys=True),
            "wiring.json": json.dumps({"wiring": regs[2]["wiring"]})}
    for name, text in (released | held).items():
        (record / name).write_text(text)
    gate_files = [f"gates/out/{p.name}" for p in sorted((run / "out").iterdir())]
    rc = LG.main([str(record), str(tmp_path / "q"), "--gates", str(tmp_path / "g"), "--report",
                  str(tmp_path / "report.json")])
    assert rc == 3
    assert sorted(f["path"] for f in json.loads((tmp_path / "report.json").read_text())["found"]) == sorted(held)
    assert sorted(p.relative_to(record).as_posix() for p in record.rglob("*") if p.is_file()) == sorted(
        [*released, *gate_files])
    rc = LG.main([str(record), str(tmp_path / "q2"), "--gates", str(tmp_path / "g"), "--report",
                  str(tmp_path / "clean.json")])
    assert rc == 0 and json.loads((tmp_path / "clean.json").read_text())["found"] == []
    assert not [p for p in (tmp_path / "q2" / "files").rglob("*") if p.is_file()]
    assert sorted(p.relative_to(record).as_posix() for p in record.rglob("*") if p.is_file()) == sorted(
        [*released, *gate_files])


def test_the_certificates_step_releases_its_declared_keys_and_keeps_the_traffics_in_the_witness(tmp_path,
                                                                                               monkeypatch):
    """``chain_record.py certs`` writes ``certs.json`` with only the keys the list of released_keys.json names, on an
    accepted run and on one the network attester's own check refuses, and every count of `verdict` the list names;
    what the step found of the traffic (`network_attester_chain.WITNESS_KEYS`: which execution phases the signer refused, how
    many it certified, each one's certified, how many problems) goes to ``--witness`` only, and a ``certs.json`` that
    held any of it would fail the record."""
    from verity.protocols.resource_accounting.network.prover import released_keys as RK

    listing = RK.load().listing("certs.json")
    assert listing is not None and not RK.load().waits("certs.json")
    accepted, refused = gate_run(tmp_path / "ok"), gate_run(tmp_path / "cut")
    check = WC.RC.check
    monkeypatch.setattr(WC.RC, "check", lambda wit: check(wit) | (
        {"accepted": False, "problems": ["its words"]} if refused in Path(wit).parents else {}))
    for run in (accepted, refused):
        out, witness = run.parent / "certs.json", run.parent / "witness" / "certs" / "certs.json"
        rc = cr.main(["certs", "--warden-run", str(run), "--out", str(out), "--witness", str(witness)])
        doc, private = json.loads(out.read_text()), json.loads(witness.read_text())
        assert (rc, doc["verdict"]) == ((0, "accepted") if run is accepted else (1, "refused"))
        assert listing.undeclared(out.read_bytes()) == {} and set(private) == set(WC.WITNESS_KEYS)
        assert not set(doc) & set(private) and private["issuer_refused"] == []
        assert set(listing.undeclared(json.dumps(doc | private).encode())) == set(WC.WITNESS_KEYS)
    full = {"accepted": True, "summary": {"failed": ["cut"], "windows": 2, "certified_windows": 2, "certificates": 6,
                                          "cell_rows": {"registered": 4, "of": 4},
                                          "load": {"values": 3, "registered": 3, "relayed": True}}}
    assert listing.undeclared(json.dumps({"warden_check": WC.verdict(full)}).encode()) == {}


@pytest.fixture(scope="module")
def certifier_run(tmp_path_factory):
    return certifier_rehearsal(tmp_path_factory.mktemp("certifier"))


def test_the_certificates_step_refuses_a_certifier_rehearsals_run_by_name(tmp_path, certifier_run):
    with pytest.raises(SystemExit, match=re.escape(WC.PROXY)):
        cr.main(["certs", "--warden-run", str(certifier_run), "--out", str(tmp_path / "certs.json")])


def _steps(tmp_path, run: Path) -> SimpleNamespace:
    """The network attester path's step outputs on `run`: its real certificates, and the endpoint, the tie, N2 and the
    scan as their drivers write them."""
    def put(name, doc):
        (tmp_path / name).write_text(json.dumps(doc))
        return str(tmp_path / name)

    rid = "00" * 32
    window = {"accepted": True, "proof": {"proved": True}, "as_expected": True}
    return SimpleNamespace(
        custody=put("custody.json", {"verdict": "accepted"}),
        certs=put("certs.json", WC.certificates(run)),
        endpoint=put("endpoint.json", {"warden_run": rid, "io": {"accepted": True, "requests": 5}, "chain": {
            "bits": {"bound": True, "rebuilt": True, "commit_windows": [{"bound": True, "rebuilt": True}]}}}),
        tie=put("tie.json", {"class": "warden", "warden_run": rid, "ok": True, "link": {"T": 32, "unit": "u"},
                             "tie": {"honest": [True, True], "forged": [False, True]}}),
        n2=put("n2.json", {"ok": True, "warden_run": rid, "members": {
            "honest": {"windows": [window, window], "network": {"accepted": True}, "ok": True},
            "tampered": {"windows": [window | {"accepted": False}, window], "network": {"accepted": False},
                         "ok": True}}}),
        scan=put("scan.json", {"ok": True, "T": 32, "warden_run": rid, "rounds": {
            "honest": {"audit": {"window_accepted": True}}, "forged": {"audit": {"window_accepted": False}}}}),
        job=put("job.json", {"path": cr.NETWORK_CERTIFIER, "rc": {"n2": "0", "scan": "0", "tie": "0"}}),
        network_certifier_run=str(run))


def test_the_network_attester_path_requires_every_step_on_one_network_attester_run_and_refuses_a_certifier_rehearsals(tmp_path,
                                                                                                certifier_run):
    steps = _steps(tmp_path, gate_run(tmp_path / "run"))
    got = cr.report(steps)
    assert got["verdict"] == "accepted" and got["stubbed"] == list(cr.NETWORK_CERTIFIER_STUBBED)
    assert got["negatives_refused"] == {"tie/forged": True, "n2/tampered": True, "scan/forged": True}
    assert got["steps"]["certificates"]["clock_certificates"] == 0 and got["d1"]["acc"] == [True] * sz.W63.windows
    assert got["steps"]["n2"]["clocks"]["source"] == "none"
    md = cr.markdown(got)
    assert f"on the {cr.NETWORK_CERTIFIER} path" in md and "| tie | accepted |" in md and "certified I/O" in md
    ep = json.loads(Path(steps.endpoint).read_text())
    Path(steps.endpoint).write_text(json.dumps(ep | {"at": cr.SERVED_NODE}))
    served = cr.report(steps)
    assert served["verdict"] == "accepted" and served["stubbed"] == list(cr.NETWORK_CERTIFIER_SERVED_STUBBED)
    assert served["steps"]["endpoint"]["in_the_clear"] == cr.CLEAR["warden_endpoint_served"]
    Path(steps.endpoint).write_text(json.dumps(ep))

    def refused(name: str, edit, why: str) -> None:
        doc = json.loads(Path(getattr(steps, name)).read_text())
        Path(getattr(steps, name)).write_text(json.dumps(edit(dict(doc))))
        r = cr.report(steps)
        step = {"certs": "certificates"}.get(name, name)
        assert r["verdict"] == "refused" and why in r["steps"][step]["why"], (name, r["steps"][step])
        Path(getattr(steps, name)).write_text(json.dumps(doc))

    proxy = WC.PROXY
    refused("n2", lambda d: {k: v for k, v in d.items() if k != "warden_run"} | {"run": str(certifier_run)},
            f"N2's summary names no network certifier run: {proxy}")
    refused("tie", lambda d: d | {"warden_run": "11" * 32}, "the tie's summary is on network certifier run " + "11" * 32)
    refused("tie", lambda d: d | {"tie": {"honest": [True, True], "forged": [True, True]}}, "forged's refused")
    refused("scan", lambda d: d | {"warden_run": None}, "the scan's summary names no network certifier run")
    refused("endpoint", lambda d: d | {"io": {"accepted": False, "violations": [[3, "x"]]}}, "read whole off its cells")
    refused("endpoint", lambda d: {k: v for k, v in d.items() if k != "warden_run"},
            f"the endpoint's report names no network certifier run: {proxy}")
    refused("certs", lambda d: {k: v for k, v in d.items() if k != "warden_run"},
            f"the certificates step's input names no network certifier run: {proxy}")
    steps.endpoint = steps.tie = None
    r = cr.report(steps)
    assert r["verdict"] == "refused" and r["steps"]["endpoint"]["verdict"] == "not run"
    assert r["steps"]["tie"]["verdict"] == "failed"
