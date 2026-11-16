"""The network chain on an active network certifier's run (`network_certifier_chain`, `chain_record`'s network certifier path),
here the synthetic run published as `network_certifier_serve.Publisher` publishes a served one, in the gate's layout (its witness
under ``witness/``, and under ``out/`` what the gate's own outcome publishes of it): the
certificates step reruns the network certifier's own check and holds the public certificates to the certified lines,
refusing a changed one and a changed cell; the run's deployment is the auditor's record's (`network_certifier_chain.deployment_of`),
and a copy of it re-signed, rewritten or in other bytes is refused; the certified I/O read off the cells is the requests
and responses served;
N2's link is the network certifier's (both heads at its digest, no sync, charge 0) and its references expect honest
accepted and tampered refused on its signed digest alone. A certifier rehearsal's run (its records through the older certifier's proxy)
is refused by name (`network_certifier_chain.PROXY`) by every reader: its files, or a certifier's deployment.json in an active
network certifier's run. The report on the network certifier path requires the certificates, the endpoint, the tie, N2
and the scan, names its stubs, and refuses a step whose input is a certifier rehearsal's or another network certifier
run's."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.append(str(Path(__file__).resolve().parents[3]))  # the root, for benchmarks.network_traces
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/protocols/compose/verifier/live_challenger"))

import auditor as G  # noqa: E402
from verity.protocols.resource_accounting.network.attester.tests.test_connection_rows_run import ANSWERS, CR, ENDS, PROMPTS, T, device_key, served  # noqa: E402
from verity.protocols.resource_accounting.network.prover.tests.test_scan_zk import published, sz  # noqa: E402

from benchmarks.network_traces import network_certifier_chain as WC  # noqa: E402
from verity.protocols.resource_accounting.network.verifier import blob_files as bf  # noqa: E402
from verity.protocols.resource_accounting.network.verifier import record_files as RF  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"network_traces_wc_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load("chain_record")
from verity.protocols.resource_accounting.network.prover import n2_zk as n2  # noqa: E402
from benchmarks.network_traces import leak_guard as LG  # noqa: E402
from benchmarks.network_traces import zk_drivers  # noqa: E402

zk_drivers.install()


def _n2(run: Path, at: Path, *args: str) -> tuple[int, dict, dict, str]:
    """n2 --dry on ``run``: its exit, its public summary, its witness summary and the witness side's log (a failure's
    text is there alone)."""
    w = at / "witness" / "summary.json"
    rc = n2.main(["prove", "--warden-run", str(run), "--dry", "--out", str(at / "n2"), "--witness", str(w), *args])
    read = lambda f: json.loads(f.read_text()) if f.is_file() else {}  # noqa: E731
    return rc, read(at / "n2" / "summary.json"), read(w), w.with_name("n2.log").read_text()


def gate_run(d: Path, seed: bytes = b"warden-run") -> Path:
    """The synthetic network certifier run in the gate's layout, under the network certifier key ``seed`` makes: its
    files under ``witness/`` with the check the runner writes there, which accepts it, and under ``out/`` what the
    gate publishes of them (`auditor.network_certifier_outcome`), its lines among them."""
    run = published(d / "flat", seed=seed)
    (d / "gate" / "out").mkdir(parents=True)
    shutil.copytree(run, d / "gate" / "witness")
    (d / "gate" / "witness" / "check.json").write_text(json.dumps(WC.RC.check(d / "gate" / "witness")))
    outcome = G.network_certifier_outcome(d / "gate" / "out", d / "gate" / "witness")
    assert outcome["verdict"]["accepted"] is True, "the gate publishes a run's lines only when its check accepted"
    assert "unruled" not in outcome, "a run's own lines are in the forms the gate publishes them in"
    return d / "gate"


def test_the_gate_publishes_a_registration_less_its_wiring_and_the_guard_holds_the_wiring(tmp_path):
    """Of each registration of an execution phase's cells, the gate's ``out/`` holds the certificate, with the link,
    direction and window the registration names, and the line as `network_certifier_serve.Publisher` writes it less the wiring
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


def test_a_network_certifier_runs_certificates_io_and_link_are_its_own(tmp_path):
    run = gate_run(tmp_path)
    got, private = WC.checked(run)
    assert got == WC.certificates(run) and not set(WC.WITNESS_KEYS) & set(got) and set(private) == set(WC.WITNESS_KEYS)
    assert (got["verdict"], got["why"], got["windows"]) == ("accepted", None, sz.W63.windows)
    assert (private["certified_windows"], private["issuer_refused"], private["problems"]) == (sz.W63.windows, [], 0)
    assert got["public_is_the_records"] is True and got["clock_certificates"] == 0 and got["sync_bits"] == 0.0
    assert got["warden_run"] == "00" * 32 and got["warden_check"]["accepted"]
    io = WC.io(run)
    assert io["accepted"] and io["requests"] == len(sz.W63.prompts) and not io["violations"]
    assert [r["output"] for r in io["records"]] == [list(r) for r in sz.W63.responses]
    assert all(r["release"] is not None and r["frames"][0][1] >= r["release"] for r in io["records"])
    lw, dep, params = WC.link(run)
    assert lw.network_certifier.digest == params.digest and lw.windows == sz.W63.windows
    assert lw.grid(0) == lw.grid(1) and lw.link == "warden" and lw.run == dep["run_id"]
    pairs, refused = WC.pairs(run, dep)
    assert len(pairs) == sz.W63.windows and refused == []
    assert all(lw.certified(w, [r.certificate for r in pair]) for w, pair in enumerate(pairs))

    public = run / "out" / "certificates.jsonl"
    lines = public.read_text().splitlines()
    public.write_text("\n".join([lines[1], lines[0], *lines[2:]]) + "\n")
    got = WC.certificates(run)
    assert got["verdict"] == "refused" and got["why"] == "the public certificates.jsonl isn't the certified records' "\
                                                         "certificates"
    run = gate_run(tmp_path / "cell")
    path = run / "witness" / "responses.records"
    blobs = bf.read_blobs(path)
    rec = RF.unpack_record(blobs[0], sz.PARAMS.cell_bytes)
    t, j = next((t, j) for t, row in enumerate(rec.grid) for j, f in enumerate(row) if f is not None)
    grid = [list(row) for row in rec.grid]
    grid[t][j] = bytes([grid[t][j][0]]) + bytes(len(grid[t][j]) - 1)
    path.unlink()
    for b in [RF.pack_record(type(rec)(rec.link, rec.window, tuple(map(tuple, grid)), rec.status)), *blobs[1:]]:
        bf.write_blobs(path, b)
    (got, private), check = WC.checked(run), WC.RC.check(run / "witness")
    assert got["verdict"] == "refused" and private["rows"][0]["certified"] is False and "rows" not in got
    assert any("responses window 0 isn't certified" in p for p in check["problems"])
    assert got["warden_check"]["accepted"] is False and "problems" not in got["warden_check"]
    assert got["why"] == WC.REFUSES and private["problems"] == len(check["problems"])
    assert not [p for p in check["problems"] if p in json.dumps(got)]


def test_the_certificates_step_writes_of_the_check_what_the_gates_end_publishes(tmp_path):
    """Of the network certifier's own check, the certificates step writes the gate's end's verdict, key for key:
    whether it accepted and the gate's own list of counts (`auditor.NETWORK_CERTIFIER_COUNTS`, its `cell_rows` cut to
    `auditor.CELL_ROWS`). The step's list is the gate's, so a key the gate does not publish is not written by the
    chain: the check's problems and their number, the summary's counts of the traffic and its other keys (a
    connection's bytes before padding, the requests' tokens, the certifier's clocks) stay in the witness, and so
    does a key the summary gains later. The record's leak guard reads
    pieces of 32 bytes and sees no shorter value, so this is held by name."""
    assert (WC.COUNTS, WC.CELL_ROWS) == (G.NETWORK_CERTIFIER_COUNTS, G.CELL_ROWS), (
        "network_certifier_chain.COUNTS is the gate's list of published counts: rule a key public in auditor.NETWORK_CERTIFIER_COUNTS "
        "first, then write it here")
    assert (WC.ACCEPTED_COUNTS, WC.ACCEPTED_CELL_ROWS) == (G.ACCEPTED_COUNTS, G.ACCEPTED_CELL_ROWS), (
        "the counts only an accepted check's verdict holds are the gate's (auditor.ACCEPTED_COUNTS)")
    run = gate_run(tmp_path)
    check = WC.RC.check(run / "witness")
    got = WC.certificates(run)["warden_check"]
    assert set(got) <= {"accepted", *G.NETWORK_CERTIFIER_COUNTS} and set(got["cell_rows"]) <= set(G.CELL_ROWS)
    assert {"requests", "ended", "refused", "cells", "wire_bits", "cells_in_their_bucket", "tick_lateness_ms",
            "data_bytes", "completion_tokens", "prompt_tokens", "finish_reasons", "arrival_lag_ms", "clocks",
            "ticker_stalls", "withheld_windows"} <= set(check["summary"]) - set(got)
    (run / "witness" / "check.json").write_text(json.dumps(check))
    (tmp_path / "end").mkdir()
    assert got == G.network_certifier_outcome(tmp_path / "end", run / "witness")["verdict"]
    later = {"accepted": True, "problems": ["connection 3: its words"],
             "summary": {"requests": 1, "a_later_key": 2, "cell_rows": {"registered": 1, "of": 2, "cost": {"ms": 3}}}}
    assert WC.verdict(later) == {"accepted": True, "cell_rows": {"registered": 1, "of": 2}}
    refused = {"accepted": False, "problems": ["egress window 1's cells aren't registered"],
               "summary": check["summary"] | {"certified_windows": 1, "certificates": 5,
                                              "cell_rows": {"registered": 3, "of": 4, "rows": [600]}}}
    (run / "witness" / "check.json").write_text(json.dumps(refused))
    assert WC.verdict(refused) == G.network_certifier_outcome(tmp_path / "end", run / "witness")["verdict"]
    assert set(WC.verdict(refused)) == {"accepted", "failed", "windows", "cell_rows"}
    assert WC.verdict(refused)["cell_rows"] == {"of": 4}
    assert WC.RC.FAILED == G.FAILED, (
        "relay_check.FAILED is the gate's list of the words its end's `failed` holds (auditor.FAILED): rule a word "
        "there first, then write it here")
    worded = {"accepted": False, "problems": [], "summary": {"failed": ["cut", "OSError: a client's own words"]}}
    assert WC.verdict(worded)["failed"] == ["cut", "error"] == G.failed_words(worded["summary"]["failed"])


def test_the_certificates_step_releases_its_declared_keys_and_keeps_the_traffics_in_the_witness(tmp_path,
                                                                                               monkeypatch):
    """``chain_record.py certs`` writes ``certs.json`` with only the keys the list of released_keys.json names, on an
    accepted run and on one the network certifier's own check refuses, and every count of `verdict` the list names;
    what the step found of the traffic (`network_certifier_chain.WITNESS_KEYS`: which execution phases the signer refused, how
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


def test_a_runs_deployment_is_the_auditors_record_or_refused_naming_its_copy_and_field(tmp_path):
    """`deployment_of`: a run whose every deployment.json (its witness/'s and out/'s, the gate's layout) is the
    auditor's record's deployment, in the bytes the record hashes, gives the record's deployment and schedule. A run
    re-signed under another key, one copy of it rewritten, or the same deployment in other bytes is refused naming the
    copy and the field; a record that isn't a `RECORD` whose key and session are its deployment's is refused."""
    run = gate_run(tmp_path / "honest")
    record = WC.own_record(run)
    assert (record["schema"], record["key"], record["session"]) == (WC.RECORD, *map(WC.deployment(run)[0].get,
                                                                                    ("key", "session")))
    assert WC.deployment_of(run, record) == (record["deployment"], WC.deployment(run)[1])
    assert record["deployment"] == {k: v for k, v in WC.deployment(run)[0].items() if k not in WC.RC.DEVICE_KEYS}
    other = gate_run(tmp_path / "other", seed=b"another network certifier")
    with pytest.raises(SystemExit, match="witness/deployment.json names another key than the auditor's record's$"):
        WC.deployment_of(other, record)
    out = run / "out" / "deployment.json"
    kept = out.read_bytes()
    shutil.copyfile(other / "out" / "deployment.json", out)
    with pytest.raises(SystemExit, match="out/deployment.json names another key than the auditor's record's$"):
        WC.deployment_of(run, record)
    out.write_text(json.dumps(json.loads(kept), indent=2))
    with pytest.raises(SystemExit, match="out/deployment.json isn't the bytes the auditor's record hashes$"):
        WC.deployment_of(run, record)
    out.write_bytes(kept)
    for bad in ({**record, "schema": "challenger-key/v1"}, {**record, "session": "00" * 32}, {**record, "key": {}},
                {k: v for k, v in record.items() if k != "deployment"}, None):
        with pytest.raises(SystemExit, match=f"the auditor's record isn't a {WC.RECORD} whose key and session are its "
                                             "deployment's$"):
            WC.deployment_of(run, bad)


@pytest.fixture(scope="module")
def certifier_run(tmp_path_factory):
    """A certifier rehearsal's run as a fetched one holds it: its plan and calibration, and a deployment.json naming the
    certifier's key and links, which no active network certifier's does."""
    run = tmp_path_factory.mktemp("certifier") / "honest"
    run.mkdir()
    docs = {"plan.json": {"windows": 2}, "calibration.json": {"grid": {}},
            "deployment.json": {"certifier_key": "00", "links": {"n2": "served"}, "session": "00" * 32, "run_id": "r"}}
    for name, doc in docs.items():
        (run / name).write_text(json.dumps(doc))
    return run


def test_a_certifier_rehearsals_run_is_refused_by_name_by_every_reader(tmp_path, certifier_run):
    proxy = re.escape(WC.PROXY)
    for read in (WC.deployment, WC.certificates, WC.io, WC.link):
        with pytest.raises(SystemExit, match=f"{proxy}.*plan.json is there"):
            read(certifier_run)
    rc, _, _, held = _n2(certifier_run, tmp_path / "n2")
    assert rc == 1 and re.search(proxy, held)
    with pytest.raises(SystemExit, match=proxy):
        cr.main(["certs", "--warden-run", str(certifier_run), "--out", str(tmp_path / "certs.json")])
    run = gate_run(tmp_path)
    shutil.copyfile(certifier_run / "deployment.json", run / "out" / "deployment.json")
    shutil.copyfile(certifier_run / "deployment.json", run / "witness" / "deployment.json")
    with pytest.raises(SystemExit, match=f"{proxy}.*its deployment.json isn't a network certifier's"):
        WC.certificates(run)


def test_the_gate_publishes_the_word_that_an_execution_phase_has_no_head_and_a_reader_checks_it(tmp_path):
    """A run at the lengths rows' form 2 whose first two execution phases end in the network certifier's word that
    they have no head, and its last two in their heads. Of each line of its `rows.jsonl` the gate publishes one
    certificate in the same six fields and nothing of the line's cost. The word's fields, with the schedule's digest,
    are the word's public record whole, which `connection_rows.check_no_head` accepts under the deployment's key and
    session for its own execution phase alone; a head's certificate read as the word is refused. So a reader tells
    the two apart from what the gate publishes, and nothing but the word's record leaves by the word's line."""
    out = tmp_path / "run"
    params = dataclasses.replace(ENDS, queue=5)
    served(out, tmp_path / "key", params=params, prompts=(*PROMPTS, (14,), (15, 16)),
           answers=(*ANSWERS[:5], (3,), (4, 5)), arrive={6: T + 2, 7: 2 * T + 2})
    key, session = device_key(out)
    lines = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert [sorted(x) for x in lines] == [["cost", G.ROWS_NO_HEAD]] * 2 + [["cost", "registration"]] * 2
    for w, line in enumerate(lines):
        public = G.row_certificate(line, G.deployment_forms(out / "deployment.json"))
        c = public["certificate"]
        assert sorted(public) == ["certificate"]
        assert sorted(c) == ["digest", "direction", "key_id", "link", "signature", "window"]
        assert (c["link"], c["direction"], c["window"]) == (WC.RC.LINK, G.ROWS_DIRECTION, w)
        word = {"format": CR.NO_HEAD, "link": c["link"], "window": c["window"], "parameters": params.digest.hex(),
                "certificate": {"key": c["key_id"], "digest": c["digest"], "signature": c["signature"]}}
        if G.ROWS_NO_HEAD in line:
            assert word == line[G.ROWS_NO_HEAD]
            CR.check_no_head(word, key, session, params, c["link"], w)
            with pytest.raises(ValueError, match="the word's window"):
                CR.check_no_head(word, key, session, params, c["link"], w + 1)
        else:
            assert c["digest"] == line["registration"]["certificate"]["digest"]
            with pytest.raises(ValueError, match="doesn't sign that the execution phase has no head"):
                CR.check_no_head(word, key, session, params, c["link"], w)


def test_n2_on_a_network_certifier_run_expects_honest_accepted_and_tampered_refused_on_its_signed_digest(tmp_path):
    run = gate_run(tmp_path)
    rc, s, w, _ = _n2(run, tmp_path, "--gpu")
    assert rc == 0 and s["warden_run"] == "00" * 32 and "no sync" in s["warden"]["syncs"] and w["gpu"] is True
    honest, tampered = (s["members"][k]["reference"] for k in ("honest", "tampered"))
    failed = {k: w["members"][k]["reference"] for k in ("honest", "tampered")}
    assert [x["expect"] for x in honest] == [1] * sz.W63.windows
    assert all(not any(x["failed"].values()) for x in failed["honest"].values())
    assert [x["expect"] for x in tampered] == [0] + [1] * (sz.W63.windows - 1)
    assert failed["tampered"]["0"]["failed"] == {"egress": ["signed"], "ingress": []}
    assert s["draw"]["selection"] == "all" and n2.OTHER not in s["members"]


def test_n2_on_a_sampled_network_certifier_period_tampers_every_window_and_proves_another_subset(tmp_path):
    """With ``--draw K``, K < n, the tampered member changes every window, so whichever K windows the sample holds refuse
    it, and other_subset (the honest records, proved on another K-subset) is a member; K outside 1 … n is refused."""
    run = gate_run(tmp_path)
    rc, s, w, _ = _n2(run, tmp_path, "--draw", "1")
    assert rc == 0 and (s["draw"]["law"], s["draw"]["k"], s["draw"]["population"]) == ("subset:1", 1, sz.W63.windows)
    tampered, other = (s["members"][k]["reference"] for k in ("tampered", n2.OTHER))
    assert [x["expect"] for x in tampered] == [0] * sz.W63.windows
    assert all(x["failed"] == {"egress": ["signed"], "ingress": []}
               for x in w["members"]["tampered"]["reference"].values())
    assert [x["expect"] for x in other] == [1] * sz.W63.windows
    rc, *_, held = _n2(run, tmp_path / "x", "--draw", "3")
    assert rc == 1 and "a period sample takes 1 to 2" in held


def _steps(tmp_path, run: Path) -> SimpleNamespace:
    """The network certifier path's step outputs on `run`: its real certificates, and the endpoint, the tie, N2 and the
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


def test_the_network_certifier_path_requires_every_step_on_one_network_certifier_run_and_refuses_a_certifier_rehearsals(tmp_path,
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
