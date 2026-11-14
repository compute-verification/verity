"""The network chain on an active network certifier's run (`warden_chain`, `chain_record`'s network certifier path),
here the synthetic run published as `warden_serve.Publisher` publishes a served one, in the gate's layout (its witness
under ``witness/``, and under ``out/`` what the gate's own outcome publishes of it): the
certificates step reruns the network certifier's own check and holds the public certificates to the certified lines,
refusing a changed one and a changed cell; the certified I/O read off the cells is the requests and responses served;
N2's link is the network certifier's (both heads at its digest, no sync, charge 0) and its references expect honest
accepted and tampered refused on its signed digest alone. A certifier rehearsal's run (its records through active.Proxy)
is refused by name (`warden_chain.PROXY`) by every reader: its files, or a certifier's deployment.json in an active
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/protocols/compose/device/live_challenger"))

import auditor as G  # noqa: E402
from test_connection_rows_run import ANSWERS, CR, ENDS, PROMPTS, T, device_key, served  # noqa: E402
from test_scan_zk import published, sz  # noqa: E402

import warden_chain as WC  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"network_traces_wc_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr, n2, LG = _load("chain_record"), _load("n2_zk"), _load("leak_guard")


def gate_run(d: Path) -> Path:
    """The synthetic network certifier run in the gate's layout: its files under ``witness/``, and under ``out/``
    what the gate publishes of them (`auditor.warden_outcome`)."""
    run = published(d / "flat")
    (d / "gate" / "out").mkdir(parents=True)
    shutil.copytree(run, d / "gate" / "witness")
    G.warden_outcome(d / "gate" / "out", d / "gate" / "witness")
    return d / "gate"


def test_the_gate_publishes_a_registration_less_its_wiring_and_the_guard_holds_the_wiring(tmp_path):
    """Of each registration of an execution phase's cells, the gate's ``out/`` holds the certificate, with the link,
    direction and window the registration names, and the line as `warden_serve.Publisher` writes it less the wiring
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
    assert WC.WS.check(WC.witness(run))["summary"]["cell_rows"]["registered"] == len(regs)
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


def test_a_warden_runs_certificates_io_and_link_are_its_own(tmp_path):
    run = gate_run(tmp_path)
    got = WC.certificates(run)
    assert (got["verdict"], got["why"], got["certified_windows"], got["windows"]) == ("accepted", None, sz.W63.windows,
                                                                                    sz.W63.windows)
    assert got["public_is_the_records"] is True and got["clock_certificates"] == 0 and got["sync_bits"] == 0.0
    assert got["warden_run"] == "00" * 32 and got["warden_check"]["accepted"]
    io = WC.io(run)
    assert io["accepted"] and io["requests"] == len(sz.W63.prompts) and not io["violations"]
    assert [r["output"] for r in io["records"]] == [list(r) for r in sz.W63.responses]
    assert all(r["release"] is not None and r["frames"][0][1] >= r["release"] for r in io["records"])
    lw, dep, params = WC.link(run)
    assert lw.warden.digest == params.digest and lw.charge == 0.0 and lw.windows == sz.W63.windows
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
    blobs = WC.WS.ar.read_blobs(path)
    rec = WC.WS.ar.unpack_record(blobs[0], sz.PARAMS.cell_bytes)
    t, j = next((t, j) for t, row in enumerate(rec.grid) for j, f in enumerate(row) if f is not None)
    grid = [list(row) for row in rec.grid]
    grid[t][j] = bytes([grid[t][j][0]]) + bytes(len(grid[t][j]) - 1)
    path.unlink()
    for b in [WC.WS.ar.pack_record(type(rec)(rec.link, rec.window, tuple(map(tuple, grid)), rec.status)), *blobs[1:]]:
        WC.WS.ar.write_blobs(path, b)
    got, check = WC.certificates(run), WC.WS.check(run / "witness")
    assert got["verdict"] == "refused" and got["rows"][0]["certified"] is False
    assert any("responses window 0 isn't certified" in p for p in check["problems"])
    assert got["warden_check"]["problems"] == len(check["problems"]) and f"{len(check['problems'])} problems" in got["why"]
    assert not [p for p in check["problems"] if p in json.dumps(got)]


def test_the_certificates_step_writes_of_the_check_what_the_gates_end_publishes(tmp_path):
    """Of the network certifier's own check, the certificates step writes the gate's end's verdict, key for key:
    whether it accepted, how many problems, and the gate's own list of counts (`auditor.WARDEN_COUNTS`, its
    `cell_rows` cut to `auditor.CELL_ROWS`). The step's list is the gate's, so a key the gate does not publish is not
    written by the chain: the summary's other keys (a connection's bytes before padding, the requests' tokens, the
    certifier's clocks) stay in the witness, and so does a key the summary gains later. The record's leak guard reads
    pieces of 32 bytes and sees no shorter value, so this is held by name."""
    assert (WC.COUNTS, WC.CELL_ROWS) == (G.WARDEN_COUNTS, G.CELL_ROWS), (
        "warden_chain.COUNTS is the gate's list of published counts: rule a key public in auditor.WARDEN_COUNTS "
        "first, then write it here")
    run = gate_run(tmp_path)
    check = WC.WS.check(run / "witness")
    got = WC.certificates(run)["warden_check"]
    assert set(got) <= {"accepted", "problems", *G.WARDEN_COUNTS} and set(got["cell_rows"]) <= set(G.CELL_ROWS)
    assert {"data_bytes", "completion_tokens", "prompt_tokens", "finish_reasons", "arrival_lag_ms", "clocks",
            "ticker_stalls", "withheld_windows"} <= set(check["summary"]) - set(got)
    (run / "witness" / "check.json").write_text(json.dumps(check))
    (tmp_path / "end").mkdir()
    assert got == G.warden_outcome(tmp_path / "end", run / "witness")["verdict"]
    later = {"accepted": True, "problems": ["connection 3: its words"],
             "summary": {"requests": 1, "a_later_key": 2, "cell_rows": {"registered": 1, "of": 2, "cost": {"ms": 3}}}}
    assert WC.verdict(later) == {"accepted": True, "problems": 1, "requests": 1, "cell_rows": {"registered": 1, "of": 2}}
    assert WC.WS.FAILED == G.FAILED, (
        "warden_serve.FAILED is the gate's list of the words its end's `failed` holds (auditor.FAILED): rule a word "
        "there first, then write it here")
    worded = {"accepted": False, "problems": [], "summary": {"failed": ["cut", "OSError: a client's own words"]}}
    assert WC.verdict(worded)["failed"] == ["cut", "error"] == G.failed_words(worded["summary"]["failed"])


@pytest.fixture(scope="module")
def certifier_run(tmp_path_factory):
    """A certifier rehearsal's run (`n2_zk.virtual`: certifier-serve's records through its proxy)."""
    out = tmp_path_factory.mktemp("certifier")
    n2.virtual(out, requests=4, tokens=2, strategies=("honest",))
    return out / "honest"


def test_a_certifier_rehearsals_run_is_refused_by_name_by_every_reader(tmp_path, certifier_run):
    proxy = re.escape(WC.PROXY)
    for read in (WC.deployment, WC.certificates, WC.io, WC.link):
        with pytest.raises(SystemExit, match=f"{proxy}.*plan.json is there"):
            read(certifier_run)
    with pytest.raises(SystemExit, match=proxy):
        n2.main(["prove", "--warden-run", str(certifier_run), "--dry", "--out", str(tmp_path / "n2")])
    with pytest.raises(SystemExit, match=proxy):
        cr.main(["certs", "--warden-run", str(certifier_run), "--out", str(tmp_path / "certs.json")])
    run = gate_run(tmp_path)
    shutil.copyfile(certifier_run / "deployment.json", run / "out" / "deployment.json")
    shutil.copyfile(certifier_run / "deployment.json", run / "witness" / "deployment.json")
    with pytest.raises(SystemExit, match=f"{proxy}.*its deployment.json isn't a warden's"):
        WC.certificates(run)
    run = gate_run(tmp_path / "extra")
    with pytest.raises(SystemExit, match="--extra and --late are certifier-serve's runs"):
        n2.main(["prove", "--warden-run", str(run), "--extra", str(certifier_run), "--dry", "--out",
                 str(tmp_path / "n2x")])


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
        public = G.row_certificate(line)
        c = public["certificate"]
        assert sorted(public) == ["certificate"]
        assert sorted(c) == ["digest", "direction", "key_id", "link", "signature", "window"]
        assert (c["link"], c["direction"], c["window"]) == (WC.WS.LINK, G.ROWS_DIRECTION, w)
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


def test_n2_on_a_warden_run_expects_honest_accepted_and_tampered_refused_on_its_signed_digest(tmp_path):
    run = gate_run(tmp_path)
    assert n2.main(["prove", "--warden-run", str(run), "--gpu", "--dry", "--out", str(tmp_path / "n2")]) == 0
    s = json.loads((tmp_path / "n2" / "summary.json").read_text())
    assert s["warden_run"] == "00" * 32 and "no sync" in s["warden"]["syncs"] and s["gpu"] is True
    honest, tampered = (s["members"][k]["reference"] for k in ("honest", "tampered"))
    assert [x["expect"] for x in honest] == [1] * sz.W63.windows and all(not any(x["failed"].values()) for x in honest)
    assert [x["expect"] for x in tampered] == [0] + [1] * (sz.W63.windows - 1)
    assert tampered[0]["failed"] == {"egress": ["signed"], "ingress": []}
    assert s["draw"]["selection"] == "all" and n2.OTHER not in s["members"]


def test_n2_on_a_sampled_warden_period_tampers_every_window_and_proves_another_subset(tmp_path):
    """With ``--draw K``, K < n, the tampered member changes every window, so whichever K windows the sample holds refuse
    it, and other_subset (the honest records, proved on another K-subset) is a member; K outside 1 … n is refused."""
    run = gate_run(tmp_path)
    assert n2.main(["prove", "--warden-run", str(run), "--draw", "1", "--dry", "--out", str(tmp_path / "n2")]) == 0
    s = json.loads((tmp_path / "n2" / "summary.json").read_text())
    assert (s["draw"]["law"], s["draw"]["k"], s["draw"]["population"]) == ("subset:1", 1, sz.W63.windows)
    tampered, other = (s["members"][k]["reference"] for k in ("tampered", n2.OTHER))
    assert [x["expect"] for x in tampered] == [0] * sz.W63.windows
    assert all(x["failed"] == {"egress": ["signed"], "ingress": []} for x in tampered)
    assert [x["expect"] for x in other] == [1] * sz.W63.windows
    with pytest.raises(SystemExit, match="a period draw takes 1 to 2"):
        n2.main(["prove", "--warden-run", str(run), "--draw", "3", "--dry", "--out", str(tmp_path / "n2x")])


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
        job=put("job.json", {"path": cr.WARDEN, "rc": {"n2": "0", "scan": "0", "tie": "0"}}), warden_run=str(run))


def test_the_warden_path_requires_every_step_on_one_warden_run_and_refuses_a_certifier_rehearsals(tmp_path,
                                                                                                certifier_run):
    steps = _steps(tmp_path, gate_run(tmp_path / "run"))
    got = cr.report(steps)
    assert got["verdict"] == "accepted" and got["stubbed"] == list(cr.WARDEN_STUBBED)
    assert got["negatives_refused"] == {"tie/forged": True, "n2/tampered": True, "scan/forged": True}
    assert got["steps"]["certificates"]["clock_certificates"] == 0 and got["d1"]["acc"] == [True] * sz.W63.windows
    assert got["steps"]["n2"]["clocks"]["source"] == "none"
    md = cr.markdown(got)
    assert "on the warden path" in md and "| tie | accepted |" in md and "certified I/O" in md
    ep = json.loads(Path(steps.endpoint).read_text())
    Path(steps.endpoint).write_text(json.dumps(ep | {"at": cr.SERVED_NODE}))
    served = cr.report(steps)
    assert served["verdict"] == "accepted" and served["stubbed"] == list(cr.WARDEN_SERVED_STUBBED)
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
            f"N2's summary names no warden run: {proxy}")
    refused("tie", lambda d: d | {"warden_run": "11" * 32}, "the tie's summary is on warden run " + "11" * 32)
    refused("tie", lambda d: d | {"tie": {"honest": [True, True], "forged": [True, True]}}, "forged's refused")
    refused("scan", lambda d: d | {"warden_run": None}, "the scan's summary names no warden run")
    refused("endpoint", lambda d: d | {"io": {"accepted": False, "violations": [[3, "x"]]}}, "read whole off its cells")
    refused("endpoint", lambda d: {k: v for k, v in d.items() if k != "warden_run"},
            f"the endpoint's report names no warden run: {proxy}")
    certifier_certs = cr.certs(certifier_run, certifier_run)
    refused("certs", lambda d: certifier_certs, f"the certificates step's input names no warden run: {proxy}")
    steps.endpoint = steps.tie = None
    r = cr.report(steps)
    assert r["verdict"] == "refused" and r["steps"]["endpoint"]["verdict"] == "not run"
    assert r["steps"]["tie"]["verdict"] == "failed"
