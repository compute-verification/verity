"""The network chain's reader on an active network attester's run (`network_attester_chain`), here the synthetic run
published as `network_attester_serve.Publisher` publishes a served one, in the gate's layout (`attester_runs.gate_run`:
its witness under ``witness/``, and under ``out/`` what the gate's own outcome publishes of it): the
certificates step reruns the network attester's own check and holds the public certificates to the certified lines,
refusing a changed one and a changed cell; the run's deployment is the auditor's record's (`network_attester_chain.deployment_of`),
and a copy of it re-signed, rewritten or in other bytes is refused; the certified I/O read off the cells is the requests
and responses served;
N2's link is the network attester's (both heads at its digest, no sync, charge 0) and its references expect honest
accepted and tampered refused on its signed digest alone. A certifier rehearsal's run (its records through the older certifier's proxy)
is refused by name (`network_attester_chain.PROXY`) by every reader: its files, or a certifier's deployment.json in an active
network attester's run. The chain's record over these runs, `chain_record.py`'s network attester path, is tested in
`tests/benchmarks/network_traces/test_chain_record_attester.py`."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest

from tests.verity.protocols.resource_accounting.network.attester.test_connection_rows_run import ANSWERS, CR, ENDS, PROMPTS, T, device_key, served
from verity.protocols.resource_accounting.network.prover import n2_zk as n2
from verity.protocols.resource_accounting.network.prover import network_attester_chain as WC
from tests.verity.protocols.resource_accounting.network.prover.attester_runs import G, certifier_rehearsal, gate_run
from tests.verity.protocols.resource_accounting.network.prover.test_scan_zk import sz
from verity.protocols.resource_accounting.network.verifier import blob_files as bf
from verity.protocols.resource_accounting.network.verifier import record_files as RF


def _n2(run: Path, at: Path, *args: str) -> tuple[int, dict, dict, str]:
    """n2 --dry on ``run``: its exit, its public summary, its witness summary and the witness side's log (a failure's
    text is there alone)."""
    w = at / "witness" / "summary.json"
    rc = n2.main(["prove", "--warden-run", str(run), "--dry", "--out", str(at / "n2"), "--witness", str(w), *args])
    read = lambda f: json.loads(f.read_text()) if f.is_file() else {}  # noqa: E731
    return rc, read(at / "n2" / "summary.json"), read(w), w.with_name("n2.log").read_text()


def test_a_network_attester_runs_certificates_io_and_link_are_its_own(tmp_path):
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
    assert lw.network_attester.digest == params.digest and lw.windows == sz.W63.windows
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
    """Of the network attester's own check, the certificates step writes the gate's end's verdict, key for key:
    whether it accepted and the gate's own list of counts (`auditor.NETWORK_CERTIFIER_COUNTS`, its `cell_rows` cut to
    `auditor.CELL_ROWS`). The step's list is the gate's, so a key the gate does not publish is not written by the
    chain: the check's problems and their number, the summary's counts of the traffic and its other keys (a
    connection's bytes before padding, the requests' tokens, the certifier's clocks) stay in the witness, and so
    does a key the summary gains later. The record's leak guard reads
    pieces of 32 bytes and sees no shorter value, so this is held by name."""
    assert (WC.COUNTS, WC.CELL_ROWS) == (G.NETWORK_CERTIFIER_COUNTS, G.CELL_ROWS), (
        "network_attester_chain.COUNTS is the gate's list of published counts: rule a key public in auditor.NETWORK_CERTIFIER_COUNTS "
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
    assert got == G.network_attester_outcome(tmp_path / "end", run / "witness")["verdict"]
    later = {"accepted": True, "problems": ["connection 3: its words"],
             "summary": {"requests": 1, "a_later_key": 2, "cell_rows": {"registered": 1, "of": 2, "cost": {"ms": 3}}}}
    assert WC.verdict(later) == {"accepted": True, "cell_rows": {"registered": 1, "of": 2}}
    refused = {"accepted": False, "problems": ["egress window 1's cells aren't registered"],
               "summary": check["summary"] | {"certified_windows": 1, "certificates": 5,
                                              "cell_rows": {"registered": 3, "of": 4, "rows": [600]}}}
    (run / "witness" / "check.json").write_text(json.dumps(refused))
    assert WC.verdict(refused) == G.network_attester_outcome(tmp_path / "end", run / "witness")["verdict"]
    assert set(WC.verdict(refused)) == {"accepted", "failed", "windows", "cell_rows"}
    assert WC.verdict(refused)["cell_rows"] == {"of": 4}
    assert WC.RC.FAILED == G.FAILED, (
        "relay_check.FAILED is the gate's list of the words its end's `failed` holds (auditor.FAILED): rule a word "
        "there first, then write it here")
    worded = {"accepted": False, "problems": [], "summary": {"failed": ["cut", "OSError: a client's own words"]}}
    assert WC.verdict(worded)["failed"] == ["cut", "error"] == G.failed_words(worded["summary"]["failed"])


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
    return certifier_rehearsal(tmp_path_factory.mktemp("certifier"))


def test_a_certifier_rehearsals_run_is_refused_by_name_by_every_reader(tmp_path, certifier_run):
    proxy = re.escape(WC.PROXY)
    for read in (WC.deployment, WC.certificates, WC.io, WC.link):
        with pytest.raises(SystemExit, match=f"{proxy}.*plan.json is there"):
            read(certifier_run)
    rc, _, _, held = _n2(certifier_run, tmp_path / "n2")
    assert rc == 1 and re.search(proxy, held)
    run = gate_run(tmp_path)
    shutil.copyfile(certifier_run / "deployment.json", run / "out" / "deployment.json")
    shutil.copyfile(certifier_run / "deployment.json", run / "witness" / "deployment.json")
    with pytest.raises(SystemExit, match=f"{proxy}.*its deployment.json isn't a network certifier's"):
        WC.certificates(run)


def test_the_gate_publishes_the_word_that_an_execution_phase_has_no_head_and_a_reader_checks_it(tmp_path):
    """A run at the lengths rows' form 2 whose first two execution phases end in the network attester's word that
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


def test_n2_on_a_network_attester_run_expects_honest_accepted_and_tampered_refused_on_its_signed_digest(tmp_path):
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


def test_n2_on_a_sampled_network_attester_period_tampers_every_window_and_proves_another_subset(tmp_path):
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
