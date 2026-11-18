"""The network attester run reader (`network_attester_chain`) on an active network attester's run, here the synthetic run
published as `network_attester_serve.Publisher` publishes a served one, in the gate's layout (its witness under
``witness/``, and under ``out/`` what the gate's own outcome publishes of it): the attestations step reruns the network
attester's own check and holds the public attestations to the attested lines, refusing a changed one and a changed
cell; the run's deployment is the auditor's record's (`network_attester_chain.deployment_of`), and a copy of it
re-signed, rewritten or in other bytes is refused; the attested I/O read off the cells is the requests and responses
served; N2's link is the network attester's (both heads at its digest, no sync, charge 0)."""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "verity/protocols/compose/verifier/live_challenger"))

import auditor as G
from tests.verity.protocols.resource_accounting.network.attester.test_connection_rows_run import ANSWERS, CR, ENDS, PROMPTS, T, device_key, served
from tests.verity.protocols.resource_accounting.network.prover.test_scan_zk import published, sz

from verity.protocols.resource_accounting.network.verifier import network_attester_chain as WC
from verity.protocols.resource_accounting.network.verifier import blob_files as bf
from verity.protocols.resource_accounting.network.verifier import record_files as RF


def gate_run(d: Path, seed: bytes = b"warden-run") -> Path:
    """The synthetic network attester run in the gate's layout, under the network attester key ``seed`` makes: its
    files under ``witness/`` with the check the runner writes there, which accepts it, and under ``out/`` what the
    gate publishes of them (`auditor.network_attester_outcome`), its lines among them."""
    run = published(d / "flat", seed=seed)
    (d / "gate" / "out").mkdir(parents=True)
    shutil.copytree(run, d / "gate" / "witness")
    (d / "gate" / "witness" / "check.json").write_text(json.dumps(WC.RC.check(d / "gate" / "witness")))
    outcome = G.network_attester_outcome(d / "gate" / "out", d / "gate" / "witness")
    assert outcome["verdict"]["accepted"] is True, "the gate publishes a run's lines only when its check accepted"
    assert "unruled" not in outcome, "a run's own lines are in the forms the gate publishes them in"
    return d / "gate"


def test_a_network_attester_runs_attestations_io_and_link_are_its_own(tmp_path):
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


def test_the_attestations_step_writes_of_the_check_what_the_gates_end_publishes(tmp_path):
    """Of the network attester's own check, the attestations step writes the gate's end's verdict, key for key:
    whether it accepted and the gate's own list of counts (`auditor.NETWORK_ATTESTER_COUNTS`, its `cell_rows` cut to
    `auditor.CELL_ROWS`). The step's list is the gate's, so a key the gate does not publish is not written by the
    chain: the check's problems and their number, the summary's counts of the traffic and its other keys (a
    connection's bytes before padding, the requests' tokens, the attester's clocks) stay in the witness, and so
    does a key the summary gains later. The record's leak guard reads
    pieces of 32 bytes and sees no shorter value, so this is held by name."""
    assert (WC.COUNTS, WC.CELL_ROWS) == (G.NETWORK_ATTESTER_COUNTS, G.CELL_ROWS), (
        "network_attester_chain.COUNTS is the gate's list of published counts: rule a key public in auditor.NETWORK_ATTESTER_COUNTS "
        "first, then write it here")
    assert (WC.ACCEPTED_COUNTS, WC.ACCEPTED_CELL_ROWS) == (G.ACCEPTED_COUNTS, G.ACCEPTED_CELL_ROWS), (
        "the counts only an accepted check's verdict holds are the gate's (auditor.ACCEPTED_COUNTS)")
    run = gate_run(tmp_path)
    check = WC.RC.check(run / "witness")
    got = WC.certificates(run)["warden_check"]
    assert set(got) <= {"accepted", *G.NETWORK_ATTESTER_COUNTS} and set(got["cell_rows"]) <= set(G.CELL_ROWS)
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


def test_the_gate_publishes_the_word_that_an_execution_phase_has_no_head_and_a_reader_checks_it(tmp_path):
    """A run at the lengths rows' form 2 whose first two execution phases end in the network attester's word that
    they have no head, and its last two in their heads. Of each line of its `rows.jsonl` the gate publishes one
    attestation in the same six fields and nothing of the line's cost. The word's fields, with the schedule's digest,
    are the word's public record whole, which `connection_rows.check_no_head` accepts under the deployment's key and
    session for its own execution phase alone; a head's attestation read as the word is refused. So a reader tells
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
