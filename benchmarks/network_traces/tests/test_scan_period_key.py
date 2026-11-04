"""The period's certifier key is the auditor's (`scan_period.auditor_key`): the rows case takes the network certifier's
key, session, LMS parameters and schedule from the auditor's own record of the certifier's deployment
(``certifier-key/v1``, CERTIFIER_KEY), never from the warden run, the prover's copy. A fetch re-signed under the
prover's own key is refused before any check. A fetch that keeps the recorded deployment.json beside records re-signed
under another key carries certificates that don't verify under the auditor's key, which the period check refuses: Lean
and its mirror at once, as period_case's cert-key, rows-cert-key and rows-reg-key changes hold them. A record that isn't
one is refused by name, and the command takes the auditor's record or says the runs are its own rehearsals."""
from __future__ import annotations

import json
import shutil

import pytest
from test_scan_zk import published, sz

import scan_period as SP
from verity.core.protocols.certifier.certificate import Certificate, Direction, LmsKey
from verity.ml.crypto import lms


def _record(tmp_path, run, **about):
    (path := tmp_path / "certifier-key.json").write_text(json.dumps(SP.key_record(
        (run / "deployment.json").read_bytes(), host="vy-auditor-2", run="r-honest", **about)))
    return SP.auditor_key(path)


def _verifies(key: dict, c: dict) -> bool:
    N, W, H = key["key"]["params"]
    return Certificate(bytes.fromhex(c["key_id"]), c["link"], Direction(c["direction"]), c["window"],
                       bytes.fromhex(c["digest"]), bytes.fromhex(c["signature"])).verify(
        LmsKey(lms.Params(N, W, H), bytes.fromhex(key["key"]["public"])), bytes.fromhex(key["session"]))


def test_a_fetch_re_signed_under_another_key_is_refused_and_the_recorded_one_is_read(tmp_path):
    honest = published(tmp_path / "honest", sz.TEST, windows=1)
    resigned = published(tmp_path / "resigned", sz.TEST, seed=b"the prover's own key", windows=1)
    key = _record(tmp_path, honest)
    theirs = json.loads((resigned / "deployment.json").read_text())
    assert theirs["key"] != key["key"] and theirs["session"] == key["session"]
    assert SP.deployment_of(honest, key) == key["deployment"] == json.loads((honest / "deployment.json").read_text())
    with pytest.raises(SystemExit) as e:
        SP.rows_case(resigned, tmp_path / "scan", key=key)
    assert str(e.value).startswith(f"{resigned}: its deployment.json (SHA-512 ") and (
        f"isn't the network certifier's deployment the auditor recorded ({key['deployment_sha512'][:16]}…, run "
        "r-honest on vy-auditor-2): the key and session are the auditor's, never the run's") in str(e.value)
    assert all(_verifies(key, c) for c in SP.certificates(honest, 0))
    shutil.copy(honest / "deployment.json", resigned / "deployment.json")
    assert SP.deployment_of(resigned, key) == key["deployment"]
    assert not any(_verifies(key, c) for c in SP.certificates(resigned, 0))


@pytest.mark.parametrize("change", ["schema", "key", "session", "sha", "unreadable"])
def test_a_record_that_isnt_the_auditors_is_refused_by_name(tmp_path, change):
    rec = SP.key_record(json.dumps({"key": {"params": [64, 1, 6], "public": "ab"}, "session": "cd"}).encode())
    rec = {"schema": rec | {"schema": "certifier-key/v0"}, "key": rec | {"key": {"public": "ef"}},
           "session": rec | {"session": "ef"}, "sha": {k: v for k, v in rec.items() if k != "deployment_sha512"},
           "unreadable": None}[change]
    path = tmp_path / "certifier-key.json"
    path.write_text("{" if rec is None else json.dumps(rec))
    with pytest.raises(SystemExit) as e:
        SP.auditor_key(path)
    want = ("no auditor's record of the network certifier" if rec is None else
            "not a certifier-key/v1 record whose key and session are its deployment's")
    assert str(e.value).startswith(f"{path}: {want}"), str(e.value)


def test_the_command_takes_the_auditors_record_or_its_own_rehearsals(tmp_path, capsys):
    argv = ["--mirror", "m.py", "--lean", "np", "--out", str(tmp_path), "a=run,scan"]
    with pytest.raises(SystemExit) as e:
        SP.main(argv)
    assert e.value.code == 2 and "one of the arguments --certifier-key --rehearsed is required" in capsys.readouterr().err
