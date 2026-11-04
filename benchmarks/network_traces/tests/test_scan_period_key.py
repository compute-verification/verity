"""The period's case takes the network certifier's key from the auditor's record (`scan_period.rows_case`'s ``record``,
`warden_chain.deployment_of`), never from the warden run, the prover's copy. A fetch re-signed under the prover's own
key is refused at the record before any check; one that keeps the recorded deployment.json beside records re-signed
under its own key carries certificates that don't verify under the record's key and session, which the period check
refuses (period_case's cert-key changes, held to network-period). The quick form of the rows-form test in
test_scan_zk.py, which runs the period's mirror on both."""
from __future__ import annotations

import shutil

import pytest
from test_scan_zk import published, sz

import scan_period as SP
from verity.core.protocols.certifier.certificate import Certificate, Direction, LmsKey
from verity.ml.crypto import lms


def _verifies(record: dict, c: dict) -> bool:
    N, W, H = record["key"]["params"]
    return Certificate(bytes.fromhex(c["key_id"]), c["link"], Direction(c["direction"]), c["window"],
                       bytes.fromhex(c["digest"]), bytes.fromhex(c["signature"])).verify(
        LmsKey(lms.Params(N, W, H), bytes.fromhex(record["key"]["public"])), bytes.fromhex(record["session"]))


def test_the_periods_case_refuses_a_fetch_re_signed_under_another_key_than_the_auditors_record(tmp_path):
    honest = published(tmp_path / "honest", sz.TEST, windows=1)
    resigned = published(tmp_path / "resigned", sz.TEST, seed=b"the prover's own key", windows=1)
    record = SP.WC.own_record(honest)
    assert SP.WC.deployment(resigned)[0]["key"] != record["key"]
    with pytest.raises(SystemExit, match="deployment.json names another key than the auditor's record's$"):
        SP.rows_case(resigned, tmp_path / "scan", record=record)
    assert all(_verifies(record, c) for c in SP.certificates(honest, 0))
    shutil.copyfile(honest / "deployment.json", resigned / "deployment.json")
    assert SP.WC.deployment_of(resigned, record) == SP.WC.deployment_of(honest, record)
    assert not any(_verifies(record, c) for c in SP.certificates(resigned, 0))
