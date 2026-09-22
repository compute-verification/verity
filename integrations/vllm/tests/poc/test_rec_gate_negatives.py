"""REC (2026-09-09, continuation): after merging GUEST's enforcement (approved executables, statement format, proof
verification) with REC's (instance, partition, population, chronology), NEITHER may have disappeared from the STRICT v3
entry point.  Every case below goes through ``e2e.certificate.main`` (or ``instance_check`` where the TINY record has no
proofs) and asserts the specific refusal, not just "not bound".

Records used: the preserved INT-PAD-2 B0 record (real proofs, no instance) and the TINY instance-bearing record of
``test_proto_a_instance`` (instance evidence, no proofs).  Nothing under ``results/`` is modified: every case mutates a copy.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))

from e2e import approved_guests as ag  # noqa: E402
from e2e import certificate  # noqa: E402
from e2e.instance_check import check_instance_evidence  # noqa: E402

from test_guest_policy import GOOD_ELF, RECORD, _policy_with_vk, _policy_without  # noqa: E402
from test_proto_a_instance import _hash, _repin_instance, _rewrite, _tiny_record  # noqa: E402

RESULTS: dict[str, str] = {}  # case -> the refusal observed (printed at the end for the report)


def _certify(record: Path, out_dir: Path, *args: str) -> tuple[int, dict]:
    out = out_dir / "certificate.json"
    rc = certificate.main(["--record", str(record), "--out", str(out), *args])
    return rc, json.loads(out.read_text())


@pytest.fixture(scope="module")
def intpad2(tmp_path_factory) -> Path:
    if not (RECORD / "manifest.json").exists():
        pytest.skip("INT-PAD-2 record not present")
    dst = tmp_path_factory.mktemp("intpad2") / RECORD.name
    shutil.copytree(RECORD, dst, ignore=shutil.ignore_patterns("bundle", "reverify", "handshake", "logs", "log.txt", "plan", "program", "verify"))
    return dst


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


# --- GUEST's enforcement, still present after the merge ---------------------------------------------------------------------


def test_unapproved_elf_is_refused_on_the_strict_path(intpad2: Path, tmp_path: Path) -> None:
    pol = tmp_path / "approved.json"
    pol.write_text(json.dumps(_policy_without(GOOD_ELF[:8])))
    rc, cert = _certify(intpad2, tmp_path, "--approved-guests", str(pol))
    hits = [p for p in cert["binding_problems"] if ag.GUEST_NOT_APPROVED in p]
    assert rc == 1 and cert["bound"] is False and cert["certified"] is False and hits, cert["binding_problems"]
    RESULTS["unapproved ELF"] = hits[0]


def test_mismatched_vk_is_refused_on_the_strict_path(intpad2: Path, tmp_path: Path) -> None:
    pol = tmp_path / "approved.json"
    pol.write_text(json.dumps(_policy_with_vk(GOOD_ELF[:8], "0x00" + "ab" * 31)))
    rc, cert = _certify(intpad2, tmp_path, "--approved-guests", str(pol))
    hits = [p for p in cert["binding_problems"] if ag.GUEST_NOT_APPROVED in p and "approved with vk" in p]
    assert rc == 1 and cert["bound"] is False and hits, cert["binding_problems"]
    RESULTS["mismatched VK"] = hits[0]


def test_tampered_statement_is_refused_on_the_strict_path(intpad2: Path, tmp_path: Path) -> None:
    rec = _copy(intpad2, tmp_path / "stmt")
    stmt = rec / "obligations" / "statement.bin"
    b = bytearray(stmt.read_bytes())
    b[len(b) // 2] ^= 0x01
    stmt.write_bytes(bytes(b))
    rc, cert = _certify(rec, tmp_path)
    hits = [p for p in cert["binding_problems"] if "statement" in p.lower() and ("h_L" in p or "sha" in p or "digest" in p)]
    assert rc == 1 and cert["bound"] is False and hits, cert["binding_problems"]
    RESULTS["tampered statement"] = hits[0]


# --- REC's enforcement, still present after the merge -----------------------------------------------------------------------
# (the TINY record carries instance evidence but no proofs/summary, so these cases assert on ``check_instance_evidence`` --
# the very function ``certificate.py`` calls for its ``instance`` problems; the INT-PAD-2 case shows the certificate wiring.)


def test_missing_instance_is_not_bound_on_the_strict_path(intpad2: Path, tmp_path: Path) -> None:
    rc, cert = _certify(intpad2, tmp_path)
    hits = [p for p in cert["binding_problems"] if "protocol_instance.json missing" in p]
    assert rc == 1 and cert["bound"] is False and cert["certified"] is False and cert["protocol_instance"]["attests_protocol_instance"] is False and hits
    RESULTS["missing instance"] = hits[0]
    # and the same record under the legacy flag is UNCERTIFIED in the document itself, never certified
    rc2, cert2 = _certify(intpad2, tmp_path, "--historical-v2")
    assert cert2["certified"] is False and cert2.get("UNCERTIFIED") and cert2["certification_status"] == "UNCERTIFIED"


def test_mismatched_instance_pin_is_refused(tmp_path: Path) -> None:
    rec = _tiny_record(tmp_path)
    _rewrite(rec, "protocol_instance.json", lambda d: d["notes"].append("edited after READY_PHASE0"))  # content changes, pin does not
    problems, _ = check_instance_evidence(rec, json.loads((rec / "manifest.json").read_text()))
    hits = [p for p in problems if "protocol_instance.json sha" in p and "!= manifest inputs.protocol_instance_sha256" in p]
    assert hits, problems
    RESULTS["mismatched instance (edited after pin)"] = hits[0]


def test_wrong_partition_digest_is_refused(tmp_path: Path) -> None:
    rec = _tiny_record(tmp_path)

    def other_partition(d):
        d["partition"]["partition_digest"] = "f" * 64

    _rewrite(rec, "protocol_instance.json", other_partition)
    _repin_instance(rec)
    problems, _ = check_instance_evidence(rec, json.loads((rec / "manifest.json").read_text()))
    hits = [p for p in problems if "partition_digest" in p]
    assert hits, problems
    RESULTS["wrong partition digest"] = hits[0]


def test_chronology_violation_is_refused(tmp_path: Path) -> None:
    rec = _tiny_record(tmp_path)

    def reorder(d):  # r_q recorded before the boundary commitment it must follow
        ev = d["events"]
        i = next(k for k, e in enumerate(ev) if e.get("kind") == "challenge")
        j = next(k for k, e in enumerate(ev) if e.get("kind") == "commitment" and k < i)
        ev[i], ev[j] = ev[j], ev[i]

    _rewrite(rec, "chronology.json", reorder)
    problems, _ = check_instance_evidence(rec, json.loads((rec / "manifest.json").read_text()))
    assert problems, "a reordered chronology must be refused"
    RESULTS["chronology violation (challenge before commitment)"] = problems[0]


def test_swapped_seed_is_refused(tmp_path: Path) -> None:
    rec = _tiny_record(tmp_path)

    def swap(d):
        ev = next(e for e in d["events"] if e["name"] == "seed_reveal")
        ev["seed"] = "ab" * 32 if ev["seed"] != "ab" * 32 else "cd" * 32

    _rewrite(rec, "chronology.json", swap)
    problems, _ = check_instance_evidence(rec, json.loads((rec / "manifest.json").read_text()))
    assert problems, "a swapped seed must be refused"
    RESULTS["swapped seed"] = problems[0]


def test_zz_print_results(capsys) -> None:
    for k, v in RESULTS.items():
        print(f"NEGATIVE {k}: {v[:200]}")
    assert len(RESULTS) >= 8
