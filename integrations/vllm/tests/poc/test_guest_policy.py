"""GUEST (GUIDANCE §57.10, S12 caveat): approved executables are refused fail-closed, and the opt-in statement
format is bound by the digest of the RECORDED bytes.

* approved (ELF, vk) -> passes (with its approval level stated);
* unknown ELF -> NOT BOUND (``GUEST_NOT_APPROVED``) on the strict path, a warning under ``--historical-v2``;
* approved ELF with a mismatched vk -> NOT BOUND;
* the record's own ``guest_features`` / ``guest_build`` claims never substitute for the policy;
* ``check_binding(..., statement_format=3)`` accepts the format-3 public values and refuses the format-2 ones
  (and the reverse), so a record's ``h_L`` names which bytes the guest verified;
* ``protocol_instance.proof.statement_format`` decodes, validates (2 or 3 only) and leaves every existing
  instance's digest unchanged when absent.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

POC = Path(__file__).resolve().parents[1]
ROOT = POC.parent
sys.path.insert(0, str(POC))

from e2e import approved_guests as ag  # noqa: E402

RECORD = POC / "results" / "INT-PAD-2" / "INTPAD2-B0-20260909T013519Z"
GOOD_ELF = "73081b7b2c9305a93f243a2137f71c851fb24bddf5e6edfbeb398acb2f309029"
GOOD_VK = "0x006ecb51c5f0e7b5e946186af4ccada3161e17fef9c0152052f7d26a8ae98d0d"
COV_ELF = "0eec2c182638ec60a71f9fa4e43ec3cd78057139a92ada6e3aafe08ae1757686"  # the PROOF-COV host's guest: still provisional-legacy
COV_VK = "0x00ec028500bc7c65a90eab54548942c25ab8c69a37a58fcd551c4ac70c6a57e1"


def _policy_without(elf_prefix: str) -> dict:
    doc = ag.load_policy()
    return {**doc, "guests": [g for g in doc["guests"] if not g["elf_sha256"].startswith(elf_prefix)]}


def _policy_with_vk(elf_prefix: str, vk: str) -> dict:
    doc = ag.load_policy()
    return {**doc, "guests": [({**g, "vk_hash": vk} if g["elf_sha256"].startswith(elf_prefix) else g) for g in doc["guests"]]}


# --- the policy module -----------------------------------------------------------------------------------------------------


def test_the_seed_policy_lists_every_elf_pinned_by_a_certified_or_committed_record() -> None:
    doc = ag.load_policy()
    listed = {g["elf_sha256"] for g in doc["guests"]}
    pinned: dict[str, str] = {}
    for cert in POC.glob("results/*/*/certificate.json"):
        c = json.loads(cert.read_text())
        pinned[(c.get("proofs") or {})["elf_sha256"]] = str(cert)
        cov = ((c.get("kind_coverage") or {}).get("program_identity") or {}).get("elf_sha256")
        if cov:
            pinned[cov] = str(cert)
    for info in POC.glob("fixtures/e2e-composed/*/proofs/info.json"):
        pinned[json.loads(info.read_text())["elf_sha256"]] = str(info)
    missing = {e: src for e, src in pinned.items() if e not in listed}
    assert not missing, f"records pin ELFs the policy does not list: {missing}"
    for g in doc["guests"]:
        if g["approval"] == "reproduced":
            assert g["reproduction"]["bit_identical"] is True and g["guest_build"]["source_tree_sha"] and not g["missing"], g["elf_sha256"][:12]
        else:
            assert g["approval"] == "provisional-legacy" and g["missing"], "a provisional entry says what is missing"
    assert {g["approval"] for g in doc["guests"]} == {"reproduced", "provisional-legacy"}
    # the three ELFs pinned by CERTIFIED records were all rebuilt bit for bit on 2026-09-09 (reports/GUEST-2026-09-09.md §6)
    assert {g["elf_sha256"][:8] for g in doc["guests"] if g["approval"] == "reproduced"} >= {"73081b7b", "a4d1efe1", "b70e1091", "9fd00e60"}


def test_approved_pair_passes_and_states_its_level() -> None:
    a = ag.approval_of(GOOD_ELF, GOOD_VK)
    assert a.ok and a.reason is None and a.approval == "reproduced" and not a.missing
    summary, problems, warnings = ag.check_guest(GOOD_ELF, GOOD_VK, None, strict=True)
    assert not problems and not warnings and summary["approval"]["ok"] and summary["guest_build"]["elf_sha256"] == GOOD_ELF
    assert summary["guest_build"]["source_tree_sha"] == "9cf2b3e5cb1c3cee4bbfc58d590e91a4cf948bda"
    # a PROVISIONAL approval is never silent
    p = ag.approval_of(COV_ELF, COV_VK)
    assert p.ok and p.approval == "provisional-legacy" and p.missing
    _, problems, warnings = ag.check_guest(COV_ELF, COV_VK, None, strict=True)
    assert not problems and any("PROVISIONALLY" in w for w in warnings)


@pytest.mark.parametrize("elf, vk, why", [
    ("00" * 32, GOOD_VK, "not in"),  # unknown ELF, a known vk
    (GOOD_ELF, "0x00deadbeef", "approved with vk"),  # approved ELF, another vk
    (None, GOOD_VK, "pins no guest"),  # nothing pinned at all
    (GOOD_ELF, None, "pins no guest"),
])
def test_unknown_elf_or_mismatched_vk_is_refused_on_the_strict_path_and_warned_on_the_legacy_path(elf, vk, why) -> None:
    a = ag.approval_of(elf, vk)
    assert not a.ok and a.reason == ag.GUEST_NOT_APPROVED and why in a.detail, a
    # the record's CLAIM of safe features / a complete build pin changes nothing
    claimed = {"source_commit": "f" * 40, "source_tree_sha": "e" * 40, "toolchain": {"cargo_prove": "x", "rustc": "y", "sp1_sdk": "6.4.0"},
               "features_resolved": [], "elf_sha256": elf, "vk_hash": vk, "built_by": "prover", "built_at": "now"}
    summary, problems, warnings = ag.check_guest(elf, vk, claimed, strict=True)
    assert problems and all(ag.GUEST_NOT_APPROVED in p for p in problems) and summary["guest_build"] is None, problems
    summary, problems, warnings = ag.check_guest(elf, vk, claimed, strict=False)
    assert not problems and warnings and ag.GUEST_NOT_APPROVED in warnings[0] and "--historical-v2" in warnings[0]


def test_a_recorded_build_pin_that_names_another_elf_is_a_problem_even_for_an_approved_guest() -> None:
    other = {"elf_sha256": "11" * 32, "vk_hash": GOOD_VK}
    _, problems, _ = ag.check_guest(GOOD_ELF, GOOD_VK, other, strict=True)
    assert problems and "guest_build names ELF" in problems[0]


def test_the_policy_file_is_validated(tmp_path: Path) -> None:
    doc = ag.load_policy()
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({**doc, "guests": doc["guests"] + [doc["guests"][0]]}))
    with pytest.raises(ValueError, match="listed twice"):
        ag.load_policy(bad)
    bad.write_text(json.dumps({**doc, "guests": [{**doc["guests"][0], "approval": "trusted"}]}))
    with pytest.raises(ValueError, match="approval"):
        ag.load_policy(bad)
    bad.write_text(json.dumps({**doc, "schema": "other"}))
    with pytest.raises(ValueError, match="schema"):
        ag.load_policy(bad)


# --- certificate.py on a real record --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record_copy(tmp_path_factory) -> Path:
    if not (RECORD / "manifest.json").exists():
        pytest.skip("INT-PAD-2 record not present")
    dst = tmp_path_factory.mktemp("rec") / RECORD.name
    # the record minus its heavy parts; certificate.py needs manifest, summary, proofs/, bin/, obligations/, sampling_policy.json
    shutil.copytree(RECORD, dst, ignore=shutil.ignore_patterns("bundle", "reverify", "handshake", "logs", "log.txt", "plan", "program", "verify"))
    return dst


def _certify(record: Path, tmp_path: Path, *args: str) -> tuple[int, dict]:
    from e2e import certificate

    out = tmp_path / "certificate.json"
    rc = certificate.main(["--record", str(record), "--out", str(out), *args])
    return rc, json.loads(out.read_text())


def test_certificate_historical_path_still_binds_the_legacy_record_and_reports_the_approval_levels(record_copy: Path, tmp_path: Path) -> None:
    rc, cert = _certify(record_copy, tmp_path, "--historical-v2")
    guest = cert["proofs"]["guest"]
    assert guest["approval"]["ok"] and guest["approval"]["approval"] == "reproduced"  # 73081b7b rebuilt from 09b98845ce87 (GUEST §6)
    assert cert["kind_coverage"]["guest"]["approval"]["approval"] == "provisional-legacy"  # the PROOF-COV host is still provisional
    assert cert["guest_policy"]["enforced"] is False
    assert cert["commitments"]["statement_format"] == 2 and cert["commitments"]["statement_v2_digest"] == cert["commitments"]["h_L"]
    assert not any(ag.GUEST_NOT_APPROVED in p for p in cert["binding_problems"]), cert["binding_problems"]
    # what the record itself established is unchanged by the policy (whatever else --historical-v2 decides about this copy)
    assert cert["proofs"]["elf_sha256"] == GOOD_ELF and cert["proofs"]["guest_features_claimed"] is None or cert["proofs"]["guest_features_claimed"] == []


def test_certificate_strict_path_refuses_an_unlisted_elf_regardless_of_claims(record_copy: Path, tmp_path: Path) -> None:
    pol = tmp_path / "approved.json"
    pol.write_text(json.dumps(_policy_without(GOOD_ELF[:8])))
    # a manifest that CLAIMS a safe, fully pinned build for the very same ELF
    manifest = json.loads((record_copy / "manifest.json").read_text())
    manifest["inputs"]["guest_features"] = []
    manifest["inputs"]["guest_build"] = {"source_commit": "f" * 40, "source_tree_sha": "e" * 40, "toolchain": {"cargo_prove": "1", "rustc": "1", "sp1_sdk": "6.4.0"},
                                        "features_resolved": [], "elf_sha256": GOOD_ELF, "vk_hash": GOOD_VK, "built_by": "prover", "built_at": "2026-09-09T00:00:00Z"}
    claimed = record_copy.parent / (record_copy.name + "-claimed")
    if not claimed.exists():
        shutil.copytree(record_copy, claimed)
    (claimed / "manifest.json").write_text(json.dumps(manifest))
    rc, cert = _certify(claimed, tmp_path, "--approved-guests", str(pol))
    assert rc == 1 and cert["bound"] is False
    refusals = [p for p in cert["binding_problems"] if ag.GUEST_NOT_APPROVED in p]
    assert refusals and "proofs guest" in refusals[0] and GOOD_ELF[:12] in refusals[0], cert["binding_problems"]
    assert cert["proofs"]["guest"]["approval"]["ok"] is False and cert["proofs"]["guest"]["guest_build"] is None
    assert cert["proofs"]["guest"]["guest_build_recorded"]["elf_sha256"] == GOOD_ELF  # the claim is recorded AS a claim
    assert cert["guest_policy"]["enforced"] is True
    # the same unlisted ELF under --historical-v2: no GUEST_NOT_APPROVED problem (a warning), the certificate is labelled unenforced
    rc2, cert2 = _certify(claimed, tmp_path, "--historical-v2", "--approved-guests", str(pol))
    assert not any(ag.GUEST_NOT_APPROVED in p for p in cert2["binding_problems"])
    assert cert2["guest_policy"]["enforced"] is False and any(ag.GUEST_NOT_APPROVED in w for w in cert2["proofs"]["guest"]["warnings"])


def test_certificate_strict_path_refuses_an_approved_elf_with_a_mismatched_vk(record_copy: Path, tmp_path: Path) -> None:
    pol = tmp_path / "approved.json"
    pol.write_text(json.dumps(_policy_with_vk(GOOD_ELF[:8], "0x00" + "ab" * 31)))
    rc, cert = _certify(record_copy, tmp_path, "--approved-guests", str(pol))
    assert rc == 1 and cert["bound"] is False
    refusals = [p for p in cert["binding_problems"] if ag.GUEST_NOT_APPROVED in p and "proofs guest" in p]
    assert refusals and "approved with vk" in refusals[0], cert["binding_problems"]


def test_certificate_strict_path_has_no_guest_problem_for_the_seeded_pair(record_copy: Path, tmp_path: Path) -> None:
    rc, cert = _certify(record_copy, tmp_path)
    assert not any(ag.GUEST_NOT_APPROVED in p for p in cert["binding_problems"]), cert["binding_problems"]
    assert cert["proofs"]["guest"]["approval"]["approval"] == "reproduced"
    assert cert["kind_coverage"]["guest"]["approval"]["ok"] is True  # the family-proof host's ELF is checked too


# --- the statement format is what h_L binds ---------------------------------------------------------------------------------


def _accept_object() -> dict:
    vec = json.loads((ROOT / "fixtures" / "typed-obligation-v0" / "accept--packed-word-row.json").read_text())
    return vec


def test_check_binding_recomputes_h_l_in_the_recorded_format() -> None:
    from veritor.interface.binding import BindingCode, bound_public_values, check_binding

    vec = _accept_object()
    obj = vec["object"]
    pv2, pv3 = bound_public_values(obj), bound_public_values(obj, statement_format=3)
    assert pv2[:32] == pv3[:32] and pv2[32:64] != pv3[32:64] and pv2[64] == pv3[64] == 1
    assert pv2[32:64].hex() == vec["statement_digest"] and pv3[32:64].hex() == vec["statement_v3_digest"]
    assert check_binding(obj, pv2).ok and check_binding(obj, pv2, statement_format=2).ok
    assert check_binding(obj, pv3, statement_format=3).ok
    r = check_binding(obj, pv3)  # a format-3 proof checked as the historical format: refused, not silently accepted
    assert not r.ok and r.code == BindingCode.E_LOWERING_MISMATCH
    r = check_binding(obj, pv2, statement_format=3)
    assert not r.ok and r.code == BindingCode.E_LOWERING_MISMATCH
    with pytest.raises(ValueError):
        bound_public_values(obj, statement_format=4)


def test_protocol_instance_proof_section_is_optional_validated_and_digest_neutral(tmp_path: Path) -> None:
    from e2e import protocol_instance as pi

    files = sorted(POC.glob("results/PROTO-A/*/protocol_instance.json"))
    if not files:
        pytest.skip("no PROTO-A instance")
    doc = json.loads(files[0].read_text())
    inst = pi.load_instance(doc)
    assert inst.proof is None and inst.statement_format == 2
    assert pi.instance_digest(inst) == pi.instance_digest(doc) and "proof" not in pi.to_json(inst)
    with_proof = {**doc, "proof": {"statement_format": 3, "source": "run_composed --statement-format 3"}}
    inst3 = pi.load_instance(with_proof)
    assert inst3.statement_format == 3 and pi.to_json(inst3)["proof"]["statement_format"] == 3
    assert pi.instance_digest(inst3) != pi.instance_digest(inst)
    with pytest.raises(pi.InstanceError, match="statement_format must be one of"):
        pi.load_instance({**doc, "proof": {"statement_format": 4, "source": "x"}})
    with pytest.raises(pi.InstanceError, match="required field missing"):
        pi.load_instance({**doc, "proof": {"statement_format": 3}})
    # against a record: an instance naming format 3 does not describe a format-2 record (and the reverse)
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "manifest.json").write_text(json.dumps({"inputs": {"program_digest": inst.program.digest, "statement_format": 2}}))
    assert any("statement_format 2 != instance proof.statement_format 3" in p for p in pi.check_against_record(inst3, rec))
    assert not any("statement_format" in p for p in pi.check_against_record(inst, rec))
