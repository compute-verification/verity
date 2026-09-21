"""The SP1 backend over the recorded ``typed-obligation/v0`` vectors: the known-good baseline and its negatives.

Every test needs the built host (``cargo build --release -p veritor-zk-host``); the module is skipped without it.
The vectors are the cross-language conformance records under ``fixtures/typed-obligation-v0``; their ``binding``
column records the public values a correct prover produced for them, so the guest here is checked against what the
veritor guest produced -- byte for byte -- rather than against itself.  Executing is light (the K=1536 vector is
10 M cycles); proving is not and belongs on a pod, so the K=16 prove test runs only when ``VERITY_SP1_PROVE=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from verity.verification.binding import BindingCode, check_binding
from verity_sp1.host import APPROVED, GuestIdentity, Host, SP1Unavailable
from verity_sp1.typed import execute_object, lower_object, prove_object, verify_object

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "typed-obligation-v0"
K16 = "accept--packed-word-row"                    # GemmCoordinate<16> in tc-ampere-bf16@1, packed-word authentication
K16_TAMPERED = "accept--packed-word-tampered-leaf"  # the same object; a witness whose leaf does not authenticate
K16_HW = "accept--tc-store-cast-nan-hw"             # tc-ampere-bf16@2 (the hardware NaN word)
K1536 = "stage2--qkv-proj-decode-0-k1536"           # one real K=1536 coordinate of a captured vLLM decode step

try:
    Host().path()
except SP1Unavailable as error:
    pytest.skip(str(error), allow_module_level=True)


def record(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def host() -> Host:
    return Host(timeout=1800)


def test_the_guest_is_the_approved_one(host: Host) -> None:
    """A rebuilt guest that prints another ELF digest or verifying key is a different prover; pin it here.

    The build is reproducible (``host/build.rs`` remaps paths), so a sound host built from this tree matches.
    A host that does not is still fine for the ``execute`` tests -- public values do not depend on the ELF --
    but its proofs are not accepted; without ``VERITY_SP1_PROVE`` that is reported as a skip, not a failure."""

    identity = host.info()
    assert identity.guest_features == (), "a diagnostic (unsound-*) guest is not the approved prover"
    if identity != APPROVED and not os.environ.get("VERITY_SP1_PROVE"):
        pytest.skip(f"{host.path()} is not the approved prover (elf {identity.elf_sha256[:12]}, vk {identity.vk_hash[:14]}); rebuild it")
    assert identity == APPROVED, identity


def test_lowering_reproduces_the_recorded_statement() -> None:
    for name in (K16, K16_TAMPERED, K16_HW, K1536):
        rec = record(name)
        lowered = lower_object(rec["object"])
        assert lowered.statement.hex() == rec["statement"], name
        assert lowered.object_digest == rec["digest"], name


@pytest.mark.parametrize("name", [K16, K16_HW, K1536])
def test_execute_reproduces_the_recorded_public_values(host: Host, name: str) -> None:
    rec = record(name)
    report, binding = execute_object(host, rec["object"], bytes.fromhex(rec["witness"]))
    assert report.verdict and not report.unsound
    assert report.public_values.hex() == rec["binding"]["public_values"]
    assert binding.ok and binding.verdict == 1, binding
    assert report.total_cycles > 0 and set(report.cycle_tracker) >= {"parse", "merkle", "gates"}


def test_a_tampered_leaf_is_a_proof_of_rejection(host: Host) -> None:
    rec = record(K16_TAMPERED)
    assert rec["witness_verdict"] == "reject:INVALID_OPENING"
    report, binding = execute_object(host, rec["object"], bytes.fromhex(rec["witness"]))
    assert not report.verdict
    assert binding.ok and binding.verdict == 0, "the public values still bind the object; the verdict byte says reject"
    assert report.public_values[:64] == bytes.fromhex(rec["binding"]["public_values"])[:64]


def test_the_33_byte_layout_binds_no_object(host: Host) -> None:
    rec = record(K16)
    lowered = lower_object(rec["object"])
    report = host.execute(lowered.statement, bytes.fromhex(rec["witness"]))  # no --typed-object-digest
    assert report.verdict and len(report.public_values) == 33
    binding = check_binding(rec["object"], report.public_values)
    assert not binding.ok and binding.code == BindingCode.E_UNBOUND_OBJECT


def test_another_objects_digest_does_not_bind(host: Host) -> None:
    """The guest carries h_O unchecked; the verifier's binding check is what refuses a foreign object."""

    rec, other = record(K16), record(K16_HW)
    lowered = lower_object(rec["object"])
    report = host.execute(lowered.statement, bytes.fromhex(rec["witness"]), typed_object_digest=other["digest"])
    assert report.verdict
    binding = check_binding(rec["object"], report.public_values)
    assert not binding.ok and binding.code == BindingCode.E_OBJECT_DIGEST_MISMATCH
    binding_other = check_binding(other["object"], report.public_values)
    assert not binding_other.ok and binding_other.code == BindingCode.E_LOWERING_MISMATCH


@pytest.mark.skipif(not os.environ.get("VERITY_SP1_PROVE"), reason="proving is heavy; set VERITY_SP1_PROVE=1 on a pod")
def test_prove_and_accept_the_known_good_k16_instance(host: Host, tmp_path: Path) -> None:
    rec = record(K16)
    proof = tmp_path / "k16.proof"
    report = prove_object(host, rec["object"], bytes.fromhex(rec["witness"]), proof, mode="core")
    assert report.execution.verdict and report.vk_hash == APPROVED.vk_hash and report.proof_bytes == proof.stat().st_size
    assert report.execution.public_values.hex() == rec["binding"]["public_values"]

    accepted = verify_object(host, rec["object"], proof)
    assert accepted.accepted, accepted.reasons()
    assert accepted.verify.statement_match is True

    # the same proof against another object of the same family: verifies, is not a proof of that object
    other = record(K16_HW)
    foreign = verify_object(host, other["object"], proof)
    assert foreign.proof_verifies and not foreign.accepted
    assert foreign.binding.code == BindingCode.E_OBJECT_DIGEST_MISMATCH and foreign.verify.statement_match is False

    # an unapproved verifying key is refused even though the proof verifies
    unapproved = verify_object(host, rec["object"], proof, approved=GuestIdentity("00" * 32, "0x" + "00" * 32, "6.4.0"))
    assert unapproved.proof_verifies and not unapproved.approved_key and not unapproved.accepted

    # a corrupted proof file does not verify
    corrupt = tmp_path / "corrupt.proof"
    data = bytearray(proof.read_bytes())
    data[len(data) // 2] ^= 0x01
    corrupt.write_bytes(bytes(data))
    broken = verify_object(host, rec["object"], corrupt)
    assert not broken.proof_verifies and not broken.accepted
