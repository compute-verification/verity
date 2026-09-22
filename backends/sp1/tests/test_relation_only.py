"""The relation-only DIAGNOSTIC (``unsound-relation-only``): the compact input and the bare gate chain.

The compact-input builder is pure Python and always tested; the guest tests need the diagnostic host, named by
``VERITY_SP1_RELATION_ONLY_HOST`` (the pod installs it as ``/workspace/bin/veritor-zk-host-cuda-unsound-relation-only``),
and are skipped without it.  Proving is heavy and runs only with ``VERITY_SP1_PROVE=1``.  Nothing here is a result:
the guest reads ``K || x || W``, authenticates nothing, and commits ``sha256(input) || sha256(output word) || 0xEE``.
The sha256 pins below are the same constants the Rust unit test (``common/src/relation_only.rs``) asserts, so the
two builders are held to the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from verity.errors import ProtocolError
from verity_sp1.host import Host, SP1Unavailable
from verity_sp1.relation_only import FEATURE, VERDICT_DIAGNOSTIC_ACCEPT, compact_input, encode_compact, public_values_for

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "typed-obligation-v0"
K16 = "accept--packed-word-row"
K16_TAMPERED = "accept--packed-word-tampered-leaf"
K16_HW = "accept--tc-store-cast-nan-hw"  # tc-ampere-bf16@2: not the family the relation-only guest evaluates
K1536 = "stage2--qkv-proj-decode-0-k1536"
PINS = {
    K16: "f72d64523d28995b114189993eb4c419bd095c23a35d575510ef68c562943656",
    K1536: "2ab508b9d12b9bfe7933c95edc0ae26e64f29b99f745867ce50f6e4344fd8561",
}
HOST_ENV = "VERITY_SP1_RELATION_ONLY_HOST"


def record(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


# --- the compact input (no host) ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,K", [(K16, 16), (K1536, 1536)])
def test_compact_input_is_the_vectors_words_and_its_committed_output(name: str, K: int) -> None:
    rec = record(name)
    compact = compact_input(rec["object"], bytes.fromhex(rec["witness"]))
    assert compact.K == K and len(compact.x) == K and len(compact.w) == K
    assert len(compact.data) == 4 + 4 * K
    assert compact.data[:4] == K.to_bytes(4, "little")
    assert compact.data == encode_compact(compact.x, compact.w)
    assert hashlib.sha256(compact.data).hexdigest() == PINS[name], "the bytes the Rust test pins"
    # the output word the object commits to is the vector's recorded boundary value (values[0] is the one obligation)
    assert compact.output_word == rec["values"][0][-1]
    pv = compact.public_values
    assert len(pv) == 65 and pv == public_values_for(compact.data, compact.output_word)
    assert pv[:32] == hashlib.sha256(compact.data).digest()
    assert pv[32:64] == hashlib.sha256(compact.output_word.to_bytes(2, "little")).digest()
    assert pv[64] == VERDICT_DIAGNOSTIC_ACCEPT == 0xEE


def test_a_witness_that_does_not_authenticate_is_refused_before_the_guest() -> None:
    rec = record(K16_TAMPERED)
    with pytest.raises(ProtocolError, match="does not authenticate"):
        compact_input(rec["object"], bytes.fromhex(rec["witness"]))


def test_another_family_is_refused() -> None:
    rec = record(K16_HW)
    with pytest.raises(ProtocolError, match="tc-ampere-bf16@1"):
        compact_input(rec["object"], bytes.fromhex(rec["witness"]))


def test_encode_compact_rejects_what_the_guest_cannot_read() -> None:
    with pytest.raises(ProtocolError):
        encode_compact([1] * 16, [1] * 15)
    with pytest.raises(ProtocolError):
        encode_compact([1] * 8, [1] * 8)  # not a multiple of 16
    with pytest.raises(ProtocolError):
        encode_compact([0x10000] + [0] * 15, [0] * 16)


# --- the guest (the diagnostic host, if given) --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diagnostic_host() -> Host:
    binary = os.environ.get(HOST_ENV)
    if not binary:
        pytest.skip(f"{HOST_ENV} not set: no relation-only diagnostic host to run")
    host = Host(binary=Path(binary), timeout=1800)
    try:
        host.path()
    except SP1Unavailable as error:
        pytest.skip(str(error))
    identity = host.info()
    assert FEATURE in identity.guest_features, f"{binary} is not a {FEATURE} build: {identity.guest_features}"
    return host


def test_the_relation_only_guest_commits_the_expected_public_values(diagnostic_host: Host) -> None:
    rec = record(K16)
    compact = compact_input(rec["object"], bytes.fromhex(rec["witness"]))
    report = diagnostic_host.execute(b"", compact.data)  # the "witness" is the compact input; the statement is ignored
    assert report.public_values == compact.public_values
    assert report.verdict and report.unsound
    assert set(report.cycle_tracker) >= {"relation_only.io", "relation_only.parse", "gates", "relation_only.digest"}
    assert report.cycle_tracker.get("parse.statement") is None and report.cycle_tracker.get("merkle.table") is None


def test_a_wrong_output_is_not_what_the_guest_commits(diagnostic_host: Host) -> None:
    rec = record(K16)
    compact = compact_input(rec["object"], bytes.fromhex(rec["witness"]))
    report = diagnostic_host.execute(b"", compact.data)
    assert report.public_values != public_values_for(compact.data, compact.output_word ^ 1)


@pytest.mark.skipif(not os.environ.get("VERITY_SP1_PROVE"), reason="proving is heavy; set VERITY_SP1_PROVE=1 on a pod")
def test_prove_and_verify_under_the_guests_own_key(diagnostic_host: Host, tmp_path: Path) -> None:
    rec = record(K16)
    compact = compact_input(rec["object"], bytes.fromhex(rec["witness"]))
    report = diagnostic_host.prove(b"", compact.data, tmp_path / "proof.bin", mode="core")
    assert report.execution.public_values == compact.public_values
    verified = diagnostic_host.verify(tmp_path / "proof.bin")
    assert verified.ok and verified.unsound and verified.verdict
    assert verified.vk_hash == diagnostic_host.info().vk_hash
    assert verified.public_values == compact.public_values
