"""`p2-audit-wire/v4` (`benchmarks/pous/p2_v1/wire.py`) against its pinned vectors: every negative transcript gives its pinned
FINAL, the small end-to-end audit is accepted and rebuilt byte for byte, every hash the spec's §7 pins is the files', and the
package's copy of the keys is `vectors.json`'s."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("verity")

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from p2_v1 import reference as R  # noqa: E402
from p2_v1 import vectors as V  # noqa: E402
from p2_v1 import wire as Wi  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_the_files_are_the_pinned_ones_under_this_randomness():
    """A failure here after `verity.ml.randomness` moves is the cue to rerun `python -m p2_v1.vectors`."""
    for path, pin in ((R.VECTORS, R.VECTORS_SHA512), (R.AUDIT_VECTORS, R.AUDIT_VECTORS_SHA512),
                      (R.NEGATIVE_VECTORS, R.NEGATIVE_VECTORS_SHA512)):
        assert hashlib.sha512(path.read_bytes()).hexdigest() == pin, path.name
        assert _load(path)["randomness"] == V.randomness_version(), f"{path.name}: rerun `python -m p2_v1.vectors`"


def test_every_negative_transcript_gives_its_pinned_final():
    doc = _load(R.NEGATIVE_VECTORS)
    assert doc["case_count"] == len(doc["cases"]) == 31
    codes = set()
    for case in doc["cases"]:
        frames = [bytes.fromhex(f) for f in case["frames"]]
        seen = None
        if case["seen_audit_pair"]:
            body = next(f for f in frames if f[5] == Wi.AUDIT_OPEN)[10:]
            seen = {(body[:64], body[64:96])}
        final = Wi.verify_audit(frames, case["elapsed_ns"], verifier_source=bytes.fromhex(case["verifier_source"]),
                                seen_audits=seen)
        assert final.hex() == case["expected_final"], case["name"]
        assert Wi.parse_final(final) == (case["expected_verdict"], case["expected_failure_code"],
                                         case["expected_record_count"]), case["name"]
        codes.add(case["expected_failure_code"])
    assert codes == {0, 1, 2, 3, 4, 6}


def test_the_small_audit_is_accepted_and_rebuilt_byte_for_byte():
    small = _load(R.AUDIT_VECTORS)["small_end_to_end"]
    frames = [bytes.fromhex(m["bytes"]) for m in small["setup_messages"]] + [bytes.fromhex(small["audit_open"])]
    for x in small["exchanges"]:
        frames += [bytes.fromhex(x["challenge"]), bytes.fromhex(x["response"])]
    final = Wi.verify_audit(frames, small["elapsed_ns"], verifier_source=bytes.fromhex(small["audit_source"]))
    assert final.hex() == small["final"] and Wi.parse_final(final) == (True, 0, 2)
    assert V.small_vector() == small
    manifest_length, n, h = len(small["manifest"]), 8, 3
    assert [len(f) - 10 for f in frames] == ([68 + manifest_length, 96] + [108 + n] * 8 + [169, 124]
                                             + [138, 112 + n + 64 * h] * 2)


def test_frame_sizes_at_the_deployment_point():
    """RESPONSE is 112 + n + 64·h: 3,384 bytes (3,394 framed) at n = 2,056 and h = 19 for B = 2^19; FINAL is 182 + 147·r,
    the largest accepted body at 15,617 bytes for r = 105."""
    n, h = R.DEPLOYMENT.width_bytes, Wi.path_length(1 << 19, 12345)
    response = Wi.response_message(bytes(64), bytes(32), 0, 12345, bytes(n), [bytes(64)] * h)
    assert h == 19 and len(response) == 3_394 and len(response) - 10 == 112 + n + 64 * h
    assert Wi._parse_response(response, n)[4] == bytes(n)
    final = Wi.final_message(bytes(64), bytes(32), 0, 0, 0, [], [Wi.record(0, 0, 0, 0, bytes(64), bytes(64))] * 105)
    assert len(final) - 10 == Wi.FINAL_FIXED + 105 * Wi.RECORD_BYTES == 15_617
    assert (Wi.OPEN_BYTES, Wi.CHALLENGE_BYTES, Wi.SETUP_ROOT_BYTES, Wi.RECORD_BYTES) == (124, 138, 169, 147)


def test_every_spec_pin_is_the_files_and_the_package_keys_are_the_vectors():
    maps, audit, negative = (_load(p) for p in (R.VECTORS, R.AUDIT_VECTORS, R.NEGATIVE_VECTORS))
    files = {p.name: p.read_bytes() for p in (R.VECTORS, R.AUDIT_VECTORS, R.NEGATIVE_VECTORS)}
    spec = V.SPEC.read_text()
    for name, value in V.pins(maps, audit, negative, files).items():
        assert value in spec, name
    assert _load(V.KEYS_FIXTURE) == V.keys_fixture(maps, R.VECTORS_SHA512)


@pytest.mark.slow
def test_the_generator_rewrites_the_files_it_pins():
    """Regeneration gives the committed bytes (about a minute: five full-width roots in Python), so the one command that
    migrates the vectors, `python -m p2_v1.vectors`, is the one that made them."""
    maps, audit = V.map_vectors(), V.audit_vectors()
    negative = V.negative_vectors(audit["small_end_to_end"])
    for doc, path in ((maps, R.VECTORS), (audit, R.AUDIT_VECTORS), (negative, R.NEGATIVE_VECTORS)):
        assert V._dump(doc) == path.read_bytes(), path.name
