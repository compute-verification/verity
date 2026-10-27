"""`p2-audit-wire/v4`'s frames (`benchmarks/pous/p2_v1/wire.py`) against its pinned vectors: the small end-to-end audit is
rebuilt byte for byte, the frames have the spec's sizes, every hash the spec's §7 pins is the files', and the package's
copy of the keys is `vectors.json`'s."""
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
    for path, pin in ((R.VECTORS, R.VECTORS_SHA512), (R.AUDIT_VECTORS, R.AUDIT_VECTORS_SHA512)):
        assert hashlib.sha512(path.read_bytes()).hexdigest() == pin, path.name
        assert _load(path)["randomness"] == V.randomness_version(), f"{path.name}: rerun `python -m p2_v1.vectors`"


def test_the_small_audit_is_rebuilt_byte_for_byte():
    small = _load(R.AUDIT_VECTORS)["small_end_to_end"]
    frames = [bytes.fromhex(m["bytes"]) for m in small["setup_messages"]] + [bytes.fromhex(small["audit_open"])]
    for x in small["exchanges"]:
        frames += [bytes.fromhex(x["challenge"]), bytes.fromhex(x["response"])]
    assert V.small_vector() == small
    manifest_length, n, h = len(small["manifest"]), 8, 3
    assert [len(f) - 10 for f in frames] == ([68 + manifest_length, 96] + [108 + n] * 8 + [169, 124]
                                             + [138, 112 + n + 64 * h] * 2)


def test_frame_sizes_at_the_deployment_point():
    """RESPONSE is 112 + n + 64·h: 3,384 bytes (3,394 framed) at n = 2,056 and h = 19 for B = 2^19; FINAL is 182 + 147·r,
    the largest body at 15,617 bytes for r = 105."""
    n, h = R.DEPLOYMENT.width_bytes, Wi.path_length(1 << 19, 12345)
    response = Wi.response_message(bytes(64), bytes(32), 0, 12345, bytes(n), [bytes(64)] * h)
    assert h == 19 and len(response) == 3_394 and len(response) - 10 == 112 + n + 64 * h
    final = Wi.final_message(bytes(64), bytes(32), 0, 0, 0, [], [Wi.record(0, 0, 0, 0, bytes(64), bytes(64))] * 105)
    assert len(final) - 10 == Wi.FINAL_FIXED + 105 * Wi.RECORD_BYTES == 15_617
    assert (Wi.OPEN_BYTES, Wi.CHALLENGE_BYTES, Wi.SETUP_ROOT_BYTES, Wi.RECORD_BYTES) == (124, 138, 169, 147)


def test_every_spec_pin_is_the_files_and_the_package_keys_are_the_vectors():
    maps, audit = (_load(p) for p in (R.VECTORS, R.AUDIT_VECTORS))
    files = {p.name: p.read_bytes() for p in (R.VECTORS, R.AUDIT_VECTORS)}
    spec = V.SPEC.read_text()
    for name, value in V.pins(maps, audit, files).items():
        assert value in spec, name
    assert _load(V.KEYS_FIXTURE) == V.keys_fixture(maps, R.VECTORS_SHA512)


@pytest.mark.slow
def test_the_generator_rewrites_the_files_it_pins():
    """Regeneration gives the committed bytes (about a minute: five full-width roots in Python), so the one command that
    migrates the vectors, `python -m p2_v1.vectors`, is the one that made them."""
    for doc, path in ((V.map_vectors(), R.VECTORS), (V.audit_vectors(), R.AUDIT_VECTORS)):
        assert V._dump(doc) == path.read_bytes(), path.name
