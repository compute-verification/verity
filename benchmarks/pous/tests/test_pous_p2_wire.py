"""`p2-audit-wire/v4` (`benchmarks/pous/p2_v1/wire.py`) against its pinned vectors: every negative transcript gives its pinned
FINAL, the small end-to-end audit is accepted and rebuilt byte for byte, every hash `reference.py` pins is the files', and the
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
        seen: set = set()
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
    final = Wi.verify_audit(frames, small["elapsed_ns"], verifier_source=bytes.fromhex(small["audit_source"]),
                            seen_audits=set())
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


def test_every_reference_pin_is_the_files_and_the_package_keys_are_the_vectors():
    """`reference.py` pins each file and each width's `vectors.json` entry; the frames' hashes are in the files."""
    maps, audit, negative = (_load(p) for p in (R.VECTORS, R.AUDIT_VECTORS, R.NEGATIVE_VECTORS))
    files = {p.name: p.read_bytes() for p in (R.VECTORS, R.AUDIT_VECTORS, R.NEGATIVE_VECTORS)}
    pins = V.pins(maps, audit, negative, files)
    assert [pins[p.name] for p in (R.VECTORS, R.AUDIT_VECTORS, R.NEGATIVE_VECTORS)] == [
        R.VECTORS_SHA512, R.AUDIT_VECTORS_SHA512, R.NEGATIVE_VECTORS_SHA512]
    assert {v["w"]: pins[f"entry/{v['id']}"] for v in maps["vectors"]} == R.ENTRY_SHA512
    assert _load(V.KEYS_FIXTURE) == V.keys_fixture(maps, R.VECTORS_SHA512)


@pytest.mark.slow
def test_the_generator_rewrites_the_files_it_pins():
    """Regeneration gives the committed bytes (about a minute: five full-width roots in Python), so the one command that
    migrates the vectors, `python -m p2_v1.vectors`, is the one that made them."""
    maps, audit = V.map_vectors(), V.audit_vectors()
    negative = V.negative_vectors(audit["small_end_to_end"])
    for doc, path in ((maps, R.VECTORS), (audit, R.AUDIT_VECTORS), (negative, R.NEGATIVE_VECTORS)):
        assert V._dump(doc) == path.read_bytes(), path.name


def _small_session():
    small = _load(R.AUDIT_VECTORS)["small_end_to_end"]
    setup = [bytes.fromhex(m["bytes"]) for m in small["setup_messages"]]
    audit = [bytes.fromhex(small["audit_open"])]
    for x in small["exchanges"]:
        audit += [bytes.fromhex(x["challenge"]), bytes.fromhex(x["response"])]
    return small, setup, audit


def _verify(frames, small, **kw):
    kw.setdefault("seen_audits", set())
    return Wi.parse_final(Wi.verify_audit(frames, small["elapsed_ns"], verifier_source=bytes.fromhex(small["audit_source"]),
                                          **kw))


def _setup_with_block(small, setup, i, block):
    """`setup` with block `i` replaced and SETUP_ROOT's root rebuilt over the new leaves, so the root still authenticates."""
    salt, bps = bytes.fromhex(small["setup_salt"]), V.SMALL.blocks_per_segment
    blocks = [bytes.fromhex(b) for b in small["blocks"]]
    blocks[i] = block
    root = R.root([R.leaf(salt, j // bps, j % bps, b) for j, b in enumerate(blocks)])
    return Wi.setup_messages(small["manifest"].encode(), bytes.fromhex(small["w_commitment"]), salt, blocks, root)


def test_a_setup_salt_of_the_wrong_length_is_malformed():
    """A body outside its type's length is code 1, before any binding is compared."""
    small, setup, audit = _small_session()
    for body in (setup[1][10:] + b"\x00", setup[1][10:-1]):
        assert _verify([setup[0], Wi.envelope(Wi.SETUP_SALT, body), *setup[2:], *audit], small) == (False, 1, 0)
def test_several_setup_faults_report_the_first_in_1_2_3_5_precedence():
    """The same 1, 2, 3, 5 precedence applies if one setup transcript has several faults, whatever the frame
    order: a binding mismatch (2) in SETUP_SALT or a SETUP_BLOCK before a malformed later frame (1) is code 1, and an
    out-of-profile block (5) with a wrong root (3) is code 3."""
    small, setup, audit = _small_session()
    params = V.SMALL
    salt_mismatch = Wi.envelope(Wi.SETUP_SALT, bytes([setup[1][10] ^ 1]) + setup[1][11:])
    noncanonical = Wi.envelope(Wi.SETUP_BLOCK, setup[5][10:-params.width_bytes] + params.p.to_bytes(params.width_bytes, "big"))
    bad_mode = setup[-1][:-1] + b"\x02"
    assert _verify([setup[0], salt_mismatch, *setup[2:5], noncanonical, *setup[6:], *audit], small) == (False, 1, 0)
    assert _verify([setup[0], salt_mismatch, *setup[2:-1], bad_mode, *audit], small) == (False, 1, 0)
    assert _verify([*setup[:2], setup[3], setup[2], *setup[4:-1], bad_mode, *audit], small) == (False, 1, 0)
    salt, commitment = bytes.fromhex(small["setup_salt"]), bytes.fromhex(small["w_commitment"])
    k, t, _ = R.keys(salt, params, commitment, 0, 0)
    wide = R.encode(params, (1 << 63) + 5, k, t).to_bytes(params.width_bytes, "big")
    wrong_root = setup[-1][:114] + bytes([setup[-1][114] ^ 1]) + setup[-1][115:]
    assert _verify([*setup[:2], Wi.envelope(Wi.SETUP_BLOCK, setup[2][10:-8] + wide), *setup[3:-1], wrong_root, *audit],
                   small) == (False, 3, 0)
def test_a_public_setup_is_decoded_back_to_the_committed_w():
    """The verifier decodes the stream and checks the reconstructed W commitment: a block that authenticates but
    decodes outside the payload profile is code 5, and one that decodes to another payload is code 3."""
    small, setup, audit = _small_session()
    params = V.SMALL
    salt, commitment = bytes.fromhex(small["setup_salt"]), bytes.fromhex(small["w_commitment"])
    k, t, _ = R.keys(salt, params, commitment, 0, 0)
    for m, code in (((1 << 63) + 5, 5), (2, 3)):
        block = R.encode(params, m, k, t).to_bytes(params.width_bytes, "big")
        assert _verify([*_setup_with_block(small, setup, 0, block), *audit], small) == (False, code, 0)


def test_a_private_setup_checks_the_declared_root_against_the_verifier_s_own():
    """In private setup the verifier constructs the root itself from W and accepts SETUP_ROOT only when it is
    that root; a disagreeing root is code 3 before the epoch is checked.  The verifier never trusts the declared one."""
    small, setup, audit = _small_session()
    root = bytes.fromhex(small["verification_root"])
    private = setup[-1][:-1] + b"\x00"
    assert _verify([*setup[:2], private, *audit], small, private_root=root) == (True, 0, 2)
    wrong = private[:114] + bytes([private[114] ^ 1]) + private[115:]
    assert _verify([*setup[:2], wrong, *audit], small, private_root=root) == (False, 3, 0)
    with pytest.raises(ValueError, match="private setup"):
        _verify([*setup[:2], private, *audit], small)


def test_seen_audit_ids_are_required_and_persist_across_audits():
    """The verifier persists used `(epoch_id, audit_id)` pairs across audits and rejects reuse before round 0."""
    small, setup, audit = _small_session()
    with pytest.raises(TypeError):
        Wi.verify_audit([*setup, *audit], small["elapsed_ns"], verifier_source=bytes.fromhex(small["audit_source"]))
    seen: set = set()
    assert _verify([*setup, *audit], small, seen_audits=seen) == (True, 0, 2)
    assert _verify([*setup, *audit], small, seen_audits=seen) == (False, 2, 0)


def test_flag_bit_1_compares_each_payload_with_the_verifier_held_w():
    """Flag bit 1 compares the decoded payload with verifier-held expected bytes: the true W passes, a W differing
    in a challenged payload is code 5 with the decode flags set, and a verifier holding no W cannot ask for it."""
    small, setup, audit = _small_session()
    o = 10 + Wi.D + Wi.S
    opened = audit[0][:o + 19] + b"\x03" + audit[0][o + 20:]
    w = bytes.fromhex(small["W"])
    assert _verify([*setup, opened, *audit[1:]], small, w_bytes=w) == (True, 0, 2)
    first = small["exchanges"][0]["index"]
    held = bytearray(w)
    held[first * 63 // 8 + 1] ^= 0x01
    final = Wi.verify_audit([*setup, opened, *audit[1:]], small["elapsed_ns"], seen_audits=set(), w_bytes=bytes(held),
                            verifier_source=bytes.fromhex(small["audit_source"]))
    assert Wi.parse_final(final) == (False, 5, 1) and final[-1 - 2 * Wi.D] == 0x0F
    with pytest.raises(ValueError, match="held W"):
        _verify([*setup, opened, *audit[1:]], small)



def test_a_manifest_json_cannot_hold_and_an_unrepresentable_elapsed_time_is_refused():
    """A manifest with `1e999` (no float) or a lone surrogate (not UTF-8) is malformed code 1, not a crash; an
    elapsed time outside a record's u64 is the verifier's own error, refused before any FINAL."""
    small, setup, audit = _small_session()
    commitment = setup[0][-Wi.D:]
    for manifest in (small["manifest"].replace('"segments":1', '"segments":1e999'),
                     small["manifest"].replace('"p2-63b"', '"\\ud800"')):
        m = manifest.encode()
        commit = Wi.envelope(Wi.SETUP_COMMIT, len(m).to_bytes(4, "big") + m + commitment)
        assert _verify([commit, *setup[1:], *audit], small) == (False, 1, 0)
    for elapsed in ([1 << 64, 0], [-1, 0]):
        with pytest.raises(ValueError, match="u64"):
            Wi.verify_audit([*setup, *audit], elapsed, verifier_source=bytes.fromhex(small["audit_source"]), seen_audits=set())
