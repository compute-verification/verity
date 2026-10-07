"""`p2-audit-wire/v4`, the P2 audit's byte-exact frames (spec §4, "Audit protocol and wire formats"): the builders the
vectors are made with, and the reference verifier, `verify_audit`, which reads a setup and one audit transcript and returns
the exact `FINAL` frame.

A port of the P2 kit's `p2_audit_verifier.py` (`p2-audit-wire/v1`) onto `reference.py` and `verity.primitives.randomness`,
with v4's hashes: every digest is SHA-512's whole 64 bytes, so the commitment, root, epoch id, sibling, transcript hash and
the frame hashes in a record are 64 bytes where v1's were 32; the salt, the audit id and the nonce stay 32 bytes. Like the
kit's, it checks a public setup's blocks against the declared root but does not decode them back to `W`; that belongs to
setup, not to the audit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableSet, Sequence
from dataclasses import dataclass

from . import reference as R

MAGIC = b"P2AU"
WIRE_VERSION = 4
TRANSCRIPT_TAG = b"verity/pous/p2/transcript/v4\0"
MAX_W = 16_448
MAX_MANIFEST = 512
MAX_K = 105
D, S = R.DIGEST_BYTES, R.SALT_BYTES

SETUP_COMMIT, SETUP_SALT, SETUP_BLOCK, SETUP_ROOT = 0x01, 0x02, 0x03, 0x04
AUDIT_OPEN, CHALLENGE, RESPONSE, FINAL = 0x10, 0x11, 0x12, 0x13
TYPES = {SETUP_COMMIT, SETUP_SALT, SETUP_BLOCK, SETUP_ROOT, AUDIT_OPEN, CHALLENGE, RESPONSE, FINAL}

FAIL_SUCCESS, FAIL_MALFORMED, FAIL_IDENTITY, FAIL_AUTHENTICATION = 0, 1, 2, 3
FAIL_LATE, FAIL_DECODE, FAIL_TRANSPORT = 4, 5, 6
FAILURE_CODES = {"0": "success", "1": "malformed", "2": "identity/round/index/replay/domain", "3": "authentication",
                 "4": "late", "5": "decode", "6": "transport"}

OPEN_BYTES = D + S + 1 + 2 + 8 + 8 + 1 + 8
CHALLENGE_BYTES = D + S + 2 + 8 + S
RESPONSE_FIXED = D + S + 2 + 8 + 4 + 2
SETUP_ROOT_BYTES = D + S + 8 + D + 1
FINAL_FIXED = D + S + 1 + 1 + 2 + 8 + 8 + D + 2
RECORD_BYTES = 2 + 8 + 8 + 1 + D + D
"""Body lengths (spec §4's table): AUDIT_OPEN 124, CHALLENGE 138, RESPONSE 112 + n + 64·h, SETUP_ROOT 169,
FINAL 182 + 147·r."""


class WireError(ValueError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


# --- builders --------------------------------------------------------------------------------------------------------------

def envelope(message_type: int, body: bytes) -> bytes:
    return MAGIC + bytes((WIRE_VERSION, message_type)) + len(body).to_bytes(4, "big") + body


def setup_messages(manifest: bytes, commitment: bytes, salt: bytes, blocks: Sequence[bytes], root: bytes) -> list[bytes]:
    """A public setup (mode 1): SETUP_COMMIT, SETUP_SALT, one SETUP_BLOCK per block, then SETUP_ROOT."""
    out = [envelope(SETUP_COMMIT, len(manifest).to_bytes(4, "big") + manifest + commitment),
           envelope(SETUP_SALT, commitment + salt)]
    out += [envelope(SETUP_BLOCK, commitment + salt + i.to_bytes(8, "big") + len(b).to_bytes(4, "big") + b)
            for i, b in enumerate(blocks)]
    return out + [envelope(SETUP_ROOT, commitment + salt + len(blocks).to_bytes(8, "big") + root + b"\x01")]


def open_message(epoch: bytes, audit_id: bytes, mode: int, k: int, delta_ns: int, rtt_ns: int, flags: int,
                 counter: int) -> bytes:
    return envelope(AUDIT_OPEN, epoch + audit_id + bytes((mode,)) + k.to_bytes(2, "big") + delta_ns.to_bytes(8, "big")
                    + rtt_ns.to_bytes(8, "big") + bytes((flags,)) + counter.to_bytes(8, "big"))


def challenge_message(epoch: bytes, audit_id: bytes, j: int, index: int, nonce: bytes) -> bytes:
    return envelope(CHALLENGE, epoch + audit_id + j.to_bytes(2, "big") + index.to_bytes(8, "big") + nonce)


def response_message(epoch: bytes, audit_id: bytes, j: int, index: int, block: bytes, path: Sequence[bytes]) -> bytes:
    return envelope(RESPONSE, epoch + audit_id + j.to_bytes(2, "big") + index.to_bytes(8, "big")
                    + len(block).to_bytes(4, "big") + block + len(path).to_bytes(2, "big") + b"".join(path))


def record(j: int, index: int, elapsed_ns: int, flags: int, challenge_hash: bytes, response_hash: bytes) -> bytes:
    return (j.to_bytes(2, "big") + index.to_bytes(8, "big") + elapsed_ns.to_bytes(8, "big") + bytes((flags,))
            + challenge_hash + response_hash)


def final_message(epoch: bytes, audit_id: bytes, code: int, delta_ns: int, rtt_ns: int, transcript: Sequence[bytes],
                  records: Sequence[bytes]) -> bytes:
    th = hashlib.sha512(TRANSCRIPT_TAG + b"".join(transcript)).digest()
    n = len(records).to_bytes(2, "big")
    return envelope(FINAL, epoch + audit_id + bytes((code == FAIL_SUCCESS, code)) + n + delta_ns.to_bytes(8, "big")
                    + rtt_ns.to_bytes(8, "big") + th + n + b"".join(records))


def path_length(count: int, index: int) -> int:
    """The exact RFC 9162 sibling count h(index, count)."""
    if not 0 <= index < count:
        raise WireError(FAIL_IDENTITY, "index is outside the block population")
    length = 0
    while count > 1:
        split = 1 << ((count - 1).bit_length() - 1)
        index, count = (index, split) if index < split else (index - split, count - split)
        length += 1
    return length


def verify_path(root: bytes, count: int, index: int, leaf: bytes, path: Sequence[bytes]) -> bool:
    """RFC 9162 §2.1.3.2's inclusion-proof verification, with `node` for its hash."""
    if not 0 <= index < count:
        return False
    fn, sn, value = index, count - 1, leaf
    for sibling in path:
        if len(sibling) != D or sn == 0:
            return False
        if fn & 1 or fn == sn:
            value = R.node(sibling, value)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            value = R.node(value, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and value == root


# --- the verifier ----------------------------------------------------------------------------------------------------------

def _canonical_json(data: bytes) -> tuple[dict, bytes]:
    def no_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise WireError(FAIL_MALFORMED, f"duplicate manifest key {key}")
            out[key] = value
        return out

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WireError(FAIL_MALFORMED, "manifest is not canonical UTF-8 JSON") from error
    if type(value) is not dict:
        raise WireError(FAIL_MALFORMED, "manifest is not an object")
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if encoded != data:
        raise WireError(FAIL_MALFORMED, "manifest is not canonical JSON")
    return value, encoded


def parse_envelope(frame_bytes: bytes, expected_type: int | None = None) -> tuple[int, bytes]:
    if type(frame_bytes) is not bytes:
        raise WireError(FAIL_MALFORMED, "frame is not bytes")
    if len(frame_bytes) < 10:
        raise WireError(FAIL_MALFORMED, "truncated frame header")
    if frame_bytes[:4] != MAGIC:
        raise WireError(FAIL_MALFORMED, "wrong magic")
    if frame_bytes[4] != WIRE_VERSION:
        raise WireError(FAIL_MALFORMED, "wrong wire version")
    message_type = frame_bytes[5]
    if len(frame_bytes) != 10 + int.from_bytes(frame_bytes[6:10], "big"):
        raise WireError(FAIL_MALFORMED, "body length mismatch or trailing bytes")
    if expected_type is not None and message_type != expected_type:
        raise WireError(FAIL_MALFORMED, "wrong message type")
    if message_type not in TYPES:
        raise WireError(FAIL_MALFORMED, "unknown message type")
    return message_type, frame_bytes[10:]


def payload_bits(profile: str, width: int) -> int:
    if width == MAX_W and profile != "p2-2054B":
        raise WireError(FAIL_MALFORMED, "deployment width requires p2-2054B")
    if profile.startswith("p2-") and profile[-1:] in ("b", "B"):
        value = profile[3:-1]
        if value.isascii() and value.isdigit():
            bits = int(value) * (8 if profile.endswith("B") else 1)
            if 1 <= bits < width:
                return bits
    raise WireError(FAIL_MALFORMED, "unknown payload profile")


@dataclass(frozen=True)
class Setup:
    manifest: bytes
    commitment: bytes
    salt: bytes
    params: R.Params
    payload_bits: int
    blocks: tuple[bytes, ...]
    root: bytes
    epoch_id: bytes
    authentication_failed: bool


@dataclass(frozen=True)
class AuditOpen:
    frame: bytes
    epoch_id: bytes
    audit_id: bytes
    mode: int
    k: int
    delta_ns: int
    rtt_ns: int
    verification_flags: int
    audit_counter: int


def _params(m: dict) -> tuple[R.Params, int]:
    if set(m) != {"blocks_per_segment", "c0", "profile", "scheme", "segments", "w"} or m["scheme"] != "p2":
        raise WireError(FAIL_MALFORMED, "wrong manifest fields")
    if any(type(m[name]) is not int for name in ("blocks_per_segment", "c0", "segments", "w")) or type(m["profile"]) is not str:
        raise WireError(FAIL_MALFORMED, "manifest value has wrong type")
    if not 1 <= m["blocks_per_segment"] <= 1 << 16 or not 1 <= m["segments"] < 1 << 64 or not 0 < m["w"] <= MAX_W:
        raise WireError(FAIL_MALFORMED, "manifest geometry is out of range")
    bits = payload_bits(m["profile"], m["w"])
    try:
        return R.Params(m["w"], m["c0"], max(1, bits // 8), m["blocks_per_segment"], m["segments"], m["profile"]), bits
    except ValueError as error:
        raise WireError(FAIL_MALFORMED, f"no P2 prime: {error}") from error


def _parse_setup(frames: Sequence[bytes]) -> tuple[Setup, int]:
    cursor = 0
    _, body = parse_envelope(frames[cursor], SETUP_COMMIT)
    cursor += 1
    if len(body) < 4 + D:
        raise WireError(FAIL_MALFORMED, "truncated SETUP_COMMIT")
    manifest_length = int.from_bytes(body[:4], "big")
    if manifest_length > MAX_MANIFEST:
        raise WireError(FAIL_MALFORMED, "manifest exceeds 512-byte limit")
    if len(body) != 4 + manifest_length + D:
        raise WireError(FAIL_MALFORMED, "SETUP_COMMIT length mismatch")
    manifest_object, manifest = _canonical_json(body[4:4 + manifest_length])
    commitment = body[-D:]
    params, bits = _params(manifest_object)

    if cursor >= len(frames):
        raise WireError(FAIL_TRANSPORT, "missing SETUP_SALT")
    _, body = parse_envelope(frames[cursor], SETUP_SALT)
    cursor += 1
    if len(body) != D + S:
        raise WireError(FAIL_MALFORMED, "SETUP_SALT length mismatch")
    if body[:D] != commitment:
        raise WireError(FAIL_IDENTITY, "SETUP_SALT commitment mismatch")
    salt = body[D:]

    blocks: list[bytes] = []
    n = params.width_bytes
    while cursor < len(frames):
        message_type, body = parse_envelope(frames[cursor])
        if message_type != SETUP_BLOCK:
            break
        cursor += 1
        if len(body) < D + S + 8 + 4:
            raise WireError(FAIL_MALFORMED, "truncated SETUP_BLOCK")
        if body[:D] != commitment or body[D:D + S] != salt:
            raise WireError(FAIL_IDENTITY, "SETUP_BLOCK identity mismatch")
        index = int.from_bytes(body[D + S:D + S + 8], "big")
        length = int.from_bytes(body[D + S + 8:D + S + 12], "big")
        block = body[D + S + 12:]
        if index != len(blocks):
            raise WireError(FAIL_IDENTITY, "SETUP_BLOCK is missing, duplicated, or reordered")
        if length != n or len(block) != length:
            raise WireError(FAIL_MALFORMED, "SETUP_BLOCK length mismatch")
        if int.from_bytes(block, "big") >= params.p:
            raise WireError(FAIL_MALFORMED, "noncanonical setup block")
        blocks.append(block)

    if cursor >= len(frames):
        raise WireError(FAIL_TRANSPORT, "missing SETUP_ROOT")
    _, body = parse_envelope(frames[cursor], SETUP_ROOT)
    cursor += 1
    if len(body) != SETUP_ROOT_BYTES:
        raise WireError(FAIL_MALFORMED, "SETUP_ROOT length mismatch")
    if body[:D] != commitment or body[D:D + S] != salt:
        raise WireError(FAIL_IDENTITY, "SETUP_ROOT identity mismatch")
    declared_count = int.from_bytes(body[D + S:D + S + 8], "big")
    declared_root, mode = body[D + S + 8:D + S + 8 + D], body[-1]
    if mode not in (0, 1):
        raise WireError(FAIL_MALFORMED, "unknown setup mode")
    if declared_count != params.blocks:
        raise WireError(FAIL_IDENTITY, "setup block count disagrees with manifest")
    if mode == 1 and len(blocks) != params.blocks:
        raise WireError(FAIL_IDENTITY, "public setup has missing blocks")
    if mode == 0 and blocks and len(blocks) != params.blocks:
        raise WireError(FAIL_MALFORMED, "partial private setup blocks")
    failed = False
    if blocks:
        bps = params.blocks_per_segment
        failed = R.root([R.leaf(salt, i // bps, i % bps, b) for i, b in enumerate(blocks)]) != declared_root
    epoch = hashlib.sha512(R.EPOCH_TAG + len(manifest).to_bytes(4, "big") + manifest + commitment + salt
                           + params.blocks.to_bytes(8, "big") + declared_root).digest()
    return Setup(manifest, commitment, salt, params, bits, tuple(blocks), declared_root, epoch, failed), cursor


def _open_fields(frame_bytes: bytes, body: bytes) -> AuditOpen:
    o = D + S
    return AuditOpen(frame_bytes, body[:D], body[D:o], body[o], int.from_bytes(body[o + 1:o + 3], "big"),
                     int.from_bytes(body[o + 3:o + 11], "big"), int.from_bytes(body[o + 11:o + 19], "big"), body[o + 19],
                     int.from_bytes(body[o + 20:o + 28], "big"))


def _parse_open(frame_bytes: bytes) -> AuditOpen:
    _, body = parse_envelope(frame_bytes, AUDIT_OPEN)
    if len(body) != OPEN_BYTES:
        raise WireError(FAIL_MALFORMED, "AUDIT_OPEN length mismatch")
    opened = _open_fields(frame_bytes, body)
    if opened.mode not in (0, 1) or not 1 <= opened.k <= MAX_K:
        raise WireError(FAIL_MALFORMED, "invalid audit mode or k")
    if opened.verification_flags & ~0x03:
        raise WireError(FAIL_MALFORMED, "reserved verification flag is set")
    if opened.verification_flags & 0x02 and not opened.verification_flags & 0x01:
        raise WireError(FAIL_MALFORMED, "payload comparison requires decode checking")
    return opened


def _salvage_open(frame_bytes: bytes) -> AuditOpen | None:
    """The FINAL's identity fields, read at their fixed offsets from a fixed-size but invalid AUDIT_OPEN."""
    try:
        _, body = parse_envelope(frame_bytes, AUDIT_OPEN)
    except WireError:
        return None
    return _open_fields(frame_bytes, body) if len(body) == OPEN_BYTES else None


def _recover_open(frames: Sequence[bytes]) -> AuditOpen | None:
    """The valid AUDIT_OPEN identity a setup failure is reported under."""
    for frame_bytes in frames:
        try:
            message_type, _ = parse_envelope(frame_bytes)
        except WireError:
            continue
        if message_type == AUDIT_OPEN:
            try:
                return _parse_open(frame_bytes)
            except WireError:
                return _salvage_open(frame_bytes)
    return None


def _parse_challenge(frame_bytes: bytes) -> tuple[bytes, bytes, int, int, bytes]:
    _, body = parse_envelope(frame_bytes, CHALLENGE)
    if len(body) != CHALLENGE_BYTES:
        raise WireError(FAIL_MALFORMED, "CHALLENGE length mismatch")
    o = D + S
    return body[:D], body[D:o], int.from_bytes(body[o:o + 2], "big"), int.from_bytes(body[o + 2:o + 10], "big"), body[o + 10:]


def _parse_response(frame_bytes: bytes, block_bytes: int) -> tuple[bytes, bytes, int, int, bytes, tuple[bytes, ...]]:
    _, body = parse_envelope(frame_bytes, RESPONSE)
    o = D + S
    if len(body) < RESPONSE_FIXED:
        raise WireError(FAIL_MALFORMED, "truncated RESPONSE")
    declared_length = int.from_bytes(body[o + 10:o + 14], "big")
    if declared_length != block_bytes:
        raise WireError(FAIL_MALFORMED, "wrong block length")
    end = o + 14 + declared_length
    if len(body) < end + 2:
        raise WireError(FAIL_MALFORMED, "truncated block")
    count = int.from_bytes(body[end:end + 2], "big")
    path_bytes = body[end + 2:]
    if len(path_bytes) != D * count:
        raise WireError(FAIL_MALFORMED, "wrong path_count or trailing bytes")
    return (body[:D], body[D:o], int.from_bytes(body[o:o + 2], "big"), int.from_bytes(body[o + 2:o + 10], "big"),
            body[o + 14:end], tuple(path_bytes[x:x + D] for x in range(0, len(path_bytes), D)))


def verify_audit(frames: Sequence[bytes], elapsed_ns: Sequence[int], *, verifier_source: bytes | None = None,
                 beacon_inputs: Mapping[int, bytes] | None = None,
                 seen_audits: MutableSet[tuple[bytes, bytes]] | None = None) -> bytes:
    """Verify a setup and one audit transcript, returning the exact FINAL frame.

    `frames` is SETUP_COMMIT, SETUP_SALT, SETUP_BLOCK*, SETUP_ROOT, AUDIT_OPEN, then alternating CHALLENGE and RESPONSE.
    A missing final response is a transport failure. `elapsed_ns[j]` is the verifier's measured challenge-to-response time
    for round j."""
    if not frames:
        raise WireError(FAIL_TRANSPORT, "empty transcript has no audit identity")
    try:
        setup, cursor = _parse_setup(frames)
    except WireError as error:
        opened = _recover_open(frames)
        if opened is None:
            raise
        return _final(opened, error.code, [opened.frame], [])
    if cursor >= len(frames):
        raise WireError(FAIL_TRANSPORT, "missing AUDIT_OPEN")
    open_frame = frames[cursor]
    try:
        opened = _parse_open(open_frame)
    except WireError as error:
        salvaged = _salvage_open(open_frame)
        if salvaged is None:
            raise
        return _final(salvaged, error.code, [open_frame], [])
    cursor += 1
    transcript: list[bytes] = [opened.frame]
    records: list[bytes] = []

    def stop(code: int) -> bytes:
        return _final(opened, code, transcript, records)

    if setup.authentication_failed:
        return stop(FAIL_AUTHENTICATION)
    if opened.epoch_id != setup.epoch_id:
        return stop(FAIL_IDENTITY)
    pair = (opened.epoch_id, opened.audit_id)
    if seen_audits is not None:
        if pair in seen_audits:
            return stop(FAIL_IDENTITY)
        seen_audits.add(pair)
    if not verifier_source or opened.audit_id != R.audit_id(verifier_source, opened.epoch_id, opened.audit_counter):
        return stop(FAIL_IDENTITY)
    audit_key = None
    if opened.mode == 0:
        audit_key = R.audit_key(verifier_source, opened.epoch_id, setup.root, setup.salt, opened.audit_counter)[1]

    bps, n = setup.params.blocks_per_segment, setup.params.width_bytes
    zero = bytes(D)
    for j in range(opened.k):
        if audit_key is not None:
            expected_index = audit_key.uniform(setup.params.blocks, j)
            expected_nonce = audit_key.stream(("nonce", j), S)
        else:
            beacon = None if beacon_inputs is None else beacon_inputs.get(j)
            if type(beacon) is not bytes or len(beacon) != S:
                return stop(FAIL_IDENTITY)
            expected_index = R.round_key(beacon, opened.audit_id, opened.epoch_id, setup.root, j).uniform(setup.params.blocks, j)
            expected_nonce = beacon

        if cursor >= len(frames):
            return stop(FAIL_TRANSPORT)
        challenge_frame = frames[cursor]
        cursor += 1
        challenge_hash = hashlib.sha512(challenge_frame).digest()
        transcript.append(challenge_frame)
        try:
            epoch, audit, rj, index, nonce = _parse_challenge(challenge_frame)
        except WireError as error:
            records.append(record(j, expected_index, 0, 0, challenge_hash, zero))
            return stop(error.code)
        if (epoch, audit, rj, index, nonce) != (opened.epoch_id, opened.audit_id, j, expected_index, expected_nonce) \
                or index >= setup.params.blocks:
            records.append(record(j, expected_index, 0, 0, challenge_hash, zero))
            return stop(FAIL_IDENTITY)

        elapsed = elapsed_ns[j] if j < len(elapsed_ns) else 0
        if cursor >= len(frames):
            records.append(record(j, index, elapsed, 0, challenge_hash, zero))
            return stop(FAIL_TRANSPORT)
        response_frame = frames[cursor]
        cursor += 1
        response_hash = hashlib.sha512(response_frame).digest()
        transcript.append(response_frame)

        def fail(code: int, flags: int = 0) -> bytes:
            records.append(record(j, index, elapsed, flags, challenge_hash, response_hash))
            return stop(code)

        try:
            r_epoch, r_audit, r_j, r_index, block, path = _parse_response(response_frame, n)
        except WireError as error:
            return fail(error.code)
        if int.from_bytes(block, "big") >= setup.params.p or len(path) != path_length(setup.params.blocks, index):
            return fail(FAIL_MALFORMED)
        if (r_epoch, r_audit, r_j, r_index) != (opened.epoch_id, opened.audit_id, j, index):
            return fail(FAIL_IDENTITY)
        segment, local = divmod(index, bps)
        if not verify_path(setup.root, setup.params.blocks, index, R.leaf(setup.salt, segment, local, block), path):
            return fail(FAIL_AUTHENTICATION)
        flags = 0x01
        if elapsed > opened.delta_ns + opened.rtt_ns:
            return fail(FAIL_LATE, flags)
        flags |= 0x02
        if opened.verification_flags & 0x01:
            flags |= 0x04
            k_value, tweak, _ = R.keys(setup.salt, setup.params, setup.commitment, segment, local)
            if R.decode(setup.params, int.from_bytes(block, "big"), k_value, tweak) >> setup.payload_bits:
                return fail(FAIL_DECODE, flags)
            flags |= 0x08
        if opened.verification_flags & 0x02:
            # comparison with verifier-held payload is setup's, outside this audit-only reference
            return fail(FAIL_DECODE, flags)
        records.append(record(j, index, elapsed, flags, challenge_hash, response_hash))

    return stop(FAIL_IDENTITY if cursor != len(frames) else FAIL_SUCCESS)


def _final(opened: AuditOpen, code: int, transcript: Sequence[bytes], records: Sequence[bytes]) -> bytes:
    return final_message(opened.epoch_id, opened.audit_id, code, opened.delta_ns, opened.rtt_ns, transcript, records)


def parse_final(final: bytes) -> tuple[bool, int, int]:
    """A FINAL's verdict, failure code and record count."""
    _, body = parse_envelope(final, FINAL)
    o = D + S
    if len(body) < FINAL_FIXED or (len(body) - FINAL_FIXED) % RECORD_BYTES:
        raise WireError(FAIL_MALFORMED, "FINAL length mismatch")
    return bool(body[o]), body[o + 1], int.from_bytes(body[o + 2:o + 4], "big")
