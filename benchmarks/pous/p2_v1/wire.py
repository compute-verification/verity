"""`p2-audit-wire/v4`, the P2 audit's byte-exact frames (spec §4, "Audit protocol and wire formats"): the builders the
vectors (`audit-vectors.json`) are made with, and the RFC 9162 path functions the difftest holds `PousCheck.Merkle` to.

Every digest is SHA-512's whole 64 bytes, so the commitment, root, epoch id, sibling, transcript hash and the frame hashes
in a record are 64 bytes; the salt, the audit id and the nonce are 32 bytes. The timed rounds' verifier is the memory
challenger's (`verity.core.protocols.pous.memory_challenger`), which reads no frame of this format.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from . import reference as R

MAGIC = b"P2AU"
WIRE_VERSION = 4
TRANSCRIPT_TAG = b"verity/pous/p2/transcript/v4\0"
D, S = R.DIGEST_BYTES, R.SALT_BYTES

SETUP_COMMIT, SETUP_SALT, SETUP_BLOCK, SETUP_ROOT = 0x01, 0x02, 0x03, 0x04
AUDIT_OPEN, CHALLENGE, RESPONSE, FINAL = 0x10, 0x11, 0x12, 0x13

FAIL_SUCCESS, FAIL_MALFORMED, FAIL_IDENTITY, FAIL_AUTHENTICATION = 0, 1, 2, 3
FAIL_LATE, FAIL_DECODE, FAIL_TRANSPORT = 4, 5, 6

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
