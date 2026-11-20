"""Setup with a server-side encoder:

1. the server commits to `W` (`commit`) before any salt exists;
2. the verifier samples the salt after the commitment (`sample_salt`, from its own secret or a later beacon round);
3. the server encodes, `C* = Enc(W, salt)`, and sends it segment by segment.  The verifier builds `vk` one of two ways:
   - **public encoder** (every served path, POUS `docs/public-encoder/protocol.md`, milestone 1): it decodes each
     segment of `C*`, streams `Dec(C*)` into `commit`'s framing, accepts iff the digest equals the commitment, and builds
     `vk` from the labels it decoded (`verifier_key_from_encoding`).  It never runs the encoder and never holds `W`;
   - **private encoder** (`verifier_key`): it encodes `W` itself and builds `vk` from its own encoding.

On a served run setup is checked by the developer, not the auditor (`challenger_session.SETUP`): the developer's tie on
the prover's case (`period.tie`, and `PousCheck.RunProtocol.tie` with `SetupProtocol.check`, in the prover's piece on the
developer's node) decodes the case's `C*` and holds `vk.root` to the root the run published, and no auditor-side entry
compares that root with anything.  The signed run takes setup's acceptance as the open premise `PousVerifier.SignedRun.SETUP_ACCEPTED`.

The encoder is deterministic and public given `W`, the salt and the primitives, so the server running it changes nothing
in the timed game; a server-chosen or predictable salt would break the model's "`W` before coins" order.  Everything
here works one segment at a time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from verity.commitments.identity import canonical_json_bytes
from verity.commitments.randomness import derive

from .codec import Codec, Geometry, encode_segments, split
from .params import SALT_BITS
from .verifier import VerifierKey, segment_leaves

COMMIT_TAG = "verity-pous/w-commitment/v2"
SALT_DOMAIN = "verity/pous/salt/v1"
_TAGGED_SHA512 = b"verity/tagged-sha512/v1\0"


def _commitment_hash(params: Geometry):
    """`commit`'s running hash before the first byte of `W`: the tagged-SHA-512 header, the payload length, then
    `u64(len manifest) || manifest`, all from `params`."""
    manifest = canonical_json_bytes(params.manifest())
    tag = COMMIT_TAG.encode("utf-8")
    return hashlib.sha512(_TAGGED_SHA512 + len(tag).to_bytes(4, "big") + tag
                          + (8 + len(manifest) + params.w_bytes).to_bytes(8, "big")
                          + len(manifest).to_bytes(8, "big") + manifest)


def commit(params: Geometry, w: bytes | Iterable[bytes]) -> bytes:
    """The server's commitment to `W` under `params`: `verity.commitments.identity.tagged_sha512(COMMIT_TAG, u64(len
    manifest) || manifest || W)`, streamed over `W`'s segments."""
    h = _commitment_hash(params)
    for seg in split(params, w):
        h.update(seg)
    return h.digest()


def sample_salt(source: bytes, commitment: bytes) -> bytes:
    """The verifier's 24-byte salt from `source` (its secret, or a beacon round after the commitment), bound to it."""
    return derive(source, SALT_DOMAIN, {"w": commitment}).stream("salt", SALT_BITS // 8)


def verifier_key(params: Geometry, prims, w: bytes | Sequence[bytes], commitment: bytes, salt: bytes,
                 encoded: Iterable[Sequence[bytes]] | None = None, codec: Codec | None = None) -> VerifierKey:
    """`vk` from the verifier's own encoding of the committed `W`: whole, or a sequence of its segments (read twice,
    for the commitment and to encode).  Pass `encoded`, the verifier's own per-segment `encode_segment` results (for
    example from a process pool), to skip encoding here."""
    if commit(params, w) != commitment:
        raise ValueError("W does not match the server's commitment")
    segments = encoded if encoded is not None else encode_segments(codec or params.codec, params, prims, w, salt)
    return VerifierKey.from_segments(params, salt, segments)


def verifier_key_from_encoding(params: Geometry, prims, commitment: bytes, salt: bytes,
                               encoded: Iterable[Sequence[bytes]], codec: Codec | None = None) -> VerifierKey:
    """Milestone 1: decode each segment of the server's `C*`, check `commit(Dec(C*)) == commitment`, and return `vk`
    built from the decoded labels.  Raises `ValueError` on any mismatch (a label's size, a segment's label count, the
    number of segments, the digest); never calls an encoder, and holds one segment at a time."""
    codec = codec or params.codec
    n, mb = params.blocks_per_segment, params.block_bytes
    h = _commitment_hash(params)
    leaves: list[bytes] = []
    count = 0
    for s, segment in enumerate(encoded):
        if s >= params.segments:
            raise ValueError(f"C* has more than the {params.segments} segments the parameters say")
        blocks = [bytes(b) if isinstance(b, (bytes, bytearray, memoryview)) else None for b in segment]
        if len(blocks) != n or any(b is None or len(b) != mb for b in blocks):
            raise ValueError(f"segment {s} of C* is not {n} blocks of {mb} bytes")
        h.update(codec.decode_segment(params, prims, salt, s, blocks))
        leaves += segment_leaves(params, salt, s, blocks)
        count += 1
    if count != params.segments:
        raise ValueError(f"C* has {count} segments, the parameters say {params.segments}")
    if h.digest() != commitment:
        raise ValueError("Dec(C*) does not match the commitment to W")
    return VerifierKey.from_leaves(params, salt, leaves)
