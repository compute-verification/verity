"""Setup with a server-side encoder:

1. the server commits to `W` (`commit`) before any salt exists;
2. the verifier samples the salt after the commitment (`draw_salt`, from its own secret or a later beacon round);
3. the server encodes, `C* = Enc(W, salt)`, and sends it segment by segment.  The verifier builds `vk` one of two ways:
   - **public encoder** (the default, POUS `docs/public-encoder/protocol.md`, milestone 1): it decodes each segment of
     `C*`, streams `Dec(C*)` into `commit`'s framing, accepts iff the digest equals the commitment, and builds `vk`
     from the labels it decoded (`verifier_key_from_encoding`).  It never runs the encoder and never holds `W`;
   - **private encoder** (`public_encoder=False`): it encodes `W` itself and builds `vk` from its own encoding
     (`verifier_key`); comparing roots (`check_server_root`) catches a wrong server encoding at setup.

`verifier_setup` picks between them.  The encoder is deterministic and public given `W`, the salt and the primitives,
so the server running it changes nothing in the timed game; a server-chosen or predictable salt would break the
model's "`W` before coins" order.  Everything here works one segment at a time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from verity.ml.commitments.identity import canonical_json_bytes
from verity.ml.randomness import derive

from .codec import Codec, Geometry, encode_segments, split
from .params import SALT_BITS
from .scheme import Encoding
from .verifier import VerifierKey, segment_leaves, tree

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
    """The server's commitment to `W` under `params`: `verity.ml.commitments.identity.tagged_sha512(COMMIT_TAG, u64(len
    manifest) || manifest || W)`, streamed over `W`'s segments."""
    h = _commitment_hash(params)
    for seg in split(params, w):
        h.update(seg)
    return h.digest()


def draw_salt(source: bytes, commitment: bytes) -> bytes:
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


PUBLIC_ENCODER = True
"""The default setup: the verifier decodes the server's `C*` (milestone 1).  `False` restores the private-encoder path,
in which the verifier encodes `W` itself."""


def verifier_setup(params: Geometry, prims, commitment: bytes, salt: bytes, *,
                   encoded: Iterable[Sequence[bytes]] | None = None, w: bytes | Sequence[bytes] | None = None,
                   public_encoder: bool | None = None, codec: Codec | None = None) -> VerifierKey:
    """Setup step 3 on the verifier's side: `verifier_key_from_encoding` of the server's `encoded` segments, or with
    `public_encoder=False` (or `PUBLIC_ENCODER = False`) `verifier_key` of `w`, which the verifier then encodes."""
    public = PUBLIC_ENCODER if public_encoder is None else public_encoder
    if public:
        if encoded is None:
            raise ValueError("the public-encoder path needs the server's C*")
        return verifier_key_from_encoding(params, prims, commitment, salt, encoded, codec)
    if w is None:
        raise ValueError("the private-encoder path needs W")
    return verifier_key(params, prims, w, commitment, salt, codec=codec)


def server_root(encoding: Encoding) -> bytes:
    """The root the server reports for its `C`."""
    return tree(encoding).root


def check_server_root(vk: VerifierKey, root: bytes) -> bool:
    return root == vk.root
