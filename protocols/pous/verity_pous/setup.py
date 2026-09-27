"""Setup with a server-side encoder (`docs/p3-scheme.md` §2c):

1. the server commits to `W` (`commit`) before any salt exists;
2. the verifier draws the salt after the commitment (`draw_salt`, from its own secret or a later beacon round);
3. the server encodes, `C = Enc(W, salt)`, and the verifier computes `vk` from its own `Enc(W, salt)` (`verifier_key`),
   never accepting it from the server.  Comparing its root with the server's (`check_server_root`) catches a wrong
   encoding at setup rather than by audit sampling.

P3's encoder is deterministic and public given `W`, the salt and the primitives, so the server running it changes
nothing in the timed game; a server-chosen or predictable salt would break the model's "`W` before coins" order.
Everything here takes `W` whole or as its segments, and works one segment at a time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from verity.commitments.identity import canonical_json_bytes
from verity.randomness import derive

from .codec import Codec, Geometry, encode_segments, split
from .params import SALT_BITS
from .scheme import Encoding
from .verifier import VerifierKey, tree

COMMIT_TAG = "verity-pous/w-commitment/v1"
SALT_DOMAIN = "verity/pous/salt/v1"
_TAGGED_SHA256 = b"veritor/tagged-sha256/v1\0"


def commit(params: Geometry, w: bytes | Iterable[bytes]) -> bytes:
    """The server's commitment to `W` under `params`: `verity.commitments.identity.tagged_sha256(COMMIT_TAG, u64(len
    manifest) || manifest || W)`, streamed over `W`'s segments."""
    manifest = canonical_json_bytes(params.manifest())
    tag = COMMIT_TAG.encode("utf-8")
    h = hashlib.sha256(_TAGGED_SHA256 + len(tag).to_bytes(4, "big") + tag
                       + (8 + len(manifest) + params.w_bytes).to_bytes(8, "big")
                       + len(manifest).to_bytes(8, "big") + manifest)
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


def server_root(encoding: Encoding) -> bytes:
    """The root the server reports for its `C`."""
    return tree(encoding).root


def check_server_root(vk: VerifierKey, root: bytes) -> bool:
    return root == vk.root
