"""Setup with a server-side encoder (`docs/p3-scheme.md` §2c):

1. the server commits to `W` (`commit`) before any salt exists;
2. the verifier draws the salt after the commitment (`draw_salt`, from its own secret or a later beacon round);
3. the server encodes, `C = Enc(W, salt)`, and the verifier computes `vk` from its own `Enc(W, salt)` (`verifier_key`),
   never accepting it from the server.  Comparing its root with the server's (`check_server_root`) catches a wrong
   encoding at setup rather than by audit sampling.

P3's encoder is deterministic and public given `W`, the salt and the primitives, so the server running it changes
nothing in the timed game; a server-chosen or predictable salt would break the model's "`W` before coins" order.
"""

from __future__ import annotations

from verity.commitments.identity import canonical_json_bytes, tagged_sha256
from verity.randomness import derive

from .params import SALT_BITS, Params
from .primitives import Primitives
from .scheme import Encoding, encode
from .verifier import VerifierKey, tree

COMMIT_TAG = "verity-pous/w-commitment/v1"
SALT_DOMAIN = "verity/pous/salt/v1"


def commit(params: Params, w: bytes) -> bytes:
    """The server's commitment to `W` under `params`: SHA-256 over the parameters' manifest, then `W`."""
    manifest = canonical_json_bytes(params.manifest())
    return bytes.fromhex(tagged_sha256(COMMIT_TAG, len(manifest).to_bytes(8, "big") + manifest + w))


def draw_salt(source: bytes, commitment: bytes) -> bytes:
    """The verifier's 24-byte salt from `source` (its secret, or a beacon round after the commitment), bound to it."""
    return derive(source, SALT_DOMAIN, {"w": commitment}).stream("salt", SALT_BITS // 8)


def verifier_key(params: Params, prims: Primitives, w: bytes, commitment: bytes, salt: bytes) -> VerifierKey:
    """`vk` from the verifier's own encoding of the committed `W`."""
    if commit(params, w) != commitment:
        raise ValueError("W does not match the server's commitment")
    return VerifierKey.build(encode(params, prims, w, salt))


def server_root(encoding: Encoding) -> bytes:
    """The root the server reports for its `C`."""
    return tree(encoding).root


def check_server_root(vk: VerifierKey, root: bytes) -> bool:
    return root == vk.root
