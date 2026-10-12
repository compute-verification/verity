"""The bandwidth audit's pinned vectors (`vectors.json`): `python bandwidth_vectors.py` rewrites them, and
`test_bandwidth_vectors.py` holds the code to them.

Blocks are generated, not stored: block `j` of `block_bytes` bytes is the first bytes of
`SHA512("verity/bandwidth/test-block/v1\\0" ‖ u64be j ‖ u32be 0) ‖ SHA512(… ‖ u32be 1) ‖ …`.  A GPU tagger fills its
store with the same blocks and checks its tags and roots against these.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.core.protocols.bandwidth import audit, tags
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.pous.verifier import VerifierKey, leaf_hash

PATH = Path(__file__).with_name("vectors.json")
BLOCK_DOMAIN = b"verity/bandwidth/test-block/v1\0"
SEED = bytes(range(32))
SOURCE = b"verity bandwidth vector source"
SALT = bytes(range(100, 124))
NONCES = (bytes(32), bytes(range(32)), bytes([0xFF]) * 32)
BLOCK_BYTES = 2056


def block(j: int, block_bytes: int = BLOCK_BYTES) -> bytes:
    out = b"".join(hashlib.sha512(BLOCK_DOMAIN + j.to_bytes(8, "big") + c.to_bytes(4, "big")).digest()
                   for c in range(-(-block_bytes // 64)))
    return out[:block_bytes]


def store(blocks: int, block_bytes: int = BLOCK_BYTES) -> audit.Store:
    """`blocks` generated blocks under a one-segment P2 geometry and `SALT`."""
    params = P2Params(8 * block_bytes, 21065 if block_bytes == BLOCK_BYTES else 173, blocks, 1)
    return audit.Store(params, SALT, [block(j, block_bytes) for j in range(blocks)])


def vk_of(s: audit.Store) -> VerifierKey:
    return VerifierKey.from_leaves(s.params, s.salt, [leaf_hash(s.params, s.salt, i, b) for i, b in enumerate(s.blocks)])


def _tag_vectors() -> list[dict]:
    out = []
    for name in sorted(tags.TAGS):
        t = tags.get(name)
        keyed = t.keyed(SEED)
        for j in (0, 1, 7, (1 << 19) - 1, 1 << 40):
            v = {"tag": name, "seed": SEED.hex(), "j": j, "block_bytes": BLOCK_BYTES,
                 "prefix": tags.prefix(tags.TAG_DOMAIN, name, SEED).hex(), "y": keyed(j, block(j)).hex()}
            if isinstance(t, tags.PolynomialTag):
                v["g"] = t.point(SEED, j)
            if isinstance(t, tags.Blake3Tag):
                v["key"] = t.key(SEED).hex()
            out.append(v)
    return out


def _salt_vectors() -> list[dict]:
    commitment = hashlib.sha512(b"verity bandwidth vector commitment").digest()
    return [{"source": SOURCE.hex(), "commitment": commitment.hex(), "device": device,
             "salt": audit.draw_salt(SOURCE, commitment, device).hex()} for device in ("GPU-a", "GPU-b")]


def _window_vectors() -> list[dict]:
    s = store(37)
    vk = vk_of(s)
    out = []
    for tag, nonce in (("ro-blake3/v1", NONCES[0]), ("ro-blake3/v1", NONCES[1]), ("gf31-poly/v1", NONCES[1]),
                       ("ro-sha512/v1", NONCES[2])):
        key = audit.window_key(SOURCE, vk, tag, nonce)
        out.append({"source": SOURCE.hex(), "vk_root": vk.root.hex(), "salt": SALT.hex(), "tag": tag,
                    "nonce": nonce.hex(), "seed": audit.seed(key).hex(),
                    "samples_of_2^19": list(audit.samples(key, 1 << 19, 8)),
                    "samples_of_37": list(audit.samples(key, 37, 8))})
    return out


def _root_vectors() -> list[dict]:
    """Every tag's root over the 37 generated blocks under `SEED`: the GPU tagger's small case."""
    s = store(37)
    return [{"tag": name, "blocks": 37, "seed": SEED.hex(), "tag_root": audit.tag_pass(s, name, SEED).root.hex()}
            for name in sorted(tags.TAGS)]


def _audit_vectors() -> list[dict]:
    """A whole honest window over 37 blocks (not a power of two, and a short last leaf for the polynomial tag)."""
    s = store(37)
    vk = vk_of(s)
    out = []
    for name in sorted(tags.TAGS):
        key = audit.window_key(SOURCE, vk, name, NONCES[1])
        commitment = audit.tag_pass(s, name, audit.seed(key))
        drawn = audit.samples(key, 37, 5)
        out.append({"tag": name, "blocks": 37, "block_bytes": BLOCK_BYTES, "salt": SALT.hex(),
                    "store_root": s.root.hex(), "nonce": NONCES[1].hex(), "samples": 5,
                    "leaves": len(commitment.tree.leaves), "tag_root": commitment.root.hex(), "drawn": list(drawn),
                    "opened_leaf_tags": [[y.hex() for y in commitment.open(i).leaf_tags] for i in drawn[:2]]})
    return out


def generate() -> dict:
    return {"block": "SHA512('verity/bandwidth/test-block/v1\\0' || u64be j || u32be c), c = 0, 1, ..., truncated",
            "tags": _tag_vectors(), "salts": _salt_vectors(), "windows": _window_vectors(),
            "roots": _root_vectors(), "audits": _audit_vectors()}


if __name__ == "__main__":
    PATH.write_text(json.dumps(generate(), indent=1) + "\n")
