"""The receipts of one pass, the seed chain that orders passes, and the trees that commit them.

~~~text
Z(x)          x zero-padded to whole 128-byte blocks (one SHA-512 block here)
seed_(t,i)    SHA512(SEED_DOMAIN ‖ σ_t ‖ u32be i ‖ root_(t,i−1)),  SEED_DOMAIN = "verity/remote-bandwidth/seed/v0\0"
              root_(t,−1) = root_(t−1,last) in a session, the registration's digest for its first window
spread(z)     z laid over b = 4096 bytes: byte u of z XORed into byte u mod b (a z of b bytes or more folds), a shorter
              z repeated
y_(i,j)       SHA512(Z(RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be j ‖ (E_j ⊕ spread(z_(i,j)))),
              RECEIPT_DOMAIN = "verity/remote-bandwidth/receipt/v0\0"                                     64 bytes
leaf c        y_(16c) ‖ … ‖ y_(16c+15), the last leaf short
root_(t,i)    frame-v3-sha512 over the leaves: CommitmentDomain(binding = seed_i, owner 0, positions 0 … ⌈N/16⌉ − 1),
              schema RECEIPTS_SCHEMA
~~~

`frame` is `verity.ml.randomness.frame`.  A receipt is seed-first: the seed fills the first SHA-512 block, so no state
over a block's bytes exists before the seed, and the seed exists only after the previous pass's root.  `z_(i,j)` is
PoUW's noised output of the tile before block j in pass i; it enters elementwise, before any compression, so a relay
that meets the block and z over a link carries `min(|z|, b)` bytes per block, not a digest of one side.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain, VerificationLimits
from verity.ml.commitments import verify_opening
from verity.ml.randomness import frame

from .codec import BLOCK_BYTES

SEED_DOMAIN = b"verity/remote-bandwidth/seed/v0\0"
RECEIPT_DOMAIN = b"verity/remote-bandwidth/receipt/v0\0"
RECEIPTS_SCHEMA = "verity/remote-bandwidth/receipts/v0"
CHALLENGE_BYTES = 32
DIGEST_BYTES = 64
PER_LEAF = 16
OWNER = 0
LIMITS = VerificationLimits()


def _digest(value: bytes, name: str) -> bytes:
    if type(value) is not bytes or len(value) != DIGEST_BYTES:
        raise ValueError(f"{name} is {DIGEST_BYTES} bytes")
    return value


def seed(challenge: bytes, i: int, previous: bytes) -> bytes:
    """`seed_(t,i)`: pass `i`'s seed under the window's challenge, after the root before it."""
    if type(challenge) is not bytes or len(challenge) != CHALLENGE_BYTES:
        raise ValueError(f"a window's challenge is {CHALLENGE_BYTES} bytes")
    if type(i) is not int or not 0 <= i < 1 << 32:
        raise ValueError("a pass index is a u32")
    return hashlib.sha512(SEED_DOMAIN + challenge + i.to_bytes(4, "big") + _digest(previous, "a root")).digest()


def prefix(seed: bytes) -> bytes:
    """`Z(RECEIPT_DOMAIN ‖ frame(seed))`, one SHA-512 block."""
    body = RECEIPT_DOMAIN + frame(_digest(seed, "a pass seed"))
    return body + bytes(-len(body) % 128)


def spread(z: bytes, b: int = BLOCK_BYTES) -> bytes:
    """`z` laid over `b` bytes: folded by XOR when it is `b` bytes or longer, repeated when it is shorter."""
    if type(z) is not bytes or not z:
        raise ValueError("a tile's output is nonempty bytes")
    if len(z) < b:
        return (z * -(-b // len(z)))[:b]
    out = int.from_bytes(z[:b], "little")
    for u in range(b, len(z), b):
        out ^= int.from_bytes(z[u:u + b], "little")
    return out.to_bytes(b, "little")


def _xor(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(len(a), "little")


class Receipter:
    """Receipts under one pass's seed: `receipter(j, block, z)` is `y_(i,j)`."""

    def __init__(self, seed: bytes) -> None:
        self.head = hashlib.sha512(prefix(seed))

    def __call__(self, j: int, block: bytes, z: bytes) -> bytes:
        if type(j) is not int or not 0 <= j < 1 << 64:
            raise ValueError("a block index is a u64")
        if type(block) is not bytes or len(block) != BLOCK_BYTES:
            raise ValueError(f"a block is {BLOCK_BYTES} bytes")
        h = self.head.copy()
        h.update(j.to_bytes(8, "big"))
        h.update(_xor(block, spread(z)))
        return h.digest()


def leaves(blocks: int) -> int:
    return math.ceil(blocks / PER_LEAF)


def domain(seed: bytes, blocks: int) -> CommitmentDomain:
    """The domain of pass `seed`'s receipt tree over a store of `blocks` blocks."""
    return CommitmentDomain(_digest(seed, "a pass seed"), OWNER, RangeIndexedDomain(0, leaves(blocks)), hash="sha512")


def tree(seed: bytes, ys: Sequence[bytes]) -> MerkleTree:
    """The receipt tree of one pass: `ys` in block order, sixteen to a leaf."""
    values = {c: b"".join(ys[PER_LEAF * c:PER_LEAF * (c + 1)]) for c in range(leaves(len(ys)))}
    return MerkleTree(domain(seed, len(ys)), values, lambda _c: RECEIPTS_SCHEMA)


def leaf_receipt(leaf: bytes, j: int, blocks: int) -> bytes | None:
    """Block `j`'s receipt inside its leaf's value, or None if the value is not that leaf's size."""
    c = j // PER_LEAF
    if len(leaf) != DIGEST_BYTES * min(PER_LEAF, blocks - PER_LEAF * c):
        return None
    k = j - PER_LEAF * c
    return leaf[DIGEST_BYTES * k:DIGEST_BYTES * (k + 1)]


def verify_leaf(seed: bytes, blocks: int, root: bytes, j: int, leaf: bytes, path: tuple[bytes, ...]) -> bool:
    """Whether `leaf`, with `path`, is block `j`'s leaf in the tree whose root is `root`."""
    d = domain(seed, blocks)
    return verify_opening(d, Commitment(root, d.count), Opening(j // PER_LEAF, leaf, path), RECEIPTS_SCHEMA, LIMITS)
