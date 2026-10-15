"""The verifier key `vk` (`docs/p3-instantiation.md` §4): the root of a SHA-512 Merkle tree over the top labels.

~~~text
row(c)  = SHA512(P(8·|c|) || c)                      sha512/row/v2, P = ("verity/sha512-row/v2\0" || u64be bits) to 128 B
leaf(i) = SHA512(0x00 || tag_s || u16(v) || row(c²_v))   block i = s * n + v, tag_s = salt || u64(s), little-endian
node    = SHA512(0x01 || left || right)
~~~

The tree has the shape of RFC 6962 (Certificate Transparency): over `k` leaves it splits at the largest power of two
below `k`, and an audit path lists the siblings from the leaf up, verified as in RFC 9162 §2.1.3.2.  Binding is SHA-512's
collision resistance, a single-stage property.  A leaf wraps the block's `sha512/row/v2` digest, the digest a
row-digest gadget checks, in one more compression (99 bytes) that binds it to its salt, segment and index.  Lean's audit
compares each answer with the true block (`Model/Audit.lean`: "no verifier key ... idealizes the per-block tags"); these
digests are that idealization's instance.

The verifier builds `vk` from its own encoding (`setup.verifier_key`) and keeps every leaf digest (0.1% of `C` at
64 KB), so it checks an answer alone; `root` is the 64-byte key a root-only memory challenger keeps, checking an answer with the
server's audit path.  `VerifierKey.from_segments` and `from_leaves` build it one segment at a time, without `C`.  Only
the geometry (`codec.Geometry`) is read, so the same `vk` serves any codec.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

from .codec import Geometry
from .labels import salt_value, tag
from .scheme import Encoding


DIGEST_BYTES = 64


def row_digest(block_bytes: int, block: bytes | Iterable[bytes]) -> bytes:
    """`sha512_row_v2_digest(block, 8 * block_bytes)`; `block` is the block or its bytes in chunks, in order."""
    h = hashlib.sha512(sha512_row_v2_prefix(8 * block_bytes))
    count = 0
    for chunk in (block,) if isinstance(block, (bytes, bytearray, memoryview)) else block:
        h.update(chunk)
        count += len(chunk)
    if count != block_bytes:
        raise ValueError(f"a block is {block_bytes} bytes, not {count}")
    return h.digest()


def row_leaf(params: Geometry, salt: bytes, i: int, row: bytes) -> bytes:
    """Leaf `i`'s digest from block `i`'s `sha512/row/v2` digest `row`."""
    s, v = divmod(i, params.blocks_per_segment)
    return hashlib.sha512(b"\x00" + tag(salt_value(salt), s).to_bytes(32, "little") + v.to_bytes(2, "little") + row).digest()


def leaf_hash(params: Geometry, salt: bytes, i: int, block: bytes | Iterable[bytes]) -> bytes:
    """Leaf `i`'s digest; `block` is the block or its bytes in chunks, in order."""
    return row_leaf(params, salt, i, row_digest(params.block_bytes, block))


def segment_leaves(params: Geometry, salt: bytes, s: int, blocks: Sequence[bytes]) -> list[bytes]:
    """The leaf digests of segment `s`'s blocks."""
    n = params.blocks_per_segment
    if len(blocks) != n:
        raise ValueError(f"segment {s} has {n} blocks")
    return [leaf_hash(params, salt, s * n + v, b) for v, b in enumerate(blocks)]


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha512(b"\x01" + left + right).digest()


def _split(k: int) -> int:
    return 1 << ((k - 1).bit_length() - 1)


class MerkleTree:
    """The server's tree over the leaf digests, to answer with audit paths."""

    def __init__(self, leaves: Sequence[bytes]) -> None:
        if not leaves:
            raise ValueError("a codeword has at least one block")
        self.leaves = tuple(leaves)
        self._nodes: dict[tuple[int, int], bytes] = {}
        self.root = self._hash(0, len(self.leaves))

    def _hash(self, lo: int, hi: int) -> bytes:
        if hi - lo == 1:
            return self.leaves[lo]
        key = (lo, hi)
        if key not in self._nodes:
            mid = lo + _split(hi - lo)
            self._nodes[key] = node_hash(self._hash(lo, mid), self._hash(mid, hi))
        return self._nodes[key]

    def path(self, i: int) -> tuple[bytes, ...]:
        """RFC 6962 `PATH(i, D)`: the siblings from the leaf up."""
        out = []
        lo, hi = 0, len(self.leaves)
        while hi - lo > 1:
            mid = lo + _split(hi - lo)
            if i < mid:
                out.append(self._hash(mid, hi))
                hi = mid
            else:
                out.append(self._hash(lo, mid))
                lo = mid
        return tuple(reversed(out))


def verify_path(root: bytes, count: int, i: int, leaf: bytes, path: Sequence[bytes]) -> bool:
    """RFC 9162 §2.1.3.2: `leaf` is leaf `i` of the `count`-leaf tree with `root`."""
    if not 0 <= i < count:
        return False
    fn, sn, r = i, count - 1, leaf
    for p in path:
        if sn == 0 or len(p) != DIGEST_BYTES:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def leaves(encoding: Encoding) -> tuple[bytes, ...]:
    params = encoding.params
    return tuple(leaf_hash(params, encoding.salt, i, encoding.block(i)) for i in range(params.blocks))


def tree(encoding: Encoding) -> MerkleTree:
    return MerkleTree(leaves(encoding))


@dataclass(frozen=True, slots=True)
class VerifierKey:
    """`vk`: the root, and the leaf digests a verifier that computed its own encoding keeps."""

    params: Geometry
    salt: bytes
    root: bytes
    leaves: tuple[bytes, ...]

    @classmethod
    def build(cls, encoding: Encoding) -> VerifierKey:
        t = tree(encoding)
        return cls(encoding.params, encoding.salt, t.root, t.leaves)

    @classmethod
    def from_leaves(cls, params: Geometry, salt: bytes, leaves: Sequence[bytes]) -> VerifierKey:
        """`vk` from every leaf digest, in block order; the root is recomputed."""
        leaves = tuple(leaves)
        if len(leaves) != params.blocks or any(type(x) is not bytes or len(x) != DIGEST_BYTES for x in leaves):
            raise ValueError(f"expected {params.blocks} {DIGEST_BYTES}-byte leaf digests")
        return cls(params, salt, MerkleTree(leaves).root, leaves)

    @classmethod
    def from_row_digests(cls, params: Geometry, salt: bytes, rows: Sequence[bytes]) -> VerifierKey:
        """`vk` from every block's `sha512/row/v2` digest, in block order: the memory challenger's table of `C*`."""
        if len(rows) != params.blocks or any(type(r) is not bytes or len(r) != DIGEST_BYTES for r in rows):
            raise ValueError(f"expected {params.blocks} {DIGEST_BYTES}-byte row digests")
        return cls.from_leaves(params, salt, [row_leaf(params, salt, i, r) for i, r in enumerate(rows)])

    @classmethod
    def from_segments(cls, params: Geometry, salt: bytes, segments: Iterable[Sequence[bytes]]) -> VerifierKey:
        """`vk` from each segment's blocks in order (a codec's `encode_segment` results), one segment at a time."""
        leaves: list[bytes] = []
        for s, blocks in enumerate(segments):
            leaves += segment_leaves(params, salt, s, blocks)
        return cls.from_leaves(params, salt, leaves)

    def check(self, i: int, block: bytes) -> bool:
        """The answer to challenge `i` is block `i`."""
        return (type(block) is bytes and len(block) == self.params.block_bytes
                and leaf_hash(self.params, self.salt, i, block) == self.leaves[i])

    def check_path(self, i: int, block: bytes, path: Sequence[bytes]) -> bool:
        """Root-only check: `block` with the server's audit `path` is block `i` under `root`."""
        return (type(block) is bytes and len(block) == self.params.block_bytes
                and verify_path(self.root, self.params.blocks, i, leaf_hash(self.params, self.salt, i, block), path))
