"""The verifier key `vk` (`docs/p3-instantiation.md` §4): the root of a SHA-256 Merkle tree over the top labels.

~~~text
leaf(i) = SHA256(0x00 || tag_s || u16(v) || c²_v)     block i = s * n + v, tag_s = salt || u64(s)
node    = SHA256(0x01 || left || right)
~~~

The tree has the shape of RFC 6962 (Certificate Transparency): over `k` leaves it splits at the largest power of two
below `k`, and an audit path lists the siblings from the leaf up, verified as in RFC 9162 §2.1.3.2.  Binding is SHA-256's
collision resistance, a single-stage property.  Lean's audit compares each answer with the true block (`Model/Audit.lean`:
"no verifier key ... idealizes the per-block tags"); these digests are that idealization's instance.

The verifier builds `vk` from its own encoding (`setup.verifier_key`) and keeps every leaf digest (0.05% of `C` at
64 KB), so it checks an answer alone; `root` is the 32-byte key a root-only verifier keeps, checking an answer with the
server's audit path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from .labels import salt_value, tag
from .params import Params
from .scheme import Encoding


def leaf_hash(params: Params, salt: bytes, i: int, block: bytes) -> bytes:
    s, v = divmod(i, params.n)
    t = tag(salt_value(salt), s).to_bytes(32, "little")
    return hashlib.sha256(b"\x00" + t + v.to_bytes(2, "little") + block).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


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
        if sn == 0:
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

    params: Params
    salt: bytes
    root: bytes
    leaves: tuple[bytes, ...]

    @classmethod
    def build(cls, encoding: Encoding) -> VerifierKey:
        t = tree(encoding)
        return cls(encoding.params, encoding.salt, t.root, t.leaves)

    def check(self, i: int, block: bytes) -> bool:
        """The answer to challenge `i` is block `i`."""
        return (type(block) is bytes and len(block) == self.params.m // 8
                and leaf_hash(self.params, self.salt, i, block) == self.leaves[i])

    def check_path(self, i: int, block: bytes, path: Sequence[bytes]) -> bool:
        """Root-only check: `block` with the server's audit `path` is block `i` under `root`."""
        return (type(block) is bytes and len(block) == self.params.m // 8
                and verify_path(self.root, self.params.blocks, i, leaf_hash(self.params, self.salt, i, block), path))
