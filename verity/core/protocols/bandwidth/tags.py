"""The bandwidth audit's tags: `y_j = Tag(σ, j, E_j)` for every block `j` of the store, and the tree that commits them.

~~~text
Z(x)           x zero-padded to whole 128-byte blocks (one SHA-512 block here, so a device starts from its midstate)
prefix(D, σ)   Z(D ‖ frame(name) ‖ frame(σ)),  D = "verity/bandwidth/tag/v1\0", or "verity/bandwidth/tag-leaf/v1\0" for leaves
ro-blake3/v1   y_j = BLAKE3-keyed(K, u64be j ‖ E_j),  K = SHA512(prefix)[0:32]            32 bytes
ro-sha512/v1   y_j = SHA512(prefix ‖ u64be j ‖ E_j)                                   64 bytes
gf31-poly/v1   y_j = Σ_i (w_i mod p) g_j^(W−1−i) mod p,  p = 2^31 − 1                   4 bytes, u32be
               w_0 … w_(W−1) the block's little-endian u32 words, evaluated by Horner (`benchmarks/pous/hbm_audit/digest.py`)
               g_j = 2 + x, x the first 31-bit value below p − 2 among the u32be words of
                     SHA512(prefix ‖ u64be j ‖ u32be c) shifted right by one, for c = 0, 1, …
leaf_c         SHA512(prefix(leaf) ‖ u64be c ‖ y_(cL) ‖ … ‖ y_(cL+L−1)),  L = `per_leaf` (1 for ro-sha512/v1, else 16);
               the last leaf may be short
root           the RFC 6962 SHA-512 tree over the leaves (`pous.verifier`: node = SHA512(0x01 ‖ l ‖ r))
~~~

`frame` is `verity.ml.randomness.frame`.  Every tag is seed-first, `H(σ, j, E_j)`: σ is in the first SHA-512 block,
or in the BLAKE3 key every compression starts from, before `j` and any byte of `E_j`, so no state over a block can be
computed before σ is known.

`ro-blake3/v1` and `ro-sha512/v1` are certified: modeled as a random oracle at the fresh seed, they are the tag of
`Bandwidth.Guarantees.AuditRO`, and a tag is right without `E_j` with probability `2^-h`.  `ro-blake3/v1` is the one
that fits the deadline on the RTX PRO 6000.  `gf31-poly/v1` (`hbm_audit`'s polynomial hash with a fresh point per
block) is an uncertified fast check: a polynomial evaluated at a point is not unpredictable per block, and a device
with spare memory can keep a structure per block that evaluates it at any point without reading the block's words
(Kedlaya–Umans), so no ingress bound holds for it.  Two blocks whose words differ mod p collide with probability at
most (W − 1) / (p − 2) over `g_j`, and its word map binds only the words mod p, `⌈W·log2 p⌉` bits (`info_bits`).
"""

from __future__ import annotations

import hashlib
import itertools
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from verity.ml.crypto import blake3
from verity.ml.randomness import frame

TAG_DOMAIN = b"verity/bandwidth/tag/v1\0"
LEAF_DOMAIN = b"verity/bandwidth/tag-leaf/v1\0"
SEED_BYTES = 32
DIGEST_BYTES = 64
MERSENNE31 = (1 << 31) - 1


def prefix(domain: bytes, name: str, seed: bytes) -> bytes:
    """`Z(domain ‖ frame(name) ‖ frame(seed))`."""
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a window's seed is {SEED_BYTES} bytes")
    body = domain + frame(name) + frame(seed)
    return body + bytes(-len(body) % 128)


class Keyed(Protocol):
    """A tag construction under one window's seed: `keyed(j, block)` is `y_j`."""

    def __call__(self, j: int, block: bytes) -> bytes: ...


class Tag(Protocol):
    name: str
    tag_bytes: int
    per_leaf: int
    certified: bool

    def keyed(self, seed: bytes) -> Keyed: ...

    def check_block(self, block_bytes: int) -> None:
        """Raise `ValueError` if the construction is not defined on blocks of `block_bytes` bytes."""
        ...

    def guess(self, block_bytes: int) -> Fraction:
        """γ: the most a tag computed without the block (or from a block that differs where the tag binds it) is
        right with, over the window's seed."""
        ...

    def info_bits(self, block_bytes: int) -> int:
        """The bits of a block the tag binds: what a prover must hold of `E_j` to compute `y_j` for a fresh seed."""
        ...


def _index(j: int) -> bytes:
    if type(j) is not int or not 0 <= j < 1 << 64:
        raise ValueError("a block index is a u64")
    return j.to_bytes(8, "big")


@dataclass(frozen=True, slots=True)
class Blake3Tag:
    name: str = "ro-blake3/v1"
    tag_bytes: int = 32
    per_leaf: int = 16
    certified: bool = True

    def key(self, seed: bytes) -> bytes:
        """`K`, the window's BLAKE3 key."""
        return hashlib.sha512(prefix(TAG_DOMAIN, self.name, seed)).digest()[:blake3.KEY_LEN]

    def keyed(self, seed: bytes) -> Keyed:
        key = self.key(seed)

        def tag(j: int, block: bytes) -> bytes:
            return blake3.keyed(key, _index(j) + block)

        return tag

    def check_block(self, block_bytes: int) -> None:
        if block_bytes < 1:
            raise ValueError("a block has at least one byte")

    def guess(self, block_bytes: int) -> Fraction:
        return Fraction(1, 1 << 256)

    def info_bits(self, block_bytes: int) -> int:
        return 8 * block_bytes


@dataclass(frozen=True, slots=True)
class RandomOracleTag:
    name: str = "ro-sha512/v1"
    tag_bytes: int = 64
    per_leaf: int = 1
    certified: bool = True

    def keyed(self, seed: bytes) -> Keyed:
        head = hashlib.sha512(prefix(TAG_DOMAIN, self.name, seed))

        def tag(j: int, block: bytes) -> bytes:
            h = head.copy()
            h.update(_index(j))
            h.update(block)
            return h.digest()

        return tag

    def check_block(self, block_bytes: int) -> None:
        if block_bytes < 1:
            raise ValueError("a block has at least one byte")

    def guess(self, block_bytes: int) -> Fraction:
        return Fraction(1, 1 << 512)

    def info_bits(self, block_bytes: int) -> int:
        return 8 * block_bytes


@dataclass(frozen=True, slots=True)
class PolynomialTag:
    name: str = "gf31-poly/v1"
    tag_bytes: int = 4
    per_leaf: int = 16
    certified: bool = False
    p: int = MERSENNE31

    def point(self, seed: bytes, j: int) -> int:
        """`g_j`, uniform on [2, p − 1]."""
        return _point(hashlib.sha512(prefix(TAG_DOMAIN, self.name, seed)), self.p, j)

    def keyed(self, seed: bytes) -> Keyed:
        head = hashlib.sha512(prefix(TAG_DOMAIN, self.name, seed))
        p = self.p

        def tag(j: int, block: bytes) -> bytes:
            return horner(block, _point(head, p, j), p).to_bytes(4, "big")

        return tag

    def check_block(self, block_bytes: int) -> None:
        if block_bytes < 4 or block_bytes % 4:
            raise ValueError("gf31-poly/v1 reads whole u32 words: a block is a positive multiple of 4 bytes")

    def guess(self, block_bytes: int) -> Fraction:
        self.check_block(block_bytes)
        return Fraction(block_bytes // 4 - 1, self.p - 2)

    def info_bits(self, block_bytes: int) -> int:
        self.check_block(block_bytes)
        return (self.p ** (block_bytes // 4) - 1).bit_length()


def _point(head, p: int, j: int) -> int:
    base = _index(j)
    for c in itertools.count():
        h = head.copy()
        h.update(base + c.to_bytes(4, "big"))
        d = h.digest()
        for k in range(0, DIGEST_BYTES, 4):
            x = int.from_bytes(d[k:k + 4], "big") >> 1
            if x < p - 2:
                return 2 + x
    raise AssertionError("unreachable")


def horner(block: bytes, g: int, p: int = MERSENNE31) -> int:
    """`Σ_i (w_i mod p) g^(W−1−i) mod p` over the block's little-endian u32 words."""
    if len(block) % 4:
        raise ValueError("a block is whole u32 words")
    acc = 0
    for (w,) in struct.iter_unpack("<I", block):
        acc = (acc * g + w) % p
    return acc


TAGS: dict[str, Tag] = {t.name: t for t in (Blake3Tag(), RandomOracleTag(), PolynomialTag())}


def get(name: str) -> Tag:
    try:
        return TAGS[name]
    except KeyError:
        raise ValueError(f"unknown tag {name!r}; known: {sorted(TAGS)}") from None


def tags_per_leaf(tag: Tag) -> int:
    return tag.per_leaf


def leaf_count(tag: Tag, blocks: int) -> int:
    per = tags_per_leaf(tag)
    return -(-blocks // per)


def leaf_head(tag: Tag, seed: bytes):
    """SHA-512 after the leaves' prefix, to copy for each leaf."""
    return hashlib.sha512(prefix(LEAF_DOMAIN, tag.name, seed))


def leaf_hash(head, c: int, ys: Sequence[bytes]) -> bytes:
    h = head.copy()
    h.update(_index(c))
    for y in ys:
        h.update(y)
    return h.digest()


def leaves(tag: Tag, seed: bytes, ys: Sequence[bytes]) -> list[bytes]:
    """The tag tree's leaves over every `y_j`, in block order."""
    head, per = leaf_head(tag, seed), tags_per_leaf(tag)
    return [leaf_hash(head, c, ys[c * per:(c + 1) * per]) for c in range(leaf_count(tag, len(ys)))]
