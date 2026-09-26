"""``hm96-sha256/v1``: Halevi–Micali hiding leaves on SHA-256, a leaf rule that wraps any SHA-256 leaf digest.

Spec: ``PROTOCOL.md`` beside this file (the construction, the hiding bound, binding, what a circuit proves and what it
costs).  Conformance vectors: ``vectors.json`` beside it, written by ``python -m verity.commitments.hm96.vectors``.  This
module is the plain reference, stdlib only; fast committers live with their producers and match its bytes.

A leaf commits a value under an *inner* layout, any :class:`~verity.commitments.scheme.LeafLayout` with hash SHA-256: the
unsalted leaf digest the port already defines (a vllm-v1 position leaf, a ``sha256/row/v1`` row digest)::

    x    = SHA-256(inner.prefix || value || inner.suffix)       the unsalted digest; private
    y    = 128 fresh bytes from the OS generator (the salt)      private; stored with the value
    c    = SHA-256(SALT_PREFIX || y)
    b    = x XOR (M_key * y)                                     M_key: the 256 x 1024 Hankel matrix of the key over GF(2)
    leaf = SHA-256(leaf_prefix(key) || b || c)

``b || c`` is the Halevi–Micali commit string: their first scheme with the affine universal family ``y -> M y + b``,
applied to the digest ``x`` (their O(k) variant).  A backend proves ``b || c`` from the private value and salt and
publishes it; the verifier computes the leaf natively.  In a circuit only ``c`` costs AND gates beyond the unsalted
digest: ``M_key * y`` is XOR-only, and the key is a circuit constant or a public input, never a witness.  Binding rests on
SHA-256's collision resistance alone, for any key.  Hiding is statistical given uniform salts: at most ``N * 2^-256`` for N
leaves under a uniform key, and under the pinned :data:`DEFAULT_KEY` ``N * 2^-192`` except for a ``2^-64`` fraction of keys,
a common-reference-string step (:func:`statistical_distance_log2`; claims ``statistical-hiding``, ``common-reference-string``).
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

from verity.errors import InvalidArtifact

from ..identity import identity_digest
from ..scheme import LeafLayout, Path
from ..vllm_v1 import VllmV1

__all__ = [
    "COLS",
    "DEFAULT_KEY",
    "DIGEST_BYTES",
    "HidingLayout",
    "Hm96Sha256",
    "KEY_BITS",
    "KEY_BYTES",
    "KEY_LABEL",
    "LEAF_TAG",
    "NAME",
    "PROVEN_BYTES",
    "ROWS",
    "SALT_BYTES",
    "SALT_PREFIX",
    "SALT_TAG",
    "byte_tables",
    "check_key",
    "columns",
    "commit_string",
    "fresh_key",
    "fresh_salt",
    "key_from_label",
    "leaf_prefix",
    "mask",
    "rows",
    "salt_digest",
    "scheme_digest",
    "statistical_distance_log2",
    "tree_leaf",
    "xor_count",
]

NAME = "hm96-sha256/v1"

DIGEST_BYTES = 32
ROWS = 8 * DIGEST_BYTES              # L: the bits of x and b
SALT_BYTES = 128
COLS = 8 * SALT_BYTES                # l: the bits of y
KEY_BITS = ROWS + COLS - 1           # 1279 bits define M
KEY_BYTES = -(-KEY_BITS // 8)        # 160; bit 1279 (the top bit of byte 159) is zero
PROVEN_BYTES = 2 * DIGEST_BYTES      # b || c

SALT_TAG = b"verity/hm96-sha256/salt/v1\0"
LEAF_TAG = b"verity/hm96-sha256/leaf/v1\0"
KEY_LABEL = b"verity/hm96-sha256/key/v1\0"
SALT_PREFIX = SALT_TAG.ljust(64, b"\0")
SCHEME_DIGEST_TAG = "verity/hm96-sha256/scheme/v1"


def _bit(data: bytes, t: int) -> int:
    return (data[t >> 3] >> (t & 7)) & 1


def check_key(key: bytes) -> bytes:
    """``key`` as bytes, refused unless it is 160 bytes with bit 1279 zero (one encoding per matrix)."""

    key = bytes(key)
    if len(key) != KEY_BYTES or _bit(key, KEY_BITS):
        raise InvalidArtifact(f"an {NAME} key is {KEY_BYTES} bytes with bit {KEY_BITS} zero")
    return key


def _clear_top(raw: bytes) -> bytes:
    out = bytearray(raw[:KEY_BYTES])
    out[-1] &= 0x7F
    return bytes(out)


def key_from_label(label: bytes) -> bytes:
    """``SHA-256(label || u32be(i))`` for i = 0..4, concatenated, bit 1279 cleared: a key anyone can re-derive."""

    return _clear_top(b"".join(hashlib.sha256(bytes(label) + i.to_bytes(4, "big")).digest() for i in range(5)))


def fresh_key() -> bytes:
    """A key from the operating system's generator (a committer's own per-run key, PROTOCOL.md section 3)."""

    return _clear_top(os.urandom(KEY_BYTES))


def fresh_salt() -> bytes:
    """One leaf's salt: 128 bytes from the operating system's generator, never reused."""

    return os.urandom(SALT_BYTES)


#: The pinned key of v1: a circuit constant, re-derivable from :data:`KEY_LABEL`.
DEFAULT_KEY = key_from_label(KEY_LABEL)


@functools.lru_cache(maxsize=8)
def rows(key: bytes) -> tuple[int, ...]:
    """Row i of M as a 1024-bit integer (bit j is ``M[i][j] = key bit i + j``)."""

    k = int.from_bytes(check_key(key), "little")
    return tuple((k >> i) & ((1 << COLS) - 1) for i in range(ROWS))


@functools.lru_cache(maxsize=8)
def columns(key: bytes) -> tuple[int, ...]:
    """Column j of M as a 256-bit integer (bit i is ``M[i][j]``)."""

    k = int.from_bytes(check_key(key), "little")
    return tuple((k >> j) & ((1 << ROWS) - 1) for j in range(COLS))


@functools.lru_cache(maxsize=8)
def byte_tables(key: bytes) -> tuple[tuple[int, ...], ...]:
    """Per salt byte position p and byte value v, the XOR of the columns ``8p + s`` over the set bits s of v."""

    cols = columns(key)
    out = []
    for p in range(SALT_BYTES):
        t = [0] * 256
        for v in range(1, 256):
            t[v] = t[v & (v - 1)] ^ cols[8 * p + (v & -v).bit_length() - 1]
        out.append(tuple(t))
    return tuple(out)


def mask(key: bytes, salt: bytes) -> bytes:
    """``M_key * y`` over GF(2), 32 bytes (bit i of the result is byte ``i >> 3``, bit ``i & 7``)."""

    salt = bytes(salt)
    if len(salt) != SALT_BYTES:
        raise InvalidArtifact(f"a salt is {SALT_BYTES} bytes")
    acc = 0
    for table, v in zip(byte_tables(check_key(key)), salt, strict=True):
        acc ^= table[v]
    return acc.to_bytes(DIGEST_BYTES, "little")


def salt_digest(salt: bytes) -> bytes:
    """``c = SHA-256(SALT_PREFIX || y)``; the prefix is one block, so a gadget starts from a constant midstate."""

    salt = bytes(salt)
    if len(salt) != SALT_BYTES:
        raise InvalidArtifact(f"a salt is {SALT_BYTES} bytes")
    return hashlib.sha256(SALT_PREFIX + salt).digest()


def commit_string(key: bytes, inner_digest: bytes, salt: bytes) -> bytes:
    """``b || c`` for the unsalted digest ``x`` and salt ``y``: what a backend proves and publishes (64 bytes)."""

    x = bytes(inner_digest)
    if len(x) != DIGEST_BYTES:
        raise InvalidArtifact("the inner digest is 32 bytes")
    b = bytes(p ^ q for p, q in zip(x, mask(key, salt), strict=True))
    return b + salt_digest(salt)


def leaf_prefix(key: bytes) -> bytes:
    """``(LEAF_TAG || SHA-256(key))`` zero-padded to 64 bytes: the leaf binds its key, at the cost of a constant midstate."""

    return (LEAF_TAG + hashlib.sha256(check_key(key)).digest()).ljust(64, b"\0")


def tree_leaf(key: bytes, proven: bytes) -> bytes:
    """``SHA-256(leaf_prefix(key) || b || c)``: the tree leaf a published commit string stands for (the verifier's native step)."""

    proven = bytes(proven)
    if len(proven) != PROVEN_BYTES:
        raise InvalidArtifact(f"a commit string b || c is {PROVEN_BYTES} bytes")
    return hashlib.sha256(leaf_prefix(key) + proven).digest()


def scheme_digest(key: bytes) -> bytes:
    """The configured leaf rule's identity, for a root binding or statement to absorb: name, sizes, and the key's digest."""

    doc = {"name": NAME, "salt_bytes": SALT_BYTES, "key_sha256": hashlib.sha256(check_key(key)).hexdigest()}
    return bytes.fromhex(identity_digest(SCHEME_DIGEST_TAG, doc))


def xor_count(key: bytes) -> int:
    """Two-input XOR gates of ``b = x XOR M y`` in a circuit: row i takes ``weight(row i) - 1`` XORs, plus one for ``x_i``."""

    return sum(r.bit_count() for r in rows(key))


def statistical_distance_log2(leaves: int, *, key_failure_log2: int | None = None) -> float:
    """log2 of the hiding bound for ``leaves`` commitments under one key (PROTOCOL.md section 3).

    A fresh uniform key: any two vectors of ``leaves`` inner digests give commit strings within ``leaves * 2^-256``.
    A pinned key (``key_failure_log2 = t``): the same holds within ``leaves * 2^(t - 256)`` for all but a ``2^-t`` fraction
    of keys (Markov over the key).
    """

    if leaves < 1:
        raise ValueError("at least one leaf")
    lhl = (ROWS + DIGEST_BYTES * 8 - COLS) / 2                     # -256: per-leaf 2^-257, doubled for two message vectors
    return math.log2(leaves) + lhl + (key_failure_log2 or 0)


# --- the layout a backend proves, and the configured scheme ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HidingLayout:
    """What a backend proves for one leaf: ``b || c`` from the private value (under ``inner``) and the private salt.

    ``inner`` is the unsalted SHA-256 leaf layout of the port; ``salt_layout`` is ``c``'s; the XOR-only ``b = x XOR M y``
    reads :func:`rows` of ``key``.  The verifier turns the published ``b || c`` into the tree leaf with :meth:`tree_leaf`.
    """

    inner: LeafLayout
    key: bytes

    def __post_init__(self) -> None:
        if self.inner.hash != "sha256":
            raise InvalidArtifact(f"{NAME} wraps a SHA-256 leaf layout, not {self.inner.hash!r}")
        check_key(self.key)

    @property
    def salt_layout(self) -> LeafLayout:
        return LeafLayout("sha256", SALT_PREFIX, SALT_BYTES)

    def commit(self, value: bytes, salt: bytes) -> bytes:
        return commit_string(self.key, self.inner.digest(value), salt)

    def tree_leaf(self, proven: bytes) -> bytes:
        return tree_leaf(self.key, proven)

    def leaf(self, value: bytes, salt: bytes) -> bytes:
        return self.tree_leaf(self.commit(value, salt))


class Hm96Sha256:
    """``hm96-sha256/v1`` leaves over a ``vllm-v1`` configuration: its ports, domains, trees and root bindings, with each
    position's unsalted leaf layout wrapped in a :class:`HidingLayout` under ``key``.

    ``leaf_layout`` returns a :class:`HidingLayout` (the proven digest is ``b || c``, 64 bytes) and ``tree_leaf`` takes it.
    The key must be bound where the verifier reads the roots from: :func:`scheme_digest` names it, and every leaf binds
    its digest too.
    """

    name = NAME
    hashes = ("sha256",)

    def __init__(self, base: VllmV1, key: bytes = DEFAULT_KEY) -> None:
        if not isinstance(base, VllmV1):
            raise TypeError(f"{NAME} wraps a vllm-v1 configuration, whose tree leaf is the leaf digest itself")
        self.base = base
        self.key = check_key(key)

    @property
    def digest(self) -> bytes:
        return scheme_digest(self.key)

    def leaf_layout(self, port: str, position: int) -> HidingLayout:
        return HidingLayout(self.base.leaf_layout(port, position), self.key)

    def tree_leaf(self, port: str, position: int, digest: bytes) -> bytes:
        return self.base.tree_leaf(port, position, tree_leaf(self.key, digest))

    def root(self, port: str, leaves: Sequence[bytes]) -> bytes:
        return self.base.root(port, leaves)

    def path(self, port: str, leaves: Sequence[bytes], index: int) -> Path:
        return self.base.path(port, leaves, index)

    def verify_path(self, port: str, root: bytes, leaf_count: int, index: int, leaf: bytes, path: Sequence[bytes | None]) -> bool:
        return self.base.verify_path(port, root, leaf_count, index, leaf, path)
