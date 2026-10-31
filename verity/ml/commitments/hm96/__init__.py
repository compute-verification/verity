"""Halevi–Micali hiding leaves: ``hm96-sha256/v1`` and ``hm96-sha512/v1``, leaf rules that wrap a SHA-256 or SHA-512 leaf digest.

Spec: ``PROTOCOL.md`` beside this file (the construction, the hiding bound, binding, the pinned keys, what a circuit proves and
what it costs).  Conformance vectors: ``vectors.json`` (SHA-256) and ``vectors_sha512.json`` beside it, written by
``python -m verity.ml.commitments.hm96.vectors``.  This module is the plain reference, stdlib only; fast committers live with
their producers and match its bytes.

A leaf commits a value under an *inner* layout, any :class:`~verity.ml.commitments.scheme.LeafLayout` with the instance's hash:
the unsalted leaf digest the port already defines (a position leaf, a ``sha256/row/v1`` or ``sha512/row/v1`` row
digest).  With H the instance's hash, n its output bits and l = n + n + 512 the salt bits::

    x    = H(inner.prefix || value || inner.suffix)             the unsalted digest; private
    y    = l / 8 fresh bytes from the OS generator (the salt)   private; stored with the value
    c    = H(salt_prefix || y)
    b    = x XOR (M_key * y)                                     M_key: the n x l Hankel matrix of the key over GF(2)
    leaf = H(leaf_prefix(key) || b || c)

``b || c`` is the Halevi–Micali commit string: their first scheme with the affine universal family ``y -> M y + b``,
applied to the digest ``x`` (their O(k) variant).  A backend proves ``b || c`` from the private value and salt and
publishes it; the verifier computes the leaf natively.  In a circuit only ``c`` costs AND gates beyond the unsalted
digest: ``M_key * y`` is XOR-only, and the key is a circuit constant or a public input, never a witness.  Binding rests on
H's collision resistance alone, for any key.  Hiding is statistical given uniform salts, with the same bound in both
instances: at most ``N * 2^-256`` for N leaves under a uniform key, and under the pinned key ``N * 2^-192`` except for a
``2^-64`` fraction of keys, a common-reference-string step (:meth:`Instance.statistical_distance_log2`; claims
``statistical-hiding``, ``common-reference-string``, ``hash-derived-key``).

The module-level names are ``hm96-sha256/v1``'s (:data:`SHA256`); :data:`SHA512` is ``hm96-sha512/v1``.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

from verity.errors import InvalidArtifact

from ..identity import identity_digest, identity_digest_sha512
from ..scheme import DIGEST_BYTES as _DIGEST_BYTES
from ..scheme import LeafLayout, sha

__all__ = [
    "COLS",
    "DEFAULT_KEY",
    "DIGEST_BYTES",
    "HidingLayout",
    "Instance",
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
    "SHA256",
    "SHA512",
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

_BLOCK_BYTES = {"sha256": 64, "sha512": 128}
#: Bits of hiding margin in the salt beyond ``|x| + |c|``: per-leaf distance 2^-257 under a uniform key (PROTOCOL.md section 3).
MARGIN_BITS = 512


def _bit(data: bytes, t: int) -> int:
    return (data[t >> 3] >> (t & 7)) & 1


@dataclass(frozen=True, slots=True)
class Instance:
    """One hm96 leaf rule: the hash, the salt size and the domain-separation strings; every other size follows from them."""

    name: str
    hash: str
    salt_bytes: int
    salt_tag: bytes
    leaf_tag: bytes
    key_label: bytes
    scheme_tag: str

    def __post_init__(self) -> None:
        if 8 * self.salt_bytes != 2 * self.rows + MARGIN_BITS:
            raise ValueError(f"{self.name}: the salt must be |x| + |c| + {MARGIN_BITS} bits")

    @property
    def digest_bytes(self) -> int:
        return _DIGEST_BYTES[self.hash]

    @property
    def rows(self) -> int:
        """L = n: the bits of x and b (and of c)."""
        return 8 * self.digest_bytes

    @property
    def cols(self) -> int:
        """l: the bits of y."""
        return 8 * self.salt_bytes

    @property
    def key_bits(self) -> int:
        return self.rows + self.cols - 1

    @property
    def key_bytes(self) -> int:
        return -(-self.key_bits // 8)

    @property
    def proven_bytes(self) -> int:
        return 2 * self.digest_bytes

    @property
    def salt_prefix(self) -> bytes:
        return self.salt_tag.ljust(_BLOCK_BYTES[self.hash], b"\0")

    @property
    def default_key(self) -> bytes:
        """The pinned key: a circuit constant, re-derivable from :attr:`key_label`."""
        return _default_key(self)

    def _h(self, data: bytes) -> bytes:
        return sha(self.hash)(data).digest()

    def _top_mask(self) -> int:
        return (1 << (self.key_bits - 8 * (self.key_bytes - 1))) - 1

    def _clear_top(self, raw: bytes) -> bytes:
        out = bytearray(raw[:self.key_bytes])
        out[-1] &= self._top_mask()
        return bytes(out)

    def check_key(self, key: bytes) -> bytes:
        """``key`` as bytes, refused unless it has :attr:`key_bytes` bytes with the bits above :attr:`key_bits` zero."""

        key = bytes(key)
        if len(key) != self.key_bytes or key[-1] & ~self._top_mask():
            raise InvalidArtifact(f"an {self.name} key is {self.key_bytes} bytes with bit {self.key_bits} and above zero")
        return key

    def key_from_label(self, label: bytes) -> bytes:
        """``H(label || u32be(i))`` for i = 0, 1, ..., concatenated to :attr:`key_bytes`, the top bits cleared: anyone re-derives it."""

        blocks = -(-self.key_bytes // self.digest_bytes)
        return self._clear_top(b"".join(self._h(bytes(label) + i.to_bytes(4, "big")) for i in range(blocks)))

    def fresh_key(self) -> bytes:
        """A key from the operating system's generator."""
        return self._clear_top(os.urandom(self.key_bytes))

    def fresh_salt(self) -> bytes:
        """One leaf's salt from the operating system's generator, never reused."""
        return os.urandom(self.salt_bytes)

    def rows_of(self, key: bytes) -> tuple[int, ...]:
        """Row i of M as an l-bit integer (bit j is ``M[i][j] = key bit i + j``)."""
        return _rows(self, self.check_key(key))

    def columns(self, key: bytes) -> tuple[int, ...]:
        """Column j of M as an n-bit integer (bit i is ``M[i][j]``)."""
        return _columns(self, self.check_key(key))

    def byte_tables(self, key: bytes) -> tuple[tuple[int, ...], ...]:
        """Per salt byte position p and byte value v, the XOR of the columns ``8p + s`` over the set bits s of v."""
        return _byte_tables(self, self.check_key(key))

    def _salt(self, salt: bytes) -> bytes:
        salt = bytes(salt)
        if len(salt) != self.salt_bytes:
            raise InvalidArtifact(f"an {self.name} salt is {self.salt_bytes} bytes")
        return salt

    def mask(self, key: bytes, salt: bytes) -> bytes:
        """``M_key * y`` over GF(2) (bit i of the result is byte ``i >> 3``, bit ``i & 7``)."""

        acc = 0
        for table, v in zip(self.byte_tables(key), self._salt(salt), strict=True):
            acc ^= table[v]
        return acc.to_bytes(self.digest_bytes, "little")

    def salt_digest(self, salt: bytes) -> bytes:
        """``c = H(salt_prefix || y)``; the prefix is one block, so a gadget starts from a constant midstate."""
        return self._h(self.salt_prefix + self._salt(salt))

    def commit_string(self, key: bytes, inner_digest: bytes, salt: bytes) -> bytes:
        """``b || c`` for the unsalted digest ``x`` and salt ``y``: what a backend proves and publishes."""

        x = bytes(inner_digest)
        if len(x) != self.digest_bytes:
            raise InvalidArtifact(f"the inner digest is {self.digest_bytes} bytes")
        return bytes(p ^ q for p, q in zip(x, self.mask(key, salt), strict=True)) + self.salt_digest(salt)

    def leaf_prefix(self, key: bytes) -> bytes:
        """``(leaf_tag || H(key))`` zero-padded to one block: the leaf binds its key, at the cost of a constant midstate."""
        return (self.leaf_tag + self._h(self.check_key(key))).ljust(_BLOCK_BYTES[self.hash], b"\0")

    def tree_leaf(self, key: bytes, proven: bytes) -> bytes:
        """``H(leaf_prefix(key) || b || c)``: the tree leaf a published commit string stands for (the verifier's native step)."""

        proven = bytes(proven)
        if len(proven) != self.proven_bytes:
            raise InvalidArtifact(f"an {self.name} commit string b || c is {self.proven_bytes} bytes")
        return self._h(self.leaf_prefix(key) + proven)

    def scheme_digest(self, key: bytes) -> bytes:
        """The configured rule's identity, for a root binding or statement to absorb: name, salt size and the key's digest.

        SHA-256: ``identity_digest(scheme_tag, doc)``.  SHA-512: ``identity_digest_sha512(scheme_tag, doc)``, whose key digest
        is SHA-512.
        """

        doc = {"name": self.name, "salt_bytes": self.salt_bytes, f"key_{self.hash}": self._h(self.check_key(key)).hex()}
        if self.hash == "sha256":
            return bytes.fromhex(identity_digest(self.scheme_tag, doc))
        return identity_digest_sha512(self.scheme_tag, doc)

    def xor_count(self, key: bytes) -> int:
        """Two-input XOR gates of ``b = x XOR M y`` in a circuit: row i takes ``weight(row i) - 1`` XORs, plus one for ``x_i``."""
        return sum(r.bit_count() for r in self.rows_of(key))

    def statistical_distance_log2(self, leaves: int, *, key_failure_log2: int | None = None) -> float:
        """log2 of the hiding bound for ``leaves`` commitments under one key (PROTOCOL.md section 3).

        A fresh uniform key: any two vectors of ``leaves`` inner digests give commit strings within ``leaves * 2^-256``.
        A pinned key (``key_failure_log2 = t``): the same holds within ``leaves * 2^(t - 256)`` for all but a ``2^-t`` fraction
        of keys (Markov over the key).
        """

        if leaves < 1:
            raise ValueError("at least one leaf")
        return math.log2(leaves) + (2 * self.rows - self.cols) / 2 + (key_failure_log2 or 0)


@functools.lru_cache(maxsize=4)
def _default_key(inst: Instance) -> bytes:
    return inst.key_from_label(inst.key_label)


@functools.lru_cache(maxsize=16)
def _rows(inst: Instance, key: bytes) -> tuple[int, ...]:
    k = int.from_bytes(key, "little")
    return tuple((k >> i) & ((1 << inst.cols) - 1) for i in range(inst.rows))


@functools.lru_cache(maxsize=16)
def _columns(inst: Instance, key: bytes) -> tuple[int, ...]:
    k = int.from_bytes(key, "little")
    return tuple((k >> j) & ((1 << inst.rows) - 1) for j in range(inst.cols))


@functools.lru_cache(maxsize=16)
def _byte_tables(inst: Instance, key: bytes) -> tuple[tuple[int, ...], ...]:
    cols = _columns(inst, key)
    out = []
    for p in range(inst.salt_bytes):
        t = [0] * 256
        for v in range(1, 256):
            t[v] = t[v & (v - 1)] ^ cols[8 * p + (v & -v).bit_length() - 1]
        out.append(tuple(t))
    return tuple(out)


SHA256 = Instance("hm96-sha256/v1", "sha256", 128, b"verity/hm96-sha256/salt/v1\0", b"verity/hm96-sha256/leaf/v1\0",
                  b"verity/hm96-sha256/key/v1\0", "verity/hm96-sha256/scheme/v1")
SHA512 = Instance("hm96-sha512/v1", "sha512", 192, b"verity/hm96-sha512/salt/v1\0", b"verity/hm96-sha512/leaf/v1\0",
                  b"verity/hm96-sha512/key/v1\0", "verity/hm96-sha512/scheme/v1")
INSTANCES = {i.name: i for i in (SHA256, SHA512)}

# hm96-sha256/v1's names, as first published.
NAME = SHA256.name
DIGEST_BYTES = SHA256.digest_bytes
ROWS = SHA256.rows
SALT_BYTES = SHA256.salt_bytes
COLS = SHA256.cols
KEY_BITS = SHA256.key_bits
KEY_BYTES = SHA256.key_bytes
PROVEN_BYTES = SHA256.proven_bytes
SALT_TAG = SHA256.salt_tag
LEAF_TAG = SHA256.leaf_tag
KEY_LABEL = SHA256.key_label
SALT_PREFIX = SHA256.salt_prefix
SCHEME_DIGEST_TAG = SHA256.scheme_tag
DEFAULT_KEY = SHA256.default_key
check_key = SHA256.check_key
key_from_label = SHA256.key_from_label
fresh_key = SHA256.fresh_key
fresh_salt = SHA256.fresh_salt
rows = SHA256.rows_of
columns = SHA256.columns
byte_tables = SHA256.byte_tables
mask = SHA256.mask
salt_digest = SHA256.salt_digest
commit_string = SHA256.commit_string
leaf_prefix = SHA256.leaf_prefix
tree_leaf = SHA256.tree_leaf
scheme_digest = SHA256.scheme_digest
xor_count = SHA256.xor_count
statistical_distance_log2 = SHA256.statistical_distance_log2


# --- the layout a backend proves ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HidingLayout:
    """What a backend proves for one leaf: ``b || c`` from the private value (under ``inner``) and the private salt.

    ``inner`` is the unsalted leaf layout of the port, with the instance's hash; ``salt_layout`` is ``c``'s; the XOR-only
    ``b = x XOR M y`` reads ``instance.rows_of(key)``.  The verifier turns the published ``b || c`` into the tree leaf with
    :meth:`tree_leaf`.
    """

    inner: LeafLayout
    key: bytes
    instance: Instance = SHA256

    def __post_init__(self) -> None:
        if self.inner.hash != self.instance.hash:
            raise InvalidArtifact(f"{self.instance.name} wraps a {self.instance.hash} leaf layout, not {self.inner.hash!r}")
        self.instance.check_key(self.key)

    @property
    def salt_layout(self) -> LeafLayout:
        return LeafLayout(self.instance.hash, self.instance.salt_prefix, self.instance.salt_bytes)

    def commit(self, value: bytes, salt: bytes) -> bytes:
        return self.instance.commit_string(self.key, self.inner.digest(value), salt)

    def tree_leaf(self, proven: bytes) -> bytes:
        return self.instance.tree_leaf(self.key, proven)

    def leaf(self, value: bytes, salt: bytes) -> bytes:
        return self.tree_leaf(self.commit(value, salt))

