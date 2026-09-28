"""The ``CommitmentScheme`` interface: what every named commitment format exposes to backends and verifiers.

Every admissible leaf has one shape, ``digest = H(prefix || value || suffix)`` with H SHA-256, SHA-512 or (keyed) BLAKE3: the
value bytes are private, everything else is public.  So a backend proves only the leaf digests
(:meth:`CommitmentScheme.leaf_layout` tells its gadget and the statement check exactly what to hash), publishes them,
and the verifier checks them against the statement's roots natively with the scheme's reference
(:meth:`~CommitmentScheme.tree_leaf`, :meth:`~CommitmentScheme.verify_path`).  Adding a scheme adds data and a native
verifier, never a circuit.

A configured scheme covers the committed *ports* of one statement (a tensor role, a tree); each port has its own
positions, leaf layout and root.  Tree methods take leaf *indices* in tree order; ``leaf_layout`` and ``tree_leaf``
take the port's *position*, which for every range domain is the index.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from verity.ml.crypto import blake3 as _blake3

Path = tuple[bytes | None, ...]

#: Table 1's hash class of every hash a :class:`LeafLayout` may use: the standard hashes only.  The algebraic Poseidon2 row
#: leaves (:data:`rowleaf.SCHEMA_ROW`) are still admissible in Table 2, flagged as algebraic and never counted toward a
#: standard-hash claim.
HASH_CLASS = {"sha256": "standard", "sha512": "standard", "blake3": "standard"}
#: Output bytes of each hash; the SHA-512 variants of the framings (``frame-v3-sha512``, ``vllm-v1-sha512``) use 64-byte digests.
DIGEST_BYTES = {"sha256": 32, "sha512": 64, "blake3": 32, "blake3-s256": 32}
_SHA = {"sha256": hashlib.sha256, "sha512": hashlib.sha512}


def sha(name: str):
    """The hashlib constructor of a framing's hash, ``sha256`` or ``sha512``."""

    try:
        return _SHA[name]
    except KeyError:
        raise ValueError(f"a framing hash is one of {sorted(_SHA)}, not {name!r}") from None


@dataclass(frozen=True, slots=True)
class LeafLayout:
    """``digest = hash(prefix || value || suffix)``, keyed by ``key`` for BLAKE3; ``value`` is ``value_len`` private bytes.

    The byte offset of the value inside the hashed message is ``len(prefix)``: a gadget absorbs from any offset, and
    a verifier may precompute the midstate of every whole leading block.
    """

    hash: str
    prefix: bytes
    value_len: int
    suffix: bytes = b""
    key: bytes | None = None

    def __post_init__(self) -> None:
        if self.hash not in HASH_CLASS:
            raise ValueError(f"leaf hash {self.hash!r} is not one of {sorted(HASH_CLASS)}")
        if (self.key is not None) != (self.hash == "blake3"):
            raise ValueError("a key is given exactly for BLAKE3 leaves (keyed mode)")

    def digest(self, value: bytes) -> bytes:
        """The reference digest of ``value`` under this layout."""

        if len(value) != self.value_len:
            raise ValueError(f"leaf value is {len(value)} bytes, the layout says {self.value_len}")
        message = bytes(self.prefix) + bytes(value) + bytes(self.suffix)
        if self.hash in _SHA:
            return _SHA[self.hash](message).digest()
        return _blake3.keyed(self.key, message) if self.key else _blake3.blake3(message)


@runtime_checkable
class CommitmentScheme(Protocol):
    #: The scheme's registered name; its spec and conformance vectors carry the same string.
    name: str
    #: Every hash the scheme's binding rests on (keys of :data:`HASH_CLASS`).
    hashes: tuple[str, ...]

    def leaf_layout(self, port: str, position: int) -> LeafLayout:
        """What the prover's gadget hashes for the leaf at ``position`` of ``port``, and what the statement check reads."""
        ...

    def tree_leaf(self, port: str, position: int, digest: bytes) -> bytes:
        """The tree leaf a published leaf digest stands for (the verifier's native step between proof and tree)."""
        ...

    def root(self, port: str, leaves: Sequence[bytes]) -> bytes:
        """The published root of ``port`` over its tree leaves in tree order."""
        ...

    def path(self, port: str, leaves: Sequence[bytes], index: int) -> Path:
        """The authentication path of leaf ``index``, one entry per level from the leaves up."""
        ...

    def verify_path(self, port: str, root: bytes, leaf_count: int, index: int, leaf: bytes, path: Sequence[bytes | None]) -> bool:
        """Does tree leaf ``leaf`` at ``index`` of ``port`` authenticate under ``root``?  Never raises; the path shape and
        the root's context come from the scheme's own configuration, never from the prover."""
        ...
