"""Transparent SHA-256 Merkle commitments over verifier-derived domains.

A :class:`CommitmentDomain` fixes *which* positions a commitment covers, in
*which* order, at *which* phase of *which* session.  Both parties derive it
from trusted data alone; the prover never describes its own domain.  Every
leaf and node hash is bound to the domain, so a root cannot be reused for a
different session, phase, owner, or position set.

Promoted from veritor ``protocol/merkle.py`` (eb865ce2) with the ``Commitment`` and ``Opening`` records of
``protocol/messages.py``.  The frame ``veritor/protocol/merkle/frame/v3`` and every tag below are wire constants
mirrored by the SP1 kernel (``frame.rs``); every recorded root depends on them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from verity.errors import ProtocolError

from .identity import JSONValue
from .indexed import IndexedDomain, iter_members
from .limits import VerificationLimits

_FRAME = b"veritor/protocol/merkle/frame/v3\0"
_LEAF = b"leaf"
_PAD = b"pad"
_NODE = b"node"
_EMPTY = b"empty"


def _uint(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ProtocolError("hash integers must be nonnegative")
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def _hash(tag: bytes, *parts: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(_FRAME)
    digest.update(len(tag).to_bytes(4, "big"))
    digest.update(tag)
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def merkle_depth(count: int) -> int:
    return 0 if count <= 1 else (count - 1).bit_length()


def _bytes32(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ProtocolError(f"{name} must be 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class Commitment:
    """A Merkle root over one verifier-derived position domain."""

    root: bytes
    count: int

    def __post_init__(self) -> None:
        _bytes32(self.root, "commitment root")
        if type(self.count) is not int or self.count < 0:
            raise ProtocolError("commitment count must be a nonnegative integer")

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return {"count": self.count, "root": self.root.hex()}


@dataclass(frozen=True, slots=True)
class Opening:
    """An authenticated value at one position of one commitment."""

    position: int
    value: bytes
    path: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            raise ProtocolError("opening position must be a nonnegative integer")
        if type(self.value) is not bytes:
            raise ProtocolError("opening value must be bytes")
        if type(self.path) is not tuple or any(
            type(item) is not bytes or len(item) != 32 for item in self.path
        ):
            raise ProtocolError("opening path must be a tuple of 32-byte digests")


@dataclass(frozen=True, slots=True)
class CommitmentDomain:
    """Everything a leaf hash is bound to besides the value itself.

    ``binding`` is the 32-byte digest of what the commitment may not outlive:
    the header for the boundary, the replay phase for an interior, a fixed
    tag for a per-model weight root.  ``owner`` is ``-2`` for the weights,
    ``-1`` for the boundary, else a replay unit index.
    """

    binding: bytes
    owner: int
    positions: IndexedDomain[int]
    domain_id: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.binding) is not bytes or len(self.binding) != 32:
            raise ProtocolError("binding must be 32 bytes")
        if type(self.owner) is not int or self.owner < -2:
            raise ProtocolError(
                "owner must be -2 (weights), -1 (boundary) or a replay unit"
            )
        object.__setattr__(
            self,
            "domain_id",
            _hash(
                b"domain",
                self.binding,
                _uint(self.owner + 2),
                bytes.fromhex(self.positions.identity_digest),
                _uint(self.positions.count),
            ),
        )

    @property
    def count(self) -> int:
        return self.positions.count

    def leaf(self, rank: int, position: int, schema: str, value: bytes) -> bytes:
        return _hash(
            _LEAF,
            self.domain_id,
            _uint(rank),
            _uint(position),
            schema.encode("utf-8"),
            value,
        )

    def leaf_prefix(self, rank: int, position: int, schema: str, value_len: int) -> bytes:
        """Every byte of :meth:`leaf`'s SHA-256 message before the value: ``leaf == SHA-256(prefix || value)``."""

        parts = (self.domain_id, _uint(rank), _uint(position), schema.encode("utf-8"))
        out = _FRAME + len(_LEAF).to_bytes(4, "big") + _LEAF
        for part in parts:
            out += len(part).to_bytes(8, "big") + part
        return out + value_len.to_bytes(8, "big")

    def node(self, level: int, index: int, left: bytes, right: bytes) -> bytes:
        return _hash(_NODE, self.domain_id, _uint(level), _uint(index), left, right)

    def empty_root(self) -> bytes:
        return _hash(_EMPTY, self.domain_id)


def tree_levels(domain: CommitmentDomain, leaves: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    """The levels over one domain's leaf digests (in rank order), padded to a power of two; ``()`` for none."""

    if len(leaves) != domain.count:
        raise ProtocolError(f"expected {domain.count} leaves, got {len(leaves)}")
    if not leaves:
        return ()
    width = 1 << merkle_depth(len(leaves))
    level = [*leaves, *(_hash(_PAD, domain.domain_id, _uint(rank)) for rank in range(len(leaves), width))]
    levels = [tuple(level)]
    depth = 0
    while len(level) > 1:
        level = [domain.node(depth, index // 2, level[index], level[index + 1]) for index in range(0, len(level), 2)]
        levels.append(tuple(level))
        depth += 1
    return tuple(levels)


class MerkleTree:
    """Prover-side committed values for one domain."""

    __slots__ = ("_domain", "_levels", "_values")

    def __init__(
        self,
        domain: CommitmentDomain,
        values: Mapping[int, bytes],
        schema: Callable[[int], str],
    ) -> None:
        if len(values) != domain.count:
            raise ProtocolError(
                f"expected values for {domain.count} positions, got {len(values)}"
            )
        ordered: list[bytes] = []
        leaves: list[bytes] = []
        for rank, member in enumerate(iter_members(domain.positions)):
            position = int(member)
            try:
                value = values[position]
            except KeyError as error:
                raise ProtocolError(f"missing value for position {position}") from error
            if type(value) is not bytes:
                raise ProtocolError("committed values must be bytes")
            ordered.append(value)
            leaves.append(domain.leaf(rank, position, schema(position), value))
        self._domain = domain
        self._values = tuple(ordered)
        self._levels = tree_levels(domain, leaves)

    @property
    def domain(self) -> CommitmentDomain:
        return self._domain

    @property
    def commitment(self) -> Commitment:
        root = self._levels[-1][0] if self._levels else self._domain.empty_root()
        return Commitment(root, self._domain.count)

    def open(self, position: int) -> Opening:
        rank = self._domain.positions.rank(position)
        path: list[bytes] = []
        cursor = rank
        for level in self._levels[:-1]:
            path.append(level[cursor ^ 1])
            cursor >>= 1
        return Opening(position, self._values[rank], tuple(path))


def validate_commitment(domain: CommitmentDomain, commitment: Commitment) -> bool:
    if commitment.count != domain.count:
        return False
    return domain.count > 0 or commitment.root == domain.empty_root()


def verify_opening(
    domain: CommitmentDomain,
    commitment: Commitment,
    opening: Opening,
    schema: str,
    limits: VerificationLimits,
) -> bool:
    """Return whether ``opening`` authenticates under ``commitment``."""

    if not validate_commitment(domain, commitment):
        return False
    try:
        rank = domain.positions.rank(opening.position)
    except KeyError:
        return False
    if len(opening.path) != merkle_depth(domain.count):
        return False
    limits.enforce("max_proof_bytes", len(opening.value) + 32 * len(opening.path))
    digest = domain.leaf(rank, opening.position, schema, opening.value)
    cursor = rank
    for level, sibling in enumerate(opening.path):
        pair = (digest, sibling) if cursor % 2 == 0 else (sibling, digest)
        digest = domain.node(level, cursor >> 1, *pair)
        cursor >>= 1
    return digest == commitment.root
