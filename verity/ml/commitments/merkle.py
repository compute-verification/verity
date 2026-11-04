"""Transparent frame-v3 Merkle commitments over verifier-derived domains: ``frame-v3-sha512`` by default, and the SHA-256
frame where a recorded format carries it (``hash="sha256"``).

A :class:`CommitmentDomain` fixes *which* positions a commitment covers, in
*which* order, at *which* phase of *which* session.  Both parties derive it
from trusted data alone; the prover never describes its own domain.  Every
leaf and node hash is bound to the domain, so a root cannot be reused for a
different session, phase, owner, or position set.

Promoted from veritor ``protocol/merkle.py`` (eb865ce2) with the ``Commitment`` and ``Opening`` records of
``protocol/messages.py``.  The frame ``veritor/protocol/merkle/frame/v3`` and every tag below are wire constants
mirrored by the SP1 kernel (``frame.rs``); every recorded root depends on them.  Every hash the frame computes is
`frame_hash`: SHA-512 (or SHA-256) of the frame, the tag and the length-framed parts.  Under SHA-512 the domain id, leaves,
pads, nodes and roots are 64 bytes, and the domain id absorbs the positions' SHA-512 identity (``frame_v3/PROTOCOL.md`` §6),
or, for the ``range-u64`` kind, its start and count as u64be words under the tag ``domain/range-u64``.  Under every frame a
``binding`` is 64 bytes, the caller's SHA-512 identity digest, or 32 bytes, a SHA-256 identity digest as recorded statements
carry it; every part is length-framed, so the two cannot collide.

A leaf, node or pad hashes its tag and domain id first, so each domain's three prefixes are hashed once (`_head`) and every
leaf, node and pad of it continues from a copy: `CommitmentDomain`'s, `tree_levels`, `levels` and `root_from_subtrees` alike.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from verity.errors import ProtocolError

from .identity import JSONValue
from .indexed import U64_BOUND, IndexedDomain, RangeU64Domain, iter_members
from .limits import VerificationLimits
from .scheme import DIGEST_BYTES, sha

_FRAME = b"veritor/protocol/merkle/frame/v3\0"
_LEAF = b"leaf"
_PAD = b"pad"
_NODE = b"node"
_EMPTY = b"empty"


def _uint(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ProtocolError("hash integers must be nonnegative")
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def _part(part: bytes) -> bytes:
    return len(part).to_bytes(8, "big") + part


def _message(tag: bytes, *parts: bytes) -> bytes:
    return _FRAME + len(tag).to_bytes(4, "big") + tag + b"".join(_part(p) for p in parts)


def frame_hash(tag: bytes, *parts: bytes, hash: str = "sha512") -> bytes:
    """The frame's hash of ``tag`` over ``parts``, each length-framed: every domain id, leaf, node, pad and empty root is one."""

    return sha(hash)(_message(tag, *parts)).digest()


def domain_message(binding: bytes, owner: int, positions: bytes, count: int, hash: str = "sha512") -> bytes:
    """The framed message whose digest under ``hash`` is `CommitmentDomain.domain_id`, ``positions`` being the positions'
    identity bytes: for a caller that hashes it with another implementation of the same hash."""

    return _message(b"domain", binding, _uint(owner + 2), positions, _uint(count))


#: The ``range-u64`` domain message's tag and its length in bytes; the count is its last 8 bytes.
RANGE_U64_TAG = b"domain/range-u64"
RANGE_U64_MESSAGE_BYTES = 173


def _u64(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value < U64_BOUND:
        raise ProtocolError(f"a range-u64 domain's {name} must be a u64")
    return value.to_bytes(8, "big")


def range_u64_domain_message(binding: bytes, owner: int, start: int, count: int) -> bytes:
    """The framed message whose SHA-512 is a ``range-u64`` domain's id: the 64-byte binding, then ``owner + 2``, ``start`` and
    ``count`` as u64be words, 173 bytes (``frame_v3/PROTOCOL.md`` §6)."""

    if type(binding) is not bytes or len(binding) != 64:
        raise ProtocolError("a range-u64 domain's binding must be 64 bytes")
    if type(owner) is not int or owner < -2:
        raise ProtocolError("owner must be -2 (weights), -1 (boundary) or a replay unit")
    if type(start) is not int or type(count) is not int or start < 0 or count < 0 or start + count >= U64_BOUND:
        raise ProtocolError("a range-u64 domain needs nonnegative start and count with start + count < 2^64")
    return _message(RANGE_U64_TAG, binding, _u64(owner + 2, "owner + 2"), _u64(start, "start"), _u64(count, "count"))


@lru_cache(maxsize=4096)
def _head(domain_id: bytes, tag: bytes, hash: str) -> Any:
    """The hash state after ``tag`` and the domain id, the start of every ``tag`` message under the domain."""

    return sha(hash)(_message(tag, domain_id))


def _under(domain_id: bytes, tag: bytes, hash: str, *parts: bytes) -> bytes:
    """``frame_hash(tag, domain_id, *parts, hash=hash)``, from the domain's hashed prefix."""

    h = _head(domain_id, tag, hash).copy()
    h.update(b"".join([len(p).to_bytes(8, "big") + p for p in parts]))
    return h.digest()


def _node(domain_id: bytes, hash: str, level: int, index: int, left: bytes, right: bytes) -> bytes:
    return _under(domain_id, _NODE, hash, _uint(level), _uint(index), left, right)


def _pad(domain_id: bytes, hash: str, rank: int) -> bytes:
    return _under(domain_id, _PAD, hash, _uint(rank))


def merkle_depth(count: int) -> int:
    return 0 if count <= 1 else (count - 1).bit_length()


def _bytes32(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) not in (32, 64):
        raise ProtocolError(f"{name} must be a 32-byte (SHA-256 frame) or 64-byte (SHA-512 frame) digest")
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
            type(item) is not bytes or len(item) not in (32, 64) for item in self.path
        ):
            raise ProtocolError("opening path must be a tuple of 32- or 64-byte digests")


@dataclass(frozen=True, slots=True)
class CommitmentDomain:
    """Everything a leaf hash is bound to besides the value itself.

    ``binding`` is the digest of what the commitment may not outlive:
    the header for the boundary, the replay phase for an interior, a fixed
    tag for a per-model weight root.  It is 64 bytes (a SHA-512 identity digest),
    or 32 (a SHA-256 one, as recorded statements carry it), under every frame.
    ``owner`` is ``-2`` for the weights, ``-1`` for the boundary, else a replay
    unit index.  Over a :class:`~verity.ml.commitments.indexed.RangeU64Domain` it is the ``range-u64`` kind: SHA-512 only,
    a 64-byte binding, and the domain id is the SHA-512 of :func:`range_u64_domain_message`.
    """

    binding: bytes
    owner: int
    positions: IndexedDomain[int]
    hash: str = "sha512"
    domain_id: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.binding) is not bytes or len(self.binding) not in (32, 64):
            raise ProtocolError("binding must be 64 bytes (a SHA-512 identity) or 32 (a SHA-256 one)")
        if type(self.owner) is not int or self.owner < -2:
            raise ProtocolError(
                "owner must be -2 (weights), -1 (boundary) or a replay unit"
            )
        sha(self.hash)
        if isinstance(self.positions, RangeU64Domain):
            if self.hash != "sha512":
                raise ProtocolError(f"a range-u64 domain hashes with SHA-512 only, not {self.hash!r}")
            message = range_u64_domain_message(self.binding, self.owner, self.positions.start, self.positions.count)
            object.__setattr__(self, "domain_id", sha("sha512")(message).digest())
            return
        if self.hash == "sha256":
            positions = bytes.fromhex(self.positions.identity_digest)
        elif hasattr(self.positions, "identity_sha512"):
            positions = self.positions.identity_sha512()
        else:
            raise ProtocolError("a SHA-512 frame needs positions with a SHA-512 identity (identity_sha512)")
        object.__setattr__(
            self,
            "domain_id",
            sha(self.hash)(domain_message(self.binding, self.owner, positions, self.positions.count, self.hash)).digest(),
        )

    @property
    def count(self) -> int:
        return self.positions.count

    @property
    def digest_bytes(self) -> int:
        return DIGEST_BYTES[self.hash]

    def leaf(self, rank: int, position: int, schema: str, value: bytes) -> bytes:
        r = _part(_uint(rank))
        p = r if position == rank and type(position) is int else _part(_uint(position))
        h = _head(self.domain_id, _LEAF, self.hash).copy()
        h.update(r + p + _part(schema.encode("utf-8")) + _part(value))
        return h.digest()

    def leaf_prefix(self, rank: int, position: int, schema: str, value_len: int) -> bytes:
        """Every byte of :meth:`leaf`'s message before the value: ``leaf == H(prefix || value)``, H the domain's hash."""

        parts = (self.domain_id, _uint(rank), _uint(position), schema.encode("utf-8"))
        return _FRAME + len(_LEAF).to_bytes(4, "big") + _LEAF + b"".join(_part(p) for p in parts) + value_len.to_bytes(8, "big")

    def node(self, level: int, index: int, left: bytes, right: bytes) -> bytes:
        return _node(self.domain_id, self.hash, level, index, left, right)

    def pad(self, rank: int) -> bytes:
        return _pad(self.domain_id, self.hash, rank)

    def empty_root(self) -> bytes:
        return frame_hash(_EMPTY, self.domain_id, hash=self.hash)


def levels(domain_id: bytes, nodes: Sequence[bytes], first: int = 0, depth: int = 0,
           hash: str = "sha512") -> tuple[tuple[bytes, ...], ...]:
    """The levels of the subtree of domain id ``domain_id`` whose nodes at level ``depth`` are ``nodes``, the first at index
    ``first``, up to its root: ``nodes`` first, its root last.  ``len(nodes)`` is a power of two and ``first`` a multiple of
    it, so the span is one aligned subtree: `tree_levels` over a whole padded level, or a chunk of one folded on its own."""

    width = len(nodes)
    if width < 1 or width & (width - 1) or first % width:
        raise ProtocolError(f"{width} nodes from index {first}: a power of two of them, aligned to it")
    level = list(nodes)
    out = [tuple(level)]
    while len(level) > 1:
        first >>= 1
        level = [_node(domain_id, hash, depth, first + j, level[2 * j], level[2 * j + 1]) for j in range(len(level) // 2)]
        out.append(tuple(level))
        depth += 1
    return tuple(out)


def tree_levels(domain: CommitmentDomain, leaves: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    """The levels over one domain's leaf digests (in rank order), padded to a power of two; ``()`` for none."""

    if len(leaves) != domain.count:
        raise ProtocolError(f"expected {domain.count} leaves, got {len(leaves)}")
    if not leaves:
        return ()
    width = 1 << merkle_depth(len(leaves))
    return levels(domain.domain_id, [*leaves, *(domain.pad(rank) for rank in range(len(leaves), width))], hash=domain.hash)


def root_from_subtrees(domain_id: bytes, count: int, have: Mapping[tuple[int, int], bytes],
                       hash: str = "sha512") -> bytes | None:
    """The root of the tree of domain id ``domain_id`` over ``count`` leaves, from the subtree roots ``have`` (``(level,
    index)`` to its root, level 0 a leaf) and the domain's pads past ``count``; None where a node below ``count`` is in
    none of them."""

    def at(level: int, index: int) -> bytes | None:
        if (level, index) in have:
            return have[(level, index)]
        if level == 0:
            return _pad(domain_id, hash, index) if index >= count else None
        left, right = at(level - 1, 2 * index), at(level - 1, 2 * index + 1)
        return None if left is None or right is None else _node(domain_id, hash, level - 1, index, left, right)
    return at(merkle_depth(count), 0)


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
    if commitment.count != domain.count or len(commitment.root) != domain.digest_bytes:
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
    if any(len(sibling) != domain.digest_bytes for sibling in opening.path):
        return False
    limits.enforce("max_proof_bytes", len(opening.value) + domain.digest_bytes * len(opening.path))
    digest = domain.leaf(rank, opening.position, schema, opening.value)
    cursor = rank
    for level, sibling in enumerate(opening.path):
        pair = (digest, sibling) if cursor % 2 == 0 else (sibling, digest)
        digest = domain.node(level, cursor >> 1, *pair)
        cursor >>= 1
    return digest == commitment.root
