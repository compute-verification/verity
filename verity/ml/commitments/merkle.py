"""Transparent SHA-256 Merkle commitments over verifier-derived domains, and their SHA-512 variant ``frame-v3-sha512``.

A :class:`CommitmentDomain` fixes *which* positions a commitment covers, in
*which* order, at *which* phase of *which* session.  Both parties derive it
from trusted data alone; the prover never describes its own domain.  Every
leaf and node hash is bound to the domain, so a root cannot be reused for a
different session, phase, owner, or position set.

Promoted from veritor ``protocol/merkle.py`` (eb865ce2) with the ``Commitment`` and ``Opening`` records of
``protocol/messages.py``.  The frame ``veritor/protocol/merkle/frame/v3`` and every tag below are wire constants
mirrored by the SP1 kernel (``frame.rs``); every recorded root depends on them.  A domain with ``hash="sha512"`` hashes the
same bytes with SHA-512, so its domain id, leaves, pads, nodes and roots are 64 bytes, and its domain id absorbs the positions'
SHA-512 identity (``frame_v3/PROTOCOL.md`` §6), or, for the ``range-u64`` kind, its start and count as u64be words under the tag
``domain/range-u64``.  Under every frame a ``binding`` is 64 bytes, the caller's SHA-512 identity
digest, or 32 bytes, a SHA-256 identity digest as recorded statements carry it; every part is length-framed, so the two cannot
collide.

A domain with ``hash="blake3"`` (``frame-b3``) is a tree mode for latency.  Its domain id is BLAKE3 over the same framed domain
message; leaves, pads and the empty root are BLAKE3 keyed by that id, and a level's nodes are BLAKE3 keyed by the level's key.
Integers are little-endian:

- leaf: ``H = 0x01 ‖ u64 rank ‖ u64 position ‖ u16 len(schema) ‖ schema``, zero-padded to a whole 64-byte block, then the value,
  which so starts block-aligned and streams through BLAKE3's own chunk tree (a 32 KB value: 33 chunks in parallel, then 6 levels);
- node at level l: ``left ‖ right`` under the key ``keyed(domain id, 0x02 ‖ u8 l)``: one 64-byte block, one compression (the
  tree's shape is fixed by the domain's count, so no index is hashed);
- pad: ``0x03 ‖ u64 rank``; empty root: ``0x04``.

A domain with ``hash="blake3-s256"`` (``frame-b3s``, Pearl-C's hashing format -h2) is ``frame-b3`` with a shallower leaf.  Its
domain id is BLAKE3 over the framed domain message with the tag ``domain/b3s``.  Nodes, pads and the empty root are
``frame-b3``'s under that id.  A leaf over a value of L bytes is a Merkle tree over its n = max(1, ⌈L / 256⌉) segments:

- segment j is ``value[256 j : 256 (j + 1)]``, the last at its true length (the empty segment when L = 0), hashed by the
  standard keyed BLAKE3 under its own key;
- segment j's key: ``keyed(domain id, 0x05 ‖ u64 rank ‖ u64 position ‖ u64 L ‖ u32 j ‖ u16 len(schema) ‖ schema)``, one
  block (a schema of at most 33 bytes), so it binds the domain id, rank, position, segment position, byte length and schema;
- a segment-tree node at level l: ``left ‖ right`` under ``keyed(domain id, 0x06 ‖ u8 l)``, one compression;
- at a level of odd width, the last node is promoted unchanged to the next level; the leaf is the top node.

The tags 0x05 and 0x06 keep every segment key, segment-level key and ``frame-b3`` level key (0x02) distinct.  L is in
every segment key, so it fixes the tree's shape.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from verity.errors import ProtocolError

from verity.ml.crypto import blake3 as _blake3
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


def _message(tag: bytes, *parts: bytes) -> bytes:
    return _FRAME + len(tag).to_bytes(4, "big") + tag + b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def _digest(message: bytes, hash: str) -> bytes:
    return _blake3.blake3(message) if hash in _BLAKE3_FRAMES else sha(hash)(message).digest()


def _hash(tag: bytes, *parts: bytes, hash: str = "sha256") -> bytes:
    return _digest(_message(tag, *parts), hash)


def domain_message(binding: bytes, owner: int, positions: bytes, count: int, hash: str = "sha256") -> bytes:
    """The framed message whose digest under ``hash``'s frame is `CommitmentDomain.domain_id`, ``positions`` being the
    positions' identity bytes: for a caller that hashes it with another implementation of the same hash."""

    return _message(b"domain/b3s" if hash == "blake3-s256" else b"domain", binding, _uint(owner + 2), positions, _uint(count))


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
def blake3_node_key(domain_id: bytes, level: int) -> bytes:
    """The key of a ``frame-b3`` tree's level: ``keyed(domain id, 0x02 ‖ u8 level)``."""

    return _blake3.keyed(domain_id, b"\x02" + level.to_bytes(1, "little"))


def blake3_node(domain_id: bytes, level: int, index: int, left: bytes, right: bytes) -> bytes:
    """A ``frame-b3`` node: ``left ‖ right`` under its level's key, one compression (module docstring); ``index`` is not hashed."""

    return _blake3.keyed(blake3_node_key(domain_id, level), left + right)


#: ``frame-b3s``: the segment size, and the longest schema name a segment key's one block holds
B3S_SEGMENT, B3S_MAX_SCHEMA = 256, 33
_BLAKE3_FRAMES = ("blake3", "blake3-s256")


def b3s_segment_key(domain_id: bytes, rank: int, position: int, schema: str, length: int, segment: int) -> bytes:
    """A ``frame-b3s`` segment's key: ``keyed(domain id, 0x05 ‖ u64 rank ‖ u64 position ‖ u64 L ‖ u32 j ‖ u16 len ‖ schema)``."""

    name = schema.encode("utf-8")
    if len(name) > B3S_MAX_SCHEMA:
        raise ProtocolError(f"a frame-b3s schema name is at most {B3S_MAX_SCHEMA} bytes")
    return _blake3.keyed(domain_id, b"\x05" + rank.to_bytes(8, "little") + position.to_bytes(8, "little")
                         + length.to_bytes(8, "little") + segment.to_bytes(4, "little") + len(name).to_bytes(2, "little") + name)


@lru_cache(maxsize=4096)
def b3s_level_key(domain_id: bytes, level: int) -> bytes:
    """The key of a ``frame-b3s`` segment tree's level: ``keyed(domain id, 0x06 ‖ u8 level)``."""

    return _blake3.keyed(domain_id, b"\x06" + level.to_bytes(1, "little"))


def b3s_segments(length: int) -> int:
    return max(1, -(-length // B3S_SEGMENT))


def b3s_levels(domain_id: bytes, rank: int, position: int, schema: str, value: bytes) -> tuple[tuple[bytes, ...], ...]:
    """A ``frame-b3s`` leaf's segment tree, level by level: the segment hashes first, the leaf last (module docstring)."""

    length = len(value)
    level = [_blake3.keyed(b3s_segment_key(domain_id, rank, position, schema, length, j),
                           value[j * B3S_SEGMENT:(j + 1) * B3S_SEGMENT]) for j in range(b3s_segments(length))]
    levels = [tuple(level)]
    depth = 0
    while len(level) > 1:
        key = b3s_level_key(domain_id, depth)
        up = [_blake3.keyed(key, level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            up.append(level[-1])
        level = up
        levels.append(tuple(level))
        depth += 1
    return tuple(levels)


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
    hash: str = "sha256"
    domain_id: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.binding) is not bytes or len(self.binding) not in (32, 64):
            raise ProtocolError("binding must be 64 bytes (a SHA-512 identity) or 32 (a SHA-256 one)")
        if type(self.owner) is not int or self.owner < -2:
            raise ProtocolError(
                "owner must be -2 (weights), -1 (boundary) or a replay unit"
            )
        if self.hash not in _BLAKE3_FRAMES:
            sha(self.hash)
        if isinstance(self.positions, RangeU64Domain):
            if self.hash != "sha512":
                raise ProtocolError(f"a range-u64 domain hashes with SHA-512 only, not {self.hash!r}")
            message = range_u64_domain_message(self.binding, self.owner, self.positions.start, self.positions.count)
            object.__setattr__(self, "domain_id", _digest(message, "sha512"))
            return
        if self.hash in ("sha256", *_BLAKE3_FRAMES):
            positions = bytes.fromhex(self.positions.identity_digest)
        elif hasattr(self.positions, "identity_sha512"):
            positions = self.positions.identity_sha512()
        else:
            raise ProtocolError("a SHA-512 frame needs positions with a SHA-512 identity (identity_sha512)")
        object.__setattr__(
            self,
            "domain_id",
            _digest(domain_message(self.binding, self.owner, positions, self.positions.count, self.hash), self.hash),
        )

    @property
    def count(self) -> int:
        return self.positions.count

    @property
    def digest_bytes(self) -> int:
        return DIGEST_BYTES[self.hash]

    def leaf(self, rank: int, position: int, schema: str, value: bytes) -> bytes:
        if self.hash == "blake3-s256":
            return b3s_levels(self.domain_id, rank, position, schema, value)[-1][0]
        if self.hash == "blake3":
            return _blake3.keyed(self.domain_id, self.leaf_prefix(rank, position, schema, len(value)) + value)
        return _hash(
            _LEAF,
            self.domain_id,
            _uint(rank),
            _uint(position),
            schema.encode("utf-8"),
            value,
            hash=self.hash,
        )

    def leaf_prefix(self, rank: int, position: int, schema: str, value_len: int) -> bytes:
        """Every byte of :meth:`leaf`'s message before the value: ``leaf == H(prefix || value)``, H the domain's hash
        (for ``frame-b3``, BLAKE3 keyed by the domain id).  A ``frame-b3s`` leaf is a segment tree and has no prefix."""

        if self.hash == "blake3-s256":
            raise ProtocolError("a frame-b3s leaf is a tree over its value's segments, not a hash of a prefix and the value")
        if self.hash == "blake3":
            name = schema.encode("utf-8")
            head = b"\x01" + rank.to_bytes(8, "little") + position.to_bytes(8, "little") + len(name).to_bytes(2, "little") + name
            return head + bytes(-len(head) % 64)
        parts = (self.domain_id, _uint(rank), _uint(position), schema.encode("utf-8"))
        out = _FRAME + len(_LEAF).to_bytes(4, "big") + _LEAF
        for part in parts:
            out += len(part).to_bytes(8, "big") + part
        return out + value_len.to_bytes(8, "big")

    def node(self, level: int, index: int, left: bytes, right: bytes) -> bytes:
        if self.hash in _BLAKE3_FRAMES:
            return blake3_node(self.domain_id, level, index, left, right)
        return _hash(_NODE, self.domain_id, _uint(level), _uint(index), left, right, hash=self.hash)

    def pad(self, rank: int) -> bytes:
        if self.hash in _BLAKE3_FRAMES:
            return _blake3.keyed(self.domain_id, b"\x03" + rank.to_bytes(8, "little"))
        return _hash(_PAD, self.domain_id, _uint(rank), hash=self.hash)

    def empty_root(self) -> bytes:
        if self.hash in _BLAKE3_FRAMES:
            return _blake3.keyed(self.domain_id, b"\x04")
        return _hash(_EMPTY, self.domain_id, hash=self.hash)


def tree_levels(domain: CommitmentDomain, leaves: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    """The levels over one domain's leaf digests (in rank order), padded to a power of two; ``()`` for none."""

    if len(leaves) != domain.count:
        raise ProtocolError(f"expected {domain.count} leaves, got {len(leaves)}")
    if not leaves:
        return ()
    width = 1 << merkle_depth(len(leaves))
    level = [*leaves, *(domain.pad(rank) for rank in range(len(leaves), width))]
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
