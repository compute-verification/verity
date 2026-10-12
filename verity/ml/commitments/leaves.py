"""Committing runtime tensor values as whole leaves: the ``verity-vllm/leaf/v1`` scheme.

This is the Verity protocol's application-level commitment to runtime values (README "Commitments"): a
*leaf* is one whole tensor row or tensor -- ``(canonical_id, dtype, shape, raw little-endian bytes)`` -- and a
*tree* commits a set of leaves in code-point order of their ids.  A verification obligation then names words
*inside* a leaf (``packed/v1:verity-vllm/leaf/v1:u16le:<count>:<index>``, :mod:`verity_sp1.proofs.schema`),
and a checker authenticates the whole leaf against the committed root before reading the word at the
statement-fixed index.  The SP1 guest re-derives exactly this chain (``archive/sp1/common/src/packed.rs``).

Hash layout, all SHA-256 over one stream with the tag as the only domain separator::

    H(tag, *parts) = sha256( u32be(len(tag)) || tag || (u64be(len(part)) || part)* )
    leaf  = H("verity-vllm/leaf/v1", canonical_id, dtype, "[d0,d1,...]", value)
    node  = H("verity-vllm/node/v1", left, right)
    lift  = H("verity-vllm/lift/v1", child)               # an odd trailing node, never duplicated
    ids   = H("verity-vllm/ids/v1", id_0, ..., id_{n-1})   # the sorted ids, one part each: the tree's domain
    empty = H("verity-vllm/empty/v1")                      # the root of a tree with no leaves

Level 0 is the leaf hashes in tree order; neighbours ``(0,1), (2,3), ...`` become ``node(left, right)`` and an
odd trailing node becomes ``lift(child)``.  An opening carries the leaf's parts, its index, and per level either
the sibling digest or ``None`` where the node was lifted; the sibling/lift pattern is a function of
``(leaf_count, index)`` (:func:`path_shape`) and a path of another shape is rejected before any hash is compared.

*Shape text* is ``"[" + ",".join(str(d)) + "]"`` with no spaces (a scalar is ``"[]"``); *dtype* is a dtype name
such as ``"bfloat16"``; *value* is the exact little-endian bytes in C order, hashed without conversion.

Provenance: the generic core of veritor ``vllm-poc/verity_vllm/commit/hashing.py`` and ``commit.py`` (eb865ce2), which
``veritor.protocol.proofs.packed`` mirrored.  Two things were left behind on purpose.  The tag strings say
``verity-vllm`` because the scheme was born in the serving integration; the *scheme* is not vLLM-specific (any
id, dtype, shape, bytes) and the tag bytes are pinned by the guest, every conformance vector and every recorded
root, so they stay.  Track A also validated every id against the vLLM canonical-id grammar before hashing; a
commitment binds whatever id string it is given, so that check belongs to the integration and is not here.
The range openings of the v2 capture format are not carried yet (nothing here opens ranges).
"""

from __future__ import annotations

import base64
import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.errors import InvalidArtifact

from .scheme import DIGEST_BYTES as _DIGEST_BYTES
from .scheme import sha as _sha

__all__ = [
    "DIGEST_BYTES",
    "EMPTY_TAG",
    "IDS_TAG",
    "LEAF_TAG",
    "LIFT_TAG",
    "NODE_TAG",
    "LeafOpening",
    "LeafTree",
    "LeafValue",
    "H",
    "empty_root",
    "fold_path",
    "ids_digest",
    "leaf_hash",
    "levels",
    "lift_hash",
    "node_hash",
    "open_path",
    "parse_shape_text",
    "path_shape",
    "shape_text",
    "tree_depth",
    "verify_opening",
]

LEAF_TAG = "verity-vllm/leaf/v1"
NODE_TAG = "verity-vllm/node/v1"
LIFT_TAG = "verity-vllm/lift/v1"
IDS_TAG = "verity-vllm/ids/v1"
EMPTY_TAG = "verity-vllm/empty/v1"

DIGEST_BYTES = 32


def H(tag: str, *parts: bytes, hash: str = "sha256") -> bytes:  # noqa: N802 - the contract names it H
    """SHA-256 over the tag and the length-prefixed parts (layout in the module docstring); ``hash="sha512"`` is the
    ``vllm-v1-sha512`` tree's form, the same bytes through SHA-512."""

    digest = _sha(hash)()
    encoded_tag = tag.encode("utf-8")
    digest.update(len(encoded_tag).to_bytes(4, "big"))
    digest.update(encoded_tag)
    for part in parts:
        if not isinstance(part, bytes | bytearray | memoryview):
            raise TypeError(f"hash parts must be bytes, got {type(part).__name__}")
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def shape_text(shape: Sequence[int]) -> str:
    """The canonical textual shape inside the leaf hash: ``[4,8]``, scalar ``[]``."""

    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise InvalidArtifact(f"shape dims must be non-negative ints, got {shape!r}")
        dims.append(str(dim))
    return "[" + ",".join(dims) + "]"


def parse_shape_text(text: str) -> tuple[int, ...]:
    if not (text.startswith("[") and text.endswith("]")):
        raise InvalidArtifact(f"malformed shape text {text!r}")
    inner = text[1:-1]
    if inner == "":
        return ()
    dims = []
    for piece in inner.split(","):
        if not piece.isdigit() or (len(piece) > 1 and piece[0] == "0"):
            raise InvalidArtifact(f"malformed shape text {text!r}")
        dims.append(int(piece))
    return tuple(dims)


def leaf_hash(canonical_id: str, dtype: str, shape: Sequence[int] | str, value: bytes) -> bytes:
    """``H("verity-vllm/leaf/v1", canonical_id, dtype, shape text, value)``; ``shape`` may already be the text."""

    text = shape if isinstance(shape, str) else shape_text(shape)
    return H(LEAF_TAG, canonical_id.encode("utf-8"), dtype.encode("utf-8"), text.encode("utf-8"), value)


def node_hash(left: bytes, right: bytes, hash: str = "sha256") -> bytes:
    _check_digest(left, "left", hash)
    _check_digest(right, "right", hash)
    return H(NODE_TAG, left, right, hash=hash)


def lift_hash(child: bytes, hash: str = "sha256") -> bytes:
    _check_digest(child, "child", hash)
    return H(LIFT_TAG, child, hash=hash)


def ids_digest(ids: Iterable[str]) -> bytes:
    """``H("verity-vllm/ids/v1", id_0, ..., id_{n-1})`` over the ids in tree (code-point) order: the tree's
    domain, so an opening cannot be replayed under another tree of the same width."""

    return H(IDS_TAG, *(cid.encode("utf-8") for cid in sorted(ids)))


def empty_root(hash: str = "sha256") -> bytes:
    return H(EMPTY_TAG, hash=hash)


def path_shape(leaf_count: int, index: int) -> list[bool]:
    """Per level from the leaves up: ``True`` where the node has a sibling, ``False`` where it is lifted."""

    if leaf_count <= 0 or index < 0 or index >= leaf_count:
        raise InvalidArtifact(f"index {index} out of range for {leaf_count} leaves")
    shape: list[bool] = []
    width, cursor = leaf_count, index
    while width > 1:
        shape.append(not (cursor == width - 1 and width % 2 == 1))
        width = (width + 1) // 2
        cursor //= 2
    return shape


def tree_depth(leaf_count: int) -> int:
    return len(path_shape(leaf_count, 0)) if leaf_count > 0 else 0


def fold_path(leaf: bytes, index: int, path: Sequence[bytes | None], hash: str = "sha256") -> bytes:
    """Walk from a leaf hash to the root: a sibling pairs by the cursor's parity, ``None`` lifts."""

    digest, cursor = leaf, index
    for sibling in path:
        if sibling is None:
            digest = lift_hash(digest, hash)
        elif cursor % 2 == 0:
            digest = node_hash(digest, sibling, hash)
        else:
            digest = node_hash(sibling, digest, hash)
        cursor //= 2
    return digest


def levels(leaf_hashes: Sequence[bytes], hash: str = "sha256") -> list[list[bytes]]:
    """The tree over ``leaf_hashes`` (already in tree order): level 0 up to the root."""

    if not leaf_hashes:
        return [[empty_root(hash)]]
    out = [list(leaf_hashes)]
    while len(out[-1]) > 1:
        level = out[-1]
        out.append(
            [node_hash(level[i], level[i + 1], hash) if i + 1 < len(level) else lift_hash(level[i], hash) for i in range(0, len(level), 2)]
        )
    return out


def open_path(tree_levels: Sequence[Sequence[bytes]], index: int) -> tuple[bytes | None, ...]:
    """The authentication path of leaf ``index`` in :func:`levels`' output."""

    path: list[bytes | None] = []
    cursor = index
    for level in tree_levels[:-1]:
        sibling = cursor ^ 1
        path.append(level[sibling] if sibling < len(level) else None)
        cursor //= 2
    return tuple(path)


@dataclass(frozen=True, slots=True)
class LeafValue:
    """One committed value: an id, a dtype name, a shape and the exact little-endian bytes."""

    canonical_id: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def leaf(self) -> bytes:
        return leaf_hash(self.canonical_id, self.dtype, self.shape, self.data)


@dataclass(frozen=True, slots=True)
class LeafOpening:
    """A leaf's parts plus its authentication path in one tree."""

    tree: str
    root: bytes
    leaf_count: int
    index: int
    canonical_id: str
    dtype: str
    shape: tuple[int, ...]
    value: bytes
    path: tuple[bytes | None, ...]

    def leaf(self) -> bytes:
        return leaf_hash(self.canonical_id, self.dtype, self.shape, self.value)

    def replace(self, **changes: Any) -> LeafOpening:
        """A copy with fields changed and nothing recomputed (what a forger would present)."""

        return dataclasses.replace(self, **changes)

    def to_json(self) -> dict[str, Any]:
        return {
            "tree": self.tree,
            "root": self.root.hex(),
            "leaf_count": self.leaf_count,
            "index": self.index,
            "canonical_id": self.canonical_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "value_b64": base64.b64encode(self.value).decode("ascii"),
            "path": [None if p is None else p.hex() for p in self.path],
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> LeafOpening:
        try:
            return cls(
                tree=str(doc["tree"]),
                root=bytes.fromhex(doc["root"]),
                leaf_count=int(doc["leaf_count"]),
                index=int(doc["index"]),
                canonical_id=str(doc["canonical_id"]),
                dtype=str(doc["dtype"]),
                shape=tuple(int(d) for d in doc["shape"]),
                value=base64.b64decode(doc["value_b64"]),
                path=tuple(None if p is None else bytes.fromhex(p) for p in doc["path"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidArtifact(f"malformed opening: {error}") from error


def verify_opening(root: bytes, leaf_count: int, opening: LeafOpening) -> bool:
    """Authenticate ``opening`` against ``root``: id, dtype, shape and bytes all bind.

    Returns ``False`` (never raises) for a malformed opening, a path of the wrong shape, an index outside the
    tree, or a root mismatch.  The opening's own ``root``/``leaf_count`` must agree with the arguments so a
    caller cannot be misled by a self-describing opening.
    """

    try:
        if opening.root != root or opening.leaf_count != leaf_count:
            return False
        if not isinstance(root, bytes) or len(root) != DIGEST_BYTES:
            return False
        if leaf_count <= 0:
            return False
        expected = path_shape(leaf_count, opening.index)
        if len(opening.path) != len(expected):
            return False
        for sibling, has_sibling in zip(opening.path, expected, strict=True):
            if has_sibling and (sibling is None or len(sibling) != DIGEST_BYTES):
                return False
            if not has_sibling and sibling is not None:
                return False
        return fold_path(opening.leaf(), opening.index, opening.path) == root
    except (ValueError, TypeError, InvalidArtifact):
        return False


class LeafTree:
    """A tree over ``canonical_id -> leaf hash``; leaves are sorted into tree order internally."""

    __slots__ = ("_index", "ids", "leaf_count", "leaves", "levels", "tree_id")

    def __init__(self, leaves: Mapping[str, bytes], tree_id: str = "") -> None:
        ids = sorted(leaves)
        for cid in ids:
            if len(leaves[cid]) != DIGEST_BYTES:
                raise InvalidArtifact(f"{cid}: leaf hash must be 32 bytes")
        self.tree_id = tree_id
        self.ids = ids
        self.leaves = {cid: bytes(leaves[cid]) for cid in ids}
        self.leaf_count = len(ids)
        self._index = {cid: i for i, cid in enumerate(ids)}
        self.levels = levels([self.leaves[cid] for cid in ids])

    @classmethod
    def from_values(cls, values: Iterable[LeafValue], tree_id: str = "") -> LeafTree:
        leaves: dict[str, bytes] = {}
        for value in values:
            if value.canonical_id in leaves:
                raise InvalidArtifact(f"{value.canonical_id}: duplicate leaf")
            leaves[value.canonical_id] = value.leaf()
        return cls(leaves, tree_id)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def ids_digest(self) -> bytes:
        return ids_digest(self.ids)

    @property
    def depth(self) -> int:
        return len(self.levels) - 1 if self.leaf_count else 0

    def __contains__(self, canonical_id: str) -> bool:
        return canonical_id in self._index

    def index_of(self, canonical_id: str) -> int:
        try:
            return self._index[canonical_id]
        except KeyError as error:
            raise KeyError(f"{canonical_id} is not a leaf of {self.tree_id or 'this tree'}") from error

    def path(self, canonical_id: str) -> tuple[bytes | None, ...]:
        return open_path(self.levels, self.index_of(canonical_id))

    def open(self, value: LeafValue) -> LeafOpening:
        """An opening for ``value``; refuses a value whose bytes do not hash to the stored leaf."""

        if value.leaf() != self.leaves.get(value.canonical_id):
            raise InvalidArtifact(f"{value.canonical_id}: value does not match the committed leaf")
        return LeafOpening(
            tree=self.tree_id,
            root=self.root,
            leaf_count=self.leaf_count,
            index=self.index_of(value.canonical_id),
            canonical_id=value.canonical_id,
            dtype=value.dtype,
            shape=value.shape,
            value=value.data,
            path=self.path(value.canonical_id),
        )

    def record(self, kind: str, **extra: Any) -> dict[str, Any]:
        """The ``commitments.json`` entry of this tree: root, leaf count and ids digest (hex)."""

        doc = {"kind": kind, "root": self.root.hex(), "leaf_count": self.leaf_count, "ids_digest": self.ids_digest.hex()}
        doc.update(extra)
        return doc


def _check_digest(value: bytes, name: str, hash: str = "sha256") -> None:
    n = _DIGEST_BYTES[hash]
    if not isinstance(value, bytes | bytearray | memoryview) or len(value) != n:
        raise InvalidArtifact(f"{name} must be a {n}-byte digest")
