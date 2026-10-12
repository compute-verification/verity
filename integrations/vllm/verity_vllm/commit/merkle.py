"""Merkle trees over canonical ids, openings, ``commitments.json`` (contract section 5).

Hash layout (also in :mod:`verity_vllm.commit.hashing`; specified as ``vllm-v1`` in
``verity/ml/commitments/vllm_v1/PROTOCOL.md`` §1–§3)::

    H(tag, *parts) = sha256( u32be(len(tag)) || tag || (u64be(len(part)) || part)* )
    leaf  = H("verity-vllm/leaf/v1", canonical_id, dtype, "[d0,d1,...]", raw little-endian bytes)
    node  = H("verity-vllm/node/v1", left, right)
    lift  = H("verity-vllm/lift/v1", child)          # odd trailing node; nothing is duplicated
    ids   = H("verity-vllm/ids/v1", id_0, ..., id_{n-1})  # sorted ids, one part each
    empty = H("verity-vllm/empty/v1")                # root of a tree with no leaves

Tree: level 0 is the leaf hashes in tree order (code-point order of the canonical id).  While a
level has more than one node, neighbours ``(0,1), (2,3), ...`` become ``node(left, right)`` and an
odd trailing node becomes ``lift(child)``.  An opening carries the value (id, dtype, shape,
bytes), its index, and for every level either the sibling digest or ``None`` where the node was
lifted.  :func:`verify_opening` recomputes the leaf from the opening's own bytes, derives the
expected sibling/lift pattern from ``(leaf_count, index)`` and rejects a path of another shape.

Three tree kinds, ids as in :mod:`verity_vllm.observe.capture_v1`: ``ru/<request_id>/<step:06d>`` (one per
replay unit), ``boundary/<request_id>`` (prompt ids, output ids, presence, sampling params, seed,
finish reason) and ``weights``.
"""

from __future__ import annotations

import base64
import dataclasses
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.ml.commitments.leaves import fold_path
from verity.errors import InvalidArtifact
from verity_vllm.commit import hashing, identity, scheme
from verity_vllm.observe.capture_v1 import COMMITMENTS_FILE, WEIGHTS_TREE_ID, CaptureError, CaptureReader, Value, _write_json, boundary_tree_id, ru_tree_id

COMMITMENTS_SCHEMA = "verity-vllm/commitments/v1"


class CommitError(ValueError):
    pass


def path_shape(leaf_count: int, index: int) -> list[bool]:
    """For each level from the leaves up: ``True`` when the node at that level has a sibling,
    ``False`` when it is the odd trailing node that gets lifted."""
    try:
        return scheme.path_shape(leaf_count, index)
    except InvalidArtifact as error:
        raise CommitError(str(error)) from error


def tree_depth(leaf_count: int) -> int:
    return len(path_shape(leaf_count, 0)) if leaf_count > 0 else 0


@dataclass(frozen=True, slots=True)
class Opening:
    """A value plus its authentication path in one tree."""

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
        return hashing.leaf_hash(self.canonical_id, self.dtype, self.shape, self.value)

    def replace(self, **changes: Any) -> Opening:
        """A copy with fields changed and nothing recomputed (what a forger would present)."""
        return dataclasses.replace(self, **changes)

    def as_value(self, origin: str = "captured") -> Value:
        return Value(self.canonical_id, self.dtype, self.shape, self.value, origin)

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
    def from_json(cls, doc: Mapping[str, Any]) -> Opening:
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
            raise CommitError(f"malformed opening: {error}") from error


def verify_opening(root: bytes, leaf_count: int, opening: Opening) -> bool:
    """Authenticate ``opening`` against ``root``: identity, dtype, shape and bytes all bind.

    Returns ``False`` (never raises) for a malformed opening, a path of the wrong shape, an
    index outside the tree, or a root mismatch.  The opening's own ``root``/``leaf_count`` fields
    must agree with the arguments so a caller cannot be tricked by a self-describing opening.
    """
    try:
        if opening.root != root or opening.leaf_count != leaf_count:
            return False
        if not isinstance(root, bytes) or len(root) != hashing.DIGEST_BYTES:
            return False
        identity.parse(opening.canonical_id)
        if leaf_count <= 0:
            return False
        expected = path_shape(leaf_count, opening.index)
        if len(opening.path) != len(expected):
            return False
        for sibling, has_sibling in zip(opening.path, expected, strict=True):
            if has_sibling and (sibling is None or len(sibling) != hashing.DIGEST_BYTES):
                return False
            if not has_sibling and sibling is not None:
                return False
        return fold_path(opening.leaf(), opening.index, opening.path) == root
    except (ValueError, TypeError, identity.IdentityError, CommitError):
        return False


@dataclass(frozen=True, slots=True)
class RangeOpening:
    """Capture format v2: a contiguous run of leaves ``[lo, hi)`` of one tree with the boundary
    siblings needed to fold them to the root.  Used to authenticate a whole weight tensor whose
    leaves are its row blocks (``w=<param>/c=<row>``): the verifier rehashes every block leaf,
    folds the run level by level and compares the root, so the cost is one hash per leaf plus
    two siblings per level instead of one full path per leaf.

    ``leaves`` are ``(canonical_id, dtype, shape, bytes)`` in tree order; ``path[level]`` is
    ``(left_sibling | None, right_sibling | None)``.  ``canonical_id``/``dtype``/``shape``/``value``
    describe the aggregate (the whole tensor) so a checker can use it like an :class:`Opening`.
    """

    tree: str
    root: bytes
    leaf_count: int
    lo: int
    hi: int
    canonical_id: str
    dtype: str
    shape: tuple[int, ...]
    value: bytes
    leaves: tuple[tuple[str, str, tuple[int, ...], bytes], ...]
    path: tuple[tuple[bytes | None, bytes | None], ...]

    @property
    def index(self) -> int:
        return self.lo

    def as_value(self, origin: str = "captured") -> Value:
        return Value(self.canonical_id, self.dtype, self.shape, self.value, origin)

    def replace(self, **changes: Any) -> RangeOpening:
        return dataclasses.replace(self, **changes)

    def path_bytes(self) -> int:
        return sum(len(s) for pair in self.path for s in pair if s is not None)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "range",
            "tree": self.tree,
            "root": self.root.hex(),
            "leaf_count": self.leaf_count,
            "lo": self.lo,
            "hi": self.hi,
            "canonical_id": self.canonical_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "value_b64": base64.b64encode(self.value).decode("ascii"),
            "leaves": [
                {"canonical_id": c, "dtype": d, "shape": list(s), "nbytes": len(b)}
                for c, d, s, b in self.leaves
            ],
            "path": [[None if s is None else s.hex() for s in pair] for pair in self.path],
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> RangeOpening:
        """Inverse of :meth:`to_json`; each leaf's bytes are the slice of ``value`` at its row
        coordinate (leaves are listed in tree order, which is not row order)."""
        try:
            value = base64.b64decode(doc["value_b64"])
            shape = tuple(int(d) for d in doc["shape"])
            if not shape or shape[0] <= 0 or len(value) % shape[0] != 0:
                raise ValueError("aggregate shape does not divide the value")
            row_nbytes = len(value) // shape[0]
            leaves: list[tuple[str, str, tuple[int, ...], bytes]] = []
            total = 0
            for leaf in doc["leaves"]:
                cid = str(leaf["canonical_id"])
                r = int(identity.parse(cid).coordinate or 0)
                n = int(leaf["nbytes"])
                leaves.append((cid, str(leaf["dtype"]), tuple(int(d) for d in leaf["shape"]), value[r * row_nbytes : r * row_nbytes + n]))
                total += n
            if total != len(value):
                raise ValueError("leaf sizes do not cover the value")
            return cls(
                tree=str(doc["tree"]),
                root=bytes.fromhex(doc["root"]),
                leaf_count=int(doc["leaf_count"]),
                lo=int(doc["lo"]),
                hi=int(doc["hi"]),
                canonical_id=str(doc["canonical_id"]),
                dtype=str(doc["dtype"]),
                shape=tuple(int(d) for d in doc["shape"]),
                value=value,
                leaves=tuple(leaves),
                path=tuple(
                    (None if pair[0] is None else bytes.fromhex(pair[0]), None if pair[1] is None else bytes.fromhex(pair[1]))
                    for pair in doc["path"]
                ),
            )
        except (KeyError, TypeError, ValueError, IndexError, identity.IdentityError) as error:
            raise CommitError(f"malformed range opening: {error}") from error


def opening_from_json(doc: Mapping[str, Any]) -> Opening | RangeOpening:
    return RangeOpening.from_json(doc) if doc.get("kind") == "range" else Opening.from_json(doc)


def leaves_tile_value(
    leaves: Sequence[tuple[str, str, tuple[int, ...], bytes]], shape: tuple[int, ...], value: bytes
) -> bool:
    """Do the row-block leaves tile the aggregate tensor exactly?  Tree order is code-point order
    (``c=10`` before ``c=2``), so each leaf is placed by its row coordinate: leaf ``w=P/c=r`` with
    block shape ``(k, *rest)`` must equal ``value[r*row : (r+k)*row]`` and the blocks must cover
    ``[0, shape[0])`` without gaps or overlaps."""
    if not shape or shape[0] <= 0 or len(value) % shape[0] != 0:
        return False
    row_nbytes = len(value) // shape[0]
    placed: list[tuple[int, int]] = []
    for cid, _dtype, block_shape, data in leaves:
        parsed = identity.parse(cid)
        if parsed.kind != "weight" or parsed.coordinate is None:
            return False
        r = int(parsed.coordinate)
        if not block_shape or tuple(block_shape[1:]) != tuple(shape[1:]):
            return False
        k = int(block_shape[0])
        if k <= 0 or len(data) != k * row_nbytes or r + k > shape[0]:
            return False
        if value[r * row_nbytes : (r + k) * row_nbytes] != data:
            return False
        placed.append((r, r + k))
    placed.sort()
    cursor = 0
    for start, end in placed:
        if start != cursor:
            return False
        cursor = end
    return cursor == shape[0]


def range_path_shape(leaf_count: int, lo: int, hi: int) -> list[tuple[bool, bool]]:
    """Per level: does the run ``[lo, hi)`` need a left sibling, a right sibling?"""
    if not (0 <= lo < hi <= leaf_count):
        raise CommitError(f"range [{lo}, {hi}) out of range for {leaf_count} leaves")
    shape: list[tuple[bool, bool]] = []
    width = leaf_count
    while width > 1:
        left = lo % 2 == 1
        last = hi - 1
        right = last % 2 == 0 and not (last == width - 1 and width % 2 == 1)
        shape.append((left, right))
        # next level: the run covers parents floor(lo'/2) .. ceil(hi'/2) with lo' = lo - left,
        # hi' = hi + right; (hi + 1) // 2 equals ceil(hi'/2) in every case (hi' is even when a
        # right sibling was added; a lifted odd tail maps to (width - 1) // 2)
        lo, hi, width = lo // 2, (hi + 1) // 2, (width + 1) // 2
    return shape


def fold_range(
    leaves: Sequence[bytes], leaf_count: int, lo: int, path: Sequence[tuple[bytes | None, bytes | None]]
) -> bytes:
    """Fold a run of leaf hashes starting at index ``lo`` to the root along ``path``."""
    nodes = list(leaves)
    width = leaf_count
    for left, right in path:
        if width <= 1:
            raise CommitError("range path longer than the tree")
        if left is not None:
            nodes.insert(0, left)
            lo -= 1
        if right is not None:
            nodes.append(right)
        if lo % 2 == 1:
            raise CommitError("range does not start on a pair boundary")
        nxt: list[bytes] = []
        for i in range(0, len(nodes) - 1, 2):
            nxt.append(hashing.node_hash(nodes[i], nodes[i + 1]))
        if len(nodes) % 2 == 1:
            if lo + len(nodes) - 1 != width - 1 or width % 2 == 0:
                raise CommitError("odd node in the middle of a level")
            nxt.append(hashing.lift_hash(nodes[-1]))
        nodes = nxt
        lo //= 2
        width = (width + 1) // 2
    if width != 1 or len(nodes) != 1:
        raise CommitError("range path shorter than the tree")
    return nodes[0]


def verify_range_opening(root: bytes, leaf_count: int, opening: RangeOpening) -> bool:
    """Authenticate a :class:`RangeOpening`: every leaf is rehashed from its own id, dtype, shape
    and bytes; the aggregate value must be exactly the concatenation of the leaf bytes; the
    sibling pattern is derived from ``(leaf_count, lo, hi)`` and the fold must reach ``root``."""
    try:
        if opening.root != root or opening.leaf_count != leaf_count:
            return False
        if not isinstance(root, bytes) or len(root) != hashing.DIGEST_BYTES:
            return False
        if opening.hi - opening.lo != len(opening.leaves) or not opening.leaves:
            return False
        identity.parse(opening.canonical_id)
        if not leaves_tile_value(opening.leaves, opening.shape, opening.value):
            return False
        ids = [c for c, _, _, _ in opening.leaves]
        if ids != identity.tree_sorted(ids):
            return False
        expected = range_path_shape(leaf_count, opening.lo, opening.hi)
        if len(opening.path) != len(expected):
            return False
        for (left, right), (need_left, need_right) in zip(opening.path, expected, strict=True):
            if need_left != (left is not None) or need_right != (right is not None):
                return False
            for s in (left, right):
                if s is not None and len(s) != hashing.DIGEST_BYTES:
                    return False
        hashes = [hashing.leaf_hash(c, d, s, b) for c, d, s, b in opening.leaves]
        return fold_range(hashes, leaf_count, opening.lo, opening.path) == root
    except (ValueError, TypeError, identity.IdentityError, CommitError):
        return False


class MerkleTree:
    """A tree over ``canonical_id -> leaf hash``; leaves are sorted into tree order internally."""

    __slots__ = ("_index", "ids", "leaf_count", "leaves", "levels", "tree_id")

    def __init__(self, leaves: Mapping[str, bytes], tree_id: str = "") -> None:
        ids = identity.tree_sorted(list(leaves))
        for cid in ids:
            identity.parse(cid)
            if len(leaves[cid]) != hashing.DIGEST_BYTES:
                raise CommitError(f"{cid}: leaf hash must be 32 bytes")
        self.tree_id = tree_id
        self.ids = ids
        self.leaves = {cid: bytes(leaves[cid]) for cid in ids}
        self.leaf_count = len(ids)
        self._index = {cid: i for i, cid in enumerate(ids)}
        self.levels = scheme.fold_levels([self.leaves[cid] for cid in ids])

    @classmethod
    def from_values(cls, values: Iterable[Value], tree_id: str = "") -> MerkleTree:
        leaves: dict[str, bytes] = {}
        for value in values:
            if value.canonical_id in leaves:
                raise CommitError(f"{value.canonical_id}: duplicate leaf")
            leaves[value.canonical_id] = value.leaf()
        return cls(leaves, tree_id)

    @property
    def root(self) -> bytes:
        if self.leaf_count == 0:
            return hashing.empty_root()
        return self.levels[-1][0]

    @property
    def ids_digest(self) -> bytes:
        return hashing.ids_digest(self.ids)

    @property
    def depth(self) -> int:
        return len(self.levels) - 1 if self.leaf_count else 0

    def __contains__(self, canonical_id: str) -> bool:
        return canonical_id in self._index

    def index_of(self, canonical_id: str) -> int:
        try:
            return self._index[canonical_id]
        except KeyError as error:
            name = self.tree_id or "this tree"
            raise KeyError(f"{canonical_id} is not a leaf of {name}") from error

    def path(self, canonical_id: str) -> tuple[bytes | None, ...]:
        cursor = self.index_of(canonical_id)
        out: list[bytes | None] = []
        for level in self.levels[:-1]:
            if cursor == len(level) - 1 and len(level) % 2 == 1:
                out.append(None)
            else:
                out.append(level[cursor ^ 1])
            cursor //= 2
        return tuple(out)

    def open(self, value: Value) -> Opening:
        """An opening for ``value``; refuses a value whose bytes do not hash to the stored leaf."""
        if value.leaf() != self.leaves.get(value.canonical_id):
            raise CommitError(f"{value.canonical_id}: value does not match the committed leaf")
        return Opening(
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

    def range_path(self, lo: int, hi: int) -> tuple[tuple[bytes | None, bytes | None], ...]:
        """Boundary siblings for the run ``[lo, hi)`` (see :class:`RangeOpening`)."""
        out: list[tuple[bytes | None, bytes | None]] = []
        for (need_left, need_right), level in zip(
            range_path_shape(self.leaf_count, lo, hi), self.levels[:-1], strict=True
        ):
            left = level[lo - 1] if need_left else None
            right = level[hi] if need_right else None
            out.append((left, right))
            lo, hi = (lo - int(need_left)) // 2, (hi + 1) // 2
        return tuple(out)

    def open_range(self, values: Sequence[Value], aggregate: Value) -> RangeOpening:
        """A :class:`RangeOpening` over ``values`` (contiguous in tree order) whose concatenated
        bytes are ``aggregate`` (the whole tensor); refuses non-matching leaves or gaps."""
        if not values:
            raise CommitError("empty range")
        ids = [v.canonical_id for v in values]
        if ids != identity.tree_sorted(ids):
            raise CommitError("range values must be in tree order")
        lo = self.index_of(ids[0])
        hi = self.index_of(ids[-1]) + 1
        if hi - lo != len(ids):
            raise CommitError("range values are not contiguous in the tree")
        for v in values:
            if v.leaf() != self.leaves.get(v.canonical_id):
                raise CommitError(f"{v.canonical_id}: value does not match the committed leaf")
        leaves = tuple((v.canonical_id, v.dtype, v.shape, v.data) for v in values)
        if not leaves_tile_value(leaves, aggregate.shape, aggregate.data):
            raise CommitError(f"{aggregate.canonical_id}: row blocks do not tile the aggregate bytes")
        return RangeOpening(
            tree=self.tree_id,
            root=self.root,
            leaf_count=self.leaf_count,
            lo=lo,
            hi=hi,
            canonical_id=aggregate.canonical_id,
            dtype=aggregate.dtype,
            shape=aggregate.shape,
            value=aggregate.data,
            leaves=leaves,
            path=self.range_path(lo, hi),
        )

    def record(self, kind: str, **extra: Any) -> dict[str, Any]:
        doc = {
            "kind": kind,
            "root": self.root.hex(),
            "leaf_count": self.leaf_count,
            "ids_digest": self.ids_digest.hex(),
        }
        doc.update(extra)
        return doc


# --- trees of a capture --------------------------------------------------------------------------


def ru_tree(
    reader: CaptureReader, request_id: str, step: int, *, recompute: bool = True
) -> MerkleTree:
    ru = reader.ru(request_id, step)
    if ru.index.tainted:
        raise CommitError(f"{ru.tree_id} is tainted (conflicting captures); refusing to commit")
    return MerkleTree(ru.leaves(recompute=recompute), ru.tree_id)


def boundary_tree(reader: CaptureReader, request_id: str) -> MerkleTree:
    boundary = reader.boundary(request_id)
    problems = boundary.validate()
    if problems:
        raise CommitError(f"boundary {request_id}: " + "; ".join(problems))
    return MerkleTree.from_values(boundary.leaves(reader.profile_id), boundary_tree_id(request_id))


def weights_tree(reader: CaptureReader, *, recompute: bool = False) -> MerkleTree:
    """The weights tree: whole-tensor leaves (format v1) and/or row-block leaves (v2).
    ``recompute`` rehashes every leaf from the stored bytes instead of trusting the index."""
    return MerkleTree(reader.weights.leaves(recompute=recompute), WEIGHTS_TREE_ID)


class TreeCache:
    """Builds and memoises the trees of one capture; ``open(canonical_id)`` for any id kind."""

    def __init__(self, reader: CaptureReader, *, recompute_leaves: bool = True) -> None:
        self.reader = reader
        self.recompute_leaves = recompute_leaves
        self._trees: dict[str, MerkleTree] = {}

    def tree(self, tree_id: str) -> MerkleTree:
        if tree_id not in self._trees:
            self._trees[tree_id] = self._build(tree_id)
        return self._trees[tree_id]

    def _build(self, tree_id: str) -> MerkleTree:
        if tree_id == WEIGHTS_TREE_ID:
            return weights_tree(self.reader, recompute=self.recompute_leaves)
        parts = tree_id.split("/")
        if parts[0] == "boundary" and len(parts) == 2:
            return boundary_tree(self.reader, parts[1])
        if parts[0] == "ru" and len(parts) == 3:
            return ru_tree(self.reader, parts[1], int(parts[2]), recompute=self.recompute_leaves)
        raise CommitError(f"unknown tree id {tree_id!r}")

    def tree_of(self, canonical_id: str) -> MerkleTree:
        tree_id = self.reader.tree_of(canonical_id)
        if tree_id is None:
            raise KeyError(f"{canonical_id} is not committed anywhere in this capture")
        return self.tree(tree_id)

    def open(self, canonical_id: str) -> Opening | RangeOpening:
        """An :class:`Opening` for a tree leaf; for a whole weight tensor committed as row
        blocks (format v2) a :class:`RangeOpening` over its blocks."""
        tree = self.tree_of(canonical_id)
        if tree.tree_id == WEIGHTS_TREE_ID and not self.reader.weights.is_leaf(canonical_id):
            blocks = self.reader.weights.block_values(canonical_id)
            return tree.open_range(blocks, self.reader.weights.value(canonical_id))
        return tree.open(self.reader.value(canonical_id))

    def tree_ids(self) -> list[str]:
        ids = [ru_tree_id(r, s) for r, s in self.reader.ru_keys()]
        for request_id in self.reader.request_ids():
            if self.reader.has_boundary(request_id):
                ids.append(boundary_tree_id(request_id))
        if self.reader.has_weights():
            ids.append(WEIGHTS_TREE_ID)
        return ids


# --- commitments.json ----------------------------------------------------------------------------


def build_commitments(
    run_dir: str | os.PathLike[str] | CaptureReader,
    *,
    recompute_leaves: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Build every tree of a capture and write ``commitments.json`` (returned as a dict)."""
    reader = run_dir if isinstance(run_dir, CaptureReader) else CaptureReader(run_dir)
    cache = TreeCache(reader, recompute_leaves=recompute_leaves)
    trees: dict[str, Any] = {}
    for request_id, step in reader.ru_keys():
        ru = reader.ru(request_id, step)
        tree = cache.tree(ru.tree_id)
        trees[ru.tree_id] = tree.record(
            "ru",
            request_id=request_id,
            step=step,
            phase=ru.index.phase,
            positions=list(ru.index.positions),
        )
    for request_id in reader.request_ids():
        if reader.has_boundary(request_id):
            tree = cache.tree(boundary_tree_id(request_id))
            trees[tree.tree_id] = tree.record("boundary", request_id=request_id)
    if reader.has_weights():
        tree = cache.tree(WEIGHTS_TREE_ID)
        trees[WEIGHTS_TREE_ID] = tree.record(
            "weights", weights_sha256=dict(reader.weights.index.source_sha256)
        )
    doc = {
        "schema": COMMITMENTS_SCHEMA,
        "profile_id": reader.profile_id,
        "hash": {
            "frame": hashing.FRAME_DESCRIPTION,
            "leaf": hashing.LEAF_TAG,
            "node": hashing.NODE_TAG,
            "lift": hashing.LIFT_TAG,
            "ids": hashing.IDS_TAG,
            "empty": hashing.EMPTY_TAG,
        },
        "trees": trees,
    }
    if write:
        _write_json(reader.run_dir / COMMITMENTS_FILE, doc)
    return doc


def load_commitments(run_dir: str | os.PathLike[str] | CaptureReader) -> dict[str, Any]:
    reader = run_dir if isinstance(run_dir, CaptureReader) else CaptureReader(run_dir)
    doc = reader.commitments
    if doc is None:
        raise CommitError(f"{reader.run_dir / COMMITMENTS_FILE} does not exist")
    if doc.get("schema") != COMMITMENTS_SCHEMA:
        raise CommitError(f"unexpected commitments schema {doc.get('schema')!r}")
    return doc


def tree_record(commitments: Mapping[str, Any], tree_id: str) -> dict[str, Any]:
    try:
        return commitments["trees"][tree_id]
    except KeyError as error:
        raise CommitError(f"{tree_id} is not in commitments.json") from error


def root_of(commitments: Mapping[str, Any], tree_id: str) -> tuple[bytes, int]:
    record = tree_record(commitments, tree_id)
    return bytes.fromhex(record["root"]), int(record["leaf_count"])


def verify_opening_in(commitments: Mapping[str, Any], opening: Opening | RangeOpening) -> bool:
    """Authenticate an opening (single leaf or range) against the root recorded for ``opening.tree``."""
    try:
        root, count = root_of(commitments, opening.tree)
    except (CommitError, KeyError, ValueError):
        return False
    if isinstance(opening, RangeOpening):
        return verify_range_opening(root, count, opening)
    return verify_opening(root, count, opening)


def verify_commitments(reader: CaptureReader, commitments: Mapping[str, Any]) -> list[str]:
    """Rebuild every tree from the capture's bytes and compare with ``commitments.json``."""
    problems: list[str] = []
    if commitments.get("schema") != COMMITMENTS_SCHEMA:
        problems.append("commitments schema")
    if commitments.get("profile_id") != reader.profile_id:
        problems.append("commitments.profile_id != manifest profile id")
    cache = TreeCache(reader, recompute_leaves=True)
    recorded = set(commitments.get("trees", {}))
    actual = set(cache.tree_ids())
    for tree_id in sorted(actual - recorded):
        problems.append(f"{tree_id}: present in capture, absent from commitments")
    for tree_id in sorted(recorded - actual):
        problems.append(f"{tree_id}: in commitments, absent from capture")
    for tree_id in sorted(recorded & actual):
        record = commitments["trees"][tree_id]
        try:
            tree = cache.tree(tree_id)
        except (CommitError, CaptureError) as error:
            problems.append(f"{tree_id}: {error}")
            continue
        if tree.root.hex() != record.get("root"):
            problems.append(f"{tree_id}: root mismatch")
        if tree.leaf_count != record.get("leaf_count"):
            problems.append(f"{tree_id}: leaf_count mismatch")
        if tree.ids_digest.hex() != record.get("ids_digest"):
            problems.append(f"{tree_id}: ids_digest mismatch")
    return problems


def verify_source_linkage(reader: CaptureReader, commitments: Mapping[str, Any]) -> list[str]:
    """N06: the weights tree must be bound to the manifest's ``weights_sha256`` and every stored
    weight's bytes to its recorded sha256 and leaf."""
    problems: list[str] = []
    if not reader.has_weights():
        return ["weights index missing"]
    manifest_shas = dict(reader.manifest.get("model", {}).get("weights_sha256", {}))
    index = reader.weights.index
    if dict(index.source_sha256) != manifest_shas:
        problems.append("weights/index.json source_sha256 != manifest.model.weights_sha256")
    try:
        record = tree_record(commitments, WEIGHTS_TREE_ID)
    except CommitError:
        return problems + ["weights tree missing from commitments"]
    if dict(record.get("weights_sha256", {})) != manifest_shas:
        problems.append("commitments weights.weights_sha256 != manifest.model.weights_sha256")
    if index.store_values:
        values_path = reader.run_dir / "weights" / "values.bin"
        if values_path.exists():
            stored_sha = hashing.sha256_hex(values_path.read_bytes())
            declared = manifest_shas.get("weights/values.bin")
            if declared is not None and declared != stored_sha:
                problems.append("weights/values.bin sha256 != manifest.model.weights_sha256")
    problems.extend(f"weights: {p}" for p in reader.weights.validate(check_bytes=True))
    tree = weights_tree(reader)
    if tree.root.hex() != record.get("root"):
        problems.append("weights root != tree over weights/index.json leaves")
    return problems


def write_openings(path: str | os.PathLike[str], openings: Iterable[Opening]) -> Path:
    path = Path(path)
    _write_json(path, [o.to_json() for o in openings])
    return path
