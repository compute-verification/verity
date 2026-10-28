"""The authentication-path shapes and the range fold of the ``vllm-v1`` trees (contract section 5).

Hash layout (also in :mod:`verity_vllm.commit.hashing`; specified as ``vllm-v1`` in
``verity/ml/commitments/vllm_v1/PROTOCOL.md`` §1–§3)::

    H(tag, *parts) = sha256( u32be(len(tag)) || tag || (u64be(len(part)) || part)* )
    leaf  = H("verity-vllm/leaf/v1", canonical_id, dtype, "[d0,d1,...]", raw little-endian bytes)
    node  = H("verity-vllm/node/v1", left, right)
    lift  = H("verity-vllm/lift/v1", child)          # odd trailing node; nothing is duplicated
    ids   = H("verity-vllm/ids/v1", id_0, ..., id_{n-1})  # sorted ids, one part each
    empty = H("verity-vllm/empty/v1")                # root of a tree with no leaves

Tree: level 0 is the leaf hashes in tree order.  While a level has more than one node, neighbours
``(0,1), (2,3), ...`` become ``node(left, right)`` and an odd trailing node becomes ``lift(child)``.
:func:`path_shape` is the sibling/lift pattern of one leaf's path, :func:`fold_path` folds it to the root,
and :func:`range_path_shape` / :func:`fold_range` do the same for a contiguous run of leaves
(:mod:`verity_vllm.commit.native_ranges`' range openings).
"""

from __future__ import annotations

from collections.abc import Sequence

from verity.ml.commitments.leaves import fold_path
from verity.errors import InvalidArtifact
from verity_vllm.commit import hashing, scheme

__all__ = ["CommitError", "fold_path", "fold_range", "path_shape", "range_path_shape"]


class CommitError(ValueError):
    pass


def path_shape(leaf_count: int, index: int) -> list[bool]:
    """For each level from the leaves up: ``True`` when the node at that level has a sibling,
    ``False`` when it is the odd trailing node that gets lifted."""
    try:
        return scheme.path_shape(leaf_count, index)
    except InvalidArtifact as error:
        raise CommitError(str(error)) from error


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
