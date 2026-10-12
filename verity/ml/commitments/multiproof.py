"""Merkle multiproofs: one shared authentication path set for many openings.

A batch opens many leaves of the same tree, and their authentication paths
share every node above the point where they meet.  Folding each path on its
own hashes those shared nodes once per opening; a *multiproof* carries each
node of the union of paths once and lets the verifier reconstruct the root
in a single bottom-up pass, so the hash count is the number of distinct
internal nodes on the union of the paths instead of ``openings x depth``.

The layout is canonical, so the encoding of a multiproof is unique and the
verifier's reconstruction is a fixed algorithm:

* the opened ranks are sorted ascending and deduplicated; the *known* nodes
  of level 0 are their leaf digests, at index ``rank``;
* level by level, the known nodes are visited in ascending index order and
  paired: an even-index node whose right neighbour ``index + 1`` is also
  known is hashed with it (both are consumed, no proof node is used);
  otherwise its sibling ``index ^ 1`` is the next digest of the proof.  Each
  pair yields the parent, known at level ``level + 1`` at index
  ``index >> 1``, with the exact node framing of :mod:`merkle`
  (``node(level, index >> 1, left, right)``);
* after ``merkle_depth(count)`` levels exactly one node is known, at index
  ``0``: the root.  The proof must be consumed exactly.

The tree is the complete binary tree of :class:`MerkleTree` (``2**depth``
leaves, padding leaves past ``count``), so every known node has a sibling
and a padding sibling is simply supplied by the proof like any other.  With
one opened rank the multiproof is that rank's authentication path.

Multiproofs are over the SHA frames only: ``frame-v3-sha512`` (``hash="sha512"``)
and the SHA-256 frame the frozen SP1 backend reads.  A BLAKE3 domain
(``frame-b3``, ``frame-b3s``) is refused.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from verity.ml.commitments.merkle import _NODE, MerkleTree, _hash, _uint, merkle_depth
from verity.ml.commitments.scheme import sha
from verity.errors import ProtocolError

type Node = Callable[[int, int, bytes, bytes], bytes]
"""``node(level, index, left, right)``: the parent of two nodes."""


def _ranks(count: int, ranks: Iterable[int]) -> tuple[int, ...]:
    """The opened ranks sorted and deduplicated; every rank must lie in the tree."""

    ordered = tuple(sorted(set(ranks)))
    if not ordered:
        raise ProtocolError("a multiproof opens at least one rank")
    for rank in ordered:
        if type(rank) is not int or not 0 <= rank < count:
            raise ProtocolError(f"rank {rank} lies outside a tree of {count} leaves")
    return ordered


def _node_of(domain_id: bytes, hash: str = "sha256") -> Node:
    sha(hash)

    def node(level: int, index: int, left: bytes, right: bytes) -> bytes:
        return _hash(_NODE, domain_id, _uint(level), _uint(index), left, right, hash=hash)

    return node


def _fold(
    count: int,
    known: Mapping[int, bytes],
    sibling: Callable[[int, int], bytes],
    node: Node,
) -> bytes:
    """Reconstruct the root from the known leaves.

    ``sibling(level, index)`` supplies every node the pairing needs that is
    not known; ``node`` hashes each pair.  ``known`` must not be empty.
    """

    level_nodes = sorted(known.items())
    for level in range(merkle_depth(count)):
        parents: list[tuple[int, bytes]] = []
        at = 0
        while at < len(level_nodes):
            index, digest = level_nodes[at]
            if (
                index % 2 == 0
                and at + 1 < len(level_nodes)
                and level_nodes[at + 1][0] == index + 1
            ):
                left, right = digest, level_nodes[at + 1][1]
                at += 2
            else:
                other = sibling(level, index ^ 1)
                left, right = (digest, other) if index % 2 == 0 else (other, digest)
                at += 1
            parents.append((index >> 1, node(level, index >> 1, left, right)))
        level_nodes = parents
    ((_root_index, root),) = level_nodes
    return root


def multiproof(tree: MerkleTree, ranks: Iterable[int]) -> list[bytes]:
    """The sibling digests a verifier needs to reconstruct ``tree``'s root from the leaves at ``ranks``."""

    domain = tree.domain
    ordered = _ranks(domain.count, ranks)
    levels = tree._levels  # the prover's own tree: the multiproof reads its nodes
    proof: list[bytes] = []

    def sibling(level: int, index: int) -> bytes:
        proof.append(levels[level][index])
        return proof[-1]

    known = {rank: levels[0][rank] for rank in ordered}
    _fold(domain.count, known, sibling, _node_of(domain.domain_id, domain.hash))
    return proof


def multiproof_from_paths(
    count: int, paths: Mapping[int, Sequence[bytes]]
) -> list[bytes]:
    """The multiproof for the ranks of ``paths`` from their individual authentication paths.

    ``paths[rank]`` is the path :meth:`MerkleTree.open` returned for ``rank``
    (sibling per level, bottom up).  Every node the multiproof needs lies on
    one of these paths; paths that disagree on a shared node are rejected,
    since no tree produced them.
    """

    ordered = _ranks(count, paths)
    depth = merkle_depth(count)
    nodes: dict[tuple[int, int], bytes] = {}
    for rank in ordered:
        path = paths[rank]
        if len(path) != depth:
            raise ProtocolError("path length differs from the tree depth")
        for level, digest in enumerate(path):
            if nodes.setdefault((level, (rank >> level) ^ 1), digest) != digest:
                raise ProtocolError("authentication paths disagree on a shared node")
    proof: list[bytes] = []

    def sibling(level: int, index: int) -> bytes:
        proof.append(nodes[(level, index)])
        return proof[-1]

    # which siblings are emitted depends on the ranks alone, not on any digest
    _fold(count, dict.fromkeys(ordered, b""), sibling, lambda *_: b"")
    return proof


def verify_multiproof(
    root: bytes,
    count: int,
    domain_id: bytes,
    leaves: Mapping[int, bytes],
    proof: Sequence[bytes],
    hash: str = "sha256",
) -> bool:
    """Whether the leaf digests ``leaves[rank]`` reconstruct ``root`` with ``proof`` (``hash`` the domain's frame hash).

    False when a rank lies outside the tree, when ``leaves`` is empty (nothing
    is authenticated) or when the proof is not consumed exactly.  A ``hash``
    that is not a SHA frame's raises ``ValueError``.
    """

    if not leaves or any(not 0 <= rank < count for rank in leaves):
        return False
    cursor = 0

    def sibling(level: int, index: int) -> bytes:
        nonlocal cursor
        if cursor >= len(proof):
            raise IndexError
        cursor += 1
        return proof[cursor - 1]

    try:
        reconstructed = _fold(count, leaves, sibling, _node_of(domain_id, hash))
    except IndexError:
        return False
    return cursor == len(proof) and reconstructed == root


@dataclass(frozen=True, slots=True)
class MultiproofCost:
    """What authenticating the opened ranks of one tree costs, per scheme."""

    leaves: int
    depth: int
    siblings: int
    node_hashes: int

    @property
    def path_hashes(self) -> int:
        """Node hashes when every opening folds its own path: ``leaves * depth``."""

        return self.leaves * self.depth


def multiproof_cost(count: int, ranks: Iterable[int]) -> MultiproofCost:
    """Sibling count and node-hash count of the multiproof for ``ranks`` (no hashing)."""

    ordered = _ranks(count, ranks)
    counters = {"siblings": 0, "hashes": 0}

    def sibling(level: int, index: int) -> bytes:
        counters["siblings"] += 1
        return b""

    def node(level: int, index: int, left: bytes, right: bytes) -> bytes:
        counters["hashes"] += 1
        return b""

    _fold(count, dict.fromkeys(ordered, b""), sibling, node)
    return MultiproofCost(
        len(ordered), merkle_depth(count), counters["siblings"], counters["hashes"]
    )


__all__ = [
    "MultiproofCost",
    "multiproof",
    "multiproof_cost",
    "multiproof_from_paths",
    "verify_multiproof",
]
