"""Merkle multiproofs against :class:`MerkleTree`: canonical, complete, and sound."""

from __future__ import annotations

import random

import pytest

from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree, merkle_depth
from verity.ml.commitments.multiproof import multiproof, multiproof_cost, multiproof_from_paths, verify_multiproof
from verity.errors import ProtocolError


SCHEMA = "u16"
BASE = 10


def tree_of(count: int) -> tuple[CommitmentDomain, MerkleTree, dict[int, bytes]]:
    domain = CommitmentDomain(
        bytes([count % 251]) * 32, -1, RangeIndexedDomain(BASE, BASE + count), hash="sha256"
    )
    values = {BASE + rank: bytes([rank & 0xFF, rank >> 8]) for rank in range(count)}
    return domain, MerkleTree(domain, values, lambda _position: SCHEMA), values


def leaves_of(domain: CommitmentDomain, values, ranks) -> dict[int, bytes]:
    return {
        rank: domain.leaf(rank, BASE + rank, SCHEMA, values[BASE + rank])
        for rank in set(ranks)
    }


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 7, 8, 9, 16, 33, 100])
def test_random_subsets_reconstruct_the_root(count: int) -> None:
    domain, tree, values = tree_of(count)
    root = tree.commitment.root
    rng = random.Random(count)
    for _ in range(25):
        ranks = [rng.randrange(count) for _ in range(rng.randint(1, count))]
        proof = multiproof(tree, ranks)
        leaves = leaves_of(domain, values, ranks)
        assert verify_multiproof(root, count, domain.domain_id, leaves, proof)
        # the same proof from the individual paths, and its size as predicted
        paths = {rank: tree.open(BASE + rank).path for rank in set(ranks)}
        assert multiproof_from_paths(count, paths) == proof
        cost = multiproof_cost(count, ranks)
        assert cost.siblings == len(proof)
        assert cost.leaves == len(set(ranks))
        assert cost.depth == merkle_depth(count)
        assert cost.node_hashes <= cost.path_hashes
        # spelled out: level by level, ascending, every sibling of a node the
        # openings determine that is not itself determined -- and nothing else
        known = [set(ranks)]
        for _level in range(cost.depth):
            known.append({index >> 1 for index in known[-1]})
        assert proof == [
            tree._levels[level][index ^ 1]
            for level in range(cost.depth)
            for index in sorted(known[level])
            if index ^ 1 not in known[level]
        ]
        assert cost.node_hashes == sum(
            len(known[level]) for level in range(1, cost.depth + 1)
        )


def test_one_leaf_is_its_authentication_path() -> None:
    for count in (1, 2, 5, 8, 13):
        _domain, tree, _values = tree_of(count)
        for rank in range(count):
            opening = tree.open(BASE + rank)
            assert multiproof(tree, [rank]) == list(opening.path)
            assert len(opening.path) == merkle_depth(count)


def test_all_leaves_need_only_the_padding_siblings() -> None:
    for count in (1, 2, 3, 4, 6, 8, 9, 16):
        domain, tree, values = tree_of(count)
        proof = multiproof(tree, range(count))
        depth = merkle_depth(count)
        cost = multiproof_cost(count, range(count))
        # one hash per ancestor of an opened leaf; a full power of two hashes every node
        assert cost.node_hashes == sum(
            len({rank >> level for rank in range(count)})
            for level in range(1, depth + 1)
        )
        # a power of two has no padding, hence no siblings at all
        assert (proof == []) == (count == 1 << depth)
        assert verify_multiproof(
            tree.commitment.root,
            count,
            domain.domain_id,
            leaves_of(domain, values, range(count)),
            proof,
        )


def test_adjacent_pairs_share_their_parent_and_need_no_sibling_there() -> None:
    domain, tree, values = tree_of(8)
    # ranks 4 and 5 share a parent: one level saved against two separate paths
    assert multiproof_cost(8, [4, 5]).siblings == 2  # levels 1 and 2 only
    assert multiproof_cost(8, [4, 6]).siblings == 3  # 4 and 6 meet one level higher
    assert multiproof_cost(8, [0, 7]).siblings == 4  # meet only at the root
    for ranks in ([4, 5], [4, 6], [0, 7], [2, 3, 4, 5]):
        assert verify_multiproof(
            tree.commitment.root,
            8,
            domain.domain_id,
            leaves_of(domain, values, ranks),
            multiproof(tree, ranks),
        )


def test_duplicates_and_order_do_not_change_the_proof() -> None:
    _domain, tree, _values = tree_of(13)
    assert multiproof(tree, [7, 2, 2, 11, 7]) == multiproof(tree, [2, 7, 11])
    assert multiproof_cost(13, [7, 2, 2, 11, 7]) == multiproof_cost(13, (2, 7, 11))


def test_a_count_that_is_not_a_power_of_two_pads_the_tree() -> None:
    domain, tree, values = tree_of(5)  # depth 3, three padding leaves
    proof = multiproof(tree, range(5))
    assert len(proof) == 2  # the pad leaf 5 and the pad subtree (6, 7)
    assert verify_multiproof(
        tree.commitment.root,
        5,
        domain.domain_id,
        leaves_of(domain, values, range(5)),
        proof,
    )


def test_a_corrupted_or_misshaped_proof_is_rejected() -> None:
    domain, tree, values = tree_of(16)
    root = tree.commitment.root
    ranks = [1, 6, 7, 12]
    proof = multiproof(tree, ranks)
    leaves = leaves_of(domain, values, ranks)
    assert verify_multiproof(root, 16, domain.domain_id, leaves, proof)
    for at in range(len(proof)):
        corrupted = list(proof)
        corrupted[at] = bytes(byte ^ 1 for byte in corrupted[at])
        assert not verify_multiproof(root, 16, domain.domain_id, leaves, corrupted)
    assert not verify_multiproof(root, 16, domain.domain_id, leaves, proof[:-1])
    assert not verify_multiproof(
        root, 16, domain.domain_id, leaves, [*proof, bytes(32)]
    )
    assert not verify_multiproof(bytes(32), 16, domain.domain_id, leaves, proof)
    # a wrong leaf, a foreign leaf, a rank outside the tree, nothing opened
    wrong = dict(leaves)
    wrong[6] = bytes(32)
    assert not verify_multiproof(root, 16, domain.domain_id, wrong, proof)
    assert not verify_multiproof(
        root, 16, domain.domain_id, {**leaves, 16: bytes(32)}, proof
    )
    assert not verify_multiproof(root, 16, domain.domain_id, {}, [])
    # the domain is bound into every node
    assert not verify_multiproof(root, 16, bytes(32), leaves, proof)


def test_provers_reject_ranks_outside_the_tree_and_inconsistent_paths() -> None:
    _domain, tree, _values = tree_of(6)
    with pytest.raises(ProtocolError, match="outside"):
        multiproof(tree, [6])
    with pytest.raises(ProtocolError, match="at least one"):
        multiproof(tree, [])
    paths = {rank: tree.open(BASE + rank).path for rank in (1, 4)}
    assert multiproof_from_paths(6, paths) == multiproof(tree, [1, 4])
    with pytest.raises(ProtocolError, match="disagree"):
        # ranks 0 and 1 share every node above level 0; a forged level-1 sibling shows
        forged = {
            0: tree.open(BASE).path,
            1: (tree.open(BASE + 1).path[0], bytes(32), *tree.open(BASE + 1).path[2:]),
        }
        multiproof_from_paths(6, forged)
    with pytest.raises(ProtocolError, match="depth"):
        multiproof_from_paths(6, {0: tree.open(BASE).path[:-1]})


@pytest.mark.parametrize("hash", ["blake3", "blake3-s256"])
def test_other_hashes_are_refused(hash: str) -> None:
    """Multiproofs are over the SHA frames only: no domain of another hash can be made, and its hash named to the verifier
    raises instead of folding nodes of another frame."""
    with pytest.raises(ValueError, match="framing hash"):
        CommitmentDomain(bytes(32), -1, RangeIndexedDomain(BASE, BASE + 7), hash=hash)
    domain, tree, values = tree_of(7)
    leaves = leaves_of(domain, values, (1, 4))
    paths = {rank: tree.open(BASE + rank).path for rank in (1, 4)}
    with pytest.raises(ValueError, match="framing hash"):
        verify_multiproof(tree.commitment.root, 7, domain.domain_id, leaves, multiproof_from_paths(7, paths), hash=hash)
