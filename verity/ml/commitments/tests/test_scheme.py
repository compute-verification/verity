"""Both named schemes sit behind ``CommitmentScheme``; frame-v3's configured scheme reproduces ``MerkleTree`` exactly,
and every layout's digest is the tree leaf rule it stands for."""

from __future__ import annotations

import hashlib

import pytest

from verity.ml.commitments.frame_v3 import FrameV3, Port, WordLeaf
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree
from verity.ml.commitments.scheme import CommitmentScheme
from verity.ml.commitments.vllm_v1 import VllmV1


def _domain(n: int) -> CommitmentDomain:
    return CommitmentDomain(hashlib.sha256(b"scheme-test").digest(), 0, RangeIndexedDomain(n), hash="sha256")


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8, 13])
def test_frame_v3_word_leaves_match_merkle_tree(n: int) -> None:
    domain = _domain(n)
    values = {p: bytes([p]) * 3 for p in range(n)}
    tree = MerkleTree(domain, values, lambda _p: "u8[3]")
    scheme = FrameV3({"v": Port(domain, WordLeaf("u8[3]", 3))})
    leaves = [scheme.tree_leaf("v", p, scheme.leaf_layout("v", p).digest(values[p])) for p in range(n)]
    assert leaves == [domain.leaf(p, p, "u8[3]", values[p]) for p in range(n)]
    assert scheme.root("v", leaves) == tree.commitment.root
    for p in range(n):
        path = scheme.path("v", leaves, p)
        assert path == tree.open(p).path
        assert scheme.verify_path("v", tree.commitment.root, n, p, leaves[p], path)
        assert not scheme.verify_path("v", tree.commitment.root, n + 1, p, leaves[p], path)
        if n > 1:
            assert not scheme.verify_path("v", tree.commitment.root, n, p, leaves[(p + 1) % n], path)


def test_both_schemes_implement_the_interface() -> None:
    frame = FrameV3({"v": Port(_domain(1), WordLeaf("u16", 2))})
    vllm = VllmV1({})
    assert isinstance(frame, CommitmentScheme) and isinstance(vllm, CommitmentScheme)
    assert {frame.name, vllm.name} == {"frame-v3", "vllm-v1"}
