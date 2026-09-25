"""Both named schemes sit behind ``CommitmentScheme``; frame-v3's adapter reproduces ``MerkleTree`` exactly."""

from __future__ import annotations

import hashlib

import pytest

from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, FrameV3, MerkleTree
from verity.commitments.scheme import CommitmentScheme
from verity.commitments.vllm_production import SCHEME as VLLM


def _domain(n: int) -> CommitmentDomain:
    return CommitmentDomain(hashlib.sha256(b"scheme-test").digest(), 0, RangeIndexedDomain(n))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8, 13])
def test_frame_v3_adapter_matches_merkle_tree(n: int) -> None:
    domain = _domain(n)
    values = {p: bytes([p]) * 3 for p in range(n)}
    tree = MerkleTree(domain, values, lambda _p: "u8[3]")
    leaves = [domain.leaf(p, p, "u8[3]", values[p]) for p in range(n)]
    scheme = FrameV3(domain)
    assert scheme.root(leaves) == tree.commitment.root
    for p in range(n):
        path = scheme.path(leaves, p)
        assert path == tree.open(p).path
        assert scheme.verify_path(tree.commitment.root, n, p, leaves[p], path)
        assert not scheme.verify_path(tree.commitment.root, n + 1, p, leaves[p], path)
        if n > 1:
            assert not scheme.verify_path(tree.commitment.root, n, p, leaves[(p + 1) % n], path)


def test_both_schemes_implement_the_interface() -> None:
    assert isinstance(FrameV3(_domain(1)), CommitmentScheme)
    assert isinstance(VLLM, CommitmentScheme)
    assert {FrameV3.name, VLLM.name} == {"frame-v3", "vllm-production/v1"}
