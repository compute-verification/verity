"""``frame-b3``: the BLAKE3 tree mode of `verity.commitments.merkle` (a ``hash="blake3"`` domain): its message layouts, openings,
multiproofs and a pinned root."""

from __future__ import annotations

from verity.commitments import (CommitmentDomain, MerkleTree, RangeIndexedDomain, VerificationLimits, multiproof,
                                verify_multiproof, verify_opening)
from verity.commitments.blake3 import BLOCK_LEN, blake3, keyed
from verity.commitments.merkle import _FRAME, _uint

BINDING = bytes(range(32))


def domain(count: int = 5, owner: int = 3, hash: str = "blake3") -> CommitmentDomain:
    return CommitmentDomain(BINDING, owner, RangeIndexedDomain(count), hash=hash)


def values(count: int = 5) -> dict[int, bytes]:
    return {i: bytes([i]) * (40 + 997 * i) for i in range(count)}


def test_messages_are_the_documented_layout():
    d = domain()
    positions = bytes.fromhex(d.positions.identity_digest)
    framed = _FRAME + (6).to_bytes(4, "big") + b"domain"
    framed += b"".join(len(p).to_bytes(8, "big") + p for p in (BINDING, _uint(5), positions, _uint(5)))
    assert d.domain_id == blake3(framed) and d.digest_bytes == 32
    value = b"\x07" * 3000
    prefix = b"\x01" + (2).to_bytes(8, "little") + (2).to_bytes(8, "little") + (4).to_bytes(2, "little") + b"rows"
    assert d.leaf_prefix(2, 2, "rows", len(value)) == prefix and d.leaf(2, 2, "rows", value) == keyed(d.domain_id, prefix + value)
    node = b"\x02" + (3).to_bytes(1, "little") + (9).to_bytes(8, "little") + b"L" * 32 + b"R" * 32
    assert len(node) <= 2 * BLOCK_LEN and d.node(3, 9, b"L" * 32, b"R" * 32) == keyed(d.domain_id, node)
    assert d.pad(6) == keyed(d.domain_id, b"\x03" + (6).to_bytes(8, "little")) and d.empty_root() == keyed(d.domain_id, b"\x04")


def test_openings_authenticate_and_tampering_fails():
    d, vals = domain(), values()
    tree = MerkleTree(d, vals, lambda _: "rows")
    limits = VerificationLimits()
    for p in range(5):
        op = tree.open(p)
        assert verify_opening(d, tree.commitment, op, "rows", limits)
        assert not verify_opening(d, tree.commitment, type(op)(p, op.value + b"\0", op.path), "rows", limits)
        assert not verify_opening(d, tree.commitment, op, "cols", limits)
        bad = (bytes(32),) + op.path[1:]
        assert not verify_opening(d, tree.commitment, type(op)(p, op.value, bad), "rows", limits)
    other = MerkleTree(domain(owner=4), vals, lambda _: "rows")
    sha = MerkleTree(domain(hash="sha256"), vals, lambda _: "rows")
    assert len({tree.commitment.root, other.commitment.root, sha.commitment.root}) == 3
    assert not verify_opening(domain(owner=4), tree.commitment, tree.open(0), "rows", limits)
    assert MerkleTree(domain(0), {}, lambda _: "rows").commitment.root == domain(0).empty_root()


def test_multiproofs_use_the_same_nodes():
    d, vals = domain(7), values(7)
    tree = MerkleTree(d, vals, lambda _: "rows")
    ranks = (1, 4, 6)
    proof = multiproof(tree, ranks)
    leaves = {r: d.leaf(r, r, "rows", vals[r]) for r in ranks}
    assert verify_multiproof(tree.commitment.root, 7, d.domain_id, leaves, proof, hash="blake3")
    assert not verify_multiproof(tree.commitment.root, 7, d.domain_id, leaves, proof, hash="sha256")


def test_root_is_pinned():
    assert MerkleTree(domain(), values(), lambda _: "rows").commitment.root.hex() == (
        "0acb01c26f4b797a821a44aa346bfabe0805451f28288fc7dda079280bb6037c")
