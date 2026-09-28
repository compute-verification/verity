"""``frame-b3``: the BLAKE3 tree mode of `verity.ml.commitments.merkle` (a ``hash="blake3"`` domain): its message layouts (a
block-aligned leaf header, one-block nodes under per-level keys), openings, pinned vectors and the root rebuilt
with the `blake3` package from the layout alone."""

from __future__ import annotations

import pytest

from verity.ml.commitments import (
    CommitmentDomain,
    MerkleTree,
    RangeIndexedDomain,
    VerificationLimits,
    verify_opening,
)
from verity.ml.crypto.blake3 import BLOCK_LEN, blake3, keyed
from verity.ml.commitments.merkle import _FRAME, _uint

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
    head = b"\x01" + (2).to_bytes(8, "little") + (2).to_bytes(8, "little") + (4).to_bytes(2, "little") + b"rows"
    prefix = head + bytes(BLOCK_LEN - len(head))
    assert d.leaf_prefix(2, 2, "rows", len(value)) == prefix and d.leaf(2, 2, "rows", value) == keyed(d.domain_id, prefix + value)
    long = "s" * 50
    assert len(d.leaf_prefix(0, 0, long, 0)) == 2 * BLOCK_LEN
    level_key = keyed(d.domain_id, b"\x02" + (3).to_bytes(1, "little"))
    assert d.node(3, 9, b"L" * 32, b"R" * 32) == keyed(level_key, b"L" * 32 + b"R" * 32) == d.node(3, 0, b"L" * 32, b"R" * 32)
    assert d.node(2, 9, b"L" * 32, b"R" * 32) != d.node(3, 9, b"L" * 32, b"R" * 32)
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


#: the vectors another implementation checks against: `domain()` (five values, binding bytes(range(32)), owner 3), schema "rows"
PINNED = {"domain_id": "fb0723b4eb9c9bd013d7c38b1d6097f70eee42600cec7c7126638b7c203fee37",
          "leaf_2": "40b8a5ee86bd3f1f9a9915911813d1ad76be25320c9ea3c7fd30e2109b6379be",
          "node_1": "10d5804f72f8ca8412a8c4724aa7c86f1e1728605e075154f774149ecd8c9add",
          "pad_6": "5bc67aaa1926efb9896d74754572c055ebbaea52cb8ca91b830c4b5f6a470600",
          "empty": "e0f8c2396ea837c8fe0cde7a9e51e7c985217eb5e1a923d0b0aa1eed3aa46b08",
          "root": "479cb1e344ecbf5583b37afc4bc85094c31b5fae8f90abb6c706295677cc68a3"}


def test_vectors_are_pinned():
    """leaf_2 is value 2's leaf; node_1 a level-1 node over bytes(range(32)) ‖ bytes(range(32, 64)); pad_6 rank 6's pad."""
    d = domain()
    got = {"domain_id": d.domain_id, "leaf_2": d.leaf(2, 2, "rows", values()[2]),
           "node_1": d.node(1, 0, bytes(range(32)), bytes(range(32, 64))), "pad_6": d.pad(6), "empty": d.empty_root(),
           "root": MerkleTree(d, values(), lambda _: "rows").commitment.root}
    assert {k: v.hex() for k, v in got.items()} == PINNED


def test_root_from_the_layout_alone():
    """The pinned root rebuilt from the documented layout with the `blake3` package (the Rust reference) alone."""
    b3 = pytest.importorskip("blake3")
    key = lambda k, m: b3.blake3(m, key=k).digest()  # noqa: E731
    positions = bytes.fromhex(domain().positions.identity_digest)
    framed = _FRAME + (6).to_bytes(4, "big") + b"domain"
    framed += b"".join(len(p).to_bytes(8, "big") + p for p in (BINDING, _uint(5), positions, _uint(5)))
    dom = b3.blake3(framed).digest()
    assert dom.hex() == PINNED["domain_id"]
    head = lambda r: b"\x01" + r.to_bytes(8, "little") * 2 + (4).to_bytes(2, "little") + b"rows"  # noqa: E731
    level = [key(dom, head(r) + bytes(64 - len(head(r))) + v) for r, v in values().items()]
    level += [key(dom, b"\x03" + r.to_bytes(8, "little")) for r in range(5, 8)]
    for lv in range(3):
        k = key(dom, b"\x02" + bytes([lv]))
        level = [key(k, level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    assert level[0].hex() == PINNED["root"]


@pytest.mark.parametrize("hash", ["blake3", "blake3-s256"])
def test_the_domain_message_hashed_is_the_domain_id(hash):
    """`domain_message` is the frame hash's message for a domain: a native hash of it is the domain's id."""
    from verity.ml.commitments.merkle import domain_message

    for count, owner in ((1, -2), (5, 3), (8192, 1 << 40)):
        d = domain(count, owner, hash)
        positions = bytes.fromhex(d.positions.identity_digest)
        assert blake3(domain_message(BINDING, owner, positions, count, hash)) == d.domain_id
