"""``frame-b3s``: the segment-tree leaf of a ``hash="blake3-s256"`` domain (Pearl-C's hashing format -h2), one test per
format condition of the theory sign-off (research store ``internal/pouw/rtx-pro/a-commit-latency.md``): distinct keys, a
segment key binding every field, the last segment at its true length, the odd-node rule. Then openings,
pinned vectors and a root rebuilt with the `blake3` package from the documented layout alone."""

from __future__ import annotations

import pytest

from verity.ml.commitments import (
    CommitmentDomain,
    MerkleTree,
    RangeIndexedDomain,
    VerificationLimits,
    verify_opening,
)
from verity.ml.crypto.blake3 import keyed
from verity.ml.commitments.merkle import (B3S_SEGMENT, _FRAME, _uint, b3s_level_key, b3s_levels, b3s_segment_key,
                                       blake3_node_key)
from verity.errors import ProtocolError

BINDING = bytes(range(32))


def domain(count: int = 5, owner: int = 3, binding: bytes = BINDING) -> CommitmentDomain:
    return CommitmentDomain(binding, owner, RangeIndexedDomain(count), hash="blake3-s256")


def values(count: int = 5) -> dict[int, bytes]:
    return {i: bytes([i]) * (40 + 997 * i) for i in range(count)}


def seg_key(d: CommitmentDomain, rank: int, position: int, schema: str, length: int, j: int) -> bytes:
    name = schema.encode()
    return keyed(d.domain_id, b"\x05" + rank.to_bytes(8, "little") + position.to_bytes(8, "little")
                 + length.to_bytes(8, "little") + j.to_bytes(4, "little") + len(name).to_bytes(2, "little") + name)


def test_condition_1_leaf_and_node_keys_are_distinct_derived_keys():
    """Every segment key (tag 0x05), every segment-level key (0x06) and every frame-b3 level key (0x02) of a domain is
    a separate keyed BLAKE3 output of the domain id, so no leaf output can pass for a node; the domain id itself differs
    from frame-b3's for the same binding."""
    d = domain()
    seg = {b3s_segment_key(d.domain_id, r, r, "rows", length, j)
           for r in range(3) for length in (0, 256, 1000) for j in range(4)}
    levels = {b3s_level_key(d.domain_id, l) for l in range(16)}
    frame_levels = {blake3_node_key(d.domain_id, l) for l in range(16)}
    assert len(seg) == 36 and len(levels) == 16 and len(frame_levels) == 16
    assert not (seg & levels) and not (seg & frame_levels) and not (levels & frame_levels) and d.domain_id not in seg | levels
    assert b3s_level_key(d.domain_id, 3) == keyed(d.domain_id, b"\x06\x03")
    assert d.domain_id != CommitmentDomain(BINDING, 3, RangeIndexedDomain(5), hash="blake3").domain_id
    levels_of = b3s_levels(d.domain_id, 0, 0, "rows", bytes(512))
    assert levels_of[1][0] == keyed(b3s_level_key(d.domain_id, 0), levels_of[0][0] + levels_of[0][1])
    assert levels_of[0][0] == keyed(seg_key(d, 0, 0, "rows", 512, 0), bytes(256))


def test_condition_2_the_segment_key_binds_every_field():
    """The key is the documented one-block message under the domain id, and changing the domain (binding or owner), the
    rank, the row index, the byte length, the segment position or the schema changes it; so the leaf of a value and of
    the same value one byte longer (the same first segments, another shape) differ."""
    d = domain()
    base = b3s_segment_key(d.domain_id, 2, 7, "rows", 1000, 1)
    assert base == seg_key(d, 2, 7, "rows", 1000, 1)
    others = [b3s_segment_key(domain(binding=bytes(32)).domain_id, 2, 7, "rows", 1000, 1),
              b3s_segment_key(domain(owner=4).domain_id, 2, 7, "rows", 1000, 1),
              b3s_segment_key(d.domain_id, 3, 7, "rows", 1000, 1), b3s_segment_key(d.domain_id, 2, 8, "rows", 1000, 1),
              b3s_segment_key(d.domain_id, 2, 7, "rows", 1001, 1), b3s_segment_key(d.domain_id, 2, 7, "rows", 1000, 2),
              b3s_segment_key(d.domain_id, 2, 7, "cols", 1000, 1)]
    assert len({base, *others}) == 8
    v = bytes(range(256)) * 4
    assert d.leaf(0, 0, "rows", v) != d.leaf(0, 0, "rows", v + b"\0") != d.leaf(0, 0, "rows", v[:-1])
    assert d.leaf(0, 0, "rows", v) != d.leaf(1, 1, "rows", v) != d.leaf(1, 1, "cols", v)
    b3s_segment_key(d.domain_id, 0, 0, "s" * 33, 0, 0)
    with pytest.raises(ProtocolError):
        b3s_segment_key(d.domain_id, 0, 0, "s" * 34, 0, 0)
    with pytest.raises(ProtocolError):
        d.leaf_prefix(0, 0, "rows", 10)


@pytest.mark.parametrize("length", [0, 1, 63, 64, 255, 256, 257, 1000, 1300, 32768 + 5])
def test_condition_3_the_last_segment_is_hashed_at_its_true_length(length):
    """Every segment is the standard keyed BLAKE3 of exactly its bytes: the last one short (or empty when L = 0), never
    zero-padded to 256."""
    d = domain()
    v = bytes((7 * i + 3) & 0xFF for i in range(length))
    first = b3s_levels(d.domain_id, 1, 1, "rows", v)[0]
    n = max(1, -(-length // B3S_SEGMENT))
    assert len(first) == n
    last = v[(n - 1) * B3S_SEGMENT:]
    assert len(last) == (length - (n - 1) * B3S_SEGMENT)
    assert first[-1] == keyed(seg_key(d, 1, 1, "rows", length, n - 1), last)
    if len(last) < B3S_SEGMENT:
        assert first[-1] != keyed(seg_key(d, 1, 1, "rows", length, n - 1), last + bytes(B3S_SEGMENT - len(last)))


def promoted_root(key_of, nodes: list[bytes], level: int = 0) -> bytes:
    """The odd-node rule, written recursively: pair from the left, promote an odd last node unchanged."""
    if len(nodes) == 1:
        return nodes[0]
    up = [keyed(key_of(level), nodes[i] + nodes[i + 1]) for i in range(0, len(nodes) - 1, 2)]
    return promoted_root(key_of, up + nodes[len(nodes) - len(nodes) % 2:], level + 1)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 9, 11, 128, 129])
def test_condition_4_an_odd_node_is_promoted_unchanged(n):
    """At a level of odd width the last node goes up unchanged: for 3 segments the leaf is node_1(node_0(g0, g1), g2);
    every width matches the recursive rule."""
    d = domain()
    v = bytes((5 * i) & 0xFF for i in range(256 * n - 17))
    levels = b3s_levels(d.domain_id, 0, 0, "rows", v)
    g = list(levels[0])
    assert len(g) == n and levels[-1] == (d.leaf(0, 0, "rows", v),)
    assert d.leaf(0, 0, "rows", v) == promoted_root(lambda l: b3s_level_key(d.domain_id, l), g)
    if n == 3:
        k0, k1 = b3s_level_key(d.domain_id, 0), b3s_level_key(d.domain_id, 1)
        assert levels[1] == (keyed(k0, g[0] + g[1]), g[2]) and levels[2][0] == keyed(k1, keyed(k0, g[0] + g[1]) + g[2])
    for below, above in zip(levels, levels[1:]):
        assert len(above) == (len(below) + 1) // 2
        if len(below) % 2:
            assert above[-1] == below[-1]


def test_openings_authenticate_and_tampering_fails():
    d, vals = domain(), values()
    tree = MerkleTree(d, vals, lambda _: "rows")
    limits = VerificationLimits()
    for p in range(5):
        op = tree.open(p)
        assert verify_opening(d, tree.commitment, op, "rows", limits)
        assert not verify_opening(d, tree.commitment, type(op)(p, op.value + b"\0", op.path), "rows", limits)
        assert not verify_opening(d, tree.commitment, type(op)(p, op.value[:-1], op.path), "rows", limits)
        assert not verify_opening(d, tree.commitment, op, "cols", limits)
    b3 = MerkleTree(CommitmentDomain(BINDING, 3, RangeIndexedDomain(5), hash="blake3"), vals, lambda _: "rows")
    assert tree.commitment.root != b3.commitment.root
    assert MerkleTree(domain(0), {}, lambda _: "rows").commitment.root == domain(0).empty_root()


#: the vectors another implementation checks against: `domain()` (five values, binding bytes(range(32)), owner 3), schema
#: "rows"; seg_key_2_1 is rank 2's segment 1 key for value 2 (2,034 bytes); leaf_3 is value 3's leaf (3,031 bytes: 12
#: segments, the last 215 bytes, widths 12, 6, 3, 2, 1: one odd level); row_32k a 32 KB value's leaf at rank 0
PINNED = {"domain_id": "1397806572d8d301bbc4761e07d033a29c6260af9f5e537c0ab711ca4058e10c",
          "seg_key_2_1": "94020f4be26cabc357b24062e037fed2b5019570c9a28636bb596d6f71c32d63",
          "level_key_3": "ffc1d87469edc7fad3563fa12f15708748815a9f279fa065ab3d2c4349a4ab47",
          "leaf_3": "00b1f748f81fcbbff11720c7b5f02ac0ac1e6549b2d459eecab37f0168eaedca",
          "row_32k": "5599f93a410b0fe87c80e90c108db65fa3f4cf6a6b70cf1b1ae138cc8947edad",
          "root": "026500f3ea571d4a139c352a980ab101f55bf2df77c3d34761e130ac31e51aab"}


def pinned() -> dict[str, str]:
    d = domain()
    got = {"domain_id": d.domain_id, "seg_key_2_1": b3s_segment_key(d.domain_id, 2, 2, "rows", len(values()[2]), 1),
           "level_key_3": b3s_level_key(d.domain_id, 3), "leaf_3": d.leaf(3, 3, "rows", values()[3]),
           "row_32k": d.leaf(0, 0, "rows", bytes(range(256)) * 128),
           "root": MerkleTree(d, values(), lambda _: "rows").commitment.root}
    return {k: v.hex() for k, v in got.items()}


def test_vectors_are_pinned():
    assert pinned() == PINNED


def test_root_from_the_layout_alone():
    """The pinned root rebuilt from the module docstring's layout with the `blake3` package (the Rust reference) alone."""
    b3 = pytest.importorskip("blake3")
    key = lambda k, m: b3.blake3(m, key=k).digest()  # noqa: E731
    positions = bytes.fromhex(domain().positions.identity_digest)
    framed = _FRAME + (10).to_bytes(4, "big") + b"domain/b3s"
    framed += b"".join(len(p).to_bytes(8, "big") + p for p in (BINDING, _uint(5), positions, _uint(5)))
    dom = b3.blake3(framed).digest()
    assert dom.hex() == PINNED["domain_id"]

    def leaf(r: int, v: bytes) -> bytes:
        n = max(1, -(-len(v) // 256))
        level = [key(key(dom, b"\x05" + r.to_bytes(8, "little") * 2 + len(v).to_bytes(8, "little") + j.to_bytes(4, "little")
                         + (4).to_bytes(2, "little") + b"rows"), v[256 * j:256 * j + 256]) for j in range(n)]
        lv = 0
        while len(level) > 1:
            k = key(dom, b"\x06" + bytes([lv]))
            level = [key(k, level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)] + level[len(level) - len(level) % 2:]
            lv += 1
        return level[0]

    level = [leaf(r, v) for r, v in values().items()] + [key(dom, b"\x03" + r.to_bytes(8, "little")) for r in range(5, 8)]
    for lv in range(3):
        k = key(dom, b"\x02" + bytes([lv]))
        level = [key(k, level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    assert level[0].hex() == PINNED["root"]


def test_every_hash_is_a_complete_keyed_blake3_call():
    """Every node of both trees (segment tree and the tree over leaves), every segment hash, key, pad and the empty root is a
    complete keyed BLAKE3 call, the ROOT flag included: each equals the Rust `blake3` package's keyed hash of the documented
    message, and no node equals the bare PARENT compression of its children, with or without ROOT."""
    b3 = pytest.importorskip("blake3")
    from verity.ml.crypto.blake3 import compress

    def words(k: bytes) -> tuple[int, ...]:
        return tuple(int.from_bytes(k[4 * i:4 * i + 4], "little") for i in range(8))

    def parent(key: bytes, block: bytes, flags: int) -> bytes:
        return b"".join(w.to_bytes(4, "little") for w in compress(words(key), block, 0, 64, flags)[:8])

    rust = lambda k, m: b3.blake3(m, key=k).digest()  # noqa: E731
    d = domain(7)
    vals = {r: bytes((r + 3 * i) & 0xFF for i in range(256 * (r + 1) + 9)) for r in range(7)}
    for r, v in vals.items():
        levels = b3s_levels(d.domain_id, r, r, "rows", v)
        for j, g in enumerate(levels[0]):
            key = b3s_segment_key(d.domain_id, r, r, "rows", len(v), j)
            assert g == rust(key, v[256 * j:256 * j + 256])
            assert key == rust(d.domain_id, b"\x05" + r.to_bytes(8, "little") * 2 + len(v).to_bytes(8, "little")
                               + j.to_bytes(4, "little") + (4).to_bytes(2, "little") + b"rows")
        for l, (below, above) in enumerate(zip(levels, levels[1:])):
            k = b3s_level_key(d.domain_id, l)
            assert k == rust(d.domain_id, b"\x06" + bytes([l]))
            for i in range(len(below) // 2):
                block = below[2 * i] + below[2 * i + 1]
                assert above[i] == rust(k, block)
                assert above[i] not in (parent(k, block, 4 | 16), parent(k, block, 4 | 8 | 16))
    tree = MerkleTree(d, vals, lambda _: "rows")
    leaves = [d.leaf(r, r, "rows", vals[r]) for r in range(7)] + [d.pad(7)]
    assert d.pad(7) == rust(d.domain_id, b"\x03" + (7).to_bytes(8, "little")) and d.empty_root() == rust(d.domain_id, b"\x04")
    level, lv = leaves, 0
    while len(level) > 1:
        k = blake3_node_key(d.domain_id, lv)
        assert k == rust(d.domain_id, b"\x02" + bytes([lv]))
        up = [rust(k, level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        assert [d.node(lv, i // 2, level[i], level[i + 1]) for i in range(0, len(level), 2)] == up
        assert all(u != parent(k, level[2 * i] + level[2 * i + 1], 4 | 8 | 16) for i, u in enumerate(up))
        level, lv = up, lv + 1
    assert level[0] == tree.commitment.root


def test_every_key_is_derived_from_the_domain():
    """No -h2 key is a constant: the segment and level keys come from the domain id, which comes from the binding, owner,
    positions and count, so the same value under another domain has other keys and another leaf, and no segment or level
    key is the all-zero or the frame's IV key."""
    from verity.ml.crypto.blake3 import IV
    iv = b"".join(w.to_bytes(4, "little") for w in IV)
    v = bytes(range(256)) * 3
    ds = [domain(), domain(owner=4), domain(binding=bytes(32)), domain(count=6)]
    assert len({d.domain_id for d in ds}) == 4
    assert len({d.leaf(1, 1, "rows", v) for d in ds}) == 4
    for d in ds:
        keys = [b3s_segment_key(d.domain_id, 1, 1, "rows", len(v), j) for j in range(3)] + [b3s_level_key(d.domain_id, l) for l in range(3)]
        assert not {bytes(32), iv, d.domain_id} & set(keys)
