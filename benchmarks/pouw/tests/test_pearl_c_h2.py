"""The -h2 kernels (`pearl_c/hash_h2.cuh`) against frame-b3s, `verity.primitives.commitments.merkle`'s reference, through
`simt_twin.py`'s thread-for-thread host build: the segment-tree level keys, row leaves at every length class (the aligned
16-byte-load path and the byte path, short last segments, odd levels), short-value leaves, and whole trees against
`MerkleTree`'s root; and rows form `s` (`pearl_c_sm120/h2_rows_stats.cuh`), the row leaves with forming's stats against the
scheme's `s5_stats`."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from verity.primitives.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.primitives.commitments.merkle import b3s_level_key
from verity_catalog.silicon import fp32
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C

HERE = Path(__file__).resolve().parents[1] / "pearl_c"
spec = importlib.util.spec_from_file_location("simt_twin", HERE / "simt_twin.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"
BINDING = bytes(range(32))


def data(tag: str, n: int) -> bytes:
    return hashlib.shake_128(tag.encode()).digest(n)


def domain(count: int, owner: int = 7) -> CommitmentDomain:
    return CommitmentDomain(BINDING, owner, RangeIndexedDomain(count), hash="blake3-s256")


def template(schema: str) -> tuple[bytes, int]:
    """The segment key's message with the rank, position, L and j left zero (the kernels fill them), and its length."""
    name = schema.encode()
    head = b"\x05" + bytes(28) + len(name).to_bytes(2, "little") + name
    return head + bytes(64 - len(head)), len(head)


def level_keys(dom: bytes, levels: int = 10) -> bytes:
    out = T.buffer(32 * max(1, levels))
    T.launch("b3s_level_keys", 1, 32, [out, T.buffer(dom)], [levels])
    return out.raw


def rows_b3s(rows: bytes, m: int, rb: int, d: CommitmentDomain, rank0: int = 0, schema: str = ROW, skeys=None) -> bytes:
    """hash_rows_b3s's leaves; `skeys` (a buffer, or None for a null p3) takes the level keys its first CTA stores."""
    tmpl, tlen = template(schema)
    nseg = max(1, -(-rb // 256))
    out = T.buffer(32 * m)
    T.launch("hash_rows_b3s", m, 32 * -(-nseg // 32),
             [T.buffer(rows if rows else b"\0"), out, T.buffer(d.domain_id), skeys, T.buffer(tmpl)], [m, rb, rank0, tlen])
    return out.raw


def test_segment_level_keys_are_frame_b3s():
    d = domain(4)
    got = level_keys(d.domain_id, 10)
    assert [got[32 * l:32 * l + 32] for l in range(10)] == [b3s_level_key(d.domain_id, l) for l in range(10)]


@pytest.mark.parametrize("rb", [0, 1, 63, 64, 255, 256, 257, 1000, 1300, 2817, 4100, 16384, 32768])
def test_row_leaves_are_frame_b3s_at_every_length(rb):
    """3 rows at ranks 5–7: lengths within one block, one segment, one byte past it, short last segments, 6 and 12
    segments (odd levels), a length that isn't a multiple of 16 (the byte path at full segments), BF16 and FP32 rows at
    k = 8,192."""
    m, d = 3, domain(8)
    rows = data(f"b3s rows {rb}", m * rb)
    got = rows_b3s(rows, m, rb, d, rank0=5)
    assert [got[32 * r:32 * r + 32] for r in range(m)] == [d.leaf(5 + r, 5 + r, ROW, rows[r * rb:(r + 1) * rb]) for r in range(m)]


@pytest.mark.parametrize("rb", [0, 256, 257, 1300, 16384, 32768])
def test_row_leaves_store_their_level_keys(rb):
    """hash_rows_b3s derives its segment tree's level keys itself (no b3s_level_keys launch before it): with p3 set, its
    first CTA stores b3s_level_keys' bytes for the row's ⌈log2 segments⌉ levels there and nothing past them, and the leaves
    are those of a null p3."""
    m, d = 3, domain(8)
    rows = data(f"b3s keys {rb}", m * rb)
    levels = max(0, (max(1, -(-rb // 256)) - 1).bit_length())
    skeys = T.buffer(b"\xa5" * 32 * 10)
    assert rows_b3s(rows, m, rb, d, rank0=5, skeys=skeys) == rows_b3s(rows, m, rb, d, rank0=5)
    assert skeys.raw[:32 * 10] == level_keys(d.domain_id, levels)[:32 * levels] + b"\xa5" * 32 * (10 - levels)


@pytest.mark.parametrize("length", [0, 1, 32, 64, 200, 256])
def test_short_value_leaves_are_frame_b3s(length):
    """hash_leaves_b3s (a thread a value, one segment): 5 values at ranks 2–6, the tile tree's 32-byte leaves among them."""
    count, d = 5, domain(8)
    vals = data(f"b3s short {length}", count * length)
    tmpl, tlen = template(TILE)
    out = T.buffer(32 * count)
    T.launch("hash_leaves_b3s", 1, 128, [T.buffer(vals if vals else b"\0"), out, T.buffer(d.domain_id), T.buffer(tmpl)],
             [count, length, 2, tlen])
    assert [out.raw[32 * t:32 * t + 32] for t in range(count)] == [
        d.leaf(2 + t, 2 + t, TILE, vals[t * length:(t + 1) * length]) for t in range(count)]


@pytest.mark.parametrize("m,rb,schema", [(32, 32768, ROW), (8, 1000, ROW), (128, 32, TILE)])
def test_whole_trees_are_merkle_trees_roots(m, rb, schema):
    """A's m = 32 tree over 32 KB rows, a small tree of 1,000-byte rows, and the decode tile tree (128 leaves of 32 bytes):
    the -h2 leaves, then frame-b3's level keys and hash_tree_small, equal to MerkleTree's root under frame-b3s."""
    d = domain(m)
    vals = data(f"b3s tree {m} {rb}", m * rb)
    tree = T.buffer(32 * 2 * m)
    if schema == TILE:
        tmpl, tlen = template(TILE)
        T.launch("hash_leaves_b3s", -(-m // 128), 128, [T.buffer(vals), tree, T.buffer(d.domain_id), T.buffer(tmpl)], [m, rb, 0, tlen])
    else:
        tree = T.buffer(rows_b3s(vals, m, rb, d, schema=schema) + bytes(32 * m))
    keys = T.buffer(32 * 16)
    T.launch("hash_node_keys", 1, 32, [keys, T.buffer(d.domain_id)], [max(0, (m - 1).bit_length())])
    T.launch("hash_tree_small", 1, max(1, m // 2), [tree, keys], [m])
    want = MerkleTree(d, {r: vals[r * rb:(r + 1) * rb] for r in range(m)}, lambda _: schema).commitment.root
    assert tree.raw[32 * (2 * m - 2):32 * (2 * m - 1)] == want


def test_kernel_nodes_are_complete_keyed_blake3_calls():
    """The kernels' segment-tree node, the node over two leaves (hash_tree_small) and a one-segment leaf are complete keyed
    BLAKE3 calls (the ROOT flag included): equal to the standard keyed hash, and not to the bare PARENT compression."""
    from verity.primitives.crypto.blake3 import compress, keyed
    from verity.primitives.commitments.merkle import b3s_levels, b3s_segment_key, blake3_node_key

    def parent(key: bytes, block: bytes, flags: int) -> bytes:
        kw = tuple(int.from_bytes(key[4 * i:4 * i + 4], "little") for i in range(8))
        return b"".join(w.to_bytes(4, "little") for w in compress(kw, block, 0, 64, flags)[:8])

    d = domain(2)
    rows = data("complete calls", 2 * 512)
    got = rows_b3s(rows, 2, 512, d)
    for r in range(2):
        g = b3s_levels(d.domain_id, r, r, ROW, rows[512 * r:512 * r + 512])[0]
        k0 = b3s_level_key(d.domain_id, 0)
        assert got[32 * r:32 * r + 32] == keyed(k0, g[0] + g[1])
        assert got[32 * r:32 * r + 32] not in (parent(k0, g[0] + g[1], 4 | 16), parent(k0, g[0] + g[1], 4 | 8 | 16))
    tree, keys = T.buffer(got + bytes(64)), T.buffer(32 * 16)
    T.launch("hash_node_keys", 1, 32, [keys, T.buffer(d.domain_id)], [1])
    T.launch("hash_tree_small", 1, 1, [tree, keys], [2])
    k = blake3_node_key(d.domain_id, 0)
    assert tree.raw[64:96] == keyed(k, got) != parent(k, got, 4 | 8 | 16)
    one = data("one segment", 256)
    assert rows_b3s(one, 1, 256, d, rank0=1) == keyed(b3s_segment_key(d.domain_id, 1, 1, ROW, 256, 0), one)


def test_kernel_keys_follow_the_domain():
    """The kernels take no key but the domain id: the same rows under three domains give three sets of leaves, each the
    reference's under its own derived keys."""
    rows = data("keys follow", 3 * 1300)
    outs = set()
    for d in (domain(8), domain(8, owner=9), CommitmentDomain(bytes(32), 7, RangeIndexedDomain(8), hash="blake3-s256")):
        got = rows_b3s(rows, 3, 1300, d)
        assert [got[32 * r:32 * r + 32] for r in range(3)] == [d.leaf(r, r, ROW, rows[1300 * r:1300 * (r + 1)]) for r in range(3)]
        outs.add(got)
    assert len(outs) == 3


def f32_rows(tag: str, rows: int, k: int) -> np.ndarray:
    """FP32 words: a row of normal draws at one binade (2^-20 .. 2^20), so that the chains' terms are alike and their order
    shows in the sums; -0, a subnormal and a tiny value on the stats' sampled positions (multiples of 8); row 1 with 3e17,
    whose square is near FP32's top; and the last row all zeros."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "little"))
    x = (rng.standard_normal((rows, k)) * np.exp2(rng.integers(-20, 21, (rows, 1)))).astype(np.float32)
    x[:, 0], x[:, 8], x[:, k - 8] = -0.0, np.float32(1e-40), np.float32(-1e-3)
    x[1, 16 % k] = np.float32(3e17)
    x[-1] = 0.0
    return x.view(np.uint32)


@pytest.mark.parametrize("k", [256, 1024, 3072, 4096])
def test_rows_form_s_writes_the_row_leaves_and_s5_stats(k):
    """hash_rows_b3s_stats over m = 12 rows, hashing m_real = 8 (ranks 3–10): the leaves and level keys are hash_rows_b3s's,
    nothing past m_real's leaves is written, and every row's alpha, beta, 1/alpha and (s, rho), the filler rows' included,
    are the scheme's s5_stats (stats_s5's order: chain c over x[256 i + 8 c], then the butterfly)."""
    m, m_real, rb, d = 12, 8, 4 * k, domain(16)
    words = f32_rows(f"rows form s {k}", m, k)
    rows = words.tobytes()
    tmpl, tlen = template(ROW)
    nseg, poison = rb // 256, b"\xa5"
    leaves, skeys = T.buffer(poison * 32 * m), T.buffer(poison * 32 * 10)
    alpha, beta, ia, aux = (T.buffer(poison * n * m) for n in (4, 4, 4, 8))
    T.launch("hash_rows_b3s_stats", m, 32 * -(-nseg // 32),
             [T.buffer(rows), leaves, T.buffer(d.domain_id), skeys, T.buffer(tmpl), alpha, beta, ia, aux],
             [m_real, rb, 3, tlen, m], smem=nseg * (32 + 33))
    want_skeys = T.buffer(poison * 32 * 10)
    want = rows_b3s(rows, m_real, rb, d, rank0=3, skeys=want_skeys)
    assert leaves.raw[:32 * m_real] == want and leaves.raw[32 * m_real:32 * m] == poison * 32 * (m - m_real)
    assert want == b"".join(d.leaf(3 + r, 3 + r, ROW, rows[r * rb:(r + 1) * rb]) for r in range(m_real))
    assert skeys.raw[:32 * 10] == want_skeys.raw[:32 * 10]
    for r in range(m):
        s_, rho, a_, b_ = C.s5_stats([int(w) for w in words[r]])
        got = [int.from_bytes(buf.raw[n * r:n * r + 4], "little") for buf, n in ((alpha, 4), (beta, 4), (ia, 4), (aux, 8))]
        got.append(int.from_bytes(aux.raw[8 * r + 4:8 * r + 8], "little"))
        assert got == [a_, b_, fp32.div(0x3F800000, a_), s_, rho], (k, r)
