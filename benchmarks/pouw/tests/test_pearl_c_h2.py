"""The -h2 kernels (`pearl_c/hash_h2.cuh`) against frame-b3s, `verity.commitments.merkle`'s reference, through
`simt_twin.py`'s thread-for-thread host build: the segment-tree level keys, row leaves at every length class (the aligned
16-byte-load path and the byte path, short last segments, odd levels), short-value leaves, and whole trees against
`MerkleTree`'s root."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from verity.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.commitments.merkle import b3s_level_key

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


def rows_b3s(rows: bytes, m: int, rb: int, d: CommitmentDomain, rank0: int = 0, schema: str = ROW) -> bytes:
    tmpl, tlen = template(schema)
    nseg = max(1, -(-rb // 256))
    out = T.buffer(32 * m)
    T.launch("hash_rows_b3s", m, 32 * -(-nseg // 32),
             [T.buffer(rows if rows else b"\0"), out, T.buffer(d.domain_id), T.buffer(level_keys(d.domain_id)), T.buffer(tmpl)],
             [m, rb, rank0, tlen])
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
    from verity.commitments.blake3 import compress, keyed
    from verity.commitments.merkle import b3s_levels, b3s_segment_key, blake3_node_key

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


def test_seed_kernel_is_the_schemes_row_and_unit_seeds():
    """b3_seeds: in mode 1 each row's seed is `pearl_c.row_seed` (the one-row unit's seed_A over the row's frame-b3s leaf in
    its unit's A tree, index·2^32 + row), in mode 0 the unit's `unit_seed_a` over its root; the kernel only ever sees the
    leaves, so the leaves come from the kernels too."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "protocols" / "pouw"))
    from verity.commitments.blake3 import blake3
    from verity_pouw.protocol import Context, Matmul, Shape
    from verity_pouw.schemes import pearl_c as PC

    m, rb, index = 8, 1024, 5
    salt, root_b = hashlib.sha256(b"seed salt").digest(), hashlib.sha256(b"seed root_B").digest()
    ctx = Context(salt, Matmul(index, Shape(m, rb // 4, 64), 0), bytes(32), root_b)
    dom = PC.activation_domain(ctx, "h3")
    rows = data("seed rows", m * rb)
    leaves = rows_b3s(rows, m, rb, dom)
    key = T.buffer(blake3(b"pearl-c/v0/seed-A"))
    out = T.buffer(32 * m)
    T.launch("b3_seeds", 1, 128, [T.buffer(leaves), out, key, T.buffer(salt), T.buffer(root_b)], [m, index, 1])
    assert [out.raw[32 * r:32 * r + 32] for r in range(m)] == [PC.row_seed(ctx, r, rows[r * rb:(r + 1) * rb]) for r in range(m)]
    root = hashlib.sha256(b"a root").digest()
    one = T.buffer(32)
    T.launch("b3_seeds", 1, 128, [T.buffer(root), one, key, T.buffer(salt), T.buffer(root_b)], [1, index, 0])
    assert one.raw == PC.unit_seed_a(salt, root, root_b, index)


@pytest.mark.parametrize("m,rb,n", [(3, 1024, 256), (2, 768, 256), (2, 32768, 8192)])
def test_the_fused_producer_writes_derive_rows_wides_a_and_hashes_those_bytes(m, rb, n):
    """derive_rows_b3s (A's producer with the -h2 segment leaves fused): its A equals derive_rows_wide's byte for byte, and its
    leaves are frame-b3s's segment hashes of the bytes it stored, so they bind what forming reads. b3s_seg_trees over them
    then gives the reference row leaves and, with seeds on, `pearl_c.row_seed` (3 segments: an odd level)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "protocols" / "pouw"))
    from verity.commitments.blake3 import blake3
    from verity.commitments.merkle import b3s_levels
    from verity_pouw.protocol import Context, Matmul, Shape
    from verity_pouw.schemes import pearl_c as PC

    index = 5
    salt, root_b = hashlib.sha256(b"fused salt").digest(), hashlib.sha256(b"fused root_B").digest()
    ctx = Context(salt, Matmul(index, Shape(m, rb // 4, 64), 0), bytes(32), root_b)
    d = PC.activation_domain(ctx, "h3")
    y = T.buffer(data(f"fused y {m} {rb} {n}", 2 * m * n))
    wide = T.buffer(m * rb)
    T.launch("derive_rows_wide", -(-m * rb // 2 // 256), 256, [y, wide], [m, rb, n])
    nseg = rb // 256
    fused, leaves = T.buffer(m * rb), T.buffer(32 * m * nseg)
    tmpl, tlen = template(ROW)
    T.launch("derive_rows_b3s", -(-m * nseg // 128), 128, [y, fused, leaves, T.buffer(d.domain_id), T.buffer(tmpl)], [m, rb, n, tlen])
    assert fused.raw == wide.raw
    rows = [fused.raw[r * rb:(r + 1) * rb] for r in range(m)]
    assert leaves.raw == b"".join(b"".join(b3s_levels(d.domain_id, r, r, ROW, rows[r])[0]) for r in range(m))
    row_leaves, seeds = T.buffer(32 * m), T.buffer(32 * m)
    T.launch("b3s_seg_trees", m, 32 * -(-nseg // 32),
             [leaves, row_leaves, T.buffer(level_keys(d.domain_id)), T.buffer(blake3(b"pearl-c/v0/seed-A")), T.buffer(salt),
              T.buffer(root_b), seeds], [m, nseg, index, 1])
    assert [row_leaves.raw[32 * r:32 * r + 32] for r in range(m)] == [d.leaf(r, r, ROW, rows[r]) for r in range(m)]
    assert [seeds.raw[32 * r:32 * r + 32] for r in range(m)] == [PC.row_seed(ctx, r, rows[r]) for r in range(m)]
    vec = T.buffer(m * rb)
    T.launch("derive_rows_vec", -(-m * nseg // 128), 128, [y, vec], [m, rb, n])
    assert vec.raw == wide.raw
    for mode in (1, 2):
        a2, leaves2, seeds2 = T.buffer(m * rb), T.buffer(32 * m), T.buffer(32 * m)
        T.launch("derive_rows_b3s", m, nseg,
                 [y, a2, leaves2, T.buffer(d.domain_id), T.buffer(tmpl), T.buffer(level_keys(d.domain_id)),
                  T.buffer(blake3(b"pearl-c/v0/seed-A")), T.buffer(salt), T.buffer(root_b), seeds2], [m, rb, n, tlen, mode, index])
        assert a2.raw == wide.raw and leaves2.raw == row_leaves.raw
        assert (seeds2.raw == seeds.raw) if mode == 2 else not any(seeds2.raw)


@pytest.mark.parametrize("m,rb,n,block", [(2, 32768, 8192, 1024), (3, 768, 256, 32), (2, 1024, 256, 64)])
def test_the_coalesced_producer_stores_the_rows_and_commits_those_bytes(m, rb, n, block):
    """derive_rows_b3s_coal (16-byte coalesced stores, segments staged in padded shared memory, then hashed): its A equals
    derive_rows_wide's byte for byte, and its row leaves and -h3 row seeds are the reference's over those stored bytes (3
    segments: an odd level); derive_rows_coal and derive_rows_v16 store the same bytes without hashing."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "protocols" / "pouw"))
    from verity.commitments.blake3 import blake3
    from verity_pouw.protocol import Context, Matmul, Shape
    from verity_pouw.schemes import pearl_c as PC

    index = 5
    salt, root_b = hashlib.sha256(b"coal salt").digest(), hashlib.sha256(b"coal root_B").digest()
    ctx = Context(salt, Matmul(index, Shape(m, rb // 4, 64), 0), bytes(32), root_b)
    d = PC.activation_domain(ctx, "h3")
    y = T.buffer(data(f"coal y {m} {rb} {n}", 2 * m * n))
    wide = T.buffer(m * rb)
    T.launch("derive_rows_wide", -(-m * rb // 2 // 256), 256, [y, wide], [m, rb, n])
    rows = [wide.raw[r * rb:(r + 1) * rb] for r in range(m)]
    tmpl, tlen = template(ROW)
    for mode in (1, 2):
        a2, leaves, seeds = T.buffer(m * rb), T.buffer(32 * m), T.buffer(32 * m)
        T.launch("derive_rows_b3s_coal", m, block,
                 [y, a2, leaves, T.buffer(d.domain_id), T.buffer(tmpl), T.buffer(level_keys(d.domain_id)),
                  T.buffer(blake3(b"pearl-c/v0/seed-A")), T.buffer(salt), T.buffer(root_b), seeds], [m, rb, n, tlen, mode, index])
        assert a2.raw == wide.raw
        assert [leaves.raw[32 * r:32 * r + 32] for r in range(m)] == [d.leaf(r, r, ROW, rows[r]) for r in range(m)]
        want = b"".join(PC.row_seed(ctx, r, rows[r]) for r in range(m)) if mode == 2 else bytes(32 * m)
        assert seeds.raw == want
    for kernel, grid, blk in (("derive_rows_coal", m, block), ("derive_rows_v16", -(-m * rb // 16 // 256), 256)):
        out = T.buffer(m * rb)
        T.launch(kernel, grid, blk, [y, out], [m, rb, n])
        assert out.raw == wide.raw, kernel


@pytest.mark.parametrize("kernel", ["derive_rows_b3s_split", "derive_rows_b3s_wide"])
@pytest.mark.parametrize("m,rb,n", [(2, 32768, 8192), (3, 8192, 4096), (2, 12288, 4096)])
def test_the_split_producer_commits_its_stores_and_resets_its_counters(m, rb, n, kernel):
    """derive_rows_b3s_split (a thread a segment in CTAs of 32; the last CTA of each row builds its segment tree and seed) and
    derive_rows_b3s_wide (a thread a 16-byte piece, 16 segments and their subtree a CTA; 12 KB rows: 3 subtrees, an odd level): its
    A is derive_rows_wide's, its row leaves and -h3 row seeds are the reference's over those bytes, and its per-row counters
    are back at zero, so a second launch (as a graph replays) gives the same."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "protocols" / "pouw"))
    from verity.commitments.blake3 import blake3
    from verity_pouw.protocol import Context, Matmul, Shape
    from verity_pouw.schemes import pearl_c as PC

    if kernel == "derive_rows_b3s_split" and (rb // 256) % 32:
        pytest.skip("the split producer's rows are whole 32-segment CTAs")
    index = 5
    salt, root_b = hashlib.sha256(b"split salt").digest(), hashlib.sha256(b"split root_B").digest()
    ctx = Context(salt, Matmul(index, Shape(m, rb // 4, 64), 0), bytes(32), root_b)
    d = PC.activation_domain(ctx, "h3")
    y = T.buffer(data(f"split y {m} {rb} {n}", 2 * m * n))
    wide = T.buffer(m * rb)
    T.launch("derive_rows_wide", -(-m * rb // 2 // 256), 256, [y, wide], [m, rb, n])
    rows = [wide.raw[r * rb:(r + 1) * rb] for r in range(m)]
    tmpl, tlen = template(ROW)
    segs = rb // 256
    a2, leaves, seeds, segl, counters = T.buffer(m * rb), T.buffer(32 * m), T.buffer(32 * m), T.buffer(32 * m * segs), T.buffer(4 * m)
    args = [y, a2, leaves, T.buffer(d.domain_id), T.buffer(tmpl), T.buffer(level_keys(d.domain_id)),
            T.buffer(blake3(b"pearl-c/v0/seed-A")), T.buffer(salt), T.buffer(root_b), seeds, segl, counters]
    for _ in range(2):
        if kernel == "derive_rows_b3s_split":
            T.launch(kernel, m * segs // 32, 32, args, [m, rb, n, tlen, 2, index])
        else:
            T.launch(kernel, m * rb // 4096, 256, args, [m, rb, n, tlen, 2, index])
        assert a2.raw == wide.raw and counters.raw[:4 * m] == bytes(4 * m)
        assert [leaves.raw[32 * r:32 * r + 32] for r in range(m)] == [d.leaf(r, r, ROW, rows[r]) for r in range(m)]
        assert seeds.raw == b"".join(PC.row_seed(ctx, r, rows[r]) for r in range(m))
