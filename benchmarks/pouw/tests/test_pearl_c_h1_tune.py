"""The -h1 kernels tuned for sm_120 (`hash.cuh`: hash_rows_w, hash_tree_small, hash_leaf_p, hash_msg_v) commit the verifier's
bytes: each against `verity.commitments`' frame-b3 and Pearl-C's h1 digests, through a thread-for-thread host build."""
from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

from verity.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.commitments.blake3 import keyed

HERE = Path(__file__).parents[1] / "pearl_c"
spec = importlib.util.spec_from_file_location("simt_twin", HERE / "simt_twin.py")
T = importlib.util.module_from_spec(spec)
sys.modules["simt_twin"] = T
spec.loader.exec_module(T)

sys.path.insert(0, str(Path(__file__).parents[3] / "protocols" / "pouw"))
from verity_pouw.schemes.pearl_c import H1_KEYS  # noqa: E402

ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"
BINDING = bytes(range(32))


def data(tag: str, n: int) -> bytes:
    return hashlib.shake_128(tag.encode()).digest(n)


def rows_w(rows: bytes, m: int, rb: int, dom: bytes, warps: int = 4) -> bytes:
    out = T.buffer(32 * m)
    T.launch("hash_rows_w", -(-m // warps), 32 * warps, [T.buffer(rows), out, T.buffer(dom)], [m, rb],
             smem=4 * warps * T.rows_w_words(rb))
    return out.raw


@pytest.mark.parametrize("rb", [1, 3, 36, 987, 988, 1024, 1025, 2011, 3072 - 37, 3072, 16384, 32768])
def test_row_leaves_are_frame_b3s_at_every_alignment(rb):
    """Rows of every length class: shorter than the prefix, exactly one chunk with it (987), one byte and 37 bytes past a
    chunk boundary (988, 1,024), odd lengths (the byte path), and BF16 and FP32 rows at k = 8,192 (17 and 33 chunks)."""
    m = 5
    rows = data(f"rows {rb}", m * rb)
    d = CommitmentDomain(BINDING, 3, RangeIndexedDomain(m), hash="blake3")
    got = rows_w(rows, m, rb, d.domain_id)
    assert [got[32 * r:32 * r + 32] for r in range(m)] == [d.leaf(r, r, ROW, rows[r * rb:(r + 1) * rb]) for r in range(m)]


@pytest.mark.parametrize("m,rb", [(32, 16384), (5, 900), (1, 64)])
def test_a_whole_row_tree_in_one_launch_is_merkle_trees_root(m, rb):
    rows = data(f"tree {m} {rb}", m * rb)
    d = CommitmentDomain(BINDING, 7, RangeIndexedDomain(m), hash="blake3")
    width = 1 << max(0, (m - 1).bit_length())
    tree, dom = T.buffer(32 * (2 * width)), T.buffer(d.domain_id)
    T.launch("hash_rows_w", -(-m // 4), 128, [T.buffer(rows), tree, dom], [m, rb], smem=4 * 4 * T.rows_w_words(rb))
    if width > m:
        T.launch("hash_pad", 1, 128, [tree, dom], [m, width])
    T.launch("hash_tree_small", 1, max(1, width // 2), [tree, dom], [width])
    want = MerkleTree(d, {r: rows[r * rb:(r + 1) * rb] for r in range(m)}, lambda _: ROW)
    levels, off = [], 0
    w = width
    while w >= 1:
        levels.append(tree.raw[off:off + 32 * w])
        off, w = off + 32 * w, w // 2
    assert levels[-1] == want.commitment.root


def digests(m: int, n: int) -> tuple[bytes, list[bytes]]:
    c = data(f"c {m} {n}", 4 * m * n)
    u = data(f"u {m} {n}", 4 * m * n)
    per = n // 32
    ref = [keyed(H1_KEYS["message"], c[4 * (r * n + 32 * h):4 * (r * n + 32 * h + 32)] + u[4 * (r * n + 32 * h):4 * (r * n + 32 * h + 32)])
           for r in range(m) for h in range(per)]
    out = T.buffer(32 * m * per)
    T.launch("hash_msg_v", -(-m * per // 128), 128, [T.buffer(c), T.buffer(u), out, T.buffer(H1_KEYS["message"])], [m, n])
    return out.raw, ref


@pytest.mark.parametrize("m,n", [(32, 128), (16, 64), (48, 64), (64, 192), (128, 128)])
def test_message_digests_and_tile_leaves_including_partial_tiles(m, n):
    """hash_msg_v's digests, and hash_leaf_p's leaves: full 64 x 64 tiles, and the partial tiles of m = 16, 32 and 48 (the
    decode headline's m = 32 among them), each the reference's leaf over the tile's own rows."""
    dg, ref = digests(m, n)
    assert dg == b"".join(ref)
    per, tn = n // 32, n // 64
    tiles = -(-m // 64) * tn
    out = T.buffer(32 * tiles)
    T.launch("hash_leaf_p", -(-tiles // 32), 128, [T.buffer(dg), out, T.buffer(H1_KEYS["leaf"])], [m, n])
    for t in range(tiles):
        ti, tj = divmod(t, tn)
        rows = range(64 * ti, min(m, 64 * ti + 64))
        joined = b"".join(ref[r * per + 2 * tj + h] for r in rows for h in range(2))
        assert out.raw[32 * t:32 * t + 32] == keyed(H1_KEYS["leaf"], joined), (m, n, t)


def test_the_tuned_leaves_equal_hash_leaf_on_full_tiles():
    m, n = 128, 128
    dg, _ = digests(m, n)
    tiles = (m // 64) * (n // 64)
    a, b = T.buffer(32 * tiles), T.buffer(32 * tiles)
    T.launch("hash_leaf", 1, 64, [T.buffer(dg), a, T.buffer(H1_KEYS["leaf"])], [m, n])
    T.launch("hash_leaf_p", 1, 128, [T.buffer(dg), b, T.buffer(H1_KEYS["leaf"])], [m, n])
    assert a.raw == b.raw


def test_the_tile_tree_in_one_launch():
    t = 128
    vals = data("tiles", 32 * t)
    d = CommitmentDomain(BINDING, 11, RangeIndexedDomain(t), hash="blake3")
    tree, dom = T.buffer(32 * 2 * t), T.buffer(d.domain_id)
    T.launch("hash_tree_leaves", 1, 128, [T.buffer(vals), tree, dom], [t])
    T.launch("hash_tree_small", 1, 64, [tree, dom], [t])
    want = MerkleTree(d, {i: vals[32 * i:32 * i + 32] for i in range(t)}, lambda _: TILE).commitment.root
    assert tree.raw[32 * (2 * t - 2):32 * (2 * t - 1)] == want
