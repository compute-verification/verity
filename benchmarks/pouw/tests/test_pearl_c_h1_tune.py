"""The -h1 kernels tuned for sm_120 (`hash_sm120.cuh`: hash_rows_w, hash_tree_small, hash_leaf_p, hash_msg_v) commit the
verifier's bytes: each against `verity.primitives.commitments`' frame-b3 (one-block leaf headers, one-compression nodes under level keys)
and Pearl-C's h1 digests, and against `hash.cuh`'s kernels, through a thread-for-thread host build."""
from __future__ import annotations

import hashlib
import os
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

from verity.primitives.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.primitives.crypto.blake3 import keyed

HERE = Path(__file__).parents[1] / "pearl_c"
spec = importlib.util.spec_from_file_location("simt_twin", HERE / "simt_twin.py")
T = importlib.util.module_from_spec(spec)
sys.modules["simt_twin"] = T
spec.loader.exec_module(T)

sys.path.insert(0, str(Path(__file__).parents[3] / "verity" / "protocols" / "accounting" / "work" / "pouw"))
from verity.protocols.accounting.work.pouw.schemes.pearl_c import H1_KEYS, TREE_HASH  # noqa: E402

ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"
BINDING = bytes(range(32))


def data(tag: str, n: int) -> bytes:
    return hashlib.shake_128(tag.encode()).digest(n)


def rows_w(rows: bytes, m: int, rb: int, dom: bytes, warps: int = 4) -> bytes:
    out = T.buffer(32 * m)
    T.launch("hash_rows_w", -(-m // warps), 32 * warps, [T.buffer(rows), out, T.buffer(dom)], [m, rb],
             smem=4 * warps * T.rows_w_words(rb))
    return out.raw


@pytest.mark.parametrize("rb", [1, 3, 36, 63, 960, 961, 987, 988, 1024, 1025, 2011, 3072 - 64, 3072, 16384, 32768])
def test_row_leaves_are_frame_b3s_at_every_alignment(rb):
    """Rows of every length class: shorter than a block, exactly one chunk with the 64-byte header (960) and one byte past it
    (961), rows 37 bytes past a chunk boundary (1,024 under the old header; 987, 988 and 2,011 odd or unaligned), and BF16 and
    FP32 rows at k = 8,192 (17 and 33 chunks)."""
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
    depth = max(0, (width - 1).bit_length())
    keys = T.buffer(32 * max(1, depth))
    T.launch("hash_node_keys", 1, 32, [keys, dom], [depth])
    T.launch("hash_tree_small", 1, max(1, width // 2), [tree, keys], [width])
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


@pytest.mark.parametrize("m", [16, 32, 48, 128])
def test_the_tuned_leaves_equal_hash_leaf_on_full_and_partial_tiles(m):
    n = 128
    dg, _ = digests(m, n)
    tiles = -(-m // 64) * (n // 64)
    a, b = T.buffer(32 * tiles), T.buffer(32 * tiles)
    T.launch("hash_leaf", -(-tiles // 64), 64, [T.buffer(dg), a, T.buffer(H1_KEYS["leaf"])], [m, n])
    T.launch("hash_leaf_p", -(-tiles // 32), 128, [T.buffer(dg), b, T.buffer(H1_KEYS["leaf"])], [m, n])
    assert a.raw == b.raw


def test_the_tile_tree_in_one_launch():
    t = 128
    vals = data("tiles", 32 * t)
    d = CommitmentDomain(BINDING, 11, RangeIndexedDomain(t), hash="blake3")
    tree, dom = T.buffer(32 * 2 * t), T.buffer(d.domain_id)
    T.launch("hash_tree_leaves", 1, 128, [T.buffer(vals), tree, dom], [t])
    keys = T.buffer(32 * 7)
    T.launch("hash_node_keys", 1, 32, [keys, dom], [7])
    T.launch("hash_tree_small", 1, 64, [tree, keys], [t])
    want = MerkleTree(d, {i: vals[32 * i:32 * i + 32] for i in range(t)}, lambda _: TILE).commitment.root
    assert tree.raw[32 * (2 * t - 2):32 * (2 * t - 1)] == want



def test_the_four_lane_compression_is_b3cs():
    """lat_chain's chain of 40 compressions, on one thread (b3c) and on four lanes (b3c4), both equal to BLAKE3's compression
    in `verity.primitives.crypto.blake3` chained the same way."""
    from verity.primitives.crypto.blake3 import IV, compress
    n, cv, m = 40, tuple(IV), [(0x9E3779B9 * (i + 1)) & 0xFFFFFFFF for i in range(16)]
    for k in range(n):
        cv = tuple(compress(cv, b"".join(w.to_bytes(4, "little") for w in m), k, 64, 16)[:8])
        m[0] ^= cv[0]
    want = b"".join(w.to_bytes(4, "little") for w in cv)
    for mode, threads in ((0, 1), (1, 4)):
        out = T.buffer(48)
        T.launch("lat_chain", 1, threads, [out], [n, mode])
        assert out.raw[16:48] == want, mode


@pytest.mark.parametrize("m,rb", [(32, 32768), (32, 16384), (5, 900)])
def test_rows_and_tree_in_one_launch_are_merkle_trees(m, rb):
    """hash_rows_wt: the row leaves, then the last CTA's pads and levels, equal to hash_rows_w + hash_pad + hash_tree_small's
    tree and MerkleTree's root; its counter is back at zero for the next call."""
    rows = data(f"wt {m} {rb}", m * rb)
    d = CommitmentDomain(BINDING, 7, RangeIndexedDomain(m), hash="blake3")
    width = 1 << max(0, (m - 1).bit_length())
    depth = max(0, (width - 1).bit_length())
    dom, keys, counter = T.buffer(d.domain_id), T.buffer(32 * max(1, depth)), T.buffer(16)
    T.launch("hash_node_keys", 1, 32, [keys, dom], [depth])
    tree = T.buffer(32 * 2 * width)
    T.launch("hash_rows_wt", -(-m // 4), 128, [T.buffer(rows), tree, dom, keys, counter], [m, rb, width],
             smem=4 * 4 * T.rows_w_words(rb))
    root = MerkleTree(d, {r: rows[r * rb:(r + 1) * rb] for r in range(m)}, lambda _: ROW).commitment.root
    assert tree.raw[32 * (2 * width - 2):32 * (2 * width - 1)] == root
    assert counter.raw[:4] == bytes(4)


def verifier(path, *extra) -> "subprocess.CompletedProcess":
    import subprocess
    env = {"PYTHONPATH": os.pathsep.join(str(Path(__file__).parents[3] / d) for d in (".",)),
           "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, str(HERE / "verify_h1.py"), *extra, "test", str(path)], capture_output=True, text=True,
                          env=env)


@pytest.mark.parametrize("hashing", ["h1", "h2"])
def test_the_verifier_accepts_the_references_transcript_and_rejects_a_flipped_byte(tmp_path, hashing):
    """verify_h1.py on a transcript made from the reference itself (A's tree, Pearl-C's digests and leaves, the tile tree;
    frame-b3 under -h1, frame-b3s under -h2) accepts it, and rejects it with one byte flipped in A's rows, A's tree, C~,
    the digests or the tile tree, and with its format relabelled."""
    import json as J

    from verity.primitives.commitments.merkle import tree_levels
    m, rb, n = 32, 256, 128
    rows = data("verify rows", m * rb)
    c, u = data("verify c", 4 * m * n), data("verify u", 4 * m * n)
    doms = {}
    for name, count in (("a", m), ("t", -(-m // 64) * (n // 64))):
        binding = hashlib.sha256(f"verify {name}".encode()).digest()
        d = CommitmentDomain(binding, 0, RangeIndexedDomain(count), hash=TREE_HASH[hashing])
        doms[name] = {"binding": binding.hex(), "owner": 0, "count": count, "domain": d.domain_id.hex(), "d": d}
    da, dt = doms["a"].pop("d"), doms["t"].pop("d")
    a_levels = tree_levels(da, [da.leaf(r, r, ROW, rows[r * rb:(r + 1) * rb]) for r in range(m)])
    dg, ref = digests(m, n)
    dgs = b"".join(keyed(H1_KEYS["message"], c[4 * (r * n + 32 * h):4 * (r * n + 32 * h + 32)] + u[4 * (r * n + 32 * h):4 * (r * n + 32 * h + 32)])
                   for r in range(m) for h in range(n // 32))
    per = n // 32
    leaves = [keyed(H1_KEYS["leaf"], b"".join(dgs[32 * (r * per + 2 * tj + h):32 * (r * per + 2 * tj + h + 1)] for r in range(m) for h in range(2)))
              for tj in range(n // 64)]
    t_levels = tree_levels(dt, [dt.leaf(t, t, TILE, v) for t, v in enumerate(leaves)])
    parts = [("a_rows", rows), ("a_tree", b"".join(b"".join(lv) for lv in a_levels)), ("c", c), ("u", u), ("digests", dgs),
             ("leaves", b"".join(leaves)), ("tile_tree", b"".join(b"".join(lv) for lv in t_levels))]

    def write(parts_, path, label=hashing):
        head = {"run": "test", "arm": "reference", "m": m, "rb": rb, "n": n, "hashing": label, "a": doms["a"], "t": doms["t"],
                "sections": [[k, len(v)] for k, v in parts_]}
        path.write_bytes(J.dumps(head).encode() + b"\n" + b"".join(v for _k, v in parts_))

    run = verifier
    good = tmp_path / "good.bin"
    write(parts, good)
    r = run(good)
    assert r.returncode == 0 and r.stdout.startswith(f"{hashing} verifier accept: test:") and "tile root" in r.stdout, r.stdout + r.stderr
    other = {"h1": "h2", "h2": "h1"}[hashing]
    write(parts, tmp_path / "relabelled.bin", other)
    r = run(tmp_path / "relabelled.bin")
    assert r.returncode == 1 and r.stdout.startswith(f"{other} verifier reject:"), r.stdout + r.stderr
    for i, (name, _v) in enumerate(parts):
        bad = list(parts)
        v = bytearray(bad[i][1])
        v[len(v) // 2] ^= 1
        bad[i] = (name, bytes(v))
        path = tmp_path / f"bad-{name}.bin"
        write(bad, path)
        r = run(path)
        assert r.returncode == 1 and r.stdout.startswith(f"{hashing} verifier reject:"), (name, r.stdout, r.stderr)


def test_the_verifier_regenerates_a_tagged_a_and_checks_its_h2_tree(tmp_path):
    """A prefill-style transcript (the tree only; row r is SHAKE128 of the tag with r): accepted with the rows' leaves
    split over processes, rejected with one byte of a leaf or of the root flipped."""
    import json as J

    from verity.primitives.commitments.merkle import tree_levels
    m, rb, tag = 16, 1000, "test tagged A {r}"
    binding = hashlib.sha256(b"verify tagged").digest()
    d = CommitmentDomain(binding, 0, RangeIndexedDomain(m), hash="blake3-s256")
    levels = tree_levels(d, [d.leaf(r, r, ROW, hashlib.shake_128(tag.format(r=r).encode()).digest(rb)) for r in range(m)])
    tree = b"".join(b"".join(lv) for lv in levels)
    head = {"run": "test", "arm": "prefill A h2", "m": m, "rb": rb, "n": 64, "hashing": "h2",
            "a": {"binding": binding.hex(), "owner": 0, "count": m, "domain": d.domain_id.hex()}, "a_rows_tag": tag,
            "sections": [["a_tree", len(tree)]]}
    for name, body, ok in (("good", tree, True), ("leaf", tree[:40] + bytes([tree[40] ^ 1]) + tree[41:], False),
                           ("root", tree[:-1] + bytes([tree[-1] ^ 1]), False)):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(J.dumps(head).encode() + b"\n" + body)
        r = verifier(path, "--jobs", "3")
        assert (r.returncode == 0) == ok and r.stdout.startswith(f"h2 verifier {'accept' if ok else 'reject'}:"), r.stdout + r.stderr
        if ok:
            assert "regenerated from their tag" in r.stdout


def segment_roots(rows: bytes, rb: int, s: int, key: bytes, levels: list[bytes]) -> bytes:
    out = b""
    for r in range(len(rows) // rb):
        level = [keyed(key, rows[r * rb + j:r * rb + j + s]) for j in range(0, rb, s)]
        lv = 0
        while len(level) > 1:
            level = [keyed(levels[lv], level[2 * i] + level[2 * i + 1]) for i in range(len(level) // 2)]
            lv += 1
        out += level[0]
    return out


@pytest.mark.parametrize("rb,s", [(32768, 256), (32768, 64), (16384, 256), (4096, 1024), (1024, 1024)])
def test_the_segment_layout_prototype_is_keyed_blake3_over_segments(rb, s):
    """hash_rows_seg (P1's prototype, not -h1): each s-byte segment's standard keyed BLAKE3 under the row-leaf key, then
    one-block nodes under the level keys, equal to `verity.primitives.crypto.blake3.keyed` composed the same way."""
    rows, key = data(f"seg rows {rb} {s}", 2 * rb), data("seg key", 32)
    levels = [data(f"seg level {l}", 32) for l in range(9)]
    out = T.buffer(64)
    T.launch("hash_rows_seg", 2, max(32, rb // s), [T.buffer(rows), out, T.buffer(key), T.buffer(b"".join(levels))], [2, rb, s])
    assert out.raw == segment_roots(rows, rb, s, key, levels)
