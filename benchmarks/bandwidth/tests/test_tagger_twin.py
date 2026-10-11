"""The GPU tagger's arithmetic and message layouts, transcribed from `tagger.cu` into Python and held to the core reference.

Each twin follows its kernel line by line (word order, byte swaps, padding words, the warp's lanes and shuffles), with the
SHA-512 constants read out of `tagger.cu`.  The device itself is checked on the node (`run.sh`: `bwtag`'s gates, then
`gpu.py windows` through `audit.Verifier`); these catch a layout or arithmetic slip before a GPU is leased.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import gpu  # noqa: E402
import reference  # noqa: E402
from verity.core.protocols.bandwidth import audit, tags  # noqa: E402
from verity.core.protocols.pous.verifier import MerkleTree, node_hash  # noqa: E402

CU = (HERE / "tagger.cu").read_text()
M64 = (1 << 64) - 1
P = tags.MERSENNE31
GF31, RO = tags.get("gf31-poly/v1"), tags.get("ro-sha512/v1")


def _consts(name: str) -> list[int]:
    body = CU[CU.index(f"#define {name}"):]
    return [int(x, 16) for x in re.findall(r"0x([0-9a-f]+)ULL", body[:body.index("}")])]


K, IV = _consts("SHA512_K"), _consts("SHA512_IV")


def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & M64


def compress(s: list[int], w: list[int]) -> list[int]:
    """`compress`: five passes of 16 rounds, the schedule updated in place at the top of each pass after the first."""
    w = list(w)
    a, b, c, d, e, f, g, h = s
    for r in range(0, 80, 16):
        for t in range(16):
            if r:
                w15, w2 = w[(t + 1) & 15], w[(t + 14) & 15]
                w[t] = (w[t] + (rotr(w15, 1) ^ rotr(w15, 8) ^ (w15 >> 7)) + w[(t + 9) & 15]
                        + (rotr(w2, 19) ^ rotr(w2, 61) ^ (w2 >> 6))) & M64
            t1 = (h + (rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)) + ((e & f) ^ (~e & g)) + K[r + t] + w[t]) & M64
            t2 = ((rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))) & M64
            h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M64, c, b, a, (t1 + t2) & M64
    return [(x + y) & M64 for x, y in zip(s, (a, b, c, d, e, f, g, h))]


def be_words(b: bytes) -> list[int]:
    return list(struct.unpack(f">{len(b) // 8}Q", b))


def digest(s: list[int]) -> bytes:
    return b"".join(x.to_bytes(8, "big") for x in s)


def bswap64(v: int) -> int:
    return int.from_bytes(v.to_bytes(8, "little"), "big")


def sha512(msg: bytes) -> bytes:
    """`sha512_host`."""
    m = msg + b"\x80"
    m += bytes((112 - len(m)) % 128) + bytes(8) + (8 * len(msg)).to_bytes(8, "big")
    s = list(IV)
    for off in range(0, len(m), 128):
        s = compress(s, be_words(m[off:off + 128]))
    return digest(s)


def prefix_bytes(domain: str, name: str, seed: bytes) -> bytes:
    """`prefix_midstate`'s block: domain ‖ NUL ‖ frame('s', name) ‖ frame('b', seed), zero-padded to 128."""
    p = domain.encode() + b"\0"
    for tag, body in ((b"s", name.encode()), (b"b", seed)):
        p += tag + len(body).to_bytes(8, "big") + body
    return p.ljust(128, b"\0")


def midstate(domain: str, name: str, seed: bytes) -> list[int]:
    return compress(list(IV), be_words(prefix_bytes(domain, name, seed)))


def red64(x: int) -> int:
    s = (x & P) + (x >> 31)
    t = (s & P) + (s >> 31)
    return t - P if t >= P else t


def mulmod(a: int, b: int) -> int:
    return red64(a * b)


SEED = bytes(range(32))
TAG_DOMAIN, LEAF_DOMAIN = "verity/bandwidth/tag/v1", "verity/bandwidth/tag-leaf/v1"


def test_sha512_and_its_constants():
    for msg in (b"", b"abc", bytes(range(256)) * 9):
        assert sha512(msg) == hashlib.sha512(msg).digest()


def test_prefixes_are_the_cores():
    for name in tags.TAGS:
        for domain, core in ((TAG_DOMAIN, tags.TAG_DOMAIN), (LEAF_DOMAIN, tags.LEAF_DOMAIN)):
            p = prefix_bytes(domain, name, SEED)
            assert p == tags.prefix(core, name, SEED)
            assert domain.encode() + b"\0" == core
            assert len(p) == 128


def test_fill_store_is_the_generated_store():
    for j in (0, 1, 36, (1 << 19) - 1):
        words = []
        for c in range(33):
            m = bytearray(128)
            m[0:30] = b"verity/bandwidth/test-block/v1"
            m[31:39] = j.to_bytes(8, "big")
            m[39:43] = c.to_bytes(4, "big")
            m[43] = 0x80
            m[127], m[126] = (43 * 8) & 0xFF, (43 * 8) >> 8
            s = compress(list(IV), be_words(bytes(m)))
            words += [bswap64(x) for x in s[:1 if c == 32 else 8]]
        assert len(words) == 257
        assert struct.pack("<257Q", *words) == reference.block(j)


def ro_tag(mid: list[int], j: int, block: bytes) -> list[int]:
    """`ro_kernel` for one lane: 16 chunks staged as (u64be j, E8[0..254]), then E8[255..256] and the padding."""
    e8 = struct.unpack("<257Q", block)
    s = list(mid)
    for t in range(16):
        w = [bswap64(e8[16 * t - 1 + wd]) if 16 * t - 1 + wd >= 0 else 0 for wd in range(16)]
        if t == 0:
            w[0] = j
        s = compress(s, w)
    w = [0] * 16
    w[0], w[1], w[2], w[15] = bswap64(e8[255]), bswap64(e8[256]), 0x80 << 56, (128 + 8 + 2056) * 8
    return compress(s, w)


def ro_leaf(leafmid: list[int], j: int, y: list[int]) -> bytes:
    w = [0] * 16
    w[0], w[1:9], w[9], w[15] = j, y, 0x80 << 56, (128 + 72) * 8
    return digest(compress(list(leafmid), w))


def test_ro_kernel_layout():
    mid, leafmid = midstate(TAG_DOMAIN, RO.name, SEED), midstate(LEAF_DOMAIN, RO.name, SEED)
    keyed, head = RO.keyed(SEED), tags.leaf_head(RO, SEED)
    for j in (0, 5, (1 << 19) - 1):
        b = reference.block(j)
        y = ro_tag(mid, j, b)
        assert digest(y) == keyed(j, b)
        assert ro_leaf(leafmid, j, y) == tags.leaf_hash(head, j, [keyed(j, b)])


def gf31_point(mid: list[int], j: int) -> int:
    c = 0
    while True:
        w = [0] * 16
        w[0], w[1], w[15] = j, (c << 32) | 0x80000000, (128 + 12) * 8
        for x in compress(list(mid), w):
            for v in (x >> 33, (x & 0xFFFFFFFF) >> 1):
                if v < P - 2:
                    return 2 + v
        c += 1


def test_gf31_points():
    mid = midstate(TAG_DOMAIN, GF31.name, SEED)
    for j in list(range(40)) + [(1 << 19) - 1, 1 << 40]:
        assert gf31_point(mid, j) == GF31.point(SEED, j)


def gf31_warp(block: bytes, g: int) -> int:
    """`gf31_kernel` for one block: 32 lanes over the row's 257 (w_2q, w_2q+1) pairs, the scan of powers of h = g², the
    per-lane Horner in H = h^32 and the butterfly sum."""
    x = struct.unpack("<514I", block)
    pair = [(x[2 * q], x[2 * q + 1]) for q in range(257)]
    v = [mulmod(g, g)] * 32
    for d in (1, 2, 4, 8, 16):
        v = [mulmod(v[l], v[l - d]) if l >= d else v[l] for l in range(32)]
    H = v[31]
    t = []
    for lane in range(32):
        c = 1 if lane == 0 else v[31 - lane]
        a = 0
        for k in range(8):
            px, py = pair[32 * k + lane]
            a = red64(a * H + red64(px * g + py))
        if lane == 0:
            a = red64(a * H + red64(pair[256][0] * g + pair[256][1]))
        t.append(mulmod(a, c))
    for d in (16, 8, 4, 2, 1):
        t = [(t[l] + t[l ^ d]) - P if t[l] + t[l ^ d] >= P else t[l] + t[l ^ d] for l in range(32)]
    assert len(set(t)) == 1
    return t[0]


def test_gf31_warp_decomposition():
    rng = random.Random(7)
    blocks = [reference.block(3), b"\xff" * 2056, bytes(2056), bytes(rng.randrange(256) for _ in range(2056))]
    for b in blocks:
        for g in (2, 3, P - 1, P - 2, rng.randrange(2, P)):
            assert gf31_warp(b, g) == tags.horner(b, g)


def gf31_leaf(leafmid: list[int], leaf: int, ys: list[int]) -> bytes:
    n = len(ys)
    w = [0] * 16
    w[0] = leaf
    for i in range(n):
        w[1 + i // 2] |= ys[i] << (0 if i & 1 else 32)
    w[1 + n // 2] |= (0x80 << 24) if n & 1 else (0x80 << 56)
    w[15] = (128 + 8 + 4 * n) * 8
    return digest(compress(list(leafmid), w))


def test_gf31_leaves_full_and_short():
    leafmid, head = midstate(LEAF_DOMAIN, GF31.name, SEED), tags.leaf_head(GF31, SEED)
    rng = random.Random(3)
    for n in range(1, 17):
        ys = [rng.randrange(P) for _ in range(n)]
        assert gf31_leaf(leafmid, 9, ys) == tags.leaf_hash(head, 9, [y.to_bytes(4, "big") for y in ys])


def tree_node(left: bytes, right: bytes) -> bytes:
    """`node_hash`: 0x01 ‖ L ‖ R as words shifted right one byte, then the 129-byte message's last byte and padding."""
    words, prev = [], 1
    for cur in be_words(left) + be_words(right):
        words.append(((prev << 56) | (cur >> 8)) & M64)
        prev = cur & 0xFF
    s = compress(list(IV), words)
    w = [0] * 16
    w[0], w[15] = (prev << 56) | (0x80 << 48), 129 * 8
    return digest(compress(s, w))


def test_node_hash():
    for k in range(3):
        left, right = hashlib.sha512(b"L%d" % k).digest(), hashlib.sha512(b"R%d" % k).digest()
        assert tree_node(left, right) == node_hash(left, right)


@pytest.mark.parametrize("n", range(1, 71))
def test_bottom_up_tree_is_rfc6962(n):
    leaves = [hashlib.sha512(b"leaf %d" % i).digest() for i in range(n)]
    levels = gpu.bottom_up(leaves, node_hash)
    tree = MerkleTree(leaves)
    assert levels[-1] == [tree.root]
    sizes = [len(level) for level in levels]
    k, expect = n, [n]
    while k > 1:
        k = (k + 1) // 2
        expect.append(k)
    assert sizes == expect
    for i in range(n):
        assert gpu.path(sizes, lambda lv, ix: levels[lv][ix], i) == tree.path(i)


SPAN = int(re.search(r"constexpr int SPAN = (\d+)", CU).group(1))
WIDE = 1 << int(re.search(r"constexpr uint64_t WIDE = 1 << (\d+);", CU).group(1))


def tree_launches(n0: int) -> tuple[int, list[tuple[int, list[int], int]]]:
    """`launch_tree` over `n0` leaves: the levels done one launch each, then the `node_span` launches (first level,
    level sizes, grid)."""
    sizes = [n0]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + 1) // 2)
    top, k = len(sizes) - 1, 0
    while k < top and sizes[k] > WIDE:
        k += 1
    wide, out = k, []
    while k < top:
        levels = min(SPAN, top - k)
        out.append((k, sizes[k:k + levels + 1], -(-sizes[k] // (1 << SPAN))))
        k += levels
    return wide, out


@pytest.mark.parametrize("n", [1, 2, 3, 255, 256, 257, 511, 512, 513, 1000, 32_768, 32_769, 70_001, 1 << 19])
def test_tree_launches_write_every_node_once_from_their_own_block(n):
    """`node_level` writes the wide levels; then `node_span`'s ranges: block b, step t writes node b·(T >> t) + tid of
    level k + t + 1 for tid < T >> t, reading nodes 2i and 2i + 1 of level k + t, which the same block wrote at step
    t − 1 (or an earlier launch did)."""
    threads = 1 << (SPAN - 1)
    sizes = [n]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + 1) // 2)
    writer: dict[tuple[int, int], int] = {(0, i): -1 for i in range(n)}
    wide, launches = tree_launches(n)
    for k in range(wide):
        assert sizes[k] > WIDE
        writer.update({(k + 1, i): -1 for i in range(sizes[k + 1])})
    for k, lv, grid in launches:
        for b in range(grid):
            for t in range(len(lv) - 1):
                for tid in range(threads >> t):
                    i = b * (threads >> t) + tid
                    if i >= lv[t + 1]:
                        continue
                    for src in (2 * i, 2 * i + 1):
                        if src < lv[t]:
                            assert writer[(k + t, src)] == b if t else (k + t, src) in writer
                    assert (k + t + 1, i) not in writer
                    writer[(k + t + 1, i)] = b
    assert set(writer) == {(level, i) for level, size in enumerate(sizes) for i in range(size)}


def test_reference_small_case_is_the_pinned_vectors():
    out = reference.small()
    vectors = json.loads(reference.VECTORS.read_text())
    assert out["seed"] == vectors["windows"][0]["seed"]
    assert out["roots"] == {a["tag"]: a["tag_root"] for a in vectors["audits"]}


def test_reference_roots_across_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(reference, "CHUNK", 16)
    leaves_out = tmp_path / "leaves.bin"
    got = reference.roots(70, SEED, 2, leaves_out)
    store = audit.Store(reference.geometry(70), reference.SALT, [reference.block(j) for j in range(70)])
    assert got == {name: audit.tag_pass(store, name, SEED).root.hex() for name in tags.TAGS}
    assert leaves_out.read_bytes() == b"".join(store.tree.leaves)


def test_geometry():
    assert reference.geometry(1 << 19).segments == 8 and reference.geometry(1 << 19).blocks == 1 << 19
    assert reference.geometry(37).blocks == 37
    with pytest.raises(ValueError):
        reference.geometry(70_000)
