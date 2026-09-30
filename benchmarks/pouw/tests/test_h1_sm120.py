"""pouw_hash/h1_sm120.cuh's h1_rows (and h1_rows_p, whose L2::256B loads are plain ones on the host) and h1_tree commit the
verifier's bytes, through pearl_c's thread-for-thread host build (simt_twin) with what h1_tree adds (the grid, 16-byte and L2 loads, the fence and the last-CTA counter; blocks run one after
another): A's row leaves at every alignment against `verity.commitments`' frame-b3 leaves, and whole trees in one launch (pads,
level keys, the tile tree's own leaves, the levels above the CTAs' roots in the last CTA) against #449's kernels level for
level and `MerkleTree`'s root, with seed_A at A's root against the scheme's derivation; and decode's commitments in one
launch each (h1_commit: A's row leaves, tree, seed_A and E_A's line key; h1_tiles: the message digests, tile leaves and tile
tree) against the same references and hash_msg_v's and hash_leaf_p's; the level-key table (h1_level_keys) against
hash_node_keys, and the three tree kernels with their level keys read from it."""
from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest
from verity.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity_pouw.schemes import pearl_c, pearl_kw

BENCH = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


T = _load("simt_twin", BENCH / "pearl_c" / "simt_twin.py")
ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"
BINDING = bytes(range(32))
SEED_KEY = bytes.fromhex("6c8abb01868eaa7a07ed62333c4c6ae75daeb48c0f4e709fe7a6d8401f2a202a")


def seed_a_keys(salt: bytes, root_a: bytes, root_b: bytes, index: int) -> bytes:
    """h1_tree's p4: seed_A from A's root as the scheme derives it, then E_A's line key and the jackpot key from it."""
    seed = pearl_c._labelled(salt + root_a + root_b + index.to_bytes(8, "little"), "seed-A")
    return seed + pearl_kw._line_key(seed) + pearl_kw.subkey("jackpot", seed)

EXTRA = r"""
static inline uint4 __ldg(const uint4* p) { return *p; }
static inline uint4 __ldcg(const uint4* p) { return *p; }
static inline unsigned atomicInc(unsigned* p, unsigned v) { const unsigned old = *p; *p = old >= v ? 0 : old + 1; return old; }
static inline int __clzll(long long x) { return x ? __builtin_clzll((unsigned long long)x) : 64; }
"""


@lru_cache(maxsize=1)
def build() -> C.CDLL:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        pytest.skip("no C++ compiler for the host build")
    body = (BENCH / "pouw_hash" / "h1_sm120.cuh").read_text().replace('#include "../pearl_c/hash_sm120.cuh"', "").replace(
        "#pragma once", "").replace("extern __shared__ uint32_t sh[];", "uint32_t* sh = twin_dyn.data();").replace(
        "extern __shared__ uint32_t smem[];", "uint32_t* smem = twin_dyn.data();")
    runs = "".join(
        f'extern "C" void run1_{k}(const Args* a, int grid, int block, int smem) {{\n'
        f"  twin_dyn.assign(smem / 4 + 1, 0u); blockDim.x = block; gridDim.x = grid;\n"
        f"  for (int b = 0; b < grid; b++) {{\n"
        f"    blockIdx.x = b; TwinBarrier bar(block); twin_bar = &bar; std::vector<std::thread> ts;\n"
        f"    for (int t = 0; t < block; t++) ts.emplace_back([&, t] {{ threadIdx.x = t; {k}(*a); bar.drop(); }});\n"
        f"    for (auto& th : ts) th.join();\n  }}\n}}\n"
        for k in ("h1_rows", "h1_rows_p", "h1_level_keys", "h1_tree", "h1_commit", "h1_tiles"))
    out = Path(tempfile.mkdtemp(prefix="h1-sm120-simt-"))
    (out / "twin.cpp").write_text(T._source().replace(T.PRELUDE, T.PRELUDE + EXTRA, 1) + body + runs)
    subprocess.run([cxx, "-O1", "-std=c++20", "-pthread", "-shared", "-fPIC", "-o", str(out / "twin.so"), str(out / "twin.cpp")],
                   check=True, capture_output=True, text=True)
    return C.CDLL(str(out / "twin.so"))


def launch(kernel: str, grid: int, block: int, ptrs, ints, smem: int = 0) -> None:
    """A kernel of h1_sm120.cuh with pointers as buffers or raw addresses (0: none)."""
    a = T.Args()
    for j, p in enumerate(ptrs):
        a.p[j] = p if isinstance(p, int) else C.addressof(p)
    for j, v in enumerate(ints):
        a.i[j] = v
    getattr(build(), f"run1_{kernel}")(C.byref(a), grid, block, smem)


def data(tag: str, n: int) -> bytes:
    return hashlib.shake_128(tag.encode()).digest(n)


def aligned(n: int, off: int = 0):
    """A buffer of n bytes and the address `off` bytes past a 16-byte boundary in it."""
    buf = C.create_string_buffer(n + 32 + off)
    return buf, (C.addressof(buf) + 15) // 16 * 16 + off


@pytest.mark.parametrize("rb", [1, 3, 36, 64, 960, 961, 987, 988, 1024, 1040, 2011, 3072 - 64, 3072, 16384, 32768])
@pytest.mark.parametrize("warps,off", [(1, 0), (4, 0), (4, 4)])
@pytest.mark.parametrize("kernel", ["h1_rows", "h1_rows_p"])
def test_h1_rows_are_frame_b3_leaves_at_every_alignment(kernel, rb, warps, off):
    """Rows shorter than a block, of one chunk with the header (960) and a byte past it, 16-byte multiples and not (1,040;
    987, 988, 2,011), FP32 rows at k = 4,096 and 8,192 (17 and 33 chunks), from a 16-byte aligned base (the vector loads) and
    4 bytes past one (every block through row_msg_word)."""
    m = 5
    rows = data(f"rows {rb}", m * rb)
    _keep, src = aligned(m * rb, off)
    C.memmove(src, rows, len(rows))
    d = CommitmentDomain(BINDING, 3, RangeIndexedDomain(m), hash="blake3")
    out, dom = T.buffer(32 * m), T.buffer(d.domain_id)
    launch(kernel, -(-m // warps), 32 * warps, [src, out, dom], [m, rb], smem=4 * warps * T.rows_w_words(rb))
    assert [out.raw[32 * r:32 * r + 32] for r in range(m)] == [d.leaf(r, r, ROW, rows[r * rb:(r + 1) * rb]) for r in range(m)]


def reference(tree: bytes, count: int, width: int, dom: bytes, leaves: bytes | None) -> tuple[bytes, bytes]:
    """#449's kernels on the same level 0 (hash_tree_leaves over `leaves`, hash_pad, hash_node_keys, hash_nodes a level):
    the whole tree and its level keys."""
    ref, dm = T.buffer(tree), T.buffer(dom)
    depth = width.bit_length() - 1
    keys = T.buffer(32 * max(1, depth))
    if leaves is not None:
        T.launch("hash_tree_leaves", -(-count // 128), 128, [T.buffer(leaves), ref, dm], [count])
    if width > count:
        T.launch("hash_pad", -(-(width - count) // 128), 128, [ref, dm], [count, width])
    T.launch("hash_node_keys", 1, 32, [keys, dm], [depth])
    off, w, level = 0, width, 0
    while w > 1:
        below, above = C.addressof(ref) + 32 * off, C.addressof(ref) + 32 * (off + w)
        a = T.Args()
        a.p[0], a.p[1], a.p[2], a.i[0], a.i[1] = below, above, C.addressof(keys) + 32 * level, level, w // 2
        T.build().run_hash_nodes(C.byref(a), -(-(w // 2) // 128), 128, 0)
        off, w, level = off + w, w // 2, level + 1
    return ref.raw, keys.raw[:32 * depth]


def level_keys(dom: bytes, width: int, table: str | None) -> tuple[int | C.Array, object]:
    """The level-key pointer of h1_tree, h1_commit and h1_tiles and what keeps it alive: 0 (`table` None: from the domain
    id), hash_node_keys' keys of `dom` ("keys"), or those with every byte flipped ("flipped"); a level past the tree's too."""
    if table is None:
        return 0, None
    depth = width.bit_length() - 1
    keys = T.buffer(32 * (depth + 1))
    T.launch("hash_node_keys", 1, 32, [keys, T.buffer(dom)], [depth + 1])
    if table == "flipped":
        keys = T.buffer(bytes(x ^ 0xFF for x in keys.raw))
    return C.addressof(keys), keys


@pytest.mark.parametrize("n,depth", [(1, 0), (1, 1), (3, 6), (2, 22), (9, 3)])
def test_h1_level_keys_is_hash_node_keys_per_domain(n, depth):
    """The table over n domains (more than one CTA's threads at 9 x 3 and 2 x 22 on CTAs of 16): each domain's row the
    level keys hash_node_keys writes for it, and nothing written past the table."""
    doms = [CommitmentDomain(BINDING, j, RangeIndexedDomain(1 << depth), hash="blake3").domain_id for j in range(n)]
    table = T.buffer(32 * n * depth + 32)
    launch("h1_level_keys", -(-n * depth // 16), 16, [T.buffer(b"".join(doms)), table], [n, depth])
    for j, dom in enumerate(doms):
        want = T.buffer(32 * max(1, depth))
        T.launch("hash_node_keys", 1, 32, [want, T.buffer(dom)], [depth])
        assert table.raw[32 * depth * j:32 * depth * (j + 1)] == want.raw[:32 * depth]
    assert table.raw[32 * n * depth:] == bytes(32)


def grid(width: int, cta: int, leaves: bool) -> tuple[int, int]:
    """run.py's Pipeline.tree_grid."""
    t = max(1, min(width // 2, cta))
    while width // (2 * t) > 2 * t:
        t *= 2
    g = max(1, width // (2 * t))
    return g, (min(width, 1024) if g == 1 and leaves else t)


TABLES = [None, "keys", "flipped"]


@pytest.mark.parametrize("count", [1, 2, 3, 5, 32, 100, 128, 256, 1000])
@pytest.mark.parametrize("cta", [1, 4, 64])
@pytest.mark.parametrize("tile", [False, True])
@pytest.mark.parametrize("table", TABLES)
def test_h1_tree_is_the_frame_b3_tree_in_one_launch(count, cta, tile, table):
    """A tree over `count` leaves at level 0 (A's rows), or with `tile` over the tile tree's 32-byte values with its leaves
    hashed in the launch: every node and level key equal to #449's kernels', the root to MerkleTree's, on one CTA or on up to
    2 × threads CTAs with the last CTA's levels (cta = 1 and 4 at these widths), and the counter back at zero after it; A's
    tree with seed_A, its line key and jackpot key at the root as the scheme derives them.  The level keys computed, or read
    from a table (level_keys): the same tree, and a flipped table's keys in every level above level 0."""
    width = 1 << max(0, (count - 1).bit_length())
    d = CommitmentDomain(BINDING, 7, RangeIndexedDomain(count), hash="blake3")
    vals = [data(f"v {count} {i}", 32) for i in range(count)]
    level0 = b"" if tile else b"".join(d.leaf(i, i, ROW, vals[i]) for i in range(count))
    tree = T.buffer(level0 + bytes(32 * (2 * width - 1) - len(level0)))
    want, want_keys = reference(tree.raw, count, width, d.domain_id, b"".join(vals) if tile else None)
    g, t = grid(width, cta, tile)
    s = min(width, 2 * t)
    assert g * s == width and g <= 2 * t
    dom, ctr, keys, out = T.buffer(d.domain_id), T.buffer(4), T.buffer(32 * 32), T.buffer(96)
    salt, root_b = data("salt", 32), data("root-B", 32)
    sp = [0, 0, 0, 0] if tile else [out, T.buffer(SEED_KEY), T.buffer(salt), T.buffer(root_b)]
    lk, lk_buf = level_keys(d.domain_id, width, table)
    launch("h1_tree", g, t, [tree, dom, ctr, T.buffer(b"".join(vals)) if tile else 0, *sp, keys, lk], [count, width, 9],
           smem=32 * (s + 32))
    assert ctr.raw[:4] == bytes(4)
    if table == "flipped":
        assert keys.raw[:len(want_keys)] == lk_buf.raw[:len(want_keys)]
        assert tree.raw[:32 * width] == want[:32 * width] and (width == 1 or tree.raw[-32:] != want[-32:])
        return
    assert tree.raw == want and keys.raw[:len(want_keys)] == want_keys
    root = MerkleTree(d, dict(enumerate(vals)), lambda _: TILE if tile else ROW).commitment.root
    assert tree.raw[-32:] == root
    if not tile:
        assert out.raw == seed_a_keys(salt, root, root_b, 9)


@pytest.mark.parametrize("count,rb,warps", [(1, 36, 1), (3, 4096, 1), (5, 987, 2), (32, 32768, 1), (33, 4096, 1), (64, 64, 1),
                                            (100, 1040, 2)])
@pytest.mark.parametrize("table", TABLES)
def test_h1_commit_is_a_s_tree_and_seed_in_one_launch(count, rb, warps, table):
    """A's rows (one to 100, decode's 32 of k = 8,192, unaligned lengths, pads) hashed, built into the tree by the last CTA
    and seed_A and E_A's line key derived at its root in one launch: every node equal to #449's kernels' over the frame-b3
    row leaves, the root to MerkleTree's, seed_A and the line key the scheme's, and the counter back at zero.  The level keys
    computed or read from a table: the same bytes, and under a flipped table another root (above one row)."""
    width = 1 << max(0, (count - 1).bit_length())
    grid, block = -(-count // warps), 32 * warps
    assert width <= 2 * block
    rows = data(f"commit rows {count} {rb}", count * rb)
    _keep, src = aligned(count * rb)
    C.memmove(src, rows, len(rows))
    d = CommitmentDomain(BINDING, 3, RangeIndexedDomain(count), hash="blake3")
    vals = [rows[r * rb:(r + 1) * rb] for r in range(count)]
    level0 = b"".join(d.leaf(r, r, ROW, vals[r]) for r in range(count))
    want, _ = reference(level0 + bytes(32 * (2 * width - 1) - len(level0)), count, width, d.domain_id, None)
    salt, root_b, index = data("salt", 32), data("root-B", 32), 11
    tree, dom, ctr, line, seed = T.buffer(32 * (2 * width - 1)), T.buffer(d.domain_id), T.buffer(4), T.buffer(32), T.buffer(32)
    unit = T.buffer(salt + root_b + index.to_bytes(8, "little") + SEED_KEY)
    smem = max(4 * warps * T.rows_w_words(rb), 32 * (width + 32))
    lk, _keys = level_keys(d.domain_id, width, table)
    launch("h1_commit", grid, block, [src, tree, dom, ctr, line, unit, seed, lk], [count, rb, width], smem=smem)
    if table == "flipped":
        assert tree.raw[:32 * width] == want[:32 * width] and (width == 1 or tree.raw[-32:] != want[-32:])
        return
    root = MerkleTree(d, dict(enumerate(vals)), lambda _: ROW).commitment.root
    assert tree.raw == want and tree.raw[-32:] == root and ctr.raw[:4] == bytes(4)
    keys = seed_a_keys(salt, root, root_b, index)
    assert seed.raw == keys[:32] and line.raw == keys[32:64]


@pytest.mark.parametrize("rows,cols", [(17, 64), (5, 256), (40, 128), (32, 1024), (64, 512), (1, 16384)])
@pytest.mark.parametrize("table", TABLES)
def test_h1_tiles_is_the_tile_tree_in_one_launch(rows, cols, table):
    """Decode's hashing in one launch, a CTA a tile (one tile to 256, full tiles of 64 rows and partial ones of 1, 5, 17,
    32 and 40 rows, so one to four chunks a leaf): the digests hash_msg_v's, the tile leaves hash_leaf_p's, the tile tree
    #449's kernels' over them and its root MerkleTree's, and the counter back at zero.  The level keys computed or read from
    a table: the same bytes, and under a flipped table another root (above one tile)."""
    tiles = -(-rows // 64) * (cols // 64)
    width = 1 << max(0, (tiles - 1).bit_length())
    words = data(f"tiles {rows} {cols}", 8 * rows * cols)
    c, u = T.buffer(words[:4 * rows * cols]), T.buffer(words[4 * rows * cols:])
    k_msg, k_leaf = T.buffer(data("k_msg", 32)), T.buffer(data("k_leaf", 32))
    ref_dg, ref_leaves = T.buffer(rows * cols), T.buffer(32 * tiles)
    T.launch("hash_msg_v", -(-rows * cols // 32 // 128), 128, [c, u, ref_dg, k_msg], [rows, cols])
    T.launch("hash_leaf_p", -(-tiles // 32), 128, [ref_dg, ref_leaves, k_leaf], [rows, cols])
    d = CommitmentDomain(BINDING, 5, RangeIndexedDomain(tiles), hash="blake3")
    want, _ = reference(bytes(32 * (2 * width - 1)), tiles, width, d.domain_id, ref_leaves.raw)
    dg, leaves, tree = T.buffer(rows * cols), T.buffer(32 * tiles), T.buffer(32 * (2 * width - 1))
    dom, ctr = T.buffer(d.domain_id), T.buffer(4)
    lk, _keys = level_keys(d.domain_id, width, table)
    launch("h1_tiles", tiles, 128, [c, u, dg, k_msg, leaves, k_leaf, tree, dom, ctr, lk], [rows, cols, width],
           smem=32 * max(132, width + 32))
    if table == "flipped":
        assert tree.raw[:32 * width] == want[:32 * width] and (width == 1 or tree.raw[-32:] != want[-32:])
        return
    root = MerkleTree(d, {t: ref_leaves.raw[32 * t:32 * t + 32] for t in range(tiles)}, lambda _: TILE).commitment.root
    assert dg.raw == ref_dg.raw and leaves.raw == ref_leaves.raw
    assert tree.raw == want and tree.raw[-32:] == root and ctr.raw[:4] == bytes(4)
