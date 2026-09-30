"""pouw_hash/h1_sm120.cuh's h1_rows and h1_tree commit the verifier's bytes, through pearl_c's thread-for-thread host build
(simt_twin) with what h1_tree adds (the grid, 16-byte and L2 loads, the fence and the last-CTA counter; blocks run one after
another): A's row leaves at every alignment against `verity.commitments`' frame-b3 leaves, and whole trees in one launch (pads,
level keys, the tile tree's own leaves, the levels above the CTAs' roots in the last CTA) against #449's kernels level for
level and `MerkleTree`'s root, with seed_A at A's root against the scheme's derivation."""
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

BENCH = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


T = _load("simt_twin", BENCH / "pearl_c" / "simt_twin.py")
F = _load("pearl_c_sm120_fixture", BENCH / "pearl_c_sm120" / "fixture.py")
ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"
BINDING = bytes(range(32))
SEED_KEY = bytes.fromhex("6c8abb01868eaa7a07ed62333c4c6ae75daeb48c0f4e709fe7a6d8401f2a202a")

EXTRA = r"""
static Dim gridDim;
static inline uint4 __ldg(const uint4* p) { return *p; }
static inline uint4 __ldcg(const uint4* p) { return *p; }
static inline void __threadfence() {}
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
        f"    for (auto& th : ts) th.join();\n  }}\n}}\n" for k in ("h1_rows", "h1_tree"))
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
def test_h1_rows_are_frame_b3_leaves_at_every_alignment(rb, warps, off):
    """Rows shorter than a block, of one chunk with the header (960) and a byte past it, 16-byte multiples and not (1,040;
    987, 988, 2,011), FP32 rows at k = 4,096 and 8,192 (17 and 33 chunks), from a 16-byte aligned base (the vector loads) and
    4 bytes past one (every block through row_msg_word)."""
    m = 5
    rows = data(f"rows {rb}", m * rb)
    _keep, src = aligned(m * rb, off)
    C.memmove(src, rows, len(rows))
    d = CommitmentDomain(BINDING, 3, RangeIndexedDomain(m), hash="blake3")
    out, dom = T.buffer(32 * m), T.buffer(d.domain_id)
    launch("h1_rows", -(-m // warps), 32 * warps, [src, out, dom], [m, rb], smem=4 * warps * T.rows_w_words(rb))
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


def grid(width: int, cta: int, leaves: bool) -> tuple[int, int]:
    """run.py's Pipeline.tree_grid."""
    t = max(1, min(width // 2, cta))
    while width // (2 * t) > 2 * t:
        t *= 2
    g = max(1, width // (2 * t))
    return g, (min(width, 1024) if g == 1 and leaves else t)


@pytest.mark.parametrize("count", [1, 2, 3, 5, 32, 100, 128, 256, 1000])
@pytest.mark.parametrize("cta", [1, 4, 64])
@pytest.mark.parametrize("tile", [False, True])
def test_h1_tree_is_the_frame_b3_tree_in_one_launch(count, cta, tile):
    """A tree over `count` leaves at level 0 (A's rows), or with `tile` over the tile tree's 32-byte values with its leaves
    hashed in the launch: every node and level key equal to #449's kernels', the root to MerkleTree's, on one CTA or on up to
    2 × threads CTAs with the last CTA's levels (cta = 1 and 4 at these widths), and the counter back at zero after it; A's
    tree with seed_A, its line key and jackpot key at the root as the scheme derives them."""
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
    launch("h1_tree", g, t, [tree, dom, ctr, T.buffer(b"".join(vals)) if tile else 0, *sp, keys], [count, width, 9],
           smem=32 * (s + 32))
    assert tree.raw == want and keys.raw[:len(want_keys)] == want_keys and ctr.raw[:4] == bytes(4)
    root = MerkleTree(d, dict(enumerate(vals)), lambda _: TILE if tile else ROW).commitment.root
    assert tree.raw[-32:] == root
    if not tile:
        assert out.raw == F.seed_a_keys(salt, root, root_b, 9)
