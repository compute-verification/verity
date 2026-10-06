"""hash.cuh and hash_h3s.cuh on the host (pearl_c/hash_host.cpp, built with g++): each hashing kernel, launched as run.py
launches it, writes the bytes the verifier's reference computes.  A's row leaves and frame-b3 trees (pads, level keys,
one-compression nodes) at rows of 1 to 75 chunks, the message digests, the tile leaves full and partial (decode's 32 x 64),
Pearl's tickets, the tile tree, a unit's seed_A over its root with its line and jackpot keys, and -h3's row seeds at the pinned
vectors (Pearl-C's and, under its own label, Pearl-C4's) and over an -h3 call's frame-b3s row leaves; -h3s's SHA-512 row
digests, its seeds and line keys over them at the pinned vectors, its line raws, tickets and tile leaves, and its hm96-sha512
frame-v3-sha512 trees under ChaCha20 salts; then run.py's commit and hashing phases end to end."""

from __future__ import annotations

import ctypes
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "pearl_c"
sys.path.insert(0, str(HERE))

import fixture as F  # noqa: E402
import run as R  # noqa: E402
from verity.primitives.commitments import MerkleTree  # noqa: E402
from verity.primitives.crypto.blake3 import blake3  # noqa: E402
from verity.primitives.commitments.merkle import tree_levels  # noqa: E402
from verity.protocols.accounting.work.pouw import audit  # noqa: E402
from verity.protocols.accounting.work.pouw.audit import ROW_SCHEMA, TILE_SCHEMA, activation_binding, rows_domain  # noqa: E402
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C  # noqa: E402
from verity.protocols.accounting.work.pouw.schemes import pearl_c4 as C4  # noqa: E402
from verity.protocols.accounting.work.pouw.schemes import pearl_kw as P  # noqa: E402


class Host:
    """run.py's `Dev` over hash.cuh built for the host: buffers are numpy arrays, and a launch runs its grid in order."""

    def __init__(self, lib):
        self.lib, self.bufs = lib, {}

    def alloc(self, n: int) -> int:
        a = np.zeros(max(n, 16), np.uint8)
        self.bufs[a.ctypes.data] = a
        return a.ctypes.data

    def put(self, data: bytes) -> int:
        p = self.alloc(len(data))
        self.bufs[p][:len(data)] = np.frombuffer(data, np.uint8)
        return p

    def read(self, p: int, n: int) -> bytes:
        base = next(b for b, a in self.bufs.items() if b <= p and p + n <= b + a.size)
        return self.bufs[base][p - base:p - base + n].tobytes()

    def free(self, p: int) -> None:
        self.bufs.pop(p, None)

    def launch(self, name: str, grid, block: int, smem: int = 0, p=(), i=(), stream=None) -> None:
        a = R.Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        assert isinstance(grid, int) and self.lib.launch(name.encode(), grid, block, ctypes.byref(a)) == 0, name


@pytest.fixture(scope="module")
def host(tmp_path_factory) -> Host:
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of hash.cuh")
    lib = tmp_path_factory.mktemp("hash-host") / "hash_host.so"
    subprocess.run([gxx, "-O2", "-std=c++20", "-pthread", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas",
                    f"-I{HERE}", "-o", str(lib), str(HERE / "hash_host.cpp")], check=True)
    cdll = ctypes.CDLL(str(lib))
    cdll.launch.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    cdll.kernel_name.restype = ctypes.c_char_p
    return Host(cdll)


def tree(host: Host, count: int, domain_id: bytes) -> SimpleNamespace:
    """What `Pipeline.levels` reads: a tree buffer, its level keys' buffer and the domain id."""
    return SimpleNamespace(dev=host, hs=None, b={"tree": host.alloc((2 * R.pow2(count) - 1) * 32), "keys": host.alloc(32 * R.depth(count))},
                           keys={"dom": host.put(domain_id)})


def test_every_hash_kernel_is_built(host):
    names = set(re.findall(r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\(', "".join((HERE / f).read_text() for f in ("hash.cuh", "hash_h3s.cuh"))))
    built = {host.lib.kernel_name(i).decode() for i in range(len(names))}
    assert built == names and host.lib.kernel_name(len(names)) is None


@pytest.mark.parametrize("rb,count", [(0, 1), (512, 3), (960, 2), (1001, 5), (4096, 3), (7168, 2), (11264, 3), (75776, 2)])
def test_row_trees_are_the_verifiers(host, rb, count):
    """hash_rows over rows of rb bytes (a row's leaf is 64 + rb bytes: 1, 1, 2, 2, 5, 8, 12 and 75 chunks; 1001 takes the
    byte-wise path), then run.py's levels: pads, level keys and nodes, laid out as `fixture.tree_bytes` lays them."""
    rng = np.random.default_rng(rb + count)
    rows = [rng.integers(0, 256, rb, dtype=np.uint8).tobytes() for _ in range(count)]
    dom = rows_domain(bytes(range(32)), 5, count, "blake3")
    t = tree(host, count, dom.domain_id)
    host.launch("hash_rows", count, 256, p=(host.put(b"".join(rows)), t.b["tree"], t.keys["dom"]), i=(count, rb))
    R.Pipeline.levels(t, "tree", count, "dom", "keys")
    got = host.read(t.b["tree"], (2 * R.pow2(count) - 1) * 32)
    assert got == F.tree_bytes(dom, rows, ROW_SCHEMA)
    assert got[-32:] == MerkleTree(dom, dict(enumerate(rows)), lambda _: ROW_SCHEMA).commitment.root


@pytest.mark.parametrize("rows", [16, 32, 48, 64, 96])
def test_messages_and_leaves_are_the_schemes(host, rows):
    """hash_msg and hash_leaf at 128 columns; a last row tile of 16, 32 or 48 rows is partial, its leaf over its digests alone
    (decode's 32 x 64: 64 digests, 2 KB)."""
    cols = 128
    rng = np.random.default_rng(rows)
    c, u = (rng.integers(0, 2**32, (rows, cols), dtype=np.uint64).astype(np.uint32) for _ in range(2))
    tiles = R.cdiv(rows, 64) * (cols // 64)
    dg, lv = host.alloc(rows * cols), host.alloc(tiles * 32)
    host.launch("hash_msg", R.cdiv(rows * cols // 32, 128), 128,
                p=(host.put(c.astype("<u4").tobytes()), host.put(u.astype("<u4").tobytes()), dg, host.put(C.H1_KEYS["message"])),
                i=(rows, cols))
    host.launch("hash_leaf", R.cdiv(tiles, 64), 64, p=(dg, lv, host.put(C.H1_KEYS["leaf"])), i=(rows, cols))
    digests = F.digests(c, u)
    assert host.read(dg, rows * cols) == b"".join(digests)
    want = [C.leaf(C.Checked(c[r0:r0 + 64, c0:c0 + 64].tolist(), u[r0:r0 + 64, c0:c0 + 64].tolist()), "h1")
            for r0 in range(0, rows, 64) for c0 in range(0, cols, 64)]
    assert host.read(lv, tiles * 32) == b"".join(want) == b"".join(F.leaves(digests, rows, cols))


def test_tickets_are_the_schemes(host):
    rows, cols, seed_a = 32, 64, bytes(range(32, 64))
    c = np.random.default_rng(3).standard_normal((rows, cols)).astype(np.float32).view(np.uint32)
    out = host.alloc((rows // 16) * (cols // 16) * 32)
    host.launch("tickets", R.cdiv((rows // 16) * (cols // 16), 128), 128,
                p=(host.put(c.astype("<u4").tobytes()), out, host.put(P.subkey("jackpot", seed_a))), i=(rows, cols))
    want = C.PearlC(forming="v1", hashing="h1").tickets(C.Noise(seed_a, bytes(32)), C.Checked(c.tolist(), c.tolist()))
    assert host.read(out, 32 * len(want)) == b"".join(want)


def test_tile_tree_is_the_verifiers(host):
    """hash_tree_leaves over 300 tile leaves (ranks past one byte), then 212 pads and nine levels."""
    count = 300
    leaves = [hashlib.sha256(i.to_bytes(4, "little")).digest() for i in range(count)]
    dom = rows_domain(bytes(32), -1, count, "blake3")
    t = tree(host, count, dom.domain_id)
    host.launch("hash_tree_leaves", R.cdiv(count, 128), 128, p=(host.put(b"".join(leaves)), t.b["tree"], t.keys["dom"]), i=(count,))
    R.Pipeline.levels(t, "tree", count, "dom", "keys")
    got = host.read(t.b["tree"], (2 * R.pow2(count) - 1) * 32)
    assert got == F.tree_bytes(dom, leaves, TILE_SCHEMA)
    assert got[-32:] == MerkleTree(dom, dict(enumerate(leaves)), lambda _: TILE_SCHEMA).commitment.root


def test_seeds_are_the_scheme_s(host):
    """b3_seeds over a unit's root (mode 0) is `pearl_c.unit_seed_a`; over row leaves (mode 1) row r's seed is the one-row unit's,
    index · 2^32 + r; seed_subkeys gives `pearl_kw.subkey`'s noise-line and jackpot keys of a seed."""
    salt, root_b, idx = bytes(range(32)), bytes(range(32, 64)), 7
    leaves = [hashlib.sha256(bytes([r])).digest() for r in range(5)]
    skey, psalt, prb = host.put(blake3(C.LABELS["seed-A"])), host.put(salt), host.put(root_b)
    src, out = host.put(b"".join(leaves)), host.alloc(32 * 5)
    host.launch("b3_seeds", 1, 128, p=(src, out, skey, psalt, prb), i=(1, idx, 0))
    assert host.read(out, 32) == C.unit_seed_a(salt, leaves[0], root_b, idx)
    host.launch("b3_seeds", 1, 128, p=(src, out, skey, psalt, prb), i=(5, idx, 1))
    assert host.read(out, 160) == b"".join(C.row_seed(salt, leaves[r], root_b, idx, r) for r in range(5))
    assert C.row_seed(salt, leaves[3], root_b, idx, 3) == C.unit_seed_a(salt, leaves[3], root_b, idx << 32 | 3)
    keys = host.alloc(64)
    host.launch("seed_subkeys", 1, 32, p=(out, keys))
    seed = host.read(out, 32)
    assert host.read(keys, 64) == P._line_key(seed) + P.subkey("jackpot", seed)
    source = (HERE / "hash.cuh").read_text()
    for name, role in (("NOISE_LINE_LABEL", "noise-line"), ("JACKPOT_LABEL", "jackpot")):
        chars = re.search(rf"__constant__ uint8_t {name}\[(\d+)\] = \{{([^}}]*)\}}", source)
        assert "".join(c.strip()[1] for c in chars.group(2).split(",")).encode() == P.LABELS[role]


#: -h3's row seeds pinned in the scheme's suite (test_pouw_pearl_c's ROW_SEEDS): (unit, row) -> seed
ROW_SEEDS = {(7, 3): "c12884464d77c2b929ce7c21ebd1600b8d5422eef522cb6436519f4ccf5f8e6a",
             (7, 200): "d6218857d90cbcd6e78ee0cbe69d50820baccbe3f7434c7346c0072a3381d139",
             (2 ** 32 - 1, 3): "3d7c8deb306724f5f05284d6d04d51b8a85387b51b454da2430a519a9744f2f8",
             (2 ** 32 - 1, 200): "ac1671c97f95166d8fd0ef09490934c7aba03d497eb08de99e84e2639b43d6a3"}


def test_row_seeds_are_the_pinned_vectors(host):
    """b3_seeds in mode 1 over 201 rows (two blocks of 128 threads), at unit 7 and at the largest unit 2^32 − 1 (i1 = −1,
    which the kernel reads as u32): every row's seed is `pearl_c.row_seed`'s, and rows 3 and 200 are the pinned bytes."""
    salt, root_b, leaf = bytes(range(32)), bytes(range(32, 64)), blake3(b"pearl-c row leaf")
    skey, psalt, prb = host.put(blake3(C.LABELS["seed-A"])), host.put(salt), host.put(root_b)
    for unit in (7, 2 ** 32 - 1):
        src, out = host.put(leaf * 201), host.alloc(32 * 201)
        host.launch("b3_seeds", 2, 128, p=(src, out, skey, psalt, prb), i=(201, unit - (unit >> 31 << 32), 1))
        got = host.read(out, 32 * 201)
        assert got == b"".join(C.row_seed(salt, leaf, root_b, unit, r) for r in range(201))
        assert [got[32 * r:32 * r + 32].hex() for r in (3, 200)] == [ROW_SEEDS[unit, 3], ROW_SEEDS[unit, 200]]


#: Pearl-C4's -h3 row seeds at the same inputs, under `pearl-c4/v0/seed-A` (test_pouw_pearl_c4's ROW_SEEDS)
C4_ROW_SEEDS = {(7, 3): "c6bf5c2d438504c44f5dda34ec27840d2403852bc8bbad0afac759da47fe3fb5",
                (7, 200): "5efa282aa56b5f5b4b48fec6c4d8142a619385b2bae0c20f77bde37d4a2af0a3",
                (2 ** 32 - 1, 3): "70d1c4a7113aec65a3a3e443253fc5de2922be2390c03ecb1d31297076e9c9ff",
                (2 ** 32 - 1, 200): "facaf01038342577857bbc390cb1aa84323eb3e24908104cc6e0eef01a3b2b91"}


def test_pearl_c4_row_seeds_are_b3_seeds_under_its_label(host):
    """b3_seeds in mode 1 keyed by blake3("pearl-c4/v0/seed-A") is `pearl_c4.row_seed`: the same kernel serves Pearl-C4's
    -h3 with its label's key, and rows 3 and 200 are the pinned bytes."""
    salt, root_b, leaf = bytes(range(32)), bytes(range(32, 64)), blake3(b"pearl-c row leaf")
    skey, psalt, prb = host.put(blake3(C4.LABELS["seed-A"])), host.put(salt), host.put(root_b)
    for unit in (7, 2 ** 32 - 1):
        src, out = host.put(leaf * 201), host.alloc(32 * 201)
        host.launch("b3_seeds", 2, 128, p=(src, out, skey, psalt, prb), i=(201, unit - (unit >> 31 << 32), 1))
        got = host.read(out, 32 * 201)
        assert got == b"".join(C4.row_seed(salt, leaf, root_b, unit, r) for r in range(201))
        assert [got[32 * r:32 * r + 32].hex() for r in (3, 200)] == [C4_ROW_SEEDS[unit, 3], C4_ROW_SEEDS[unit, 200]]


def h3s_row(k: int) -> bytes:
    """The scheme suite's pinned FP32 row of k words (test_pouw_pearl_c's `h3s_row`)."""
    return bytes((7 * i + k) % 256 for i in range(4 * k))


#: -h3s's row digests and row seeds pinned in the scheme's suite (test_pouw_pearl_c's H3S_DIGESTS and H3S_SEEDS)
H3S_DIGESTS = {
    4096: "823a21352763ecee3ba7b06da44a04525577197dbee48ef82b12927e6eb6a847360933bed78d80f85b798b77096a8746ce81dc3f48e6891d95d80d0b9432788f",
    14336: "9743c48588f788041e55fc2ef5cf44aa01225029edcf0539230053fedfcf240032ee695fd36f194c832834df8da839899cdb251e74c0090c150affc90aecadb8"}
H3S_SEEDS = {(7, 3): "48f3f41d75a3d208007bdf3f7ffd1f4649de6ac8cbeae2bc03bdc306b3358896",
             (7, 200): "f1ccc13e0ae11eb9e61f36ef1a1ccfc2c8d059f5f9bd7811fd166982d1763900",
             (2 ** 32 - 1, 3): "e7c2a2ca4c4c207c3ed6075fd0b4a9055d8aacb725a930ce3d8d1f6e7d4b6a4a",
             (2 ** 32 - 1, 200): "a8f6f7a7e2f8b1672167196ad066e4857954a064f44aa010c95fe8b95dcb3b2d"}


@pytest.mark.parametrize("k,rows", [(256, 3), (768, 2), (4096, 2), (14336, 1), (32768, 1)])
def test_s512_rows_are_row_seg_v1_digests(host, k, rows):
    """s512_rows, a block a row of 128 threads, is `pearl_c.row_digest` (``sha512/row-seg/v1``) of every row: 1, 3, 16, 56 and
    128 segments (an odd count ends the outer hash in the digests' block, an even one in a padding block of its own), and
    the pinned rows' digests are the pinned bytes."""
    rng = np.random.default_rng(k)
    xs = [h3s_row(k)] + [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(rows - 1)]
    out = host.alloc(64 * rows)
    host.launch("s512_rows", rows, 128, p=(host.put(b"".join(xs)), out), i=(rows, k))
    got = host.read(out, 64 * rows)
    assert got == b"".join(C.row_digest(x) for x in xs)
    if k in H3S_DIGESTS:
        assert got[:64].hex() == H3S_DIGESTS[k]


def unit_block(salt: bytes, root_b: bytes, unit: int) -> bytes:
    """-h3s's unit block (hash_h3s.cuh's `s512_seeds`): salt ‖ root_B (64 B) ‖ the unit index u64 LE."""
    return salt + root_b + unit.to_bytes(8, "little")


def test_h3s_row_seeds_are_the_pinned_vectors(host):
    """s512_seeds in mode 1 over 201 64-byte row digests (168 bytes after the label, two SHA-512 blocks) is `pearl_c.row_seed`
    over each, and rows 3 and 200 over the pinned row's digest are the pinned bytes, at unit 7 and at 2^32 − 1; each seed's
    line key is `pearl_c._s512_line_key`'s.  In mode 0 over root_A it is `pearl_c.unit_seed_a`."""
    salt, root_b, d = bytes(range(32)), bytes(range(32, 96)), C.row_digest(h3s_row(4096))
    digests = [d if r in (3, 200) else hashlib.sha512(bytes([r])).digest() for r in range(201)]
    for unit in (7, 2 ** 32 - 1):
        src, out, keys = host.put(b"".join(digests)), host.alloc(32 * 201), host.alloc(32 * 201)
        host.launch("s512_seeds", 2, 128, p=(src, out, keys, host.put(unit_block(salt, root_b, unit))), i=(201, 0, 1))
        got = host.read(out, 32 * 201)
        seeds = [C.row_seed(salt, x, root_b, unit, r, "h3s") for r, x in enumerate(digests)]
        assert got == b"".join(seeds)
        assert [got[32 * r:32 * r + 32].hex() for r in (3, 200)] == [H3S_SEEDS[unit, 3], H3S_SEEDS[unit, 200]]
        assert host.read(keys, 32 * 201) == b"".join(C._s512_line_key(x) for x in seeds)
    root_a, out = hashlib.sha512(b"root_A").digest(), host.alloc(32)
    host.launch("s512_seeds", 1, 128, p=(host.put(root_a), out, 0, host.put(unit_block(salt, root_b, 9))), i=(1, 0, 0))
    assert host.read(out, 32) == C.unit_seed_a(salt, root_a, root_b, 9, "h3s")


def test_h3s_line_raws_and_tickets_are_the_schemes(host):
    """s512_line_raws is H_line(key ‖ side ‖ factor ‖ line u32 LE) at lines past one byte and for own-keyed rows (line 0), the
    32 bytes `pearl_c.sample_line` maps to codes; s512_tickets is `PearlC.tickets` under -h3s, H_jackpot(seed_A ‖ fold)."""
    seed = C.unit_seed_a(bytes(32), bytes(64), bytes(64), 3, "h3s")
    key, own = C._s512_line_key(seed), [C._s512_line_key(bytes([r]) * 32) for r in range(5)]
    out = host.alloc(32 * 300)
    host.launch("s512_line_raws", 3, 128, p=(host.put(key), out, 0, 0, host.put(b"".join(own))), i=(1, 2, 300, 0, 5))
    want = [C.s512("line", (own[t] if t < 5 else key) + bytes([1, 2]) + (0 if t < 5 else t).to_bytes(4, "little")) for t in range(300)]
    assert host.read(out, 32 * 300) == b"".join(want)
    rows, cols = 32, 64
    c = np.random.default_rng(4).standard_normal((rows, cols)).astype(np.float32).view(np.uint32)
    out = host.alloc((rows // 16) * (cols // 16) * 32)
    host.launch("s512_tickets", 1, 128, p=(host.put(c.astype("<u4").tobytes()), out, host.put(seed)), i=(rows, cols))
    want = C.PearlC(forming="v1", hashing="h3s").tickets(C.Noise(seed, bytes(32)), C.Checked(c.tolist(), c.tolist()))
    assert host.read(out, 32 * len(want)) == b"".join(want)


@pytest.mark.parametrize("rows", [16, 32, 64, 96])
def test_h3s_tile_leaves_are_the_schemes(host, rows):
    """s512_tile_leaves at 128 columns: each tile's leaf is `pearl_c.leaf` under -h3s, ``sha512/row-seg/v1`` over its messages
    (32 segments a full tile; a last row block of 16 or 32 rows, decode's, is partial, 8 or 16)."""
    cols = 128
    rng = np.random.default_rng(rows)
    c, u = (rng.integers(0, 2**32, (rows, cols), dtype=np.uint64).astype(np.uint32) for _ in range(2))
    tiles = R.cdiv(rows, 64) * (cols // 64)
    lv = host.alloc(tiles * 64)
    host.launch("s512_tile_leaves", tiles, 32, p=(host.put(c.astype("<u4").tobytes()), host.put(u.astype("<u4").tobytes()), lv),
                i=(rows, cols))
    want = [C.leaf(C.Checked(c[r0:r0 + 64, c0:c0 + 64].tolist(), u[r0:r0 + 64, c0:c0 + 64].tolist()), "h3s")
            for r0 in range(0, rows, 64) for c0 in range(0, cols, 64)]
    assert host.read(lv, tiles * 64) == b"".join(want)


def s512_levels(host: Host, base: int, count: int, dom: int, small: int = 4) -> None:
    """Pads, then s512_nodes up to a level of `small` nodes, then s512_tree_small: run.py's levels under -h3s."""
    width = R.pow2(count)
    if width > count:
        host.launch("s512_pads", R.cdiv(width - count, 64), 64, p=(base, dom), i=(count, width))
    level = 0
    while width > small:
        host.launch("s512_nodes", R.cdiv(width // 2, 64), 64, p=(base, base + 64 * width, dom), i=(level, width // 2))
        base, width, level = base + 64 * width, width // 2, level + 1
    host.launch("s512_tree_small", 1, 64, p=(base, dom), i=(width, level))


@pytest.mark.parametrize("count,digests,nonce", [(1, False, 0), (5, False, 7), (300, True, 1 << 63), (64, False, (1 << 62) + 3)])
def test_hm96_trees_are_the_verifiers(host, count, digests, nonce):
    """hm_leaves over 64-byte values (row digests, or with ``digests`` tile leaves) under ChaCha20 salts at each tree's nonce
    (an activation tree's unit index, a weight's 2^62 + w, the transcript's 2^63), then pads, nodes and the small tree: every
    level is the frame-v3-sha512 tree `audit.commit_rows` builds, its root the `MerkleTree`'s.  300 leaves put ranks past one
    byte; the nonce is read from the unit block's index at 7 and given as i1, i2 otherwise."""
    rng = np.random.default_rng(count)
    rows = [rng.integers(0, 256, 128 * (r % 3 + 1), dtype=np.uint8).tobytes() for r in range(count)]
    xs = [r[:64] for r in rows] if digests else [C.row_digest(r) for r in rows]
    values = xs if digests else rows
    salt_key, binding = bytes(range(64, 96)), hashlib.sha256(b"binding").digest()
    dom = rows_domain(binding, 3, count, audit.HM96, digests)
    width = R.pow2(count)
    buf, pdom = host.alloc(64 * (2 * width - 1)), host.put(dom.domain_id)
    const = host.put(audit.hm96_kernel_constants())
    block = host.put(unit_block(bytes(32), bytes(64), nonce)) if nonce == 7 else 0
    host.launch("hm_leaves", R.cdiv(count, 64), 64, p=(host.put(b"".join(xs)), buf, host.put(salt_key), const, pdom, block + 96 if block else 0),
                i=(count, 0, 0) if block else (count, nonce & 0xFFFFFFFF, nonce >> 32))
    s512_levels(host, buf, count, pdom)
    salted = [v + audit.hm96_salt(salt_key, nonce, i) for i, v in enumerate(values)]
    want = b"".join(b"".join(level) for level in tree_levels(dom, [dom.leaf(i, i, ROW_SCHEMA, v) for i, v in enumerate(salted)]))
    assert host.read(buf, 64 * (2 * width - 1)) == want
    root = audit.commit_rows(binding, 3, values, audit.HM96, salt_key, nonce, digests).commitment.root
    assert want[-64:] == root


def test_row_seeds_over_a_calls_leaves_are_the_schemes(host):
    """An -h3 call's seeds as the device forms them: b3_seeds in mode 1 over level 0 of A's frame-b3s tree is
    `PearlC.row_seeds`, and seed_subkeys's line key of each row seed is the key E_A's row line is drawn from."""
    s, m, k = C.PearlC(k_min=32, tile=(16, 16), forming="v1", hashing="h3"), 32, 128
    ctx, rng = F.context(m, 16, k), np.random.default_rng(11)
    rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(m)]
    dom = rows_domain(activation_binding(ctx.salt, ctx.matmul.index), ctx.matmul.index, m, s.commitment_hash)
    leaves = [dom.leaf(i, i, ROW_SCHEMA, r) for i, r in enumerate(rows)]
    skey, psalt, prb = host.put(blake3(C.LABELS["seed-A"])), host.put(ctx.salt), host.put(ctx.root_b)
    out, keys = host.alloc(32 * m), host.alloc(64)
    host.launch("b3_seeds", 1, 128, p=(host.put(b"".join(leaves)), out, skey, psalt, prb), i=(m, ctx.matmul.index, 1))
    seeds = s.row_seeds(ctx, rows, range(m))
    assert host.read(out, 32 * m) == b"".join(seeds)
    slot, out2 = host.put(ctx.matmul.index.to_bytes(8, "little")), host.alloc(32 * m)   # the index read from a graph's key slot
    host.launch("b3_seeds", 1, 128, p=(host.put(b"".join(leaves)), out2, skey, psalt, prb, slot), i=(m, 0, 1))
    assert host.read(out2, 32 * m) == b"".join(seeds)
    host.launch("seed_subkeys", 1, 32, p=(out + 32 * 5, keys))
    assert host.read(keys, 32) == P._line_key(seeds[5])


def test_run_commit_and_hashing_phases_are_the_verifiers(host):
    """run.py's `Pipeline` at 128 x 384 x 128 (v1) on given A rows, C̃ and U: the commit phase (A's tree, the unit's seed_A over its
    root, and seed_A's line and jackpot keys) and the post-GEMM hashing (the digests, the leaves, the tickets under that jackpot
    key and the tile tree, 12 tiles, so 4 pads), each equal to the fixture's reference."""
    m, n, k = 128, 384, 128
    ctx, rng = F.context(m, n, k), np.random.default_rng(5)
    tiles = (m // 64) * (n // 64)
    rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(m)]
    c = rng.standard_normal((m, n)).astype(np.float32).view(np.uint32)
    u = rng.integers(0, 2**32, (m, n), dtype=np.uint64).astype(np.uint32)
    keys = {"salt": ctx.salt, "root_b": ctx.root_b, "seed_key": bytes.fromhex(R.SEED_KEY), "line_b": bytes(32),
            "dom_a": F.a_domain(ctx).domain_id, "dom_t": F.t_domain(ctx, tiles).domain_id, "k_msg": C.H1_KEYS["message"],
            "k_leaf": C.H1_KEYS["leaf"]}
    pipe = R.Pipeline(host, m, n, k, "v1", {name: host.put(v) for name, v in keys.items()}, host.put(b"".join(rows)),
                      host.alloc(n * 4 * k), unit=ctx.matmul.index)
    for name, words in (("c", c), ("u", u)):
        host.bufs[pipe.b[name]][:m * n * 4] = np.frombuffer(words.astype("<u4").tobytes(), np.uint8)
    for _, phase, step in pipe.steps():
        if phase in ("commit", "hashing"):
            step()
    tree_a = F.tree_bytes(F.a_domain(ctx), rows, ROW_SCHEMA)
    seed_a = C.unit_seed_a(ctx.salt, tree_a[-32:], ctx.root_b, ctx.matmul.index)
    digests = F.digests(c, u)
    leaves = F.leaves(digests, m, n)
    want = {"tree_a": tree_a, "seed_a": seed_a, "a_keys": P._line_key(seed_a) + P.subkey("jackpot", seed_a),
            "digests": b"".join(digests), "leaves": b"".join(leaves), "tickets": b"".join(F.tickets(c, seed_a)),
            "tree_t": F.tree_bytes(F.t_domain(ctx, tiles), leaves, TILE_SCHEMA)}
    assert {name: host.read(pipe.b[name], len(w)) == w for name, w in want.items()} == dict.fromkeys(want, True)
