"""hash.cuh on the host (pearl_c/hash_host.cpp, built with g++): each hashing kernel, launched as run.py launches it, writes
the bytes the verifier's reference computes.  A's row leaves and frame-b3 trees (pads, level keys, one-compression nodes)
at rows of 1 to 75 chunks, the message digests, the tile leaves full and partial (decode's 32 x 64), Pearl's tickets, the
tile tree, a unit's seed_A over its root with its line and jackpot keys, and -h3's row seeds at the pinned vectors; then
run.py's commit and hashing phases end to end."""

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
from verity.protocols.accounting.work.pouw.audit import ROW_SCHEMA, TILE_SCHEMA, rows_domain  # noqa: E402
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C  # noqa: E402
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
    names = set(re.findall(r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\(', (HERE / "hash.cuh").read_text()))
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
