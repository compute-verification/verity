"""hash.cuh on the host (pearl_c/hash_host.cpp, built with g++): each hashing kernel, launched as run.py launches it, writes
the bytes the verifier's reference computes.  A's row leaves and frame-b3 trees (pads, level keys, one-compression nodes)
at rows of 1 to 75 chunks, the message digests, the tile leaves full and partial (decode's 32 x 64), Pearl's tickets and
the tile tree; then run.py's hashing phase end to end."""

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
from verity.commitments import MerkleTree  # noqa: E402
from verity_pouw.audit import ROW_SCHEMA, TILE_SCHEMA, rows_domain  # noqa: E402
from verity_pouw.schemes import pearl_c as C  # noqa: E402
from verity_pouw.schemes import pearl_kw as P  # noqa: E402


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


def test_run_hashing_phase_is_the_verifiers(host):
    """run.py's `Pipeline` hashing phase at 128 x 384 x 128 (v1) on given A rows, C̃ and U: A's tree, the digests, the leaves,
    the tickets and the tile tree (12 tiles, so 4 pads), each equal to the fixture's reference."""
    m, n, k = 128, 384, 128
    ctx, rng, seed_a = F.context(m, n, k), np.random.default_rng(5), bytes(range(64, 96))
    tiles = (m // 64) * (n // 64)
    rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(m)]
    c = rng.standard_normal((m, n)).astype(np.float32).view(np.uint32)
    u = rng.integers(0, 2**32, (m, n), dtype=np.uint64).astype(np.uint32)
    keys = {"line_a": bytes(32), "line_b": bytes(32), "jackpot": P.subkey("jackpot", seed_a), "dom_a": F.a_domain(ctx).domain_id,
            "dom_t": F.t_domain(ctx, tiles).domain_id, "k_msg": C.H1_KEYS["message"], "k_leaf": C.H1_KEYS["leaf"]}
    pipe = R.Pipeline(host, m, n, k, "v1", {name: host.put(v) for name, v in keys.items()}, host.put(b"".join(rows)),
                      host.alloc(n * 4 * k))
    for name, words in (("c", c), ("u", u)):
        host.bufs[pipe.b[name]][:m * n * 4] = np.frombuffer(words.astype("<u4").tobytes(), np.uint8)
    for _, phase, step in pipe.steps():
        if phase == "hashing":
            step()
    digests = F.digests(c, u)
    leaves = F.leaves(digests, m, n)
    want = {"tree_a": F.tree_bytes(F.a_domain(ctx), rows, ROW_SCHEMA), "digests": b"".join(digests), "leaves": b"".join(leaves),
            "tickets": b"".join(F.tickets(c, seed_a)), "tree_t": F.tree_bytes(F.t_domain(ctx, tiles), leaves, TILE_SCHEMA)}
    assert {name: host.read(pipe.b[name], len(w)) == w for name, w in want.items()} == dict.fromkeys(want, True)
