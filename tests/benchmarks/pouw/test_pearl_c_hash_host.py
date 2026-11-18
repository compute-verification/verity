"""hash_h3.cuh and pearl_c_sm120/hash_fast.cuh on the host (pearl_c/hash_host.cpp, built with g++, which builds hash_wide.cuh
too: test_pearl_c_hash_wide.py): each -h3 hashing kernel,
launched as pearl_c_sm120/run.py launches it, writes the bytes the verifier's reference computes: the SHA-512 row digests
(s512_rows, and hash_fast.cuh's segment and fold launches), the seeds and line keys over them at the pinned vectors, the line
raws, tickets and tile leaves (both ways), and the hm96-sha512 frame-v3-sha512 trees over the salts they are handed; then run.py's epoch,
unit, commit and hashing phases end to end with either kernel set."""

from __future__ import annotations

import ctypes
import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import load

from verity.commitments import MerkleTree, hm96
from verity.commitments.merkle import tree_levels
from verity.protocols.resource_accounting.compute.pouw.verifier import audit
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import ROW_SCHEMA, activation_binding, rows_domain
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C

HERE = Path(__file__).resolve().parents[3] / "verity" / "protocols" / "resource_accounting" / "compute" / "pouw" / "prover" / "kernels" / "pearl_c"
FAST = HERE.parent / "pearl_c_sm120" / "hash_fast.cuh"
WIDE = HERE / "hash_wide.cuh"
R = load("pearl_c_sm120/run.py")


class Host:
    """run.py's `Dev` over hash_h3.cuh built for the host: buffers are numpy arrays, and a launch runs its grid in order."""

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

    def sync(self) -> None:
        pass

    def ck(self, name: str, to, data, n: int) -> None:
        """The one driver call a pipeline makes of this device: a host copy into one of its buffers (`put_salts`)."""
        assert name == "cuMemcpyHtoD_v2", name
        ctypes.memmove(to.value, data, n)

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
        pytest.skip("no g++ for the host build of hash_h3.cuh")
    lib = tmp_path_factory.mktemp("hash-host") / "hash_host.so"
    subprocess.run([gxx, "-O2", "-std=c++20", "-pthread", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas",
                    f"-I{HERE}", "-o", str(lib), str(HERE / "hash_host.cpp")], check=True)
    cdll = ctypes.CDLL(str(lib))
    cdll.launch.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    cdll.kernel_name.restype = ctypes.c_char_p
    return Host(cdll)


def test_every_hash_kernel_is_built(host):
    names = set(re.findall(r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\(',
                           (HERE / "hash_h3.cuh").read_text() + FAST.read_text() + WIDE.read_text()))
    built = {host.lib.kernel_name(i).decode() for i in range(len(names))}
    assert built == names and host.lib.kernel_name(len(names)) is None


def h3_row(k: int) -> bytes:
    """The scheme suite's pinned FP32 row of k words (test_pouw_pearl_c's `h3_row`)."""
    return bytes((7 * i + k) % 256 for i in range(4 * k))


#: -h3's row digests and row seeds pinned in the scheme's suite (test_pouw_pearl_c's H3_DIGESTS and H3_SEEDS)
H3_DIGESTS = {
    4096: "823a21352763ecee3ba7b06da44a04525577197dbee48ef82b12927e6eb6a847360933bed78d80f85b798b77096a8746ce81dc3f48e6891d95d80d0b9432788f",
    14336: "9743c48588f788041e55fc2ef5cf44aa01225029edcf0539230053fedfcf240032ee695fd36f194c832834df8da839899cdb251e74c0090c150affc90aecadb8"}
H3_SEEDS = {(7, 3): "48f3f41d75a3d208007bdf3f7ffd1f4649de6ac8cbeae2bc03bdc306b3358896",
             (7, 200): "f1ccc13e0ae11eb9e61f36ef1a1ccfc2c8d059f5f9bd7811fd166982d1763900",
             (2 ** 32 - 1, 3): "e7c2a2ca4c4c207c3ed6075fd0b4a9055d8aacb725a930ce3d8d1f6e7d4b6a4a",
             (2 ** 32 - 1, 200): "a8f6f7a7e2f8b1672167196ad066e4857954a064f44aa010c95fe8b95dcb3b2d"}


@pytest.mark.parametrize("k,rows", [(256, 3), (768, 2), (4096, 2), (14336, 1), (32768, 1)])
def test_s512_rows_are_row_seg_v1_digests(host, k, rows):
    """s512_rows, a block a row of 128 threads, is `pearl_c.row_digest` (``sha512/row-seg/v1``) of every row: 1, 3, 16, 56 and
    128 segments (an odd count ends the outer hash in the digests' block, an even one in a padding block of its own), and
    the pinned rows' digests are the pinned bytes."""
    rng = np.random.default_rng(k)
    xs = [h3_row(k)] + [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(rows - 1)]
    out = host.alloc(64 * rows)
    host.launch("s512_rows", rows, 128, p=(host.put(b"".join(xs)), out), i=(rows, k))
    got = host.read(out, 64 * rows)
    assert got == b"".join(C.row_digest(x) for x in xs)
    if k in H3_DIGESTS:
        assert got[:64].hex() == H3_DIGESTS[k]


def fast_heads(host: Host) -> None:
    """s512_heads as run.py's Pipeline launches it once a module: the states after every (j, s) segment header and every Q."""
    host.launch("s512_heads", R.cdiv(R.RS_MAX_SEGS * (R.RS_MAX_SEGS + 1) // 2 + R.RS_MAX_SEGS + 1, 128), 128)


@pytest.mark.parametrize("k,rows", [(256, 3), (768, 2), (4096, 2), (14336, 1), (32768, 1)])
def test_fast_row_digests_are_s512_rows(host, k, rows):
    """run.py's `row_digests` with hash_fast.cuh's kernels (s512_row_segs, a thread a segment over every row, then s512_folds,
    a thread a row) writes s512_rows' bytes, `pearl_c.row_digest` of every row, at 1, 3, 16, 56 and 128 segments."""
    rng = np.random.default_rng(k + 1)
    xs = [h3_row(k)] + [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(rows - 1)]
    src, want = host.put(b"".join(xs)), b"".join(C.row_digest(x) for x in xs)
    fast_heads(host)
    for sha in ("fast", "ref"):
        out = host.alloc(64 * rows)
        R.Pipeline.row_digests(SimpleNamespace(dev=host, k=k, sha=sha), src, out, host.alloc(64 * (k // 256) * rows), rows)
        assert host.read(out, 64 * rows) == want, sha


@pytest.mark.parametrize("k", [33024, 128256, 1000, 0])
@pytest.mark.parametrize("sha,scratch", [("fast", 64), ("ref", 64), ("fast", 0)])
def test_row_digests_refuse_a_row_their_kernels_cant_take(host, monkeypatch, k, sha, scratch):
    """pearl_c_sm120/run.py's `row_digests` launches nothing for a row past RS_MAX_SEGS segments, or of k not a positive
    multiple of 256: s512_rows, s512_row_segs and s512_folds hold no bound of their own, and past it they read and write out
    of bounds (the fast path can return a wrong digest with no error)."""
    launched = []
    monkeypatch.setattr(host, "launch", lambda name, *a, **kw: launched.append(name))
    with pytest.raises(ValueError, match="^s512_rows takes"):
        R.Pipeline.row_digests(SimpleNamespace(dev=host, k=k, sha=sha), 0, 0, scratch, 1)
    assert not launched


def test_a_row_digest_launch_its_kernel_cant_take_never_reaches_the_device():
    """`rs_refusal`, which pearl_c_sm120/run.py's `Dev.launch` raises before anything else, so whatever calls it: each of
    RS_KERNELS past RS_MAX_SEGS segments (hash_h3.cuh's constant), or with k not a positive multiple of 256.  Up to 128
    segments, and the tile leaves' folds, go through."""
    assert R.RS_MAX_SEGS == int(re.search(r"\bRS_MAX_SEGS = (\d+)", (HERE / "hash_h3.cuh").read_text()).group(1))
    assert set(R.RS_KERNELS) == {"s512_rows", "s512_row_segs", "s512_folds"}
    for name, i in (("s512_rows", (3, 32768)), ("s512_row_segs", (3, 256)), ("s512_folds", (5, 128, 5, 128)), ("s512_folds", (9, 32, 8, 31))):
        assert R.rs_refusal(name, i) is None
    for name, i in (("s512_rows", (1, 33024)), ("s512_row_segs", (1, 128256)), ("s512_folds", (4, 129, 4, 129)),
                    ("s512_folds", (4, 32, 2, 501)), ("s512_rows", (1, 1000)), ("s512_row_segs", (1, 0)), ("s512_rows", ())):
        with pytest.raises(ValueError, match=f"^{name} takes"):       # before `self` is read: nothing is launched
            R.Dev.launch(None, name, 1, 128, p=(0, 0), i=i)


def test_a_fold_count_is_refused_as_the_kernel_would_read_it():
    """s512_folds' segment counts (`rs_fold_counts`): an int slot is an int32, so a negative count would reach the kernel as
    its low 32 bits (200 - 2^32 as 200, past RS_MAX_SEGS): refused always.  A zero is refused only where a unit would fold
    it: the tile leaves pass one for the last row block when every block is full, and that goes through."""
    wrapped, a = 200 - 2**32, R.Args()
    a.i[1] = wrapped
    assert a.i[1] == 200 > R.RS_MAX_SEGS                             # what the kernel would have been given
    for i in ((4, wrapped, 4, 32), (4, 32, 2, wrapped), (4, 32, 4, wrapped), (4, -1, 0, 32), (4, 0, 4, 32), (4, 0, 1, 32),
              (4, 32, 2, 0), (4, 32, 0, 0)):
        with pytest.raises(ValueError, match="^s512_folds takes"):    # before `self` is read: nothing is launched
            R.Dev.launch(None, "s512_folds", 1, R.RS_FOLD_NT, p=(0, 0), i=i)
    for i in ((8, 32, 8, 0), (8, 32, 0, 8), (8, 0, 0, 8), (0, 0, 0, 0), (5, 128, 5, 128)):
        assert R.rs_refusal("s512_folds", i) is None


def unit_block(salt: bytes, root_b: bytes, unit: int) -> bytes:
    """-h3's unit block (hash_h3.cuh's `s512_seeds`): salt ‖ root_B (64 B) ‖ the unit index u64 LE."""
    return salt + root_b + unit.to_bytes(8, "little")


def test_h3_row_seeds_are_the_pinned_vectors(host):
    """s512_seeds in mode 1 over 201 64-byte row digests (168 bytes after the label, two SHA-512 blocks) is `pearl_c.row_seed`
    over each, and rows 3 and 200 over the pinned row's digest are the pinned bytes, at unit 7 and at 2^32 − 1; each seed's
    line key is `pearl_c._s512_line_key`'s.  In mode 0 over root_A it is `pearl_c.unit_seed_a`."""
    salt, root_b, d = bytes(range(32)), bytes(range(32, 96)), C.row_digest(h3_row(4096))
    digests = [d if r in (3, 200) else hashlib.sha512(bytes([r])).digest() for r in range(201)]
    for unit in (7, 2 ** 32 - 1):
        src, out, keys = host.put(b"".join(digests)), host.alloc(32 * 201), host.alloc(32 * 201)
        host.launch("s512_seeds", 2, 128, p=(src, out, keys, host.put(unit_block(salt, root_b, unit))), i=(201, 0, 1))
        got = host.read(out, 32 * 201)
        seeds = [C.row_seed(salt, x, root_b, unit, r) for r, x in enumerate(digests)]
        assert got == b"".join(seeds)
        assert [got[32 * r:32 * r + 32].hex() for r in (3, 200)] == [H3_SEEDS[unit, 3], H3_SEEDS[unit, 200]]
        assert host.read(keys, 32 * 201) == b"".join(C._s512_line_key(x) for x in seeds)
    root_a, out = hashlib.sha512(b"root_A").digest(), host.alloc(32)
    host.launch("s512_seeds", 1, 128, p=(host.put(root_a), out, 0, host.put(unit_block(salt, root_b, 9))), i=(1, 0, 0))
    assert host.read(out, 32) == C.unit_seed_a(salt, root_a, root_b, 9, "h3")


def test_h3_line_raws_and_tickets_are_the_schemes(host):
    """s512_line_raws is H_line(key ‖ side ‖ factor ‖ line u32 LE) at lines past one byte and for own-keyed rows (line 0), the
    32 bytes `pearl_c.sample_line` maps to codes; s512_tickets is `PearlC.tickets` under -h3, H_jackpot(seed_A ‖ fold)."""
    seed = C.unit_seed_a(bytes(32), bytes(64), bytes(64), 3, "h3")
    key, own = C._s512_line_key(seed), [C._s512_line_key(bytes([r]) * 32) for r in range(5)]
    out = host.alloc(32 * 300)
    host.launch("s512_line_raws", 3, 128, p=(host.put(key), out, 0, 0, host.put(b"".join(own))), i=(1, 2, 300, 0, 5))
    want = [C.s512("line", (own[t] if t < 5 else key) + bytes([1, 2]) + (0 if t < 5 else t).to_bytes(4, "little")) for t in range(300)]
    assert host.read(out, 32 * 300) == b"".join(want)
    rows, cols = 32, 64
    c = np.random.default_rng(4).standard_normal((rows, cols)).astype(np.float32).view(np.uint32)
    out = host.alloc((rows // 16) * (cols // 16) * 32)
    host.launch("s512_tickets", 1, 128, p=(host.put(c.astype("<u4").tobytes()), out, host.put(seed)), i=(rows, cols))
    want = C.PearlC(forming="v1", hashing="h3").tickets(C.Noise(seed, bytes(32)), C.Checked(c.tolist(), c.tolist()))
    assert host.read(out, 32 * len(want)) == b"".join(want)


@pytest.mark.parametrize("rows", [16, 32, 64, 96])
def test_h3_tile_leaves_are_the_schemes(host, rows):
    """s512_tile_leaves at 128 columns: each tile's leaf is `pearl_c.leaf` under -h3, ``sha512/row-seg/v1`` over its messages
    (32 segments a full tile; a last row block of 16 or 32 rows, decode's, is partial, 8 or 16)."""
    cols = 128
    rng = np.random.default_rng(rows)
    c, u = (rng.integers(0, 2**32, (rows, cols), dtype=np.uint64).astype(np.uint32) for _ in range(2))
    tiles = R.cdiv(rows, 64) * (cols // 64)
    lv = host.alloc(tiles * 64)
    host.launch("s512_tile_leaves", tiles, 32, p=(host.put(c.astype("<u4").tobytes()), host.put(u.astype("<u4").tobytes()), lv),
                i=(rows, cols))
    want = [C.leaf(C.Checked(c[r0:r0 + 64, c0:c0 + 64].tolist(), u[r0:r0 + 64, c0:c0 + 64].tolist()), "h3")
            for r0 in range(0, rows, 64) for c0 in range(0, cols, 64)]
    assert host.read(lv, tiles * 64) == b"".join(want)


@pytest.mark.parametrize("rows,cols", [(16, 128), (32, 128), (64, 192), (96, 128), (176, 192)])
def test_fast_tile_leaves_are_s512_tile_leaves(host, rows, cols):
    """run.py's `tile_leaves` with hash_fast.cuh's kernels (s512_tile_segs over every tile's segments, the full row blocks'
    tiles first, then s512_folds at 32 segments a full tile and the last block's rows / 2) writes s512_tile_leaves' bytes,
    each tile's `pearl_c.leaf` under -h3: no full block, one, and one or two beside a partial block of 16, 32 or 48 rows."""
    rng = np.random.default_rng(rows + cols)
    c, u = (rng.integers(0, 2**32, (rows, cols), dtype=np.uint64).astype(np.uint32) for _ in range(2))
    tiles = R.cdiv(rows, 64) * (cols // 64)
    want = b"".join(C.leaf(C.Checked(c[r0:r0 + 64, c0:c0 + 64].tolist(), u[r0:r0 + 64, c0:c0 + 64].tolist()), "h3")
                    for r0 in range(0, rows, 64) for c0 in range(0, cols, 64))
    fast_heads(host)
    for sha in ("fast", "ref"):
        b = {"c": host.put(c.astype("<u4").tobytes()), "u": host.put(u.astype("<u4").tobytes()), "leaves": host.alloc(64 * tiles),
             "segdig_t": host.alloc(64 * 32 * tiles)}
        R.Pipeline.tile_leaves(SimpleNamespace(dev=host, b=b, m_real=rows, n=cols, tiles=tiles, sha=sha))
        assert host.read(b["leaves"], 64 * tiles) == want, sha


def s512_levels(host: Host, base: int, count: int, dom: int, small: int = 4) -> None:
    """Pads, then s512_nodes up to a level of `small` nodes, then s512_tree_small: run.py's levels under -h3."""
    width = R.pow2(count)
    if width > count:
        host.launch("s512_pads", R.cdiv(width - count, 64), 64, p=(base, dom), i=(count, width))
    level = 0
    while width > small:
        host.launch("s512_nodes", R.cdiv(width // 2, 64), 64, p=(base, base + 64 * width, dom), i=(level, width // 2))
        base, width, level = base + 64 * width, width // 2, level + 1
    host.launch("s512_tree_small", 1, 64, p=(base, dom), i=(width, level))


def salts_of(label: str, count: int) -> bytes:
    """`count` leaves' salts for a test, `R.HM_SALT` bytes a leaf in leaf order: any bytes, which the kernels only read."""
    return hashlib.shake_256(f"pearl-c-test-salts/{label}".encode()).digest(R.HM_SALT * count)


def salted(values: list[bytes], salts: bytes) -> list[bytes]:
    """Each value followed by its salt, as `audit.Hm96Domain.leaf` takes one."""
    assert len(salts) == R.HM_SALT * len(values)
    return [v + salts[R.HM_SALT * i:R.HM_SALT * (i + 1)] for i, v in enumerate(values)]


def tree_of(dom, values: list[bytes], salts: bytes) -> bytes:
    """An `audit.HM96` tree's levels as the kernels lay them out, each value under its salt of `salts`."""
    return b"".join(b"".join(lv) for lv in tree_levels(dom, [dom.leaf(i, i, ROW_SCHEMA, v) for i, v in enumerate(salted(values, salts))]))


def test_a_leafs_salt_is_192_bytes_as_hm96s():
    """The buffer's stride and the kernels' is the commitment's salt size, and the 16-byte loads divide it."""
    src = (HERE / "hash_h3.cuh").read_text()
    assert R.HM_SALT == hm96.SHA512.salt_bytes == audit.HM96_SALT == 192 and "constexpr int HM_SALT = 192;" in src
    assert R.HM_SALT % 16 == 0 and not re.search(r"chacha|salt_key|nonce", src, re.I)


@pytest.mark.parametrize("count,digests", [(1, False), (5, False), (300, True), (64, False)])
def test_hm96_trees_are_the_verifiers(host, count, digests):
    """hm_leaves over 64-byte values (row digests, or with ``digests`` tile leaves) and the buffer of their salts, 192 bytes a
    leaf in leaf order: each hl is `verity.commitments.hm96`'s leaf of that value under those salt bytes (byte n of a salt at
    offset n).  Then s512_frame_leaves, pads, nodes and the small tree: every level is the frame-v3-sha512 tree over the
    salted values, its root the `MerkleTree`'s.  300 leaves put ranks past one byte.  The kernel reads the salts only."""
    rng = np.random.default_rng(count)
    rows = [rng.integers(0, 256, 128 * (r % 3 + 1), dtype=np.uint8).tobytes() for r in range(count)]
    xs = [r[:64] for r in rows] if digests else [C.row_digest(r) for r in rows]
    values = xs if digests else rows
    binding, salts = hashlib.sha256(b"binding").digest(), salts_of(f"tree-{count}", count)
    dom = rows_domain(binding, 3, count, audit.HM96, digests)
    width = R.pow2(count)
    buf, pdom = host.alloc(64 * (2 * width - 1)), host.put(dom.domain_id)
    const, at = host.put(audit.hm96_kernel_constants()), host.put(salts)
    host.launch("hm_leaves", R.cdiv(count, 64), 64, p=(host.put(b"".join(xs)), buf, at, const), i=(count,))
    key = audit.HM96_KEY
    assert host.read(buf, 64 * count) == b"".join(
        hm96.SHA512.tree_leaf(key, hm96.SHA512.commit_string(key, x, salts[192 * i:192 * i + 192])) for i, x in enumerate(xs))
    assert host.read(at, len(salts)) == salts
    host.launch("s512_frame_leaves", R.cdiv(count, 64), 64, p=(buf, pdom), i=(0, count))
    s512_levels(host, buf, count, pdom)
    want = tree_of(dom, values, salts)
    assert host.read(buf, 64 * (2 * width - 1)) == want
    assert want[-64:] == MerkleTree(dom, dict(enumerate(salted(values, salts))), lambda _: ROW_SCHEMA).commitment.root


def test_a_pipeline_without_a_trees_salts_refuses_it(host):
    """A tree's hm96 level 0 reads `salts_<tree>`, which its caller fills: a pipeline given none launches nothing and says
    whose bytes they are."""
    p = SimpleNamespace(dev=host, keys={"hm": 1, "dom_b": 1}, b={"tree_b": 1})
    launched, launch = [], host.launch
    host.launch = lambda *a, **kw: launched.append(a)
    try:
        with pytest.raises(NotImplementedError, match="salts_b .192 bytes a leaf.*derive no salt"):
            R.Pipeline.hm_level0(p, "b", 1, 4, "dom_b")
    finally:
        host.launch = launch
    assert not launched


@pytest.mark.parametrize("sha", ["fast", pytest.param("ref", marks=pytest.mark.slow)])
def test_sm120_h3_phases_are_the_verifiers(host, monkeypatch, sha):
    """pearl_c_sm120/run.py's `Pipeline` under -h3 at 64 x 128 x 1024 (m_real 48: a partial row block), its epoch, unit, commit
    and hashing phases on the host with either SHA-512 kernel set (``PEARLC_SHA``: hash_fast.cuh's segment and fold launches,
    or hash_h3.cuh's s512_rows and s512_tile_leaves): B's HM96 tree, A's and the tiles', each over the salts its caller put in
    `salts_<tree>`, seed_A over root_A with its line key, each row's seed and line key, the tile leaves over C~ and U and the
    transcript tree, each equal to the reference's (`audit.Hm96Domain`'s leaves, `pearl_c`)."""
    S = R
    m, n, k, mr, w, idx = 64, 128, 1024, 48, 3, 5
    rng = np.random.default_rng(12)
    a_rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(m)]
    b_rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(n)]
    c = rng.integers(0, 2**32, (m, n), dtype=np.uint64).astype(np.uint32)
    u = rng.integers(0, 2**32, (m, n), dtype=np.uint64).astype(np.uint32)
    salt, tiles = bytes(range(32)), S.cdiv(mr, 64) * (n // 64)
    salts = {"a": salts_of("phases-a", mr), "b": salts_of("phases-b", n), "t": salts_of("phases-t", tiles)}
    dom_a, dom_b = rows_domain(activation_binding(salt, idx), idx, mr, audit.HM96), rows_domain(audit.weight_binding(w), -2, n, audit.HM96)
    tree_b = tree_of(dom_b, b_rows, salts["b"])
    root_b = tree_b[-64:]
    epoch = audit.Epoch(audit.Workload((C.Matmul(idx, C.Shape(mr, k, n), w),)), {w: root_b}, salt)
    dom_t = audit.transcript_domain(epoch, tiles, audit.HM96)
    keys = {"hm": audit.hm96_kernel_constants(), "unit": salt + root_b + idx.to_bytes(8, "little"),
            "dom_a": dom_a.domain_id, "dom_b": dom_b.domain_id, "dom_t": dom_t.domain_id, "line_b": bytes(32)}
    host.sms = 4
    launched, launch = set(), host.launch
    monkeypatch.setattr(host, "launch", lambda name, *a, **kw: launched.add(name) or launch(name, *a, **kw))
    pipe = S.Pipeline(host, m, n, k, "sm120", {name: host.put(v) for name, v in keys.items()}, host.put(b"".join(a_rows)),
                      host.put(b"".join(b_rows)), m_real=mr, sha=sha)
    assert {t: pipe.size[f"salts_{t}"] for t in R.SALTED} == {t: len(x) for t, x in salts.items()}
    with pytest.raises(NotImplementedError, match="salts_b"):   # its own buffers hold no salts until they are put
        pipe.salts("b")
    for t, x in salts.items():
        pipe.put_salts(t, x)
    for name, words in (("c", c), ("u", u)):
        host.bufs[pipe.b[name]][:m * n * 4] = np.frombuffer(words.astype("<u4").tobytes(), np.uint8)
    names = [nm for nm, phase, step in pipe.steps() if phase in ("epoch", "unit", "commit", "hashing") and (step() or True)]
    assert names == ["rowdig_b", "hash_rows_b", "tree_b", "pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a",
                     "hash_leaf", "tree_leaves_t", "tree_t"]
    fast, ref = {"s512_row_segs", "s512_tile_segs", "s512_folds"}, {"s512_rows", "s512_tile_leaves"}
    assert (launched & fast, launched & ref) == ((fast, set()) if sha == "fast" else (set(), ref))

    tree_a = tree_of(dom_a, a_rows[:mr], salts["a"])
    seed_a = C.unit_seed_a(salt, tree_a[-64:], root_b, idx, "h3")
    seeds = [C.row_seed(salt, C.row_digest(r), root_b, idx, i) for i, r in enumerate(a_rows[:mr])]
    cr, ur = c[:mr], u[:mr]
    leaves = [C.leaf(C.Checked(cr[r0:r0 + 64, c0:c0 + 64].tolist(), ur[r0:r0 + 64, c0:c0 + 64].tolist()), "h3")
              for r0 in range(0, mr, 64) for c0 in range(0, n, 64)]
    want = {"tree_b": tree_b, "tree_a": tree_a, "seed_a": seed_a,
            "line_a": C._s512_line_key(seed_a), "seeds_a": b"".join(seeds), "lkeys_a": b"".join(C._s512_line_key(x) for x in seeds),
            "leaves": b"".join(leaves), "tree_t": tree_of(dom_t, leaves, salts["t"])}
    assert {name: host.read(pipe.b[name], len(v)) == v for name, v in want.items()} == dict.fromkeys(want, True)


def test_a_caller_without_the_scratch_runs_the_reference_kernels(host, monkeypatch):
    """A caller that owns the buffers and gives no `segdig_<tree>` (pouw_pearl_c_device's) gets s512_rows and s512_tile_leaves
    under the default sha, writing the scheme's row digests and tile leaves."""
    m, n, k, mr = 64, 128, 1024, 16
    rng = np.random.default_rng(3)
    a_rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(m)]
    c, u = (rng.integers(0, 2**32, (m, n), dtype=np.uint64).astype(np.uint32) for _ in range(2))
    bufs = {name: host.alloc(size) for name, size in R.Pipeline.sizes(m, n, k, mr).items() if name not in R.SEGDIG}
    for name, words in (("c", c), ("u", u)):
        host.bufs[bufs[name]][:m * n * 4] = np.frombuffer(words.astype("<u4").tobytes(), np.uint8)
    launched, launch = [], host.launch
    monkeypatch.setattr(host, "launch", lambda name, *a, **kw: launched.append(name) or launch(name, *a, **kw))
    monkeypatch.delenv("PEARLC_SHA", raising=False)
    pipe = R.Pipeline(host, m, n, k, "sm120", dict.fromkeys(R.TIMING_KEYS, 0), host.put(b"".join(a_rows)), 0, bufs=bufs, m_real=mr)
    by = {name: step for name, _, step in pipe.steps()}
    by["rowdig_a"](), by["hash_leaf"]()
    assert pipe.sha == "fast" and launched == ["s512_rows", "s512_tile_leaves"]
    assert host.read(bufs["rowdig_a"], 64 * mr) == b"".join(C.row_digest(r) for r in a_rows[:mr])
    assert host.read(bufs["leaves"], 64 * pipe.tiles) == b"".join(
        C.leaf(C.Checked(c[:mr, c0:c0 + 64].tolist(), u[:mr, c0:c0 + 64].tolist()), "h3") for c0 in range(0, n, 64))


def y_trees(host, monkeypatch):
    """served_y's `DeviceTrees` over the host build, for a run of the sm120 variant whose salt is fixed."""
    torch = pytest.importorskip("torch")
    from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import served_y as SY
    from verity_torch.pouw.pearl_c import served_y_device as SYD
    run = SimpleNamespace(kernel=SimpleNamespace(run=R, dev=host), device=torch.device("cpu"), variant="sm120",
                          salt=bytes(range(64)), salt_key=bytes(range(100, 132)),
                          const_keys={"hm": host.put(audit.hm96_kernel_constants())}, cur=SimpleNamespace(poison=0xA5))
    monkeypatch.setattr(host, "stream", None, raising=False)
    return torch, SY, run, SYD.DeviceTrees(run)


def y_salt_bytes(torch, label: str, m: int):
    """`m` rows' test salts as `served_y_device.y_tree` takes them: uint8 on the host, `R.HM_SALT` bytes a row."""
    return torch.frombuffer(bytearray(salts_of(label, m)), dtype=torch.uint8)


@pytest.mark.parametrize("m", [1, 5, 129, 300])
def test_served_y_device_trees_are_the_trees_over_their_salts(host, monkeypatch, m):
    """served_y's `DeviceTrees` (a retained Pearl+ call's y tree as its device builds it, by run.py's launches) on the host
    build, handed the rows' salts: over a call's real rows of y, BF16 widened to FP32 7 rows at a time, the tree is the
    frame-v3-sha512 tree over y's FP32 rows under those salts, level 0 padded and every level above it, under the call's y
    domain, at 1, 5, 129 and 300 rows.  It is `commit_y`'s tree under the same salt bytes, which is also what `y_tree`
    gives where y is on the CPU (the fallback), and other salt bytes give another root."""
    n, index = 1024, 77
    torch, SY, run, trees = y_trees(host, monkeypatch)
    from verity_torch.pouw.pearl_c import served_y_device as SYD
    monkeypatch.setattr(SY, "Y_CHUNK_BYTES", 7 * 4 * n)
    y = torch.randn(m, n, generator=torch.Generator().manual_seed(m)).to(torch.bfloat16)
    raw = salts_of(f"y-{m}", m)
    salts, y_bytes = y_salt_bytes(torch, f"y-{m}", m), y.view(torch.uint8).numpy().tobytes()
    got = trees._tree(trees._pipe(n), y, index, salts).numpy().tobytes()
    root, tree = SY.commit_y(y_bytes, n, run.salt, index, [raw[i:i + R.HM_SALT] for i in range(0, len(raw), R.HM_SALT)])
    assert got == tree == tree_of(SY.y_domain(run.salt, index, m), SY.f32_rows(y_bytes, n), raw) and got[-64:] == root
    assert SYD.y_tree(run, y, index, salts).numpy().tobytes() == got
    other = trees._tree(trees._pipe(n), y, index, y_salt_bytes(torch, f"other-{m}", m)).numpy().tobytes()
    assert other[-64:] != root


def test_served_y_refuses_a_wrong_count_or_size_of_salts(host, monkeypatch):
    """`commit_y`, and `y_tree`'s fallback through it, take one salt of `R.HM_SALT` bytes a row of y: a salt too few, one
    too many and a short salt are each refused, naming the count and the size."""
    n, m, index = 1024, 5, 3
    torch, SY, run, _ = y_trees(host, monkeypatch)
    from verity_torch.pouw.pearl_c import served_y_device as SYD
    y = torch.randn(m, n, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    raw = salts_of("y-count", m + 1)
    each = [raw[i:i + R.HM_SALT] for i in range(0, len(raw), R.HM_SALT)]
    refusal = f"y's {m} rows take {m} salts of {R.HM_SALT} bytes"
    for salts in (each[:m - 1], each, [*each[:m - 1], each[m - 1][:-1]]):
        with pytest.raises(ValueError, match=refusal):
            SY.commit_y(y.view(torch.uint8).numpy().tobytes(), n, run.salt, index, salts)
    for size in (R.HM_SALT * (m - 1), R.HM_SALT * (m + 1), R.HM_SALT * m - 1):
        with pytest.raises(ValueError, match=refusal):
            SYD.y_tree(run, y, index, torch.frombuffer(bytearray(raw[:size]), dtype=torch.uint8))


def test_device_trees_refuse_a_wrong_count_or_size_of_salts_before_any_launch(host, monkeypatch):
    """The device path (`DeviceTrees`, whose kernels read `R.HM_SALT` bytes a leaf from the salts' address with no length)
    refuses a salt too few, one too many, a short salt, and salts not contiguous uint8 on y's device, each before any
    launch: its width's pipeline is built first, and every launch after it fails."""
    n, m, index = 1024, 5, 3
    torch, _, _, trees = y_trees(host, monkeypatch)
    y = torch.randn(m, n, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    raw = salts_of("y-device-count", m + 1)
    trees.pipe(n, index)

    def launched(*a, **kw):
        raise AssertionError("a launch")
    monkeypatch.setattr(host, "launch", launched, raising=False)
    wrong = [torch.frombuffer(bytearray(raw[:size]), dtype=torch.uint8)
             for size in (R.HM_SALT * (m - 1), R.HM_SALT * (m + 1), R.HM_SALT * m - 1)]
    whole = torch.frombuffer(bytearray(raw[:R.HM_SALT * m]), dtype=torch.uint8)
    wrong += [whole.view(torch.int8), torch.frombuffer(bytearray(raw[:2 * R.HM_SALT * m]), dtype=torch.uint8)[::2]]
    for salts in wrong:
        with pytest.raises(ValueError, match=f"y's {m} rows take {m} salts of {R.HM_SALT} bytes, contiguous uint8"):
            trees(y, index, salts)


def test_served_y_device_trees_refuse_a_width_whose_first_tree_isnt_commit_ys(host, monkeypatch):
    """A width's first device tree is held to `commit_y` over a fixed 129-row y under the same fresh salts, so a tree with a
    node off is refused, naming the node, and the width's pipeline isn't kept: the next call checks again.  A tree left as
    built passes the check, and the call's tree is `commit_y`'s under the salts it was handed."""
    n = 1024
    torch, SY, run, trees = y_trees(host, monkeypatch)
    real = trees._tree

    def off(p, y, index, salts):
        t = real(p, y, index, salts)
        t[64 * 3] ^= 1
        return t

    trees._tree = off
    y, salts = torch.zeros(4, n, dtype=torch.bfloat16), y_salt_bytes(torch, "y-check", 4)
    for _ in range(2):
        with pytest.raises(ValueError, match=f"width {n} differs from served_y.commit_y over 129 rows, first at node 3 of 511"):
            trees(y, 5, salts)
        assert not trees.pipes
    trees._tree = real
    raw = salts_of("y-check", 4)
    want = SY.commit_y(y.view(torch.uint8).numpy().tobytes(), n, run.salt, 5, [raw[i:i + R.HM_SALT] for i in range(0, len(raw), R.HM_SALT)])[1]
    assert trees(y, 5, salts).numpy().tobytes() == want and list(trees.pipes) == [n]
