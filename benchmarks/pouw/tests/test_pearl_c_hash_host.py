"""hash_h3.cuh and pearl_c_sm120/hash_fast.cuh on the host (pearl_c/hash_host.cpp, built with g++): each -h3 hashing kernel,
launched as pearl_c_sm120/run.py launches it, writes the bytes the verifier's reference computes: the SHA-512 row digests
(s512_rows, and hash_fast.cuh's segment and fold launches), the seeds and line keys over them at the pinned vectors, the line
raws, tickets and tile leaves (both ways), and the hm96-sha512 frame-v3-sha512 trees under ChaCha20 salts; then run.py's epoch,
unit, commit and hashing phases end to end with either kernel set."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "pearl_c"
FAST = HERE.parent / "pearl_c_sm120" / "hash_fast.cuh"
_spec = importlib.util.spec_from_file_location("pearl_c_sm120_run", HERE.parent / "pearl_c_sm120" / "run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

from verity.ml.commitments.merkle import tree_levels  # noqa: E402
from verity.core.protocols.pouw import audit  # noqa: E402
from verity.core.protocols.pouw.audit import ROW_SCHEMA, activation_binding, rows_domain  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402


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
                           (HERE / "hash_h3.cuh").read_text() + FAST.read_text()))
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


@pytest.mark.parametrize("count,digests,nonce", [(1, False, 0), (5, False, 7), (300, True, 1 << 63), (64, False, (1 << 62) + 3)])
def test_hm96_trees_are_the_verifiers(host, count, digests, nonce):
    """hm_leaves then s512_frame_leaves over 64-byte values (row digests, or with ``digests`` tile leaves) under ChaCha20 salts at each tree's nonce
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
    host.launch("hm_leaves", R.cdiv(count, 64), 64, p=(host.put(b"".join(xs)), buf, host.put(salt_key), const, 0, block + 96 if block else 0),
                i=(count, 0, 0) if block else (count, nonce & 0xFFFFFFFF, nonce >> 32))
    host.launch("s512_frame_leaves", R.cdiv(count, 64), 64, p=(buf, pdom), i=(0, count))
    s512_levels(host, buf, count, pdom)
    salted = [v + audit.hm96_salt(salt_key, nonce, i) for i, v in enumerate(values)]
    want = b"".join(b"".join(level) for level in tree_levels(dom, [dom.leaf(i, i, ROW_SCHEMA, v) for i, v in enumerate(salted)]))
    assert host.read(buf, 64 * (2 * width - 1)) == want
    root = audit.commit_rows(binding, 3, values, audit.HM96, salt_key, nonce, digests).commitment.root
    assert want[-64:] == root


@pytest.mark.parametrize("sha", ["fast", pytest.param("ref", marks=pytest.mark.slow)])
def test_sm120_h3_phases_are_the_verifiers(host, monkeypatch, sha):
    """pearl_c_sm120/run.py's `Pipeline` under -h3 at 64 x 128 x 1024 (m_real 48: a partial row block), its epoch, unit, commit
    and hashing phases on the host with either SHA-512 kernel set (``PEARLC_SHA``: hash_fast.cuh's segment and fold launches,
    or hash_h3.cuh's s512_rows and s512_tile_leaves): B's HM96 tree under weight 3's nonce, A's under the unit index read from
    the unit block, seed_A over root_A with its line key, each row's seed and line key, the tile leaves over C~ and U and the
    transcript tree, each equal to the reference's (`audit.commit_rows`, `pearl_c`)."""
    S = R
    m, n, k, mr, w, idx = 64, 128, 1024, 48, 3, 5
    rng = np.random.default_rng(12)
    a_rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(m)]
    b_rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(n)]
    c = rng.integers(0, 2**32, (m, n), dtype=np.uint64).astype(np.uint32)
    u = rng.integers(0, 2**32, (m, n), dtype=np.uint64).astype(np.uint32)
    salt, salt_key, tiles = bytes(range(32)), bytes(range(100, 132)), S.cdiv(mr, 64) * (n // 64)
    b_tree = audit.commit_rows(audit.weight_binding(w), -2, b_rows, audit.HM96, salt_key, audit.HM96_WEIGHTS + w)
    root_b = b_tree.commitment.root
    dom_a, dom_b = rows_domain(activation_binding(salt, idx), idx, mr, audit.HM96), b_tree.domain
    epoch = audit.Epoch(audit.Workload((C.Matmul(idx, C.Shape(mr, k, n), w),)), {w: root_b}, salt)
    dom_t = audit.transcript_domain(epoch, tiles, audit.HM96)
    keys = {"salt_key": salt_key, "hm": audit.hm96_kernel_constants(), "unit": salt + root_b + idx.to_bytes(8, "little"),
            "dom_a": dom_a.domain_id, "dom_b": dom_b.domain_id, "dom_t": dom_t.domain_id, "line_b": bytes(32)}
    host.sms = 4
    launched, launch = set(), host.launch
    monkeypatch.setattr(host, "launch", lambda name, *a, **kw: launched.add(name) or launch(name, *a, **kw))
    pipe = S.Pipeline(host, m, n, k, "sm120", {name: host.put(v) for name, v in keys.items()}, host.put(b"".join(a_rows)),
                      host.put(b"".join(b_rows)), m_real=mr, weight=w, sha=sha)
    for name, words in (("c", c), ("u", u)):
        host.bufs[pipe.b[name]][:m * n * 4] = np.frombuffer(words.astype("<u4").tobytes(), np.uint8)
    names = [nm for nm, phase, step in pipe.steps() if phase in ("epoch", "unit", "commit", "hashing") and (step() or True)]
    assert names == ["rowdig_b", "hash_rows_b", "tree_b", "pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a",
                     "hash_leaf", "tree_leaves_t", "tree_t"]
    fast, ref = {"s512_row_segs", "s512_tile_segs", "s512_folds"}, {"s512_rows", "s512_tile_leaves"}
    assert (launched & fast, launched & ref) == ((fast, set()) if sha == "fast" else (set(), ref))

    def levels(dom, values, nonce):
        salted = [v + audit.hm96_salt(salt_key, nonce, i) for i, v in enumerate(values)]
        return b"".join(b"".join(lv) for lv in tree_levels(dom, [dom.leaf(i, i, ROW_SCHEMA, v) for i, v in enumerate(salted)]))

    tree_a = levels(dom_a, a_rows[:mr], idx)
    seed_a = C.unit_seed_a(salt, tree_a[-64:], root_b, idx, "h3")
    seeds = [C.row_seed(salt, C.row_digest(r), root_b, idx, i) for i, r in enumerate(a_rows[:mr])]
    cr, ur = c[:mr], u[:mr]
    leaves = [C.leaf(C.Checked(cr[r0:r0 + 64, c0:c0 + 64].tolist(), ur[r0:r0 + 64, c0:c0 + 64].tolist()), "h3")
              for r0 in range(0, mr, 64) for c0 in range(0, n, 64)]
    want = {"tree_b": levels(dom_b, b_rows, audit.HM96_WEIGHTS + w), "tree_a": tree_a, "seed_a": seed_a,
            "line_a": C._s512_line_key(seed_a), "seeds_a": b"".join(seeds), "lkeys_a": b"".join(C._s512_line_key(x) for x in seeds),
            "leaves": b"".join(leaves), "tree_t": levels(dom_t, leaves, audit.HM96_TRANSCRIPT)}
    assert want["tree_b"][-64:] == root_b
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
