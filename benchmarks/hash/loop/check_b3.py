"""The rows-blake3 arm checked bit for bit before it is timed (``job.sh`` runs it with the committer's GPU tests): the hash lane's
hm96-blake3 reference (``ref/hm96_ref.py``, ``Prim("blake3k")``) at the committer's step -- its domain, salt key, ``sha512/row/v2``
rows and leaf schema -- against ``verity_vllm.commit.step_rows_b3`` and against ``native_rows_b3.cu`` through the committer's
``StepHasher``: masks, derived and given salts, b || c, h, leaves and pads across the position widths, rows of every tail length,
nodes at large indices and high levels, whole trees and windowed subtrees, and the wide-level launch schedule.

    $VENV/bin/python -m pytest benchmarks/hash/loop/check_b3.py      (from the source tree's root, GPU node)
"""
from __future__ import annotations

import hashlib
import os
import random
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref"))
import hm96_ref as H  # noqa: E402

from verity.ml.commitments.merkle import domain_message  # noqa: E402
from verity.ml.commitments.rowleaf import SHA512_ROW_V2_TAG  # noqa: E402
from verity_vllm.commit import step_rows as R  # noqa: E402
from verity_vllm.commit import step_rows_b3 as B  # noqa: E402

PD, CTX, GEO = (hashlib.sha256(t).digest() for t in (b"program", b"ctx", b"geo"))
LB = R.ROW_BYTES
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rows-blake3 kernels need a GPU")


class Committed(H.Format):
    """hm96_ref's blake3k format at the committer's step: the step's domain and salt key, x = H(Q(l) || row) with Q the
    sha512/row/v2 prefix in one 64-byte block, the hm96-sha512/row/v2 leaf schema."""

    def __init__(self, n: int, ks: bytes, row_bytes: int = LB):
        super().__init__(H.Prim("blake3k"), n)
        self.dom = R.domain(PD, CTX, GEO, n)
        self.dm = domain_message(self.dom.binding, self.dom.owner, self.dom.positions.identity_sha512(), n, "sha512")
        self.did = self.p.h(self.dm)
        self.ks = ks
        self.q = (SHA512_ROW_V2_TAG + (8 * row_bytes).to_bytes(8, "big")).ljust(self.p.block, b"\0")

    def row_digest(self, row: bytes) -> bytes:
        return self.p.h(self.q + row)

    def frame_leaf(self, pos: int, h: bytes) -> bytes:
        return self.p.h(H.message(b"leaf", self.did, H.uint(pos), H.uint(pos), R.SCHEMA.encode(), h))

    def levels(self, rows: list[bytes]) -> list[list[bytes]]:
        n = len(rows)
        width = 1 << max(0, (n - 1).bit_length())
        lv = [[self.leaf(i, r)[1] for i, r in enumerate(rows)] + [self.pad(r) for r in range(n, width)]]
        for d in range(width.bit_length() - 1):
            cur = lv[-1]
            lv.append([self.node(d, i, cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) // 2)])
        return lv


def test_step_rows_b3_is_hm96_ref_at_the_committers_step():
    rng = random.Random(1)
    ks = rng.randbytes(B.KEY_BYTES)
    f = Committed(1 << 25, ks)
    assert B.key() == f.key and B.SALT_PREFIX == f.salt_prefix and B.leaf_prefix() == f.leaf_prefix and B.t_table() == f.tbytes
    assert B.row_prefix(8 * LB) == f.q and B.did(f.dom) == f.did
    for pos in (0, 1, 255, 256, 65535, 65536, (1 << 24) + 3):
        row, y = rng.randbytes(LB), B.salt(ks, pos)
        assert y == f.prf(pos)
        bc, h = B.hiding_leaf(row, y, 8 * LB)
        assert bc == f.bc(f.row_digest(row), y) and h == f.hleaf(bc)
        assert B.leaf(f.did, pos, h) == f.frame_leaf(pos, h) and B.pad(f.did, pos) == f.pad(pos)
        assert B.node(f.did, 7, pos, row[:32], row[32:64]) == f.node(7, pos, row[:32], row[32:64])


def test_step_root_takes_the_kernels_root_slot():
    """The committer hands step_root the tree's root as the kernels store it (a 64-byte slot); the step root is SHA-512 over that slot."""
    d, args = os.urandom(B.DIGEST_BYTES), (os.urandom(32), os.urandom(32), os.urandom(32), os.urandom(32), 5)
    root = B.step_root(*args, B.slot(d))
    assert len(root) == 64 and root == B.step_root(*args, d) == R.step_root(*args, B.slot(d))


@pytest.fixture(scope="module")
def ext():
    from verity_vllm.commit.committer import native_collect as NC

    return NC.ext()


@pytest.fixture(scope="module")
def hasher(ext):
    return ext.StepHasher()


def _st() -> int:
    return torch.cuda.current_stream().cuda_stream


def _dev(b: bytes, at_least: int = 16) -> torch.Tensor:
    return torch.frombuffer(bytearray(b.ljust(at_least, b"\0")), dtype=torch.uint8).cuda()


def _rows(rows: list[bytes], stride: int) -> torch.Tensor:
    return _dev(b"".join(r.ljust(stride, b"\0") for r in rows))


def _list(t: torch.Tensor, w: int = 64) -> list[bytes]:
    b = t.cpu().numpy().tobytes()
    return [b[w * i:w * i + w] for i in range(len(b) // w)]


def _tn() -> torch.Tensor:
    return _dev(B.t_table())


def _step(ext, f: Committed, row_bytes: int = LB):
    return ext.RowStep(f.did, f.ks, f.q, f.salt_prefix, f.leaf_prefix, R.SCHEMA, row_bytes, hash="blake3", hash_key=H.B3_KEY)


S = B.slot


@gpu
@pytest.mark.parametrize("lb", [1, 15, 16, 63, 64, 65, 960, 961, 1023, 1024, 1025, 2047, 3072, 4031, 4096, 5000, 17 * 1024 + 9])
def test_given_salts_rows_of_every_tail(ext, hasher, lb):
    """Two rows a length, at a stride wider than the row: the row hash's chunks, parents and last-block length; given salts."""
    rng = random.Random(lb)
    f = Committed(2, bytes(32), lb)
    rows, ys = [rng.randbytes(lb) for _ in range(2)], [rng.randbytes(B.SALT_BYTES) for _ in range(2)]
    stride = -(-lb // 16) * 16 + 32
    lv, bc, hs = hasher.rows_leaves(_step(ext, f, lb), _rows(rows, stride), 2 * stride, 0, 2, 2, _tn(), _st(), stride=stride,
                                    salts=_dev(b"".join(ys)), want_bc=True, want_h=True)
    want_bc = [f.bc(f.row_digest(r), y) for r, y in zip(rows, ys, strict=True)]
    assert _list(bc, 128) == [w.ljust(128, b"\0") for w in want_bc]
    assert _list(hs) == [S(f.hleaf(w)) for w in want_bc]
    assert _list(lv) == [S(f.frame_leaf(i, f.hleaf(w))) for i, w in enumerate(want_bc)]


@gpu
@pytest.mark.parametrize("first,m,n_total", [(0, 300, 290), (250, 12, 258), (65530, 12, 65538), ((1 << 24) - 6, 12, (1 << 24) + 2),
                                             ((1 << 32) - 8, 8, (1 << 32) - 3)])
def test_derived_salts_leaves_and_pads_at_every_position_width(ext, hasher, first, m, n_total):
    rng = random.Random(first)
    f = Committed(n_total, rng.randbytes(B.KEY_BYTES))
    n_rows = min(m, n_total - first)
    rows = [rng.randbytes(LB) for _ in range(n_rows)]
    lv, bc, hs = hasher.rows_leaves(_step(ext, f), _rows(rows, LB), LB * n_rows, first, m, n_total, _tn(), _st(), want_bc=True, want_h=True)
    want = [f.leaf(first + i, r) for i, r in enumerate(rows)]
    assert [b[:64] for b in _list(bc, 128)[:n_rows]] == [w[0] for w in want]
    assert _list(lv) == [S(w[1]) for w in want] + [S(f.pad(p)) for p in range(first + n_rows, first + m)]


@gpu
def test_nodes_at_large_indices_and_high_levels(ext, hasher):
    rng = random.Random(7)
    f = Committed(1 << 32, bytes(32))
    rs, first = _step(ext, f), (1 << 24) - 8
    nodes = [rng.randbytes(32) for _ in range(8)]
    host = [torch.empty(64 * (8 >> L), dtype=torch.uint8) for L in range(4)]
    top = hasher.rows_tree_window(rs, _dev(b"".join(map(S, nodes))).view(8, 64), 4, first, _st(), host)
    want, cur = [nodes], nodes
    for L in range(3):
        cur = [f.node(L, (first >> (L + 1)) + i, cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) // 2)]
        want.append(cur)
    assert [_list(h) for h in host] == [[S(x) for x in w] for w in want] and _list(top) == [S(want[-1][0])]
    levels, root, _ = hasher.rows_tree(rs, _dev(b"".join(map(S, nodes[:4]))).view(4, 64), 29, _st(), levels_host=True)
    a, b = f.node(29, 0, nodes[0], nodes[1]), f.node(29, 1, nodes[2], nodes[3])
    assert [_list(t) for t in levels] == [[S(x) for x in nodes[:4]], [S(a), S(b)], [S(f.node(30, 0, a, b))]] and root == S(f.node(30, 0, a, b))


@gpu
@pytest.mark.parametrize("n", [1, 2, 3, 5, 33, 257])
def test_trees_and_windowed_subtrees(ext, hasher, n):
    """The whole step at once and window by window as the committer's worker hashes it (windows of 1, 4 and 32 leaves, each window's
    levels to host, the tops' tree to the root), against the reference's levels; and step_rows_b3.levels is the same tree."""
    rng = random.Random(n)
    rows, f = [rng.randbytes(LB) for _ in range(n)], Committed(n, rng.randbytes(B.KEY_BYTES))
    want = f.levels(rows)
    if n <= 5:
        assert [list(x) for x in B.levels(f.dom, rows, f.ks, 8 * LB)] == want
    width, depth = len(want[0]), len(want) - 1
    rs, tn, st, stream = _step(ext, f), _tn(), _st(), _rows(rows, LB)
    lv, _, _ = hasher.rows_leaves(rs, stream, LB * n, 0, width, n, tn, st)
    levels, root, _ = hasher.rows_tree(rs, lv, 0, st, levels_host=True)
    assert [_list(t) for t in levels] == [[S(x) for x in w] for w in want] and root == S(want[-1][0])
    for j in (0, 2, 5):
        w, n_levels = 1 << j, min(j, depth) + 1
        tops = []
        for k in range(max(1, width >> j)):
            first = k * w
            m, n_rows = min(w, width - first), max(0, min(w, n - first))
            window = stream[LB * first:] if n_rows else _dev(b"")
            leaves, _, _ = hasher.rows_leaves(rs, window, LB * n_rows, first, m, n, tn, st)
            host = [torch.empty(64 * (m >> L), dtype=torch.uint8) for L in range(n_levels)]
            tops.append(hasher.rows_tree_window(rs, leaves, n_levels, first, st, host))
            for L, h in enumerate(host):
                assert _list(h) == [S(x) for x in want[L][first >> L:(first + m) >> L]], (j, k, L)
        _, wroot, _ = hasher.rows_tree(rs, torch.cat(tops), n_levels - 1, st)
        assert wroot == S(want[-1][0]), j


@gpu
@pytest.mark.parametrize("n,level0", [(2048, 0), (4096, 3), (1 << 17, 0)])
def test_wide_levels_by_the_launch_schedule(ext, hasher, n, level0):
    """rows_tree over two blocks' slices, from level 3, and over one wide level (2^17 nodes, launches of a thread a parent): sampled
    nodes of every level against the reference from the device's own children, the root included."""
    rng = random.Random(n + level0)
    f = Committed(1 << 20, bytes(32))
    nodes = [rng.randbytes(32) for _ in range(n)]
    levels, root, _ = hasher.rows_tree(_step(ext, f), _dev(b"".join(map(S, nodes))).view(n, 64), level0, _st(), levels_host=True)
    got = [_list(t) for t in levels]
    assert got[0] == [S(x) for x in nodes] and all(len(g) == n >> d for d, g in enumerate(got)) and root == got[-1][0]
    for d in range(1, len(got)):
        for i in sorted({0, len(got[d]) - 1, *rng.sample(range(len(got[d])), min(len(got[d]), 24))}):
            assert got[d][i] == S(f.node(level0 + d - 1, i, got[d - 1][2 * i][:32], got[d - 1][2 * i + 1][:32])), (d, i)
