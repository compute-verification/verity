"""``native_rows.cu`` (the ``rows-sha512`` kernels, through ``StepHasher.rows_leaves`` / ``rows_h`` + ``rows_frame`` / ``rows_tree_window``
/ ``rows_tree``) hashes what ``verity_vllm.commit.step_rows`` does, bit for bit: frame-v3-sha512's hiding-leaf vectors and hm96-sha512's
``sha512/row/v1`` leaf vectors (given salts), both hm96 keys' masks, the derived salts, leaves and pads at every position width, the
leaves in two halves (h, then the frame once the step's length is known) as ``rows_leaves``' bytes, nodes at large indices and high
levels, and whole trees and windowed subtrees (one window a fold, or several with their levels copied on another stream) against
``step_rows.levels``."""
from __future__ import annotations

import hashlib
import json
import os
import random

import pytest

pytestmark = pytest.mark.pod
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the rows-sha512 kernels need a GPU", allow_module_level=True)

from verity.ml.commitments import merkle  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain  # noqa: E402
from verity_vllm.commit import step_rows as R  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402

_CORE = os.path.dirname(merkle.__file__)
FRAME = json.load(open(os.path.join(_CORE, "frame_v3", "vectors_sha512.json")))
HM96 = json.load(open(os.path.join(_CORE, "hm96", "vectors_sha512.json")))
OTHER_KEY = bytes.fromhex(HM96["other_key"])
PD, CTX, GEO = (hashlib.sha256(t).digest() for t in (b"program", b"ctx", b"geo"))
BITS = 8 * R.ROW_BYTES


@pytest.fixture(scope="module")
def ext():
    return NC.ext()


@pytest.fixture(scope="module")
def hasher(ext):
    return ext.StepHasher()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _dev(b: bytes, at_least: int = 16) -> torch.Tensor:
    return torch.frombuffer(bytearray(b.ljust(at_least, b"\0")), dtype=torch.uint8).cuda()


def _rows(rows: list[bytes], stride: int) -> torch.Tensor:
    return _dev(b"".join(r.ljust(stride, b"\0") for r in rows))


def _list(t: torch.Tensor) -> list[bytes]:
    b, w = t.cpu().numpy().tobytes(), t.shape[1]
    return [b[w * i:w * i + w] for i in range(t.shape[0])]


def _tn(key: bytes = R.KEY) -> torch.Tensor:
    return _dev(R.t_nibbles(key))


def _step(ext, dom: CommitmentDomain, ks: bytes, prefix: bytes, row_bytes: int, key: bytes = R.KEY):
    return ext.RowStep(dom.domain_id, ks, prefix, R.INST.salt_prefix, R.INST.leaf_prefix(key), R.SCHEMA, row_bytes)


def _vector_domain(d: dict) -> CommitmentDomain:
    return CommitmentDomain(bytes.fromhex(d["binding"]), d["owner"], RangeIndexedDomain(*d["range"]), hash=d["hash"])


@pytest.mark.parametrize("vec", FRAME["bit_row_hiding_leaves"], ids=lambda v: v["name"])
def test_hiding_leaves_are_frame_v3_sha512s_vectors(ext, hasher, vec):
    row, n_bits, y, rank = bytes.fromhex(vec["row_bytes"]), vec["n_bits"], bytes.fromhex(vec["salt"]), vec["rank"]
    dom, stride = _vector_domain(vec["domain"]), -(-len(row) // 16) * 16
    rs = _step(ext, dom, bytes(64), R.row_prefix(n_bits), len(row))
    lv, bc, hs = hasher.rows_leaves(rs, _rows([row], stride), stride, rank, 1, dom.count, _tn(), _stream(), stride=stride, salts=_dev(y),
                                    want_bc=True, want_h=True)
    assert (_list(bc)[0].hex(), _list(hs)[0].hex(), _list(lv)[0].hex()) == (vec["bc"], vec["hm96_leaf"], vec["leaf"])


@pytest.mark.parametrize("vec", [v for v in HM96["leaves"] if v["inner"] == "sha512/row/v1"], ids=lambda v: str(len(v["value"]) // 2))
def test_row_v1_leaves_are_hm96_sha512s_vectors(ext, hasher, vec):
    """A 128-byte inner prefix is the kernel's P block: these rows (2 and 3072 bytes) run its message padding at both block edges."""
    prefix, value, y = (bytes.fromhex(vec[k]) for k in ("inner_prefix", "value", "salt"))
    assert len(prefix) == 128 and not vec["inner_suffix"]
    dom, stride = R.domain(PD, CTX, GEO, 1), -(-len(value) // 16) * 16
    rs = _step(ext, dom, bytes(64), prefix, len(value))
    _, bc, hs = hasher.rows_leaves(rs, _rows([value], stride), stride, 0, 1, 1, _tn(), _stream(), stride=stride, salts=_dev(y), want_bc=True,
                                   want_h=True)
    assert (_list(bc)[0].hex(), _list(hs)[0].hex()) == (vec["b"] + vec["c"], vec["leaf"])


@pytest.mark.parametrize("key", [R.KEY, OTHER_KEY], ids=["default", "other"])
@pytest.mark.parametrize("stride", [R.ROW_BYTES, R.ROW_BYTES + 48])
def test_hiding_leaves_under_both_keys(ext, hasher, key, stride):
    """300 rows (two thread blocks) with given salts, at the step's stride and a wider one: b || c and h are hm96's under either key."""
    rng = random.Random(stride)
    rows = [rng.randbytes(R.ROW_BYTES) for _ in range(300)]
    ys = [rng.randbytes(R.SALT_BYTES) for _ in rows]
    dom = R.domain(PD, CTX, GEO, len(rows))
    rs = _step(ext, dom, bytes(64), R.row_prefix(BITS), R.ROW_BYTES, key)
    lv, bc, hs = hasher.rows_leaves(rs, _rows(rows, stride), stride * len(rows), 0, len(rows), len(rows), _tn(key), _stream(), stride=stride,
                                    salts=_dev(b"".join(ys)), want_bc=True, want_h=True)
    want = [R.hiding_leaf(r, y, BITS, key) for r, y in zip(rows, ys, strict=True)]
    assert _list(bc) == [w[0] for w in want] and _list(hs) == [w[1] for w in want]
    assert _list(lv) == [R.leaf(dom, i, w[1]) for i, w in enumerate(want)]


@pytest.mark.parametrize("first,m,n_total", [(0, 3, 2), (250, 12, 258), (65530, 12, 65538), ((1 << 24) - 6, 12, (1 << 24) + 2),
                                             ((1 << 32) - 8, 8, (1 << 32) - 3)])
def test_derived_salts_leaves_and_pads_at_every_position_width(ext, hasher, first, m, n_total):
    """Salts from the step's key, leaves below n_total and pads from it on, across the 1-to-4-byte position boundaries."""
    rng = random.Random(first)
    ks = rng.randbytes(R.KEY_BYTES)
    n_rows = min(m, n_total - first)
    rows = [rng.randbytes(R.ROW_BYTES) for _ in range(n_rows)]
    dom = R.domain(PD, CTX, GEO, n_total)
    rs = _step(ext, dom, ks, R.row_prefix(BITS), R.ROW_BYTES)
    lv, _, hs = hasher.rows_leaves(rs, _rows(rows, R.ROW_BYTES), R.ROW_BYTES * n_rows, first, m, n_total, _tn(), _stream(), want_h=True)
    want_h = [R.hiding_leaf(r, R.salt(ks, first + i), BITS)[1] for i, r in enumerate(rows)]
    assert _list(hs)[:n_rows] == want_h
    assert _list(lv) == [R.leaf(dom, first + i, h) for i, h in enumerate(want_h)] + [dom.pad(p) for p in range(first + n_rows, first + m)]


@pytest.mark.parametrize("first,m,n_total,given,cap", [(0, 300, 290, False, 0), (0, 300, 290, True, 0), (250, 12, 258, False, 0),
                                                       (65530, 12, 65538, True, 0), ((1 << 24) - 6, 12, (1 << 24) + 2, False, 0),
                                                       (512, 1024, 1536, False, 1), (512, 1024, 1300, True, 1)])
def test_h_then_frame_is_rows_leaves(ext, hasher, first, m, n_total, given, cap):
    """The two halves a learning step that keeps no bytes takes (rows_h on each window before the step's length is known, rows_frame
    after) are rows_leaves' bytes: h and b || c from a RowStep of another domain under the same salt key, the leaves and pads from the
    step's own, with derived and given salts, across position widths, past one thread block and in capped launches."""
    rng = random.Random(first ^ m)
    ks = rng.randbytes(R.KEY_BYTES)
    n_rows = min(m, n_total - first)
    rows = _dev(rng.randbytes(R.ROW_BYTES * n_rows))
    salts = _dev(rng.randbytes(R.SALT_BYTES * n_rows)) if given else None
    tn, st = _tn(), _stream()
    rs = _step(ext, R.domain(PD, CTX, GEO, n_total), ks, R.row_prefix(BITS), R.ROW_BYTES)
    rs_any = _step(ext, R.domain(PD, CTX, GEO, 1), ks, R.row_prefix(BITS), R.ROW_BYTES)
    rs.max_blocks = rs_any.max_blocks = cap
    lv, bc, hs = hasher.rows_leaves(rs, rows, R.ROW_BYTES * n_rows, first, m, n_total, tn, st, salts=salts, want_bc=True, want_h=True)
    h, bc2 = hasher.rows_h(rs_any, rows, R.ROW_BYTES * n_rows, first, tn, st, salts=salts, want_bc=True)
    lv2 = hasher.rows_frame(rs, h, first, m, n_total, st)
    torch.cuda.synchronize()
    assert torch.equal(h, hs[:n_rows]) and torch.equal(bc2, bc[:n_rows])
    assert torch.equal(lv2, lv)


def test_rows_frame_from_no_h_is_all_pads(ext, hasher):
    rs = _step(ext, R.domain(PD, CTX, GEO, 3), bytes(64), R.row_prefix(BITS), R.ROW_BYTES)
    lv, _, _ = hasher.rows_leaves(rs, _dev(b""), 0, 3, 5, 3, _tn(), _stream())
    lv2 = hasher.rows_frame(rs, torch.empty(0, 64, dtype=torch.uint8, device="cuda"), 3, 5, 3, _stream())
    assert torch.equal(lv2, lv)


def test_nodes_at_large_indices_and_high_levels(ext, hasher):
    rng = random.Random(7)
    dom = R.domain(PD, CTX, GEO, 1 << 32)
    rs = _step(ext, dom, bytes(64), R.row_prefix(BITS), R.ROW_BYTES)
    first = (1 << 24) - 8
    nodes = [rng.randbytes(64) for _ in range(8)]
    host = [torch.empty(64 * (8 >> L), dtype=torch.uint8) for L in range(4)]
    top = hasher.rows_tree_window(rs, _dev(b"".join(nodes)).view(8, 64), 4, first, _stream(), host)
    want, cur = [nodes], nodes
    for L in range(3):
        cur = [dom.node(L, (first >> (L + 1)) + i, cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) // 2)]
        want.append(cur)
    assert [[h.numpy().tobytes()[64 * i:64 * i + 64] for i in range(len(w))] for h, w in zip(host, want, strict=True)] == want
    assert _list(top) == want[-1]
    levels, root, _ = hasher.rows_tree(rs, _dev(b"".join(nodes[:4])).view(4, 64), 29, _stream(), levels_host=True)
    a, b = dom.node(29, 0, nodes[0], nodes[1]), dom.node(29, 1, nodes[2], nodes[3])
    assert [_list(t) for t in levels] == [nodes[:4], [a, b], [dom.node(30, 0, a, b)]] and root == dom.node(30, 0, a, b)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 31, 32, 33, 100, 257])
def test_trees_and_windowed_subtrees_are_step_rows_levels(ext, hasher, n):
    """The whole step at once, and window by window as the worker hashes it (window k = positions [k 2^j, (k + 1) 2^j), each with its
    pads, every window's levels to host, the tops' tree to the root), for windows of 1, 4 and 32 leaves."""
    rng = random.Random(n)
    rows = [rng.randbytes(R.ROW_BYTES) for _ in range(n)]
    ks = rng.randbytes(R.KEY_BYTES)
    dom = R.domain(PD, CTX, GEO, n)
    want = R.levels(dom, rows, ks, BITS)
    width, depth = len(want[0]), len(want) - 1
    rs, tn, st = _step(ext, dom, ks, R.row_prefix(BITS), R.ROW_BYTES), _tn(), _stream()
    stream = _rows(rows, R.ROW_BYTES)
    lv, _, _ = hasher.rows_leaves(rs, stream, R.ROW_BYTES * n, 0, width, n, tn, st)
    levels, root, _ = hasher.rows_tree(rs, lv, 0, st, levels_host=True)
    assert [tuple(_list(t)) for t in levels] == list(want) and root == want[-1][0]
    for j in (0, 2, 5):
        w, n_levels = 1 << j, min(j, depth) + 1
        tops = []
        for k in range(max(1, width >> j)):
            first = k * w
            m, n_rows = min(w, width - first), max(0, min(w, n - first))
            window = stream[R.ROW_BYTES * first:] if n_rows else _dev(b"")
            leaves, _, _ = hasher.rows_leaves(rs, window, R.ROW_BYTES * n_rows, first, m, n, tn, st)
            host = [torch.empty(64 * (m >> L), dtype=torch.uint8) for L in range(n_levels)]
            tops.append(hasher.rows_tree_window(rs, leaves, n_levels, first, st, host))
            for L, h in enumerate(host):
                got = h.numpy().tobytes()
                assert [got[64 * i:64 * i + 64] for i in range(m >> L)] == list(want[L][first >> L:(first + m) >> L]), (j, k, L)
        _, wroot, _ = hasher.rows_tree(rs, torch.cat(tops), n_levels - 1, st)
        assert wroot == want[-1][0], j


@pytest.mark.parametrize("n", [5, 100, 257])
def test_windows_folded_together_copy_their_levels_on_another_stream(ext, hasher, n):
    """Consecutive windows' leaves folded at once, g windows a fold (the worker's held leaves), each fold's levels copied into pinned
    host slices on a copy stream with no sync, the copy stream joined before the tops' tree: every level and the root are step_rows'."""
    rng = random.Random(n)
    rows = [rng.randbytes(R.ROW_BYTES) for _ in range(n)]
    ks = rng.randbytes(R.KEY_BYTES)
    dom = R.domain(PD, CTX, GEO, n)
    want = R.levels(dom, rows, ks, BITS)
    width, depth = len(want[0]), len(want) - 1
    rs, tn, st = _step(ext, dom, ks, R.row_prefix(BITS), R.ROW_BYTES), _tn(), _stream()
    stream = _rows(rows, R.ROW_BYTES)
    copy = torch.cuda.Stream()
    for j, g in ((0, 3), (2, 2), (2, 3), (5, 3)):
        w, n_levels = 1 << j, min(j, depth) + 1
        host = [torch.empty(64 * (width >> L), dtype=torch.uint8).pin_memory() for L in range(n_levels)]
        tops: list[torch.Tensor] = []
        held: list[torch.Tensor] = []

        def fold() -> None:
            lv = torch.cat(held) if len(held) > 1 else held[0]
            held.clear()
            first, m = sum(int(t.shape[0]) for t in tops) << (n_levels - 1), int(lv.shape[0])
            dsts = [h[64 * (first >> L):64 * ((first + m) >> L)] for L, h in enumerate(host)]
            tops.append(hasher.rows_tree_window(rs, lv, n_levels, first, st, dsts, copy.cuda_stream))

        for k in range(max(1, width >> j)):
            first = k * w
            m, n_rows = min(w, width - first), max(0, min(w, n - first))
            window = stream[R.ROW_BYTES * first:] if n_rows else _dev(b"")
            held.append(hasher.rows_leaves(rs, window, R.ROW_BYTES * n_rows, first, m, n, tn, st)[0])
            if len(held) == g:
                fold()
        if held:
            fold()
        torch.cuda.current_stream().wait_stream(copy)
        _, wroot, _ = hasher.rows_tree(rs, torch.cat(tops), n_levels - 1, st)
        for L, h in enumerate(host):
            got = h.numpy().tobytes()
            assert [got[64 * i:64 * i + 64] for i in range(width >> L)] == list(want[L]), (j, g, L)
        assert wroot == want[-1][0], (j, g)


@pytest.mark.parametrize("n,level0,first", [(2048, 0, 0), (4096, 3, 0), (1536, 0, 4 * 1536), (3072, 0, 2 * 3072), (1 << 17, 0, 0),
                                            (3 << 16, 0, 2 * (3 << 16))])
def test_wide_trees_and_windows_fold_by_the_launch_schedule(ext, hasher, n, level0, first):
    """A level of more than 65536 nodes a launch, the narrower ones ten a launch, each block folding an aligned slice of 1024 nodes:
    rows_tree from 2048 nodes (two blocks, then a launch of one level over their tops), from 4096 at level 3 and from 2^17 (one wide
    level), and windows' subtrees of 1536 nodes (three blocks of 512), 3072 (three of 1024) and 3 * 2^16 (two wide levels) at nonzero
    first indices with their levels to host, all against dom.node."""
    rng = random.Random(n + level0)
    dom = R.domain(PD, CTX, GEO, 1 << 20)
    rs, st = _step(ext, dom, bytes(64), R.row_prefix(BITS), R.ROW_BYTES), _stream()
    nodes = [rng.randbytes(64) for _ in range(n)]
    n_up = (n & -n).bit_length() - 1
    want, cur = [nodes], nodes
    for L in range(n_up):
        cur = [dom.node(level0 + L, (first >> (L + 1)) + i, cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) // 2)]
        want.append(cur)
    if first:
        host = [torch.empty(64 * len(w), dtype=torch.uint8) for w in want]
        top = hasher.rows_tree_window(rs, _dev(b"".join(nodes)).view(n, 64), n_up + 1, first, st, host)
        assert [_list(h.view(-1, 64)) for h in host] == want and _list(top) == want[-1]
        assert torch.equal(hasher.rows_tree_window(rs, _dev(b"".join(nodes)).view(n, 64), n_up + 1, first, st, []), top)
    else:
        levels, root, _ = hasher.rows_tree(rs, _dev(b"".join(nodes)).view(n, 64), level0, st, levels_host=True)
        assert [_list(t) for t in levels] == want and root == want[-1][0]


@pytest.mark.parametrize("cap", [1, 3])
def test_capped_launches_hash_the_whole_grids_bytes(ext, hasher, cap):
    """RowStep.max_blocks > 0 hashes the leaves and each wide level in launches of at most that many blocks: leaves, b || c and h (given
    and derived salts; 1000 positions from a nonzero first, the 700 rows ending inside a launch, pads after them), a window's levels from
    2^17 nodes (a wide level) at a nonzero first index, and a whole tree's root are the whole-grid launch's."""
    rng = random.Random(cap)
    first, m, n_rows, n = 3 << 10, 1000, 700, 1 << 17
    dom = R.domain(PD, CTX, GEO, 1 << 20)
    rs, tn, st = _step(ext, dom, rng.randbytes(R.KEY_BYTES), R.row_prefix(BITS), R.ROW_BYTES), _tn(), _stream()
    rows = _rows([rng.randbytes(R.ROW_BYTES) for _ in range(n_rows)], R.ROW_BYTES)
    salts, nodes = _dev(rng.randbytes(R.SALT_BYTES * n_rows)), _dev(rng.randbytes(64 * n)).view(n, 64)

    def run(max_blocks: int) -> list:
        rs.max_blocks = max_blocks
        lv, bc, hs = hasher.rows_leaves(rs, rows, R.ROW_BYTES * n_rows, first, m, first + n_rows, tn, st, salts=salts, want_bc=True,
                                        want_h=True)
        dlv, _, dhs = hasher.rows_leaves(rs, rows, R.ROW_BYTES * n_rows, first, m, first + n_rows, tn, st, want_h=True)
        host = [torch.empty(64 * (n >> L), dtype=torch.uint8) for L in range(18)]
        top = hasher.rows_tree_window(rs, nodes, 18, 5 * n, st, host)
        _, root, _ = hasher.rows_tree(rs, nodes, 0, st)
        torch.cuda.synchronize()
        return [_list(t) for t in (lv, bc, hs, dlv, dhs, top)] + [h.numpy().tobytes() for h in host] + [root]

    assert run(cap) == run(0)


def test_refusals(ext, hasher):
    dom = R.domain(PD, CTX, GEO, 4)
    with pytest.raises(RuntimeError, match="64 bytes"):
        ext.RowStep(dom.domain_id[:32], bytes(64), R.row_prefix(BITS), R.INST.salt_prefix, R.INST.leaf_prefix(R.KEY), R.SCHEMA, R.ROW_BYTES)
    with pytest.raises(RuntimeError, match="does not fit the schema"):
        ext.RowStep(dom.domain_id, bytes(64), R.row_prefix(BITS), R.INST.salt_prefix, R.INST.leaf_prefix(R.KEY), "hm96-sha512/row-seg/v1",
                    R.ROW_BYTES)
    rs = _step(ext, dom, bytes(64), R.row_prefix(BITS), R.ROW_BYTES)
    rows = _dev(bytes(4 * R.ROW_BYTES))
    with pytest.raises(RuntimeError, match="16-byte"):
        hasher.rows_leaves(rs, rows, 4 * R.ROW_BYTES, 0, 4, 4, _tn(), _stream(), stride=R.ROW_BYTES - 16)
    with pytest.raises(RuntimeError, match="mask table"):
        hasher.rows_leaves(rs, rows, 4 * R.ROW_BYTES, 0, 4, 4, _dev(bytes(4096)), _stream())
    with pytest.raises(RuntimeError, match="power of two"):
        hasher.rows_tree(rs, _dev(bytes(3 * 64)).view(3, 64), 0, _stream())
