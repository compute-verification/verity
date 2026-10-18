"""`registry.dsv4_decode_metadata` (vLLM d9105ea80): the window row (`_compute_swa_indices_and_lens_kernel`) and the ratio-128 row
(`_build_c128a_topk_metadata_kernel`'s decode branch) against models of the Triton kernels' stores (the valid byte, positions
before the window fills, the block edges, the last position of a row at max_model_len, speculative decode's query tokens, the
int32 and int64 ends, a block number whose product wraps, lanes past the row), the twin against the reference, the statics, and
each Boolean Definition against its word."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, self_check
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity_vllm.program import boolean as SW_
from verity_vllm.program.kernels import dsv4_decode_metadata_twin as T
from verity_vllm.program.registry import boolean_dsv4_decode_metadata as BX
from verity_vllm.program.registry import dsv4_decode_metadata as X

U = np.uint32
M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
DEFS = {d.id: d for d in X.DSV4_DECODE_METADATA_COMPOSITES}
TWIN_TARGETS = T._targets(DEFS)
SWA_SERVED, SWA_4096 = X.DSV4_DECODE_METADATA["swa"], X.DSV4_DECODE_METADATA["swa-4096"]
C128_SERVED, C128_4096 = X.DSV4_DECODE_METADATA["c128"], X.DSV4_DECODE_METADATA["c128-4096"]
SWA_SMALL = dict(SW=8, BS=4, NB=8)
SWA_ODD = dict(SW=5, BS=2, NB=3)
C_SMALL = dict(W=16, R=4, BS=2, NB=8)
C_PAST = dict(W=12, R=3, BS=4, NB=2)


class Outside(Exception):
    """The kernel reads a block outside the token's own row (another request's), or a masked lane is live."""


def _s32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def _s64(v: int) -> int:
    v &= M64
    return v - (1 << 64) if v >> 63 else v


def _cdiv(a: int, b: int) -> int:
    """C's (and Triton's) integer division: truncation toward 0."""
    return a // b if a >= 0 else -(-a // b)


def swa_kernel(t, qs, qe, seq, blocks, valid, SW: int, BS: int) -> list[int]:
    """`_compute_swa_indices_and_lens_kernel` (v1/attention/backends/mla/sparse_swa.py:950-1008) for one token at window_size =
    index_width = SW, HAS_IMAGE false, read line by line in int32: an invalid byte stores length 0 and a row of -1; else the
    position, start, end and length, and per lane its block number loaded where pos_offset < end_pos, `block_numbers *
    block_size + block_offsets` and -1 from the length on.  The row, then the length, as uint32 words; `Outside` where a live
    lane reads past the row or is masked."""
    if int(valid) & 0xFF == 0:
        return [M32] * SW + [0]
    t, qs, qe, seq = (_s32(int(v)) for v in (t, qs, qe, seq))
    query_len = _s32(qe - qs)
    prefix_len = _s32(seq - query_len)
    pos = _s32(_s32(prefix_len + t) - qs)
    left_add = max(0 - (SW - 1), 0)
    start = max(_s32(_s32(pos - (SW - 1)) - left_add), 0)
    end = _s32(_s32(pos + 0) + 1)
    length = _s32(end - start)
    out = []
    for j in range(SW):
        p = _s32(start + j)
        b = _cdiv(p, BS)
        loaded = p < end
        if j < length and not (loaded and 0 <= b < len(blocks)):
            raise Outside(j)
        number = _s32(int(blocks[b])) if loaded and 0 <= b < len(blocks) else 0
        slot = _s32(number * BS + (p - b * BS))
        out.append(slot if j < length else -1)
    return [v & M32 for v in out] + [length & M32]


def c128a_kernel(position, slot, blocks, W: int, R: int, BS: int) -> list[int]:
    """`_build_c128a_topk_metadata_kernel`'s decode branch (models/deepseek_v4/sparse_mla.py:378-416) for one token at
    max_compressed_tokens W, compress_ratio R and block_size BS: num_compressed = min((position + 1) // R, W) in int64
    (truncating), lane j valid where j < num_compressed, its block number blocks[j // BS] loaded there, the slot in int32, -1
    elsewhere, and the int32 count of valid lanes where the token's int64 slot_mapping is >= 0.  The row, then the length;
    `Outside` where a valid lane's block lies past the row."""
    nc = min(_cdiv(_s64(_s64(int(position)) + 1), R), W)
    out, count = [], 0
    for j in range(W):
        live = j < nc
        if live and j // BS >= len(blocks):
            raise Outside(j)
        number = _s32(int(blocks[j // BS])) if live else 0
        out.append(_s32(number * BS + j % BS) if live else -1)
        count += live
    return [v & M32 for v in out] + [count if _s64(int(slot)) >= 0 else 0]


def _w32(xs) -> np.ndarray:
    return (np.asarray(xs, dtype=np.int64) & M32).astype(U)


def _w64(xs) -> np.ndarray:
    return np.asarray([int(x) & M64 for x in xs], dtype=np.uint64)


def _blocks(rng, n: int, NB: int) -> np.ndarray:
    return _w32(np.where(rng.random((n, NB)) < 0.5, rng.integers(0, 1 << 17, (n, NB)), rng.integers(I32_MIN, I32_MAX, (n, NB))))


def _swa_tokens(st: dict, seed: int) -> list[np.ndarray]:
    """Edge tokens at statics ``st``: every position of the window's fill (0 .. SW), the block edges, the row's last positions, as
    the first to fourth of up to four query tokens of a request; the bytes 1, 0, 2, 0x80 and 0xFF."""
    SW, BS, NB = st["SW"], st["BS"], st["NB"]
    rng = np.random.default_rng(seed)
    span = NB * BS
    positions = sorted({p for p in [*range(SW + 2), BS - 1, BS, BS + 1, 2 * BS - 1, SW + BS - 1, span - SW - 1, span - SW,
                                    span - 2, span - 1] if 0 <= p < span})
    cols = [[] for _ in range(6)]
    for i, pos in enumerate(positions):
        qlen = 1 + i % 4
        qs = int(rng.integers(0, 1 << 20))
        t = qs + (i // 4) % qlen
        seq = pos + qlen - (t - qs)
        for c, v in zip(cols, (t, qs, qs + qlen, seq)):
            c.append(v)
        cols[4].append(rng.integers(0, 1 << 17, NB) if i % 3 else rng.integers(I32_MIN, I32_MAX, NB))
        cols[5].append([1, 2, 0x80, 0xFF, 0][i % 5])
    return [*(_w32(c) for c in cols[:4]), _w32(cols[4]), np.asarray(cols[5], dtype=np.uint8)]


def _swa_want(st: dict, args) -> np.ndarray:
    return np.asarray([swa_kernel(*a, st["SW"], st["BS"]) for a in zip(*args)], dtype=np.uint64)


def _c128a_tokens(st: dict, seed: int) -> list[np.ndarray]:
    """Edge tokens at statics ``st``: num_compressed 0, 1, every block edge and the most the row holds, each at the first and last
    position that gives it; position -1 and -R; the int64 ends (position + 1 wraps at INT64_MAX); slot_mapping >= 0, -1 and
    INT64_MIN."""
    W, R, BS, NB = st["W"], st["R"], st["BS"], st["NB"]
    rng = np.random.default_rng(seed)
    most = min(W, BS * NB)
    qs = sorted({q for q in [0, 1, 2, BS - 1, BS, BS + 1, most - BS, most - 1, most] if 0 <= q <= most})
    positions = [p for q in qs for p in (q * R - 1, q * R + R - 2)] + [-1, -R, -R - 1, I64_MIN, I64_MIN + 1, I64_MAX]
    slots = [[0, 5, (1 << 40), -1, I64_MIN, I64_MAX][i % 6] for i in range(len(positions))]
    return [_w64(positions), _w64(slots), _blocks(rng, len(positions), NB)]


def _c128a_want(st: dict, args) -> np.ndarray:
    return np.asarray([c128a_kernel(*a, st["W"], st["R"], st["BS"]) for a in zip(*args)], dtype=np.uint64)


def _reference(fn, args) -> np.ndarray:
    n = len(args[0])
    return np.asarray(evaluate_batch(fn, *args, kernel="reference"), dtype=np.uint64).reshape(n, -1)


# --- the window row against the kernel --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("st", [SWA_SMALL, SWA_ODD, dict(SW=3, BS=8, NB=4), dict(SW=128, BS=64, NB=4), dict(SW=1, BS=1, NB=2)],
                         ids=["small", "odd", "narrow", "served-window", "one-lane"])
def test_the_window_row_is_the_kernels_stores_on_edge_tokens(st):
    args = _swa_tokens(st, 1)
    assert np.array_equal(_reference(bind(X.Dsv4SwaIndices, **st), args), _swa_want(st, args))


def test_the_window_row_at_the_served_statics():
    """Window 128, block 64, the SWA group's row of 16,384 blocks at max_model_len 1,048,576: the last position (1,048,575) reads
    blocks 16,382 and 16,383 from offset 0; position 0 is one slot and length 1; position 127 fills the window from block 0;
    a speculative decode's second of four query tokens is two before its request's last (297); an invalid token is -1 and 0."""
    fn = bind(X.Dsv4SwaIndices, **SWA_SERVED)
    SW, BS, NB = SWA_SERVED["SW"], SWA_SERVED["BS"], SWA_SERVED["NB"]
    blocks = _w32(np.random.default_rng(2).integers(0, 1 << 20, NB))
    span = NB * BS
    for (t, qs, qe, seq, valid), n in [((7, 7, 8, span, 1), SW), ((0, 0, 1, 1, 1), 1), ((3, 3, 4, 128, 1), 128),
                                       ((11, 10, 14, 300, 1), 128), ((5, 5, 6, 900, 0), 0)]:
        out = [int(v) for v in evaluate(fn, t, qs, qe, seq, blocks, valid)]
        assert out == swa_kernel(t, qs, qe, seq, blocks, valid, SW, BS)
        assert out[-1] == n
    last = swa_kernel(7, 7, 8, span, blocks, 1, SW, BS)
    assert last[SW - 1] == (int(blocks[NB - 1]) * BS + BS - 1) & M32
    assert last[0] == (int(blocks[NB - 2]) * BS + 0) & M32


def test_a_window_block_number_whose_product_wraps():
    """`block_numbers * block_size` is int32: a block number of 2^26 + 3 at block size 64 wraps to 192, and 2^31 - 1 to -64."""
    st = dict(SW=8, BS=64, NB=4)
    blocks = _w32([(1 << 26) + 3, I32_MAX, -5, 9])
    for pos in (7, 63, 64, 70, 130):
        args = (pos, pos, pos + 1, pos + 1, blocks, 1)
        assert [int(v) for v in evaluate(bind(X.Dsv4SwaIndices, **st), *args)] == swa_kernel(*args, 8, 64)
    assert swa_kernel(7, 7, 8, 8, blocks, 1, 8, 64)[0] == 192


def test_the_window_row_where_the_kernel_reads_another_row():
    """A position past the row (the window's blocks past NB) reads the next request's row in the kernel: the lookup selects -1, so
    such a lane's slot is -BS + p % BS, and the twin declines the token."""
    st = SWA_SMALL
    SW, BS, NB = st["SW"], st["BS"], st["NB"]
    blocks = _w32(range(10, 10 + NB))
    pos = NB * BS + 1
    args = (0, 0, 1, pos + 1, blocks, 1)
    with pytest.raises(Outside):
        swa_kernel(*args, SW, BS)
    out = [int(v) for v in evaluate(bind(X.Dsv4SwaIndices, **st), *args)]
    start = pos - (SW - 1)
    want = [(int(blocks[(start + j) // BS]) * BS if (start + j) // BS < NB else -BS) + (start + j) % BS for j in range(SW)]
    assert out == [v & M32 for v in want] + [SW]
    _, _, declined = T.swa_indices(*([np.asarray([v]) for v in args[:4]]), blocks[None], np.asarray([1], np.uint8), **st)
    assert declined.tolist() == [True]


def test_the_window_pieces():
    """The window gives (start % BS, the stored length, the K blocks from start / BS times BS); the lanes select among them."""
    st = SWA_SMALL
    blocks = _w32(range(100, 108))
    r0, length, *base = (int(v) for v in evaluate(bind(X.Dsv4SwaWindow, **st), 20, 20, 21, 21, blocks, 1))
    start = 20 - 7
    assert (r0, length, base) == (start % 4, 8, [(100 + start // 4 + i) * 4 for i in range(3)])
    assert [int(v) for v in evaluate(bind(X.Dsv4SwaWindow, **st), 20, 20, 21, 21, blocks, 0)][1] == 0
    lanes = [int(v) for v in evaluate(bind(X.Dsv4SwaLanes, SW=8, BS=4), r0, 5, _w32(base))]
    assert lanes == [(100 + (start + j) // 4) * 4 + (start + j) % 4 for j in range(5)] + [M32] * 3


# --- the ratio-128 row against the kernel -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("st", [C_SMALL, dict(W=7, R=3, BS=1, NB=7), dict(W=128, R=128, BS=2, NB=64), dict(W=4, R=1, BS=4, NB=1)],
                         ids=["small", "odd", "served-block", "one-block"])
def test_the_c128a_row_is_the_kernels_stores_on_edge_tokens(st):
    args = _c128a_tokens(st, 3)
    assert np.array_equal(_reference(bind(X.Dsv4C128aGlobalTopk, **st), args), _c128a_want(st, args))


def test_the_c128a_row_at_the_served_statics():
    """Width 8192 (the widest `active_topk_width` at max_model_len 1,048,576), ratio 128, block 2, the compressed group's row of
    4096 blocks: the last position (1,048,575) makes every lane valid and its last lane reads block 4095; position 4095 makes 32
    valid; position 126 none; a padding token (slot_mapping -1) keeps its row and stores length 0."""
    fn = bind(X.Dsv4C128aGlobalTopk, **C128_SERVED)
    W, R, BS, NB = (C128_SERVED[k] for k in ("W", "R", "BS", "NB"))
    blocks = _w32(np.random.default_rng(4).integers(0, 1 << 20, NB))
    for position, slot, n in [((1 << 20) - 1, 7, 8192), (4095, 0, 32), (126, 3, 0), (4095, -1, 0)]:
        out = [int(v) for v in evaluate(fn, position & M64, slot & M64, blocks)]
        assert out == c128a_kernel(position, slot, blocks, W, R, BS)
        assert out[-1] == n
    assert c128a_kernel((1 << 20) - 1, 0, blocks, W, R, BS)[W - 1] == (int(blocks[NB - 1]) * BS + 1) & M32


def test_the_c128a_row_at_the_int64_ends():
    """position + 1 is int64: INT64_MAX wraps to INT64_MIN, whose quotient is negative, so no lane is valid; a negative t rounds
    toward 0, so position -R gives 0 too; the length reads the slot's sign only."""
    st = C_SMALL
    blocks = _w32(range(3, 11))
    for position in (I64_MAX, I64_MIN, -1, -2, -st["R"], -st["R"] - 1):
        out = [int(v) for v in evaluate(bind(X.Dsv4C128aGlobalTopk, **st), position & M64, 0, blocks)]
        assert out == [M32] * st["W"] + [0], position
    for slot, n in [(0, 16), (I64_MAX, 16), (-1, 0), (I64_MIN, 0)]:
        assert int(evaluate(bind(X.Dsv4C128aGlobalTopk, **st), 1000, slot & M64, blocks)[-1]) == n


def test_a_c128a_block_number_whose_product_wraps():
    st = dict(W=4, R=1, BS=2, NB=2)
    blocks = _w32([I32_MIN + 5, I32_MIN])
    out = [int(v) for v in evaluate(bind(X.Dsv4C128aGlobalTopk, **st), 3, 0, blocks)]
    assert out == c128a_kernel(3, 0, blocks, 4, 1, 2)
    assert out[:4] == [10, 11, 0, 1]


def test_c128a_lanes_past_the_row():
    """Lanes j / BS >= NB: never valid on a served token (the kernel would read the next request's row; the twin declines a token
    where one is), -1 while not valid, and x % BS - BS where valid (the block read as -1, `GatherI32x{NB}_v1`'s fill)."""
    st = C_PAST
    W, R, BS, NB = (st[k] for k in ("W", "R", "BS", "NB"))
    blocks = _w32([20, 30])
    fn = bind(X.Dsv4C128aGlobalTopk, **st)
    inside = [int(v) for v in evaluate(fn, NB * BS * R - 1, 0, blocks)]
    assert inside == c128a_kernel(NB * BS * R - 1, 0, blocks, W, R, BS)
    assert inside[NB * BS:W] == [M32] * (W - NB * BS)
    past = [int(v) for v in evaluate(fn, W * R - 1, 0, blocks)]
    with pytest.raises(Outside):
        c128a_kernel(W * R - 1, 0, blocks, W, R, BS)
    assert past[:NB * BS] == inside[:NB * BS] and past[NB * BS:W] == [(j % BS - BS) & M32 for j in range(NB * BS, W)]
    assert past[-1] == W
    _, _, declined = T.c128a_global_topk(_w64([W * R - 1]), _w64([0]), blocks[None], **st)
    assert declined.tolist() == [True]


def test_the_c128a_pieces():
    """The slot gives each lane of its block (x >= 0 ? bn * BS + x % BS : -1, x >= 0); the past slot reads bn as -1; the length
    counts the lives it is given where the slot's int64 sign is clear."""
    slot = bind(X.Dsv4C128aSlot, BS=4)
    out = [int(v) for v in evaluate(slot, _w32([8, 9, -1, I32_MIN]), 7)]
    assert out == [28, 29, M32, M32, 1, 1, 0, 0]
    past = bind(X.Dsv4C128aPastSlot, BS=4)
    assert [int(v) for v in evaluate(past, 9)] == [(1 - 4) & M32, 1]
    assert [int(v) for v in evaluate(past, -1 & M32)] == [M32, 0]
    length = bind(X.Dsv4C128aLen, W=5)
    for live, slot_word, want in [([1, 0, 1, 1, 0], 0, 3), ([1] * 5, I64_MAX, 5), ([1] * 5, -1 & M64, 0)]:
        assert [int(v) for v in evaluate(length, live, slot_word)] == [want]


# --- the twin --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", TWIN_TARGETS, ids=[t.id for t in TWIN_TARGETS])
def test_the_twin_is_the_reference(target):
    sc = self_check(target, kernel=T.KERNEL, n=32, seed=0)
    assert sc.ok, str(sc)
    assert 0 < sc.n - sc.declined, str(sc)


@pytest.mark.parametrize("st", [SWA_SMALL, SWA_ODD], ids=["small", "odd"])
def test_the_window_twin_is_the_kernel_on_edge_tokens(st):
    args = _swa_tokens(st, 5)
    rows, lens, declined = T.swa_indices(*args, **st)
    assert not declined.any()
    assert np.array_equal(np.concatenate([rows, lens[:, None]], axis=1).astype(np.uint64), _swa_want(st, args))


@pytest.mark.parametrize("st", [C_SMALL, dict(W=7, R=3, BS=1, NB=7)], ids=["small", "odd"])
def test_the_c128a_twin_is_the_kernel_on_edge_tokens(st):
    args = _c128a_tokens(st, 6)
    rows, lens, declined = T.c128a_global_topk(*args, **st)
    assert not declined.any()
    assert np.array_equal(np.concatenate([rows, lens[:, None]], axis=1).astype(np.uint64), _c128a_want(st, args))


# --- the statics -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fn,kw,name", [
    (X.Dsv4SwaWindow, dict(SW=0, BS=4, NB=4), "Dsv4SwaWindow"),
    (X.Dsv4SwaWindow, dict(SW=8, BS=3, NB=4), "Dsv4SwaWindow"),
    (X.Dsv4SwaWindow, dict(SW=8, BS=1 << 30, NB=1), "Dsv4SwaWindow"),
    (X.Dsv4SwaWindow, dict(SW=8, BS=64, NB=0), "Dsv4SwaWindow"),
    (X.Dsv4SwaWindow, dict(SW=8, BS=64, NB=(1 << 25) + 1), "Dsv4SwaWindow"),
    (X.Dsv4SwaLanes, dict(SW=1 << 30, BS=4), "Dsv4SwaLanes"),
    (X.Dsv4SwaIndices, dict(SW=8, BS=6, NB=4), "Dsv4SwaWindow"),
    (X.Dsv4C128aSlot, dict(BS=3), "Dsv4C128aSlot"),
    (X.Dsv4C128aPastSlot, dict(BS=1 << 31), "Dsv4C128aSlot"),
    (X.Dsv4C128aLen, dict(W=0), "Dsv4C128aLen"),
    (X.Dsv4C128aGlobalTopk, dict(W=7, R=3, BS=2, NB=4), "Dsv4C128aGlobalTopk"),
    (X.Dsv4C128aGlobalTopk, dict(W=8, R=3, BS=2, NB=0), "Dsv4C128aGlobalTopk"),
    (X.Dsv4C128aGlobalTopk, dict(W=1 << 20, R=1 << 12, BS=2, NB=4), "Dsv4ShortContextTopK"),
    (BX.Dsv4SwaWindowV2, dict(SW=8, BS=5, NB=4), "Dsv4SwaWindow"),
    (BX.Dsv4C128aSlotV2, dict(BS=6), "Dsv4C128aSlot"),
    (BX.Dsv4C128aLenV2, dict(W=0), "Dsv4C128aLen"),
], ids=["sw-0", "bs-3", "bs-2^30", "nb-0", "nb-past-int32", "lanes-sw-2^30", "row-bs-6", "slot-bs-3", "past-bs-2^31", "len-0",
        "w-not-a-multiple", "c-nb-0", "w-r-past-2^31", "v2-window-bs-5", "v2-slot-bs-6", "v2-len-0"])
def test_bad_statics_are_refused_by_name(fn, kw, name):
    with pytest.raises(ValueError, match=name):
        bind(fn, **kw).body


# --- the Boolean Definitions ---------------------------------------------------------------------------------------------------------

def test_each_v2s_word_and_the_switch():
    """Each `_v2` names its `_v1` as its word; the switch takes each to its `_v2` but the length, whose lives are 1-bit elements
    (not the switch's bit form of Array<W, bit>, one W-bit word), so it is reached through its row."""
    pairs = [(bind(X.Dsv4SwaWindow, **SWA_SMALL), bind(BX.Dsv4SwaWindowV2, **SWA_SMALL)),
             (bind(X.Dsv4SwaLanes, SW=8, BS=4), bind(BX.Dsv4SwaLanesV2, SW=8, BS=4)),
             (bind(X.Dsv4SwaIndices, **SWA_SMALL), bind(BX.Dsv4SwaIndicesV2, **SWA_SMALL)),
             (bind(X.Dsv4C128aSlot, BS=2), bind(BX.Dsv4C128aSlotV2, BS=2)),
             (bind(X.Dsv4C128aPastSlot, BS=2), bind(BX.Dsv4C128aPastSlotV2, BS=2)),
             (bind(X.Dsv4C128aLen, W=8), bind(BX.Dsv4C128aLenV2, W=8)),
             (bind(X.Dsv4C128aGlobalTopk, **C_SMALL), bind(BX.Dsv4C128aGlobalTopkV2, **C_SMALL))]
    for word, bits in pairs:
        assert word_of(bits).id == word.id
    for word, bits in [*pairs[:5], pairs[6],
                       (bind(X.Dsv4SwaIndices, **SWA_SERVED), bind(BX.Dsv4SwaIndicesV2, **SWA_SERVED)),
                       (bind(X.Dsv4C128aGlobalTopk, **C128_SERVED), bind(BX.Dsv4C128aGlobalTopkV2, **C128_SERVED))]:
        assert SW_.boolean_version(word).id == bits.id


@pytest.mark.parametrize("st", [SWA_SMALL, SWA_ODD, dict(SW=3, BS=8, NB=4), dict(SW=1, BS=1, NB=2)],
                         ids=["small", "odd", "narrow", "one-lane"])
def test_the_boolean_window_row_is_its_word(st):
    target = bind(BX.Dsv4SwaIndicesV2, **st)
    for args in (_swa_tokens(st, 7), T._swa_sample(np.random.default_rng(8), 64, None, **st)):
        n = len(args[0])
        got = np.asarray(evaluate_words(target, *args), dtype=np.uint64).reshape(n, -1)
        assert np.array_equal(got, _reference(bind(X.Dsv4SwaIndices, **st), args))


def test_the_boolean_window_row_at_4096s_statics_is_its_word():
    args = T._swa_sample(np.random.default_rng(9), 48, None, **SWA_4096)
    got = np.asarray(evaluate_words(bind(BX.Dsv4SwaIndicesV2, **SWA_4096), *args), dtype=np.uint64).reshape(48, -1)
    assert np.array_equal(got, _reference(bind(X.Dsv4SwaIndices, **SWA_4096), args))


@pytest.mark.parametrize("st", [C_SMALL, C_PAST, dict(W=7, R=3, BS=1, NB=5), dict(W=16, R=4, BS=2, NB=3), C128_4096],
                         ids=["small", "past", "odd", "short-row", "4096"])
def test_the_boolean_c128a_row_is_its_word(st):
    target = bind(BX.Dsv4C128aGlobalTopkV2, **st)
    for args in (_c128a_tokens(st, 10), T._c128a_sample(np.random.default_rng(11), 64, None, **st)):
        n = len(args[0])
        got = np.asarray(evaluate_words(target, *args), dtype=np.uint64).reshape(n, -1)
        assert np.array_equal(got, _reference(bind(X.Dsv4C128aGlobalTopk, **st), args))


def _slot_words(BS: int) -> list[int]:
    rng = np.random.default_rng(BS)
    return [0, 1, BS - 1, BS, I32_MAX, -1, -BS, I32_MIN, *rng.integers(I32_MIN, I32_MAX, 120)]


@pytest.mark.parametrize("BS", [1, 2, 4, 64])
def test_the_boolean_slot_is_its_word(BS):
    """Over edge and random words, a block of one lane to 64, each lane's live its own 1-bit word."""
    xs = _slot_words(BS)
    x = _w32(xs + [0] * (-len(xs) % BS)).reshape(-1, BS)
    bn = _w32(np.random.default_rng(BS + 1).integers(I32_MIN, I32_MAX, len(x)))
    got = np.asarray(evaluate_words(bind(BX.Dsv4C128aSlotV2, BS=BS), x, bn), dtype=np.uint64).reshape(len(x), -1)
    assert got.shape[1] == 2 * BS
    assert np.array_equal(got, _reference(bind(X.Dsv4C128aSlot, BS=BS), [x, bn]))


@pytest.mark.parametrize("BS", [1, 2, 4, 64, 1 << 30])
def test_the_boolean_past_slot_is_its_word(BS):
    """Over edge and random words, to log2 BS = 30, where only the sign lies above x's low bits."""
    x = _w32(_slot_words(BS))
    got = np.asarray(evaluate_words(bind(BX.Dsv4C128aPastSlotV2, BS=BS), x), dtype=np.uint64).reshape(len(x), -1)
    assert np.array_equal(got, _reference(bind(X.Dsv4C128aPastSlot, BS=BS), [x]))


def test_the_boolean_length_is_its_word():
    for W in (1, 5, 8, 128):
        word, bits = bind(X.Dsv4C128aLen, W=W), bind(BX.Dsv4C128aLenV2, W=W)
        rng = np.random.default_rng(W)
        live = (rng.random((40, W)) < rng.random((40, 1))).astype(np.uint8)
        live[0], live[1] = 1, 0
        slot = _w64([0, 0, -1, I64_MIN, I64_MAX, *rng.integers(-(1 << 40), 1 << 40, 35)])
        got = np.asarray(evaluate_words(bits, live, slot), dtype=np.uint64).reshape(-1)
        assert np.array_equal(got, _reference(word, [live, slot]).reshape(-1))
        assert int(got[0]) == W and int(got[1]) == 0
