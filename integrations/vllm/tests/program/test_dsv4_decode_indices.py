"""`registry.dsv4_decode_indices` (`compute_global_topk_indices_and_lens`, vLLM d9105ea80): the Definition against a model of the
Triton kernel's stores (empty, short and full rows, the padding token's byte, the block edges, the last block of a row at
max_model_len, the int32 ends, a block number whose product wraps), the extension past the row, the twin against the reference,
the statics, and each Boolean Definition against its word."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import numpy as np
import pytest

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, self_check
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.numerics import dsv4_decode_indices_twin as T
from verity.ml.definitions.registry import boolean_dsv4_decode_indices as BX
from verity.ml.definitions.registry import dsv4_decode_indices as X

U = np.uint32
M32 = 0xFFFFFFFF
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
DEFS = {d.id: d for d in X.DSV4_DECODE_INDICES_COMPOSITES}
TWIN_TARGETS = T._targets(DEFS)
SERVED = X.DSV4_DECODE_INDICES["c4"]
SMALL = dict(TOPK=8, BS=4, NB=8)
ODD = dict(TOPK=5, BS=2, NB=3)


def _s32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def kernel(topk, blocks, valid, BS: int) -> list[int]:
    """`_compute_global_topk_indices_and_lens_kernel` (common/ops/cache_utils.py:469-517) for one token, read line by line: the
    int32 `//` and `%` truncate (C's), the block number is loaded only where the index is valid, `slot_ids = block_numbers *
    block_size + block_offsets` in int32, -1 where the index is negative, and the count stored where `is_valid_token` (a bool
    loaded from its byte) holds; the row's slots, then the length, as uint32 words."""
    count, out = 0, []
    for w in topk:
        local = _s32(int(w))
        ok = local >= 0
        b = local // BS if local >= 0 else -(-local // BS)
        number = _s32(int(blocks[b])) if ok else 0
        offset = local - b * BS
        out.append(_s32(number * BS + offset) if ok else -1)
        count += ok
    return [v & M32 for v in out] + [count if int(valid) != 0 else 0]


def _words(xs) -> np.ndarray:
    return (np.asarray(xs, dtype=np.int64) & M32).astype(U)


def _reference(st: dict, args) -> np.ndarray:
    return np.asarray(evaluate_batch(bind(X.Dsv4GlobalTopk, **st), *args, kernel="reference"), dtype=np.uint64)


def _rows(st: dict, seed: int) -> list[np.ndarray]:
    """Edge rows at statics ``st``: empty (every index -1), short (a prefix of indices, then -1, as the indexer pads), full, a row
    at the row's first and last slots and every block edge, the negative ends; each with a valid byte of 1, 0 and others."""
    TOPK, BS, NB = st["TOPK"], st["BS"], st["NB"]
    rng = np.random.default_rng(seed)
    last = NB * BS - 1
    edges = [e for e in [0, 1, BS - 1, BS, last - BS, last - 1, last] if 0 <= e <= last]
    edges += [-1, -2, -BS, -BS - 1, I32_MIN, I32_MIN + 1]
    rows = [[-1] * TOPK]
    for n in sorted({1, TOPK // 2, TOPK - 1, TOPK}):
        rows.append([*rng.integers(0, last + 1, n), *[-1] * (TOPK - n)])
    rows.append([edges[j % len(edges)] for j in range(TOPK)])
    rows.append([edges[(j + 7) % len(edges)] for j in range(TOPK)])
    rows.append([last - j for j in range(TOPK)])
    bytes_ = [1, 0, 2, 0x80, 0xFF, 1, 0]
    topk, blocks, valid = [], [], []
    for i, r in enumerate(rows):
        for byte in (bytes_[i % len(bytes_)], 0 if bytes_[i % len(bytes_)] else 1):
            topk.append(r)
            blocks.append(np.where(rng.random(NB) < 0.5, rng.integers(0, 1 << 17, NB), rng.integers(0, 1 << 32, NB)))
            valid.append(byte)
    return [_words(topk), _words(blocks), np.asarray(valid, dtype=np.uint8)]


def _want(st: dict, args) -> np.ndarray:
    return np.asarray([kernel(t, b, v, st["BS"]) for t, b, v in zip(*args)], dtype=np.uint64)


# --- the Definition against the kernel ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("st", [SMALL, ODD, dict(TOPK=16, BS=64, NB=16), dict(TOPK=3, BS=1, NB=1)],
                         ids=["small", "odd", "served-block", "one-block"])
def test_the_definition_is_the_kernels_stores_on_edge_rows(st):
    args = _rows(st, 1)
    assert np.array_equal(_reference(st, args), _want(st, args))


def test_the_row_and_its_length_at_the_served_statics():
    """Top-k 1024, block size 64, the row of 4096 blocks at max_model_len 1,048,576: a full row whose indices reach the last block
    (b = 4095, index 262,143 = 1,048,576 / 4 - 1) is 1024 slots and length 1024; the indexer's short row of 37 indices then -1 is
    37 slots, then -1, and length 37; the padding token (byte 0) stores the same slots and length 0; an empty row is all -1."""
    TOPK, BS, NB = SERVED["TOPK"], SERVED["BS"], SERVED["NB"]
    fn = bind(X.Dsv4GlobalTopk, **SERVED)
    rng = np.random.default_rng(2)
    blocks = _words(rng.integers(0, 1 << 20, NB))
    full = _words([NB * BS - 1 - 255 * j for j in range(TOPK)])
    short = _words([*rng.integers(0, NB * BS, 37), *[-1] * (TOPK - 37)])
    for topk, valid, n in [(full, 1, 1024), (short, 1, 37), (short, 0, 0), (_words([-1] * TOPK), 1, 0)]:
        out = [int(v) for v in evaluate(fn, topk, blocks, valid)]
        assert out == kernel(topk, blocks, valid, BS)
        assert out[-1] == n
    assert kernel(full, blocks, 1, BS)[0] == (int(blocks[NB - 1]) * BS + BS - 1) & M32


def test_a_block_number_whose_product_wraps():
    """`block_numbers * block_size` is int32: a block number of 2^26 + 3 at block size 64 wraps to 192, and one of 2^31 - 1 to
    -64; the slot adds the offset after the wrap."""
    st = SMALL | {"BS": 64}
    blocks = _words([(1 << 26) + 3, I32_MAX, -5, 0, 1, 2, 3, 4])
    topk = _words([0, 63, 64, 127, 128, -1, 64 + 5, 7])
    out = [int(v) for v in evaluate(bind(X.Dsv4GlobalTopk, **st), topk, blocks, 1)]
    assert out == kernel(topk, blocks, 1, 64)
    assert out[:3] == [192, 192 + 63, (-64) & M32] and out[3] == (-64 + 63) & M32 and out[4] == (-5 * 64) & M32


def test_an_index_past_the_row_selects_the_fill():
    """b >= NB, where the kernel reads the next request's row: the lookup selects -1, so the slot is -BS + x % BS, and the twin
    declines the token."""
    st = SMALL
    BS, NB = st["BS"], st["NB"]
    topk = _words([NB * BS, NB * BS + 3, I32_MAX, 0, -1, -1, -1, -1])
    blocks = _words(range(10, 10 + NB))
    out = [int(v) for v in evaluate(bind(X.Dsv4GlobalTopk, **st), topk, blocks, 1)]
    assert out[:3] == [(-BS) & M32, (-BS + 3) & M32, (-BS + I32_MAX % BS) & M32]
    assert out[3] == 10 * BS and out[4:8] == [M32] * 4 and out[8] == 4
    _, _, declined = T.global_topk(topk[None], blocks[None], np.asarray([1], dtype=np.uint8), **st)
    assert declined.tolist() == [True]


def test_the_slot_and_the_length_pieces():
    """The slot gives (its slot, x >= 0) and the length counts the lives it is given, gated by the byte's 8 bits."""
    slot = bind(X.Dsv4GlobalTopkSlot, BS=4, NB=3)
    blocks = _words([5, 6, 7])
    for x, want in [(0, (20, 1)), (11, (31, 1)), (-1, (M32, 0)), (I32_MIN, (M32, 0)), (12, ((-4) & M32, 1))]:
        assert tuple(int(v) for v in evaluate(slot, x & M32, blocks)) == want, x
    length = bind(X.Dsv4GlobalTopkLen, TOPK=5)
    for live, byte, want in [([1, 0, 1, 1, 0], 1, 3), ([1] * 5, 0x40, 5), ([1] * 5, 0, 0), ([0] * 5, 0xFF, 0)]:
        assert [int(v) for v in evaluate(length, live, byte)] == [want]


# --- the twin --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", TWIN_TARGETS, ids=[t.id for t in TWIN_TARGETS])
def test_the_twin_is_the_reference(target):
    sc = self_check(target, kernel=T.KERNEL, n=32, seed=0)
    assert sc.ok, str(sc)
    assert 0 < sc.n - sc.declined, str(sc)


@pytest.mark.parametrize("st", [SMALL, ODD], ids=["small", "odd"])
def test_the_twin_is_the_kernel_on_edge_rows(st):
    args = _rows(st, 3)
    rows, lens, declined = T.global_topk(*args, **st)
    assert not declined.any()
    assert np.array_equal(np.concatenate([rows, lens[:, None]], axis=1).astype(np.uint64), _want(st, args))


# --- the statics -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fn,kw,name", [
    (X.Dsv4GlobalTopkSlot, dict(BS=3, NB=4), "Dsv4GlobalTopkSlot"),
    (X.Dsv4GlobalTopkSlot, dict(BS=0, NB=4), "Dsv4GlobalTopkSlot"),
    (X.Dsv4GlobalTopkSlot, dict(BS=1 << 31, NB=1), "Dsv4GlobalTopkSlot"),
    (X.Dsv4GlobalTopkSlot, dict(BS=64, NB=0), "Dsv4GlobalTopkSlot"),
    (X.Dsv4GlobalTopkSlot, dict(BS=64, NB=(1 << 25) + 1), "Dsv4GlobalTopkSlot"),
    (X.Dsv4GlobalTopkLen, dict(TOPK=0), "Dsv4GlobalTopkLen"),
    (BX.Dsv4GlobalTopkSlotV2, dict(BS=6, NB=4), "Dsv4GlobalTopkSlot"),
    (BX.Dsv4GlobalTopkLenV2, dict(TOPK=0), "Dsv4GlobalTopkLen"),
], ids=["bs-3", "bs-0", "bs-2^31", "nb-0", "nb-past-int32", "topk-0", "v2-bs-6", "v2-topk-0"])
def test_bad_statics_are_refused_by_name(fn, kw, name):
    with pytest.raises(ValueError, match=name):
        bind(fn, **kw).body


# --- the Boolean Definitions ---------------------------------------------------------------------------------------------------------

def test_each_v2s_word_and_the_switch():
    """Each `_v2` names its `_v1` as its word; the switch takes the row and the slot to theirs.  The length's lives are 1-bit
    elements, which the switch's bit form of Array<TOPK, bit> (one TOPK-bit word) is not, so it is reached through its row."""
    for word, bits in [(bind(X.Dsv4GlobalTopkSlot, BS=4, NB=8), bind(BX.Dsv4GlobalTopkSlotV2, BS=4, NB=8)),
                       (bind(X.Dsv4GlobalTopkLen, TOPK=8), bind(BX.Dsv4GlobalTopkLenV2, TOPK=8)),
                       (bind(X.Dsv4GlobalTopk, **SMALL), bind(BX.Dsv4GlobalTopkV2, **SMALL))]:
        assert word_of(bits).id == word.id
    for word, bits in [(bind(X.Dsv4GlobalTopkSlot, BS=4, NB=8), bind(BX.Dsv4GlobalTopkSlotV2, BS=4, NB=8)),
                       (bind(X.Dsv4GlobalTopk, **SMALL), bind(BX.Dsv4GlobalTopkV2, **SMALL)),
                       (bind(X.Dsv4GlobalTopk, **SERVED), bind(BX.Dsv4GlobalTopkV2, **SERVED))]:
        assert SW.boolean_version(word).id == bits.id


@pytest.mark.parametrize("st", [SMALL, ODD, dict(TOPK=3, BS=1, NB=1)], ids=["small", "odd", "one-block"])
def test_the_boolean_row_is_its_word(st):
    target = bind(BX.Dsv4GlobalTopkV2, **st)
    for args in (_rows(st, 4), T._sample(np.random.default_rng(5), 64, None, **st)):
        n = len(args[2])
        got = np.asarray(evaluate_words(target, *args), dtype=np.uint64).reshape(n, -1)
        assert np.array_equal(got, _reference(st, args).reshape(n, -1))


@pytest.mark.parametrize("BS,NB", [(4, 8), (2, 3), (1, 1), (64, 16), (64, 5), (1 << 30, 2), (1 << 28, 8)],
                         ids=["4x8", "2x3", "1x1", "served-block", "64x5", "2^30x2", "2^28x8"])
def test_the_boolean_slot_is_its_word(BS, NB):
    """Over the edge indices and random words: one block (no index bit), a block count that is not a power of two, and index bits
    that run up to bit 30 (2^30 x 2, 2^28 x 8), where only the sign lies above them."""
    word, bits = bind(X.Dsv4GlobalTopkSlot, BS=BS, NB=NB), bind(BX.Dsv4GlobalTopkSlotV2, BS=BS, NB=NB)
    rng = np.random.default_rng(BS * 131 + NB)
    last = NB * BS - 1
    xs = [0, 1, BS - 1, BS, last, last + 1, (last + 1) * 2, I32_MAX, -1, -BS, I32_MIN, *rng.integers(I32_MIN, I32_MAX, 200)]
    x = _words(xs)
    blocks = _words(rng.integers(0, 1 << 32, (len(xs), NB)))
    got = np.asarray(evaluate_words(bits, x, blocks), dtype=np.uint64).reshape(len(xs), -1)
    want = np.asarray(evaluate_batch(word, x, blocks, kernel="reference"), dtype=np.uint64).reshape(len(xs), -1)
    assert np.array_equal(got, want)


def test_the_boolean_length_is_its_word():
    for TOPK in (1, 5, 8, 1024):
        word, bits = bind(X.Dsv4GlobalTopkLen, TOPK=TOPK), bind(BX.Dsv4GlobalTopkLenV2, TOPK=TOPK)
        rng = np.random.default_rng(TOPK)
        live = (rng.random((40, TOPK)) < rng.random((40, 1))).astype(np.uint8)
        live[0], live[1] = 1, 0
        valid = rng.integers(0, 256, 40).astype(np.uint8)
        valid[:4] = [1, 0, 0x80, 0xFF]
        got = np.asarray(evaluate_words(bits, live, valid), dtype=np.uint64).reshape(-1)
        want = np.asarray(evaluate_batch(word, live, valid, kernel="reference"), dtype=np.uint64).reshape(-1)
        assert np.array_equal(got, want)
        assert int(got[0]) == TOPK and int(got[1]) == 0
