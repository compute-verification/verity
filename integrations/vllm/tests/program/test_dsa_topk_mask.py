"""`registry.dsa_topk_mask` (the DSA top-k of a row past TOPK keys in the mask form, no advice) and its Boolean versions
`boolean_dsa_topk_mask`: the mask is `topk_set` under the tie rule, the counted mask that of a row's first n keys; the concentrator at the mask's settings lists the members'
rows in index order, so the long row is `sorted(topk_set)`; each Boolean version agrees with its word view, on rows that tie at
the boundary, straddle signed zero and hold NaNs and infinities, and on masks of any popcount and random settings."""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.codec import lookup_primitive
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import boolean_dsa_topk_mask as BM
from verity.ml.definitions.registry import dsa_topk as T
from verity.ml.definitions.registry import dsa_topk_mask as M

POOL = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC00000, 0xFFC00000, 0x7F800000, 0xFF800000, 0x00000001, 0x80000001]


def _scores(rng, N: int) -> list[int]:
    """Words with repeats, specials and random bits, so rows tie, straddle signed zero and hold NaNs."""
    return [rng.choice(POOL) if rng.random() < 0.5 else rng.getrandbits(32) for _ in range(N)]


def _rows(rng, N: int, n: int) -> list[list[int]]:
    """Random rows, rows of a few values (ties at the boundary), and constant ones (every key tied)."""
    few = [[rng.choice(POOL[:3]) for _ in range(N)] for _ in range(n // 4)]
    return [_scores(rng, N) for _ in range(n)] + few + [[w] * N for w in (POOL[0], POOL[1], POOL[4], POOL[7])]


def _bits(words, width: int) -> np.ndarray:
    """Words as `evaluate_bits` takes them: each split into its ``width`` bits, LSB first, on a last axis."""
    w = np.asarray(words, dtype=object)
    return np.stack([(w >> j) & 1 for j in range(width)], axis=-1).astype(np.uint8)


def _words(bits: np.ndarray) -> np.ndarray:
    """`evaluate_bits`' output bits back into words along the last axis (LSB first)."""
    return sum(bits[..., j].astype(object) << j for j in range(bits.shape[-1]))


#: (N, TOPK): equal, one, a power of two and one past it, GLM-5.3's shape scaled down, b = 1
SHAPES = [(6, 4), (6, 6), (5, 1), (8, 5), (9, 8), (17, 16), (2, 1), (33, 8)]


@pytest.mark.parametrize("N,TOPK", SHAPES)
def test_the_mask_is_the_top_set_and_its_boolean_version_agrees(N, TOPK) -> None:
    target = bind(BM.DsaTopKMaskV2, N=N, TOPK=TOPK)
    word = TR.word_of(target)
    rows = _rows(random.Random(N * 100 + TOPK), N, 24)
    ref = np.asarray(E.evaluate_batch(word, rows)).reshape(len(rows), -1)
    assert ref.tolist() == [[int(j in T.topk_set(r, TOPK)) for j in range(N)] for r in rows]
    got = np.asarray(BS.evaluate_bits(target, _bits(rows, 32))).reshape(len(rows), -1)
    assert np.array_equal(got, ref), target.id


def _counted(row: list[int], n: int, TOPK: int) -> list[int]:
    top = T.topk_set(row[:n], TOPK)
    return [int(j in top) for j in range(len(row))]


@pytest.mark.parametrize("N,TOPK", [(6, 4), (6, 6), (5, 1), (9, 8), (17, 16), (20, 16), (33, 8), (2, 1), (1, 1)])
def test_the_counted_mask_is_the_top_set_of_the_first_n_keys_and_its_boolean_version_agrees(N, TOPK) -> None:
    """`DsaTopKMaskCounted` at every n from 0 to N + 1 and at u32's top bit and top: every key below n while n <= TOPK, then the
    top set of the first n (all N past N), never a key from n on; `DsaTopKMaskCounted_v2` agrees, so `DsaRowValid_v1`'s trie is
    n > j at every n and N, a power of two or not."""
    target = bind(BM.DsaTopKMaskCountedV2, N=N, TOPK=TOPK)
    word = TR.word_of(target)
    ns = list(range(N + 2)) + [1 << 31, (1 << 32) - 1]
    cases = [(r, n) for r in _rows(random.Random(N * 100 + TOPK), N, 4) for n in ns]
    rows, counts = [r for r, _ in cases], [n for _, n in cases]
    ref = np.asarray(E.evaluate_batch(word, rows, counts)).reshape(len(cases), -1)
    assert ref.tolist() == [_counted(r, n, TOPK) for r, n in cases]
    got = np.asarray(BS.evaluate_bits(target, _bits(rows, 32), _bits(counts, 32))).reshape(len(cases), -1)
    assert np.array_equal(got, ref), target.id


def test_the_counted_mask_at_n_is_the_mask_of_the_first_n_keys() -> None:
    """At n = N it is `DsaTopKMask`'s bits, and at TOPK <= n < N those of `DsaTopKMask{n}` over the first n keys, the rest 0: one
    Definition over a bucket of N keys stands for the mask at each row length up to it."""
    N, TOPK = 12, 4
    rows = _rows(random.Random(5), N, 6)
    counted = TR.word_of(bind(BM.DsaTopKMaskCountedV2, N=N, TOPK=TOPK))
    for n in range(TOPK, N + 1):
        got = np.asarray(E.evaluate_batch(counted, rows, [n] * len(rows))).reshape(len(rows), -1)
        own = np.asarray(E.evaluate_batch(bind(M.DsaTopKMask, N=n, TOPK=TOPK), [r[:n] for r in rows])).reshape(len(rows), -1)
        assert np.array_equal(got[:, :n], own) and not got[:, n:].any(), n


def test_the_concentrator_lists_every_masks_members_in_index_order() -> None:
    """Every mask of K members over N <= 9 rows: at `butterfly_settings` the network puts the s-th member at output s (no two
    members contend for a switch's output), and random ones at GLM-5.3's row lengths past TOPK."""
    for N in range(2, 10):
        for K in range(1, N + 1):
            for members in itertools.combinations(range(N), K):
                mask = [int(j in members) for j in range(N)]
                assert M.butterfly_apply(M.butterfly_settings(mask, K), list(range(N)), K) == list(members), (N, members)
    rng = random.Random(0)
    for N, K in [(rng.randrange(10, 300), None) for _ in range(30)] + [(2049, 2048), (2080, 2048), (4096, 2048), (5000, 2048)]:
        K = K or rng.randrange(1, N + 1)
        members = sorted(rng.sample(range(N), K))
        mask = [int(j in set(members)) for j in range(N)]
        assert M.butterfly_apply(M.butterfly_settings(mask, K), list(range(N)), K) == members, (N, K)


def test_the_switch_count_is_the_pruned_butterflys() -> None:
    """K = 2048 at GLM-5.3's row lengths; unpruned (K = N = 2^L) it is the butterfly's L 2^(L-1)."""
    assert [M.butterfly_switches(N, 2048) for N in (2049, 4096, 32768, 131072)] == [15359, 24576, 210944, 849920]
    for L in range(1, 9):
        assert M.butterfly_switches(1 << L, 1 << L) == L << (L - 1)


#: (N, K): two rows, odd and even N with every pruning (K = N, N - 1, a pick at a K short of a power of two, one output)
NETS = [(2, 1), (2, 2), (3, 2), (3, 3), (5, 4), (6, 5), (7, 3), (8, 8), (9, 8), (17, 16), (4, 1), (33, 8)]


@pytest.mark.parametrize("N,K", NETS)
def test_the_settings_and_their_boolean_version_agree_on_any_mask(N, K) -> None:
    """`DsaButterflySettings` is `butterfly_settings` on masks of K members and of any popcount (whose members may contend), and
    its Boolean version agrees."""
    target = bind(BM.DsaButterflySettingsV2, N=N, K=K)
    word = TR.word_of(target)
    rng = random.Random(N * 10 + K)
    masks = [[int(j in s) for j in range(N)] for s in (set(rng.sample(range(N), K)) for _ in range(16))]
    masks += [[rng.getrandbits(1) for _ in range(N)] for _ in range(16)] + [[1] * N, [0] * N]
    ref = np.asarray(E.evaluate_batch(word, masks)).reshape(len(masks), -1)
    assert ref.tolist() == [M.butterfly_settings(m, K) for m in masks]
    got = np.asarray(BS.evaluate_bits(target, np.asarray(masks, dtype=np.uint8))).reshape(len(masks), -1)
    assert np.array_equal(got, ref), target.id


@pytest.mark.parametrize("N,K,W,WIDTH", [(2, 1, 2, 16), (3, 2, 2, 16), (6, 5, 2, 32), (7, 3, 1, 16), (9, 8, 3, 16), (17, 16, 1, 32)])
def test_the_rows_network_is_butterfly_apply_and_its_boolean_version_agrees(N, K, W, WIDTH) -> None:
    """At a mask's settings the outputs are the members' rows in index order; at any settings (random bits too) it is
    `butterfly_apply` with empty positions zero rows, and `DsaButterflyRows_v2` agrees."""
    target = bind(BM.DsaButterflyRowsV2, N=N, K=K, W=W, E=WIDTH)
    word = TR.word_of(target)
    rng = random.Random(N * 1000 + K * 10 + W)
    sws, rows, want = [], [], []
    for i in range(30):
        members = sorted(rng.sample(range(N), K))
        sw = (M.butterfly_settings([int(j in members) for j in range(N)], K) if i % 3
              else [rng.getrandbits(1) for _ in range(M.butterfly_switches(N, K))])
        r = [[rng.getrandbits(WIDTH) for _ in range(W)] for _ in range(N)]
        out = M.butterfly_apply(sw, r, K, dead=[0] * W)
        if i % 3:
            assert out == [r[j] for j in members]
        sws.append(sw)
        rows.append(r)
        want.append([w for row in out for w in row])
    ref = np.asarray(E.evaluate_batch(word, sws, rows)).reshape(len(sws), -1)
    assert ref.tolist() == want
    got = BS.evaluate_bits(target, np.asarray(sws, dtype=np.uint8), _bits(rows, WIDTH))
    assert np.array_equal(_words(np.asarray(got).reshape(len(sws), K * W, WIDTH)), ref), target.id


@pytest.mark.parametrize("N,TOPK", [(6, 4), (5, 1), (9, 8), (17, 16), (33, 8), (40, 32)])
def test_the_long_row_is_the_top_set_in_ascending_order(N, TOPK) -> None:
    """`DsaTopKRow_v2`, the mask, its settings and the concentrator over the indices, is `sorted(topk_set)` on rows with ties."""
    target = bind(BM.DsaTopKRowV2, N=N, TOPK=TOPK)
    word = TR.word_of(target)
    rows = _rows(random.Random(N + TOPK), N, 16)
    ref = np.asarray(E.evaluate_batch(word, rows)).reshape(len(rows), -1)
    assert ref.tolist() == [sorted(T.topk_set(r, TOPK)) for r in rows]
    got = np.asarray(BS.evaluate_bits(target, _bits(rows, 32))).reshape(len(rows), TOPK, 32)
    assert np.array_equal(_words(got), ref), target.id


@pytest.mark.parametrize("target", [bind(BM.DsaTopKMaskV2, N=9, TOPK=8), bind(BM.DsaTopKMaskCountedV2, N=9, TOPK=8),
                                    bind(BM.DsaButterflySettingsV2, N=9, K=8),
                                    bind(BM.DsaButterflyRowsV2, N=9, K=8, W=3, E=16), bind(BM.DsaTopKRowV2, N=9, TOPK=8),
                                    bind(BM.DsaTopKKeyV2)], ids=lambda t: t.id)
def test_the_boolean_version_is_boolean_and_its_word_view_is_v1(target) -> None:
    word = TR.word_of(target)
    defn, bindings = (word.defn, word.bindings) if hasattr(word, "defn") else (word, {})
    assert BL.is_boolean(target) and (defn.name, defn.version, bindings) == (target.defn.name, 1, target.bindings)
    assert SW.boolean_version(word).id == target.id


def test_the_accelerators_resolve_by_id_in_a_cold_decode() -> None:
    assert lookup_primitive(REGISTRY, "DsaTopKThresholdKey131072x2048_v1") is M.threshold(131072, 2048)[0]
    assert lookup_primitive(REGISTRY, "DsaTopKThresholdIndex131072x2048_v1") is M.threshold(131072, 2048)[1]
    assert lookup_primitive(REGISTRY, "DsaTopKSlot131072x2048_v1") is M.slot(131072, 2048)
    assert lookup_primitive(REGISTRY, "DsaTopKCountedThresholdKey131072x2048_v1") is M.counted_threshold(131072, 2048)[0]
    assert lookup_primitive(REGISTRY, "DsaTopKCountedThresholdIndex131072x2048_v1") is M.counted_threshold(131072, 2048)[1]
    with pytest.raises(ValueError, match="1 <= TOPK <= N"):
        M.threshold(4, 5)
    with pytest.raises(ValueError, match="1 <= TOPK <= N"):
        M.counted_threshold(4, 5)
