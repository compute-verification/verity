"""`registry.dsa_topk` (the selection relation of a DSA row past TOPK keys, and the Waksman network that reads the rows it
selects) and its Boolean versions `boolean_dsa_topk`: the key is the logits' order; the top-TOPK set is the first TOPK under the
tie rule; the relation accepts exactly the permutations of that set; the Boolean versions agree with their word views on accepted
and refused orders (ties, signed zeros, NaNs and infinities, slots past N, negative, aliasing or repeated); every order has
settings, at which the network reads the rows in that order and routes the indices to that order."""
from __future__ import annotations

import itertools
import random
import struct

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.codec import lookup_primitive
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import boolean_dsa_topk as BT
from verity.ml.definitions.registry import dsa_topk as T

POOL = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC00000, 0xFFC00000, 0x7F800000, 0xFF800000, 0x00000001, 0x80000001]


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _scores(rng, N: int) -> list[int]:
    """Words with repeats, specials and random bits, so rows tie, straddle signed zero and hold NaNs."""
    return [rng.choice(POOL) if rng.random() < 0.4 else rng.getrandbits(32) for _ in range(N)]


def _orders(rng, scores, TOPK: int, n: int) -> list[list[int]]:
    """Permutations of the top set, and orders off it by one slot (another key, past N, negative, aliasing an index in its low
    bits, a repeat), and random words."""
    N, b = len(scores), BT.index_bits(len(scores))
    top = sorted(T.topk_set(scores, TOPK))
    out = []
    for i in range(n):
        o = rng.sample(top, TOPK)
        kind = i % 4
        if kind == 1:
            s = rng.randrange(TOPK)
            o[s] = rng.choice([j for j in range(N) if j not in top] or [N]) if rng.random() < 0.5 else rng.choice(
                [N, (1 << b) | o[s], 0xFFFFFFFF, 1 << 31 | o[s], o[(s + 1) % TOPK]])
        elif kind == 2:
            o = [rng.getrandbits(32) if rng.random() < 0.5 else rng.randrange(N) for _ in range(TOPK)]
        out.append(o)
    return out


def test_the_key_is_the_logits_order() -> None:
    """`convert_to_uint32_v2`: a larger float a larger key, -0 below +0, NaNs past the infinities."""
    xs = sorted({float(x) for x in np.random.default_rng(0).normal(0, 1e3, 200).astype(np.float32)} | {0.0, 1e-45, -1e-45, 3e38, -3e38})
    keys = [T.topk_key(_f32(x)) for x in xs]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert T.topk_key(0x80000000) < T.topk_key(0) < T.topk_key(1)
    assert T.topk_key(0xFFC00000) < T.topk_key(0xFF800000) and T.topk_key(0x7F800000) < T.topk_key(0x7FC00000)


@pytest.mark.parametrize("seed", range(4))
def test_the_top_set_is_every_member_before_every_non_member(seed) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        N = rng.randrange(1, 12)
        K = rng.randrange(1, N + 1)
        sc = _scores(rng, N)
        top = T.topk_set(sc, K)
        assert len(top) == K
        assert all((T.topk_key(sc[i]), -i) > (T.topk_key(sc[j]), -j) for i in top for j in set(range(N)) - top)


def test_the_relation_accepts_exactly_the_permutations_of_the_top_set() -> None:
    """Every order of 3 slots over [0, 7) and two words past N, on a row of 5 with a tie at the boundary."""
    sc = [_f32(1.0), _f32(2.0), _f32(1.0), _f32(-0.0), _f32(1.0)]
    assert T.topk_set(sc, 3) == {0, 1, 2}
    want = {tuple(p) for p in itertools.permutations((0, 1, 2))}
    for o in itertools.product([*range(7), 0xFFFFFFFF, 1 << 31], repeat=3):
        assert T.order_ok(sc, o, 3) == (o in want), o


#: (N, TOPK): equal, one, a power of two and one past it, GLM-5.3's shape scaled down, b = 1
SHAPES = [(6, 4), (6, 6), (5, 1), (8, 5), (9, 8), (17, 16), (2, 1), (1, 1)]


@pytest.mark.parametrize("N,TOPK", SHAPES)
def test_the_boolean_relation_agrees_with_its_word_view(N, TOPK) -> None:
    target = bind(BT.DsaTopKOrderV2, N=N, TOPK=TOPK)
    word = TR.word_of(target)
    rng = random.Random(N * 100 + TOPK)
    rows = [_scores(rng, N) for _ in range(24)]
    rows += [[POOL[0]] * N, [POOL[1]] * N, [POOL[4]] * N, [POOL[0], POOL[1]] * (N // 2) + [POOL[0]] * (N % 2)]
    scores, orders = [], []
    for sc in rows:
        for o in _orders(rng, sc, TOPK, 8):
            scores.append(sc)
            orders.append(o)
    ref = np.asarray(E.evaluate_batch(word, scores, orders)).reshape(-1)
    got = np.asarray(BS.evaluate_words(target, scores, orders)).reshape(-1)
    assert ref.sum() and (len(ref) - ref.sum()), "both verdicts are exercised"
    assert ref.tolist() == [T.order_ok(s, o, TOPK) for s, o in zip(scores, orders)]
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ, first {[(scores[i], orders[i]) for i in bad[:3]]}"


def test_the_one_hot_decodes_in_range_indices_only() -> None:
    fn = bind(BT.DsaTopKOneHot, N=5)
    idx = [0, 1, 4, 5, 7, 8, 1 << 31, 0xFFFFFFFF, (1 << 3) | 2]
    got = np.asarray(BS.evaluate_words(fn, [[i] for i in idx])).reshape(-1)   # one 5-bit element: bit j is idx == j
    assert got.tolist() == [1 << i if i < 5 else 0 for i in idx]


def test_every_order_has_settings_that_route_it() -> None:
    """Every order of K distinct rows of N <= 7, and random ones up to N = 300 and at GLM-5.3's first row past TOPK: the network
    at `waksman_settings` puts row order[s] at output s."""
    for N in range(1, 8):
        for K in range(1, N + 1):
            for o in itertools.permutations(range(N), K):
                assert T.waksman_apply(T.waksman_settings(o, N), list(range(N)), K) == list(o), (N, o)
    rng = random.Random(0)
    for N, K in [(rng.randrange(8, 300), None) for _ in range(40)] + [(2049, 2048), (2080, 2048)]:
        K = K or rng.randrange(1, N + 1)
        o = rng.sample(range(N), K)
        assert T.waksman_apply(T.waksman_settings(o, N), list(range(N)), K) == o, (N, K)


def test_the_switch_count_is_the_arbitrary_size_waksman_networks() -> None:
    """Unpruned, N ceil(log2 N) - 2^ceil(log2 N) + 1; K = N - 1 prunes none at N = 2049 and 13 at N = 2050."""
    for N in range(2, 300):
        b = (N - 1).bit_length()
        assert T.waksman_switches(N, N) == N * b - (1 << b) + 1, N
    assert (T.waksman_switches(2049, 2048), T.waksman_switches(2050, 2048), T.waksman_switches(2050, 2050)) == (20493, 20505, 20505)


def test_settings_refuse_an_order_that_is_no_selection() -> None:
    for o, N in [([0, 0], 3), ([3], 3), ([-1], 3), ([], 3), ([0, 1, 2, 3], 3)]:
        with pytest.raises(ValueError, match="distinct indices"):
            T.waksman_settings(o, N)


def _pack(sw) -> int:
    """Settings as `evaluate_words` takes an ``Array<n, bit>``: one n-bit word, bit i setting i."""
    return sum(b << i for i, b in enumerate(sw))


#: (N, K, W, E): two rows, an odd and an even n with every pruning (K = N, N - 1, a pick at an odd K, one live output), GLM-5.3's
#: shape scaled down, and index tags
NETWORKS = [(2, 1, 2, 16), (2, 2, 1, 16), (3, 2, 2, 16), (3, 3, 1, 32), (5, 4, 3, 16), (6, 5, 2, 16), (6, 6, 1, 16), (7, 3, 1, 16),
            (8, 8, 2, 32), (9, 8, 2, 16), (17, 16, 1, 16), (4, 1, 1, 16)]


@pytest.mark.parametrize("N,K,W,WIDTH", NETWORKS)
def test_the_network_reads_the_ordered_rows_and_its_boolean_version_agrees(N, K, W, WIDTH) -> None:
    """At an order's settings the word network's outputs are the rows in that order; at any settings (random bits too) it is
    `waksman_apply`, and `DsaWaksmanRows_v2` agrees with it."""
    target = bind(BT.DsaWaksmanRowsV2, N=N, K=K, W=W, E=WIDTH)
    word = TR.word_of(target)
    rng = random.Random(N * 1000 + K * 10 + W)
    sws, rows, want = [], [], []
    for i in range(40):
        o = rng.sample(range(N), K)
        sw = T.waksman_settings(o, N) if i % 3 else [rng.getrandbits(1) for _ in range(T.waksman_switches(N, K))]
        r = [[rng.getrandbits(WIDTH) for _ in range(W)] for _ in range(N)]
        if i % 3:
            assert T.waksman_apply(sw, r, K) == [r[j] for j in o]
        sws.append(sw)
        rows.append(r)
        want.append([w for row in T.waksman_apply(sw, r, K) for w in row])
    ref = np.asarray(E.evaluate_batch(word, sws, rows)).reshape(40, -1)
    got = np.asarray(BS.evaluate_words(target, [[_pack(s)] for s in sws], rows)).reshape(40, -1)
    assert ref.tolist() == want
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("N,TOPK", [(5, 4), (9, 8), (17, 16), (3, 1), (6, 6)])
def test_the_routed_indices_are_the_order_and_the_relation_reads_them(N, TOPK) -> None:
    """`DsaWaksmanOrder` at an order's settings is that order, which `DsaTopKOrder` accepts when it is the top set; its Boolean
    version agrees, at random settings too."""
    target = bind(BT.DsaWaksmanOrderV2, N=N, TOPK=TOPK)
    word = TR.word_of(target)
    rng = random.Random(N + TOPK)
    orders = [rng.sample(range(N), TOPK) for _ in range(20)]
    sws = [T.waksman_settings(o, N) for o in orders] + [[rng.getrandbits(1) for _ in range(T.waksman_switches(N, TOPK))]
                                                        for _ in range(10)]
    ref = np.asarray(E.evaluate_batch(word, sws)).reshape(len(sws), -1)
    assert ref[:20].tolist() == orders
    assert np.array_equal(np.asarray(BS.evaluate_words(target, [[_pack(s)] for s in sws])).reshape(len(sws), -1), ref)
    sc = _scores(rng, N)
    top = rng.sample(sorted(T.topk_set(sc, TOPK)), TOPK)
    assert list(evaluate(bind(T.DsaTopKOrder, N=N, TOPK=TOPK), sc, evaluate(word, T.waksman_settings(top, N)))) == [1]


@pytest.mark.parametrize("target", [bind(BT.DsaTopKOrderV2, N=20, TOPK=16), bind(BT.DsaWaksmanRowsV2, N=9, K=8, W=3, E=16),
                                    bind(BT.DsaWaksmanOrderV2, N=9, TOPK=8), bind(BT.DsaWaksmanSwapV2, W=3, E=32),
                                    bind(BT.DsaWaksmanPickV2, W=3, E=16), bind(BT.Xor3x16), bind(BT.Xor3x32)], ids=lambda t: t.id)
def test_the_boolean_version_is_boolean_and_its_word_view_is_v1(target) -> None:
    word = TR.word_of(target)
    defn, bindings = (word.defn, word.bindings) if hasattr(word, "defn") else (word, {})
    assert BL.is_boolean(target) and (defn.name, defn.version, bindings) == (target.defn.name, 1, target.bindings)
    assert SW.boolean_version(word).id == target.id


def test_the_order_check_resolves_by_id_in_a_cold_decode() -> None:
    """A Program past TOPK names `DsaTopKOrderCheck{N}x{TOPK}_v1`, made while deriving; a cold decode resolves it by id."""
    assert lookup_primitive(REGISTRY, "DsaTopKOrderCheck2049x2048_v1") is T.order_check(2049, 2048)
    with pytest.raises(ValueError, match="1 <= TOPK <= N"):
        T.order_check(4, 5)
