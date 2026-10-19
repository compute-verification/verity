"""`registry.dsa_topk` (the selection relation of a DSA row past TOPK keys, and the rows it selects) and its Boolean versions
`boolean_dsa_topk`: the key is the logits' order; the top-TOPK set is the first TOPK under the tie rule; the relation accepts
exactly the permutations of that set; the Boolean versions agree with their word views on accepted and refused orders (ties,
signed zeros, NaNs and infinities, slots past N, negative, aliasing or repeated); the gather is `Embedding` per slot."""
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
from verity_vllm.program import boolean as SW
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import boolean_dsa_topk as BT
from verity_vllm.program.registry import dsa_topk as T

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


@pytest.mark.parametrize("N,TOPK,W", [(5, 4, 3), (3, 3, 1)])
def test_the_gather_is_embedding_per_slot_and_its_boolean_version_agrees(N, TOPK, W) -> None:
    target = bind(BT.DsaGatherRowsV2, N=N, TOPK=TOPK, W=W)
    word = TR.word_of(target)
    rng = random.Random(N + TOPK + W)
    orders = [[rng.choice([*range(N + 1), 0xFFFFFFFF]) for _ in range(TOPK)] for _ in range(40)]
    rows = [[[rng.getrandbits(16) for _ in range(W)] for _ in range(N)] for _ in range(40)]
    ref = np.asarray(E.evaluate_batch(word, orders, rows)).reshape(40, -1)
    got = np.asarray(BS.evaluate_words(target, orders, rows)).reshape(40, -1)
    assert np.array_equal(got, ref)
    emb = bind(b1.Embedding, V=N, H=W)
    for o, r, out in zip(orders[:5], rows, ref):
        flat = [w for row in r for w in row]
        assert out.tolist() == [w for s in o for w in evaluate(emb, [s], flat)]
        assert out.tolist() == [w for s in o for w in (r[s] if s < N else [0x7FFF] * W)]


@pytest.mark.parametrize("target", [bind(BT.DsaTopKOrderV2, N=20, TOPK=16), bind(BT.DsaGatherRowsV2, N=5, TOPK=4, W=3)],
                         ids=lambda t: t.id)
def test_the_boolean_version_is_boolean_and_its_word_view_is_v1(target) -> None:
    word = TR.word_of(target)
    assert BL.is_boolean(target) and (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, target.bindings)
    assert SW.boolean_version(word).id == target.id


def test_the_order_check_resolves_by_id_in_a_cold_decode() -> None:
    """A Program past TOPK names `DsaTopKOrderCheck{N}x{TOPK}_v1`, made while deriving; a cold decode resolves it by id."""
    assert lookup_primitive(REGISTRY, "DsaTopKOrderCheck2049x2048_v1") is T.order_check(2049, 2048)
    with pytest.raises(ValueError, match="1 <= TOPK <= N"):
        T.order_check(4, 5)
