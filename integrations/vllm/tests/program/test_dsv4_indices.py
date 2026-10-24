"""`registry.dsv4_indices` (`combine_topk_swa_indices`, vLLM d9105ea80): the twin against the reference, the Definition against
the kernel's stores at every boundary of a position sweep and at the int32 ends, the rows whose window leaves them, the ratio-128
prefill row as `Dsv4ShortContextTopK_v1`, the statics, and the Boolean Definition against its word."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import numpy as np
import pytest

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, self_check
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.numerics import dsv4_indices_twin as T
from verity.ml.definitions.registry import boolean_dsv4_indices as BX
from verity.ml.definitions.registry import dsv4_indexer as I
from verity.ml.definitions.registry import dsv4_indices as X

U = np.uint32
DEFS = {d.id: d for d in X.DSV4_INDICES_COMPOSITES}
TWIN_TARGETS = T._targets(DEFS)
SMALL = dict(TOPK=8, RATIO=4, WINDOW=4, WIDTH=16)
ODD = dict(TOPK=5, RATIO=2, WINDOW=3, WIDTH=9)
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def _words(xs) -> np.ndarray:
    return (np.asarray(xs, dtype=np.int64) & 0xFFFFFFFF).astype(U)


def _sweep(st: dict) -> np.ndarray:
    """Every position from -3 * WINDOW past the top-k's end, and the int32 ends with the window's width around them."""
    W, top = st["WINDOW"], st["TOPK"] * st["RATIO"]
    ends = [*range(I32_MIN, I32_MIN + 2 * W + 2), *range(I32_MAX - 2 * W, I32_MAX + 1)]
    return _words([*range(-3 * W, top + 2 * W + st["RATIO"]), *ends])


def _args(st: dict, pos: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(pos)
    topk = rng.integers(-1, 1 << 12, (n, st["TOPK"])) & 0xFFFFFFFF
    mb, nn, gs = (rng.integers(0, 1 << 32, n, dtype=np.uint64) for _ in range(3))
    return [topk.astype(U), pos, mb.astype(U), nn.astype(U), gs.astype(U)]


def _reference(st: dict, args) -> np.ndarray:
    return np.asarray(evaluate_batch(bind(X.Dsv4CombineTopkSwa, **st), *args, kernel="reference"), dtype=np.uint64)


# --- the twin --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", TWIN_TARGETS, ids=[t.id for t in TWIN_TARGETS])
def test_the_twin_is_the_reference(target):
    sc = self_check(target, kernel=T.KERNEL, n=48, seed=0)
    assert sc.ok, str(sc)
    assert 0 < sc.n - sc.declined, str(sc)


# --- the Definition against the kernel's stores ------------------------------------------------------------------------------------

@pytest.mark.parametrize("st", [SMALL, ODD, dict(TOPK=4, RATIO=128, WINDOW=8, WIDTH=16)], ids=["small", "odd", "ratio-128"])
def test_the_definition_is_the_kernels_stores_over_a_position_sweep(st):
    """At every position from below 0 past TOPK * RATIO and at the int32 ends, the Definition's row and length are the twin's
    stores (its j < len against the kernel's o < swa_len), and the twin declines exactly the rows whose window leaves them."""
    pos = _sweep(st)
    args = _args(st, pos, 1)
    rows, lens, declined = T.combine(*args, **st)
    want = np.concatenate([rows, lens[:, None]], axis=1).astype(np.uint64)
    got = _reference(st, args)
    keep = ~declined
    assert np.array_equal(got[keep], want[keep])
    s = T.s32(pos)
    assert np.array_equal(declined, (s == I32_MAX) | (s < I32_MIN + st["WINDOW"] - 1))


def test_a_row_whose_window_leaves_it_is_all_minus_one_with_the_kernels_length():
    """pos = INT32_MAX: t wraps to INT32_MIN, so topk_len = -2^(31 - k) and swa_len = WINDOW; below INT32_MIN + WINDOW - 1 the
    window start wraps, swa_len = WINDOW again.  The row's own slots stay -1 and the length is topk_len + swa_len."""
    st = SMALL
    k, W = 2, st["WINDOW"]
    pos = _words([I32_MAX, I32_MIN, I32_MIN + W - 2])
    got = _reference(st, _args(st, pos, 2))
    assert (got[:, :-1] == 0xFFFFFFFF).all()
    tl = [-(1 << (31 - k)), -((-(I32_MIN + 1)) >> k), -((-(I32_MIN + W - 1)) >> k)]
    assert [int(v) for v in got[:, -1]] == [(t + W) & 0xFFFFFFFF for t in tl]


def test_the_row_at_a_served_position():
    """Ratio 4, position 4101: topk_len = min(4102 // 4, 1024) = 1024, the window positions 3974..4101 at slots 1024..1151."""
    st = X.DSV4_INDICES["c4"]
    topk = np.arange(st["TOPK"], dtype=U)
    mb, n, gs = 7 * 9000, 4096, 4101 - 255
    out = [int(v) for v in evaluate(bind(X.Dsv4CombineTopkSwa, **st), topk, 4101, mb, n, gs)]
    assert out[:1024] == [i + mb for i in range(1024)]
    assert out[1024:1152] == [mb + n + 3974 + o - gs for o in range(128)]
    assert out[-1] == 1152


# --- the ratio-128 prefill row ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("K", [128, 256])
def test_the_ratio_128_prefill_row_is_the_short_context_fill(K):
    """`_build_c128a_topk_metadata_kernel`'s prefill branch (sparse_mla.py:417-427): slot i is i when i < min((position + 1) //
    128, K), else -1, over the int64 position; `Dsv4ShortContextTopK_v1{K, 128}` row for row."""
    fill = bind(I.Dsv4ShortContextTopK, TOPK=K, RATIO=128)
    for p in [-1, 0, 126, 127, 128, 255, 128 * K - 2, 128 * K - 1, 128 * K + 5, (1 << 40), (1 << 63) - 1, -(1 << 63)]:
        n = min(((p + 1) // 128) if p + 1 >= 0 else -((-(p + 1)) // 128), K) if p != (1 << 63) - 1 else -1
        want = [i if i < n else 0xFFFFFFFF for i in range(K)]
        assert [int(v) for v in evaluate(fill, p & ((1 << 64) - 1))] == want, p


# --- the statics -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(TOPK=8, RATIO=3, WINDOW=4, WIDTH=16), dict(TOPK=8, RATIO=1, WINDOW=4, WIDTH=16),
                                dict(TOPK=0, RATIO=4, WINDOW=4, WIDTH=16), dict(TOPK=8, RATIO=4, WINDOW=0, WIDTH=16),
                                dict(TOPK=8, RATIO=4, WINDOW=4, WIDTH=11), dict(TOPK=8, RATIO=4, WINDOW=4, WIDTH=1 << 30)],
                         ids=["ratio-3", "ratio-1", "topk-0", "window-0", "narrow", "wide"])
def test_bad_statics_are_refused_by_name(kw):
    with pytest.raises(ValueError, match="Dsv4CombineTopkSwa"):
        bind(X.Dsv4CombineTopkSwa, **kw).body


# --- the Boolean Definition ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("st", [SMALL, ODD], ids=["small", "odd"])
def test_the_boolean_definition_is_its_word(st):
    target = bind(BX.Dsv4CombineTopkSwaV2, **st)
    word = word_of(target)
    assert SW.boolean_version(word).id == target.id
    sweep = _args(st, _sweep(st), 3)
    drawn = T._sample(np.random.default_rng(4), 64, None, **st)
    for args in (sweep, drawn):
        n = len(args[1])
        got = np.asarray(evaluate_words(target, *args), dtype=np.uint64).reshape(n, -1)
        assert np.array_equal(got, _reference(st, args).reshape(n, -1))
