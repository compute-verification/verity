"""`verity.ml.circuits.ops.boolean.rowwise`: each softmax piece is its stated dataflow, the word primitives each step names
composed in order (`F32Add_v2`, `F32Mul_v2`, `F32Div_v2`, `MufuEx2Ftz_v1`, `F2fpBf16_v1`), on rows of scores, uniform words,
special words and ties; the three softmax pieces, each reading the one before's 16-bit outputs, compose to a softmax within
bf16's rounding of the exact one; the embedding's coordinate is `Bf16AddF2fp_v1` of the column's word at the token (0x7FFF
past the vocabulary) and the position's, at every 16-bit token for V = 256 and on ragged vocabularies; each RMSNorm half is its
`ops.boolean.rmsnorm` part's 16 bits, and Phase 1's RMSNorm from halves is `RmsNormRow_v1`; each is Boolean with 16 output
bits; and a row of no words, a third half, a vocabulary outside [1, 2^16] or a part's bad statics is refused."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import mufu as M
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import rmsnorm as R
from verity.ml.circuits.ops.boolean import rowwise as BR

#: signed zeros, the smallest and largest subnormals, the smallest normal, 1.0 and its neighbours, 2, the largest finite,
#: infinities, a signalling and two quiet NaNs, each of either sign
BF16_SPECIALS = np.array([m | s for m in (0x0000, 0x0001, 0x007F, 0x0080, 0x3F7F, 0x3F80, 0x3F81, 0x4000, 0x7F7F, 0x7F80,
                                          0x7F81, 0x7FC0, 0x7FFF) for s in (0, 0x8000)], np.uint16)
#: the same for f32, with 1/32 (a reciprocal sum at N = 32) and RN32(log2 e)
F32_SPECIALS = np.array([m | s for m in (0x00000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3D000000, 0x3F7FFFFF, 0x3F800000,
                                         0x3F800001, BR.LOG2E, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF)
                         for s in (0, 0x80000000)], np.uint32)


def _scores(rng, shape) -> np.ndarray:
    """bf16 words of either sign with magnitudes in [2^-7, 64): attention scores."""
    s = rng.integers(0, 2, shape) << 15
    return (s | rng.integers(120, 133, shape) << 7 | rng.integers(0, 128, shape)).astype(np.uint16)


def _rows(rng, n: int, N: int) -> np.ndarray:
    """``n`` rows of N bf16 words, each a score, a uniform word or a special word."""
    pick = rng.integers(0, 3, (n, N))
    words = [_scores(rng, (n, N)), rng.integers(0, 1 << 16, (n, N)), rng.choice(BF16_SPECIALS, (n, N))]
    return np.choose(pick, words).astype(np.uint16)


def _ties(rng, n: int, N: int) -> np.ndarray:
    """Rows drawn from a few words, so equal values recur and +0 meets -0."""
    return rng.choice(np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000, 0x0001], np.uint16), (n, N))


def _f32(h: np.ndarray) -> np.ndarray:
    return h.astype(np.uint32) << np.uint32(16)


def _ev(defn, *args) -> np.ndarray:
    return np.asarray(E.evaluate_batch(defn, *args)).astype(np.uint32).reshape(-1)


def _const(w: int, n: int) -> np.ndarray:
    return np.full(n, w, np.uint32)


def _first_greatest(rows: np.ndarray) -> np.ndarray:
    v = _f32(rows).view(np.float32)
    out, best = rows[:, 0].copy(), v[:, 0].copy()
    with np.errstate(invalid="ignore"):
        for j in range(1, rows.shape[1]):
            take = v[:, j] > best
            out, best = np.where(take, rows[:, j], out), np.where(take, v[:, j], best)
    return np.where(np.isnan(v).any(axis=1), 0x7FFF, out).astype(np.uint16)


def _exp(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    d = _ev(W.F32AddV2, _f32(x), _f32(m) ^ np.uint32(0x80000000))
    return _ev(M.MufuEx2Ftz, _ev(W.F32MulV2, d, _const(BR.LOG2E, len(d))))


def _recip(rows: np.ndarray, m: np.ndarray) -> np.ndarray:
    s = _const(0, len(rows))
    for j in range(rows.shape[1]):
        s = _ev(W.F32AddV2, s, _exp(rows[:, j], m))
    return _ev(W.F32DivV2, _const(0x3F800000, len(s)), s)


def _words(target, *args) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *args)).astype(np.uint32).reshape(-1)


@pytest.mark.parametrize("N", [1, 2, 3, 7])
def test_the_max_is_the_first_greatest_word_or_the_f2fp_nan(N: int) -> None:
    rng = np.random.default_rng(100 + N)
    rows = np.concatenate([_scores(rng, (2000, N)), _rows(rng, 2000, N), _ties(rng, 2000, N)])
    assert np.array_equal(_words(bind(BR.SoftmaxMax, N=N), rows), _first_greatest(rows))


@pytest.mark.parametrize("N", [1, 2, 5])
def test_the_reciprocal_halves_are_rn_of_one_over_the_exponentials_summed_from_zero(N: int) -> None:
    """At the row's max and at an unrelated m (so a term overflows, underflows or is a NaN)."""
    rng = np.random.default_rng(200 + N)
    rows = np.concatenate([_scores(rng, (1500, N)), _rows(rng, 1500, N), _ties(rng, 500, N)])
    m = np.concatenate([_first_greatest(rows), rng.choice(BF16_SPECIALS, len(rows)), _scores(rng, len(rows))])
    rows = np.concatenate([rows] * 3)
    r = _recip(rows, m)
    for half in (0, 1):
        got = _words(bind(BR.SoftmaxRecip, N=N, HALF=half), rows, m)
        assert np.array_equal(got, (r >> np.uint32(16 * half)) & np.uint32(0xFFFF)), half


def test_an_output_is_the_bf16_rounding_of_its_exponential_times_the_reciprocal() -> None:
    rng = np.random.default_rng(300)
    n = 30_000
    x, m = _rows(rng, n, 2).T
    near = (rng.integers(0x3D000000, 0x3F800001, n, dtype=np.uint64)).astype(np.uint32)
    r = np.choose(rng.integers(0, 3, n), [near, rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32),
                                          rng.choice(F32_SPECIALS, n)]).astype(np.uint32)
    want = _ev(P.F2fpBf16, _ev(W.F32MulV2, _exp(x, m), r))
    assert np.array_equal(_words(bind(BR.SoftmaxOut), x, m, r), want)


def test_the_softmax_pieces_compose_to_a_softmax_within_bf16_rounding() -> None:
    """Rows of N = 5 scores: m from `SoftmaxMax`, r from `SoftmaxRecip`'s two halves and each p_j from `SoftmaxOut`, every
    piece reading the 16-bit words the one before output, are within 2^-8 + 2^-15 relative of the exact softmax, or 2^-125
    absolute where MUFU.EX2 flushes a tiny term. 2^-8 is bf16's half ulp. The rest is MUFU.EX2's error and the f32 roundings,
    the largest those of x - m and of its product with L, each at most 2^-24 of an exponent below 2^8 (the measured worst is
    0.003865, just under 2^-8)."""
    rng = np.random.default_rng(500)
    N = 5
    rows = _scores(rng, (3000, N))
    m = _words(bind(BR.SoftmaxMax, N=N), rows).astype(np.uint16)
    r = _words(bind(BR.SoftmaxRecip, N=N, HALF=0), rows, m) | _words(bind(BR.SoftmaxRecip, N=N, HALF=1), rows, m) << 16
    p = np.stack([_words(bind(BR.SoftmaxOut), rows[:, j], m, r) for j in range(N)], axis=1).astype(np.uint16)
    v = _f32(rows).view(np.float32).astype(np.float64)
    exact = np.exp(v - v.max(axis=1, keepdims=True))
    exact /= exact.sum(axis=1, keepdims=True)
    got = _f32(p).view(np.float32).astype(np.float64)
    assert np.all(np.abs(got - exact) <= (2.0 ** -8 + 2.0 ** -15) * exact + 2.0 ** -125)


def _embedding(tok: np.ndarray, col: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """``Bf16AddF2fp_v1(col[tok], pos)``, col[tok] 0x7FFF past the column."""
    V = col.shape[1]
    h = np.where(tok < V, col[np.arange(len(tok)), np.minimum(tok, V - 1)], BR.EMBEDDING_FILL).astype(np.uint32)
    return _ev(W.Bf16AddF2fp, h, pos.astype(np.uint32))


@pytest.mark.parametrize("V,n", [(256, 1 << 16), (200, 6000), (3, 3000), (1, 2000)])
def test_the_embedding_coordinate_adds_the_tokens_word_to_the_positions(V: int, n: int) -> None:
    """At V = 256 every 16-bit token once; elsewhere the in-range tokens, the first past the vocabulary, the largest and
    uniform ones. Each instance has its own column (uniform and special words) and position word."""
    rng = np.random.default_rng(600 + V)
    edge = np.array([0, V - 1, V, (1 << 16) - 1], np.uint32)
    tok = (np.arange(n, dtype=np.uint32) if n == 1 << 16 else
           np.concatenate([edge, rng.integers(0, V, n // 2), rng.integers(0, 1 << 16, n - n // 2 - len(edge))]).astype(np.uint32))
    col = np.where(rng.integers(0, 4, (n, V), dtype=np.uint8) == 0, rng.choice(BF16_SPECIALS, (n, V)),
                   rng.integers(0, 1 << 16, (n, V), dtype=np.uint16))
    pos = np.choose(rng.integers(0, 3, n), [_scores(rng, n), rng.integers(0, 1 << 16, n), rng.choice(BF16_SPECIALS, n)])
    pos = pos.astype(np.uint16)
    assert np.array_equal(_words(bind(BR.EmbeddingCoordinate, V=V), tok, col, pos), _embedding(tok, col, pos))


def _activations(rng, n: int, N: int) -> np.ndarray:
    """bf16 words of either sign whose squares are normal (exponent fields in [112, 136)), overflow, or are subnormal or 0."""
    e = np.choose(rng.integers(0, 3, (n, N)), [rng.integers(112, 136, (n, N)), rng.integers(191, 255, (n, N)),
                                                rng.integers(0, 64, (n, N))])
    return (rng.integers(0, 2, (n, N)) << 15 | e << 7 | rng.integers(0, 128, (n, N))).astype(np.uint16)


def _f32_words(rng, n: int, G: int) -> np.ndarray:
    """Chunk sums of activations' squares, uniform words and special words."""
    sums = _words(bind(R.RmsSumSqPart, C=3), _activations(rng, n * G, 3)).reshape(n, G)
    uniform = rng.integers(0, 1 << 32, (n, G), dtype=np.uint64).astype(np.uint32)
    return np.choose(rng.integers(0, 4, (n, G)), [sums, sums, uniform, rng.choice(F32_SPECIALS, (n, G))]).astype(np.uint32)


def _join(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo | hi << np.uint32(16)


def test_each_half_is_its_parts_16_bits() -> None:
    """`RmsSumSqPartHalf` on rows of activations, scores, uniform and special words; `F32SumPartHalf` and `RmsNormRecipHalf`
    on chunk sums, uniform and special f32 words, at Phase 1's and Phase 2's N and G."""
    rng = np.random.default_rng(700)
    cases = []
    for C in (1, 4):
        x = np.concatenate([_activations(rng, 2000, C), _rows(rng, 2000, C)])
        cases.append((bind(BR.RmsSumSqPartHalf, C=C), bind(R.RmsSumSqPart, C=C), x))
    for G in (1, 3):
        cases.append((bind(BR.F32SumPartHalf, G=G), bind(R.F32SumPart, G=G), _f32_words(rng, 3000, G)))
    for N, G, eps in ((128, 1, 1e-5), (16384, 2, 1e-5), (4, 2, 1e-6)):
        cases.append((bind(BR.RmsNormRecipHalf, N=N, G=G, EPS=eps), bind(R.RmsNormRecip, N=N, G=G, EPS=eps),
                      _f32_words(rng, 3000, G)))
    for half, part, s in cases:
        want = _words(part, s)
        for h in (0, 1):
            assert np.array_equal(_words(bind(half, HALF=h), s), (want >> np.uint32(16 * h)) & np.uint32(0xFFFF)), (half.id, h)


@pytest.mark.parametrize("N", [1, 8])
def test_phase_1s_rmsnorm_from_halves_is_the_row(N: int) -> None:
    """One chunk and G = 1, each unit reading the 16-bit words the one before output: the chunk sum's two halves, rs's two
    halves from it, and `RmsNormOut_v1{C=1}` at each coordinate, against `RmsNormRow_v1{N}`."""
    rng = np.random.default_rng(800 + N)
    x, w = np.concatenate([_activations(rng, 1500, N), _rows(rng, 500, N)]), _activations(rng, 2000, N)
    acc = _join(*(_words(bind(BR.RmsSumSqPartHalf, C=N, HALF=h), x) for h in (0, 1))).reshape(-1, 1)
    rs = _join(*(_words(bind(BR.RmsNormRecipHalf, N=N, G=1, EPS=1e-5, HALF=h), acc) for h in (0, 1)))
    out = np.stack([_words(bind(R.RmsNormOut, C=1), x[:, j:j + 1], rs, w[:, j:j + 1]) for j in range(N)], axis=1)
    want = np.asarray(BS.evaluate_words(bind(R.RmsNormRow, N=N, EPS=1e-5), x, w)).astype(np.uint32).reshape(-1, N)
    assert np.array_equal(out, want)


@pytest.mark.parametrize("target", [bind(BR.SoftmaxMax, N=3), bind(BR.SoftmaxRecip, N=3, HALF=0),
                                    bind(BR.SoftmaxRecip, N=3, HALF=1), bind(BR.SoftmaxOut),
                                    bind(BR.EmbeddingCoordinate, V=256), bind(BR.RmsSumSqPartHalf, C=2, HALF=0),
                                    bind(BR.F32SumPartHalf, G=2, HALF=1), bind(BR.RmsNormRecipHalf, N=8, G=1, EPS=1e-5, HALF=1)],
                         ids=lambda t: t.id)
def test_each_is_boolean_with_16_output_bits(target) -> None:
    assert BL.is_boolean(target)
    assert BL.elements(target.signature()[1]) == [16]


def test_a_row_of_no_words_a_third_half_or_a_vocabulary_outside_16_bits_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        bind(BR.SoftmaxMax, N=0).signature()
    with pytest.raises(ValueError, match="HALF is 0"):
        bind(BR.SoftmaxRecip, N=2, HALF=2).signature()
    for V in (0, (1 << 16) + 1, "256"):
        with pytest.raises(ValueError, match="V is the vocabulary"):
            bind(BR.EmbeddingCoordinate, V=V).signature()


def test_a_half_refuses_a_third_half_and_its_parts_bad_statics() -> None:
    for d, ok in ((BR.RmsSumSqPartHalf, {"C": 2}), (BR.F32SumPartHalf, {"G": 2}),
                  (BR.RmsNormRecipHalf, {"N": 8, "G": 1, "EPS": 1e-5})):
        with pytest.raises(ValueError, match="HALF is 0"):
            bind(d, **ok, HALF=2).signature()
    bad = ((BR.RmsSumSqPartHalf, {"C": 0}), (BR.RmsSumSqPartHalf, {"C": "2"}), (BR.F32SumPartHalf, {"G": 0}),
           (BR.RmsNormRecipHalf, {"N": 0, "G": 1, "EPS": 1e-5}), (BR.RmsNormRecipHalf, {"N": 8, "G": 0, "EPS": 1e-5}),
           (BR.RmsNormRecipHalf, {"N": 8, "G": 1, "EPS": 0.0}), (BR.RmsNormRecipHalf, {"N": 8, "G": 1, "EPS": "1e-5"}))
    for d, s in bad:
        with pytest.raises(ValueError):
            bind(d, **s, HALF=0).signature()
