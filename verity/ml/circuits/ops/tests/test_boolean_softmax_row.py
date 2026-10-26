"""`verity.ml.circuits.ops.boolean.softmax_row`: `SoftmaxRow_v1{N}` is its stated dataflow, the word primitives each step names
composed in order (`F32Add_v2`, `F32Mul_v2`, `F32Div_v2`, `MufuEx2Ftz_v1`, `F2fpBf16_v1`), on rows of scores, uniform words,
special words and ties, at N = 1 up to CMP-92's context of 32; it is within bf16's rounding of the exact softmax; and
`QkSoftmaxRow_v1{T, D=64}` on the H200's step is `GemmCoordinate_v2{K=64}`'s words over the key rows followed by that
composition. Each is Boolean with one 16-bit output a key, and a row of no words, no keys or a D off the step is refused."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu as M
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import softmax_row as SR

#: signed zeros, the smallest and largest subnormals, the smallest normal, 1.0 and its neighbours, 2, the largest finite,
#: infinities, a signalling and two quiet NaNs, each of either sign
BF16_SPECIALS = np.array([m | s for m in (0x0000, 0x0001, 0x007F, 0x0080, 0x3F7F, 0x3F80, 0x3F81, 0x4000, 0x7F7F, 0x7F80,
                                          0x7F81, 0x7FC0, 0x7FFF) for s in (0, 0x8000)], np.uint16)


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


def _softmax_words(rows: np.ndarray) -> np.ndarray:
    """The dataflow, each step its word primitive."""
    m = _f32(_first_greatest(rows)) ^ np.uint32(0x80000000)
    n = len(rows)
    es = [_ev(M.MufuEx2Ftz, _ev(W.F32MulV2, _ev(W.F32AddV2, _f32(rows[:, j]), m), _const(SR.LOG2E, n)))
          for j in range(rows.shape[1])]
    s = _const(0, n)
    for e in es:
        s = _ev(W.F32AddV2, s, e)
    r = _ev(W.F32DivV2, _const(0x3F800000, n), s)
    return np.stack([_ev(P.F2fpBf16, _ev(W.F32MulV2, e, r)) for e in es], axis=1)


def _row(rows: np.ndarray) -> np.ndarray:
    N = rows.shape[1]
    return np.asarray(BS.evaluate_words(bind(SR.SoftmaxRow, N=N), rows)).astype(np.uint32).reshape(len(rows), N)


@pytest.mark.parametrize("N", [1, 2, 3, 7])
def test_the_row_is_the_dataflow_of_its_word_primitives(N: int) -> None:
    rng = np.random.default_rng(100 + N)
    rows = np.concatenate([_scores(rng, (1000, N)), _rows(rng, 1000, N), _ties(rng, 500, N)])
    assert np.array_equal(_row(rows), _softmax_words(rows))


@pytest.mark.slow
def test_the_row_at_the_context_length_is_the_dataflow_of_its_word_primitives() -> None:
    """N = 32, CMP-92's longest row: the sum's order over 32 terms, which shorter rows can't show."""
    rng = np.random.default_rng(132)
    rows = np.concatenate([_scores(rng, (200, 32)), _rows(rng, 100, 32), _ties(rng, 50, 32)])
    assert np.array_equal(_row(rows), _softmax_words(rows))


def test_the_row_is_a_softmax_within_bf16_rounding() -> None:
    """Rows of N = 5 scores are within 2^-8 + 2^-15 relative of the exact softmax, or 2^-125 absolute where MUFU.EX2 flushes a
    tiny term. 2^-8 is bf16's half ulp. The rest is MUFU.EX2's error and the f32 roundings, the largest those of x - m and of
    its product with L, each at most 2^-24 of an exponent below 2^8 (the measured worst is 0.003865, just under 2^-8)."""
    rng = np.random.default_rng(500)
    rows = _scores(rng, (3000, 5))
    got = _f32(_row(rows).astype(np.uint16)).view(np.float32).astype(np.float64)
    v = _f32(rows).view(np.float32).astype(np.float64)
    exact = np.exp(v - v.max(axis=1, keepdims=True))
    exact /= exact.sum(axis=1, keepdims=True)
    assert np.all(np.abs(got - exact) <= (2.0 ** -8 + 2.0 ** -15) * exact + 2.0 ** -125)


def _operands(rng, shape) -> np.ndarray:
    """bf16 words of either sign with magnitudes in [2^-9, 2), so a 64-long dot is a score of at most a few hundred."""
    s = rng.integers(0, 2, shape) << 15
    return (s | rng.integers(118, 128, shape) << 7 | rng.integers(0, 128, shape)).astype(np.uint16)


def _heads(rng, n: int, T: int, D: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """``n`` heads' q (n, D) and key rows (n, T, D): a third clean operands, a third with one word of the head a uniform or
    special word (so one score may be large, infinite or NaN), and a third uniform words."""
    third = n // 3
    head = _operands(rng, (n, (T + 1) * D))
    one = third + np.arange(third)
    at = rng.integers(0, (T + 1) * D, third)
    head[one, at] = np.where(rng.random(third) < 0.5, rng.integers(0, 1 << 16, third), rng.choice(BF16_SPECIALS, third))
    head[2 * third:] = rng.integers(0, 1 << 16, head[2 * third:].shape)
    return head[:, :D], head[:, D:].reshape(n, T, D)


def _scores_words(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    coord = bind(G.GemmCoordinate, K=q.shape[1], **G.GEMM_HOPPER)
    return np.stack([np.asarray(E.evaluate_batch(coord, q, k[:, j])).astype(np.uint16).reshape(-1)
                     for j in range(k.shape[1])], axis=1)


def _qk(T: int) -> object:
    return bind(SR.QkSoftmaxRow, T=T, D=64, **BG.GEMM_HOPPER)


def _fused(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    T = k.shape[1]
    return np.asarray(BS.evaluate_words(_qk(T), q, k)).astype(np.uint32).reshape(len(q), T)


@pytest.mark.parametrize("T", [1, 2, 5])
def test_the_fused_unit_is_gemm_coordinate_words_then_the_softmax_composition(T: int) -> None:
    rng = np.random.default_rng(600 + T)
    q, k = _heads(rng, 600, T)
    assert np.array_equal(_fused(q, k), _softmax_words(_scores_words(q, k)))


@pytest.mark.slow
def test_the_fused_unit_at_the_context_length_is_gemm_coordinate_words_then_the_softmax_composition() -> None:
    """T = 32, CMP-92's longest attention row, which the sidecar binds."""
    rng = np.random.default_rng(632)
    q, k = _heads(rng, 90, 32)
    assert np.array_equal(_fused(q, k), _softmax_words(_scores_words(q, k)))


@pytest.mark.parametrize("target", [bind(SR.SoftmaxRow, N=1), bind(SR.SoftmaxRow, N=3), _qk(1), _qk(3)], ids=lambda t: t.id)
def test_each_is_boolean_with_one_16_bit_output_a_key(target) -> None:
    assert BL.is_boolean(target)
    n = target.bindings.get("N", target.bindings.get("T"))
    assert BL.elements(target.signature()[1]) == [16] * n


def test_a_row_of_no_words_no_keys_or_a_d_off_the_step_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        bind(SR.SoftmaxRow, N=0).signature()
    with pytest.raises(ValueError, match="at least 1"):
        bind(SR.QkSoftmaxRow, T=0, D=64, **BG.GEMM_HOPPER).signature()
    with pytest.raises(ValueError, match="multiple of the 16-wide step"):
        bind(SR.QkSoftmaxRow, T=2, D=40, **BG.GEMM_HOPPER)
