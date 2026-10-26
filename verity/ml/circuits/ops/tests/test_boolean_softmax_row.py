"""`verity.ml.circuits.ops.boolean.softmax_row`: `SoftmaxRow_v1{N}` is its stated dataflow, the word primitives each step names
composed in order (`F32Add_v2`, `F32Mul_v2`, `F32Div_v2`, `MufuEx2Ftz_v1`, `F2fpBf16_v1`), on rows of scores, uniform words,
special words and ties, at N = 1 up to CMP-92's context of 32; it is within bf16's rounding of the exact softmax; and
`QkSoftmaxRow_v1{T, D=64}` on the H200's step is `GemmCoordinate_v2{K=64}`'s words over the key rows followed by that
composition. Each is Boolean with one 16-bit output a key, and a row of no words, no keys or a D off the step is refused.

`SoftmaxRowPad_v1{N}` at n is `SoftmaxRow_v1{N=n}` of the first n words then +0, and `QkSoftmaxRowPad_v1{T, D=64}` at n is
`QkSoftmaxRow_v1{T=n}` on the first n key rows then +0 (`qk_softmax_row_pad_reference`), whatever the masked words or rows
hold (NaN, inf, uniform words), at every n of a short row and at n = 1, 2, 5, 17, 31 and 32 of T = 32; n = 0 writes +0 and
n > T is n = T; n is a position-kind port; the reference refuses n outside 1..T."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits import ports as PT
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu as M
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import softmax_row as SR
from verity.ml.circuits.program import Program

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


def _junk(rng, shape) -> np.ndarray:
    """Masked words: a third uniform words, a third special words (NaN, inf), a third +inf or a quiet NaN, so a masked score
    would win the max or poison the row if it took part."""
    pick = rng.integers(0, 3, shape)
    return np.choose(pick, [rng.integers(0, 1 << 16, shape), rng.choice(BF16_SPECIALS, shape),
                            rng.choice(np.array([0x7F80, 0x7FC0, 0xFFC0], np.uint16), shape)]).astype(np.uint16)


def _row_pad(rows: np.ndarray, n: int) -> np.ndarray:
    N = rows.shape[1]
    return np.asarray(BS.evaluate_words(bind(SR.SoftmaxRowPad, N=N), rows, np.full(len(rows), n, np.uint16))
                      ).astype(np.uint32).reshape(len(rows), N)


def _pad(T: int) -> object:
    return bind(SR.QkSoftmaxRowPad, T=T, D=64, **BG.GEMM_HOPPER)


def _fused_pad(q: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    T = k.shape[1]
    return np.asarray(BS.evaluate_words(_pad(T), q, k, np.full(len(q), n, np.uint16))).astype(np.uint16).reshape(len(q), T)


def _masked_heads(rng, heads: int, T: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """`_heads` at T keys with every key row j >= n junk (`_junk`), and some such rows +inf or NaN whole."""
    q, k = _heads(rng, heads, T)
    k[:, n:] = _junk(rng, k[:, n:].shape)
    whole = rng.random((heads, T - n)) < 0.25
    k[:, n:][whole] = rng.choice(np.array([0x7F80, 0x7FC0], np.uint16), (int(whole.sum()), 1))
    return q, k


def test_the_padded_row_is_the_row_of_its_first_n_words() -> None:
    """Every n of N = 3, the masked words junk: the first n outputs are `SoftmaxRow_v1{N=n}`'s (its word primitives'
    composition) and the rest +0. n = 0 writes +0, and n = 4 or 0xFFFF is n = 3."""
    rng = np.random.default_rng(700)
    N = 3
    base = np.concatenate([_scores(rng, (150, N)), _rows(rng, 150, N), _ties(rng, 75, N)])
    for n in range(1, N + 1):
        rows = base.copy()
        rows[:, n:] = _junk(rng, rows[:, n:].shape)
        want = np.zeros(rows.shape, np.uint32)
        want[:, :n] = _softmax_words(rows[:, :n])
        assert np.array_equal(_row_pad(rows, n), want), n
    assert not _row_pad(base, 0).any()
    full = _row_pad(base, N)
    assert np.array_equal(full, _row(base))
    for n in (N + 1, 0xFFFF):
        assert np.array_equal(_row_pad(base, n), full), n


def test_the_padded_unit_is_the_unit_at_n_keys_then_zeros() -> None:
    """Every n of T = 2, the masked key rows junk: `qk_softmax_row_pad_reference`, `QkSoftmaxRow_v1{T=n}` on the first n rows
    then +0. n = 0 writes +0 and n = 3 or 0xFFFF is n = 2."""
    rng = np.random.default_rng(802)
    T = 2
    for n in range(1, T + 1):
        q, k = _masked_heads(rng, 150, T, n)
        assert np.array_equal(_fused_pad(q, k, n), SR.qk_softmax_row_pad_reference(q, k, n, BG.GEMM_HOPPER["DOT"])), n
    q, k = _heads(rng, 150, T)
    assert not _fused_pad(q, k, 0).any()
    full = _fused_pad(q, k, T)
    for n in (T + 1, 0xFFFF):
        assert np.array_equal(_fused_pad(q, k, n), full), n


@pytest.mark.slow
def test_the_padded_unit_at_the_context_length_is_the_unit_at_n_keys_then_zeros() -> None:
    """T = 32, the sidecar's binding, at n = 1, 2, 5, 17, 31 and 32 with junk in the masked key rows: equal to
    `QkSoftmaxRow_v1{T=n}` word for word, then +0. Shorter rows can't show the mask at the far end of a 32-word sum."""
    rng = np.random.default_rng(832)
    for n in (1, 2, 5, 17, 31, 32):
        q, k = _masked_heads(rng, 60, 32, n)
        got = _fused_pad(q, k, n)
        assert np.array_equal(got, SR.qk_softmax_row_pad_reference(q, k, n, BG.GEMM_HOPPER["DOT"])), n
        assert not got[:, n:].any(), n


def test_an_infinite_or_nan_masked_score_does_not_leak() -> None:
    """Clean heads whose masked key rows score +inf (which would be the max) or NaN (which would make every word NaN): the
    outputs are the unmasked unit's, finite, and +0 past n."""
    rng = np.random.default_rng(900)
    T, n = 2, 1
    q = np.full((2, 64), 0x3F80, np.uint16)
    k = _operands(rng, (2, T, 64))
    k[0, n:], k[1, n:] = 0x7F80, 0x7FC0
    got = _fused_pad(q, k, n)
    assert np.array_equal(got, SR.qk_softmax_row_pad_reference(q, k, n, BG.GEMM_HOPPER["DOT"]))
    assert not got[:, n:].any() and not ((got[:, :n] & 0x7F80) == 0x7F80).any()


def test_n_is_a_position_kind_port() -> None:
    """The unit's key count is a run input of kind position, so it is public; a caller passing it a value it didn't declare
    as a position is refused (`ports.E_PORT_KIND`)."""
    prog = Program(_pad(2))
    assert PT.facts(prog.root).node(2).one.kind == "position"
    params, ret = _pad(2).signature()
    undeclared = CompositeDefinition("SoftmaxRowPadTestUndeclared", 1, (), lambda S: (params, ret),
                                     lambda B, S, q, k, n: B.call(_pad(2), q, k, n), register=False)
    with pytest.raises(PT.PortError, match="takes kind=position") as e:
        Program(undeclared)
    assert e.value.code == "E_PORT_KIND"


@pytest.mark.parametrize("n", [0, 4, -1, True, 1.0])
def test_the_reference_refuses_a_key_count_outside_the_row(n) -> None:
    q, k = _heads(np.random.default_rng(1), 3, 3)
    with pytest.raises(ValueError, match="key count"):
        SR.qk_softmax_row_pad_reference(q, k, n, BG.GEMM_HOPPER["DOT"])


@pytest.mark.parametrize("target", [bind(SR.SoftmaxRow, N=1), bind(SR.SoftmaxRow, N=3), _qk(1), _qk(3),
                                    bind(SR.SoftmaxRowPad, N=3), _pad(3)], ids=lambda t: t.id)
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
