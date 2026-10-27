"""`verity.ml.circuits.ops.boolean.rmsnorm`: each Definition is its stated dataflow, the word primitives each step names composed
in order (`F32Mul_v2`, `F32Add_v2`, `F32Div_v2`, `Bf16MulF32_v1`, `Bf16MulBf16_v1`, and IEEE's correctly rounded square root
with the GPU's NaN), on rows of activations, uniform words, special words, and words whose squares overflow or underflow; the
copied square root is sqrt.rn.f32; the two-level form sums in its stated grouping and, at one chunk and G = 1, is the row's
function; `RmsSumSqChunks`' chunks are `RmsSumSqPart`s and `RmsNormOutSums` is the four-part composition word for word; a row of
activations is RMSNorm within bf16's rounding; and a row, chunk or sum count below 1, a count that doesn't divide its whole, or
an EPS whose f32 is zero or infinite is refused."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean import rmsnorm as R

GPU_NAN = 0x7FFFFFFF
#: signed zeros, the smallest and largest subnormals, the smallest normal, 1.0 and its neighbours, 2, the largest finite,
#: infinities, a signalling and two quiet NaNs, each of either sign
BF16_SPECIALS = np.array([m | s for m in (0x0000, 0x0001, 0x007F, 0x0080, 0x3F7F, 0x3F80, 0x3F81, 0x4000, 0x7F7F, 0x7F80,
                                          0x7F81, 0x7FC0, 0x7FFF) for s in (0, 0x8000)], np.uint16)
F32_SPECIALS = np.array([m | s for m in (0x00000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3F7FFFFF, 0x3F800000, 0x3F800001,
                                         0x40000000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF)
                         for s in (0, 0x80000000)], np.uint32)
EPS = (1e-5, 1e-6)


def _bf16(rng, shape, lo: int, hi: int) -> np.ndarray:
    """bf16 words of either sign with exponent fields in [lo, hi)."""
    s = rng.integers(0, 2, shape) << 15
    return (s | rng.integers(lo, hi, shape) << 7 | rng.integers(0, 128, shape)).astype(np.uint16)


def _rows(rng, n: int, N: int) -> np.ndarray:
    """``n`` rows of N bf16 words: activations of one scale (magnitudes in [2^-3, 2^3), so the squares' sum rounds at most
    steps and its order shows), activations of many (in [2^-15, 2^9)), uniform words, activations with one special word, words
    whose squares overflow, and words whose squares are subnormal or zero."""
    k = n // 8
    one = _bf16(rng, (k, N), 112, 136)
    one[np.arange(k), rng.integers(0, N, k)] = rng.choice(BF16_SPECIALS, k)
    return np.concatenate([_bf16(rng, (n - 5 * k, N), 124, 130), _bf16(rng, (k, N), 112, 136),
                           rng.integers(0, 1 << 16, (k, N)).astype(np.uint16), one, _bf16(rng, (k, N), 191, 255),
                           _bf16(rng, (k, N), 0, 64)])


def _f32(h: np.ndarray) -> np.ndarray:
    return h.astype(np.uint32) << np.uint32(16)


def _ev(defn, *args) -> np.ndarray:
    return np.asarray(E.evaluate_batch(defn, *args)).astype(np.uint32).reshape(-1)


def _const(w: int, n: int) -> np.ndarray:
    return np.full(n, w, np.uint32)


def _sqrt(u: np.ndarray) -> np.ndarray:
    """IEEE's correctly rounded f32 square root (numpy's), every NaN the GPU's."""
    with np.errstate(invalid="ignore"):
        r = np.sqrt(u.astype(np.uint32).view(np.float32)).view(np.uint32)
    return np.where(np.isnan(r.view(np.float32)), np.uint32(GPU_NAN), r)


def _sum(words: list[np.ndarray]) -> np.ndarray:
    acc = _const(0, len(words[0]))
    for s in words:
        acc = _ev(W.F32AddV2, acc, s)
    return acc


def _squares(x: np.ndarray) -> list[np.ndarray]:
    return [_ev(W.F32MulV2, _f32(x[:, i]), _f32(x[:, i])) for i in range(x.shape[1])]


def _recip(acc: np.ndarray, N: int, eps: float) -> np.ndarray:
    n = len(acc)
    mean = _ev(W.F32DivV2, acc, _const(R.f32_bits(N), n))
    return _ev(W.F32DivV2, _const(R.ONE_F, n), _sqrt(_ev(W.F32AddV2, mean, _const(R.f32_bits(eps), n))))


def _out(x: np.ndarray, rs: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.stack([_ev(W.Bf16MulBf16, w[:, j], _ev(W.Bf16MulF32, x[:, j], rs)) for j in range(x.shape[1])], axis=1)


def _words(target, *args) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *args)).astype(np.uint32)


def test_the_copied_root_is_sqrt_rn_with_the_gpu_nan() -> None:
    """Special words, every exponent field with random significands, exact squares and uniform words."""
    rng = np.random.default_rng(1)
    fields = (np.arange(256, dtype=np.uint32).repeat(64) << np.uint32(23)) | rng.integers(0, 1 << 23, 256 * 64).astype(np.uint32)
    squares = (np.arange(1, 4097, dtype=np.float32) ** 2).view(np.uint32)
    a = np.concatenate([F32_SPECIALS, fields, fields | np.uint32(0x80000000), squares,
                        rng.integers(0, 1 << 32, 50_000, dtype=np.uint64).astype(np.uint32)])
    probe = definition("RmsNormSqrtProbe", 1, R.f32_sqrt_rn, (("a", R.F32),), R.F32, register=False)
    assert np.array_equal(_words(probe, a).reshape(-1), _sqrt(a))


@pytest.mark.parametrize("N", [1, 2, 3, 8])
@pytest.mark.parametrize("eps", EPS)
def test_a_row_is_the_reference_dataflow(N: int, eps: float) -> None:
    rng = np.random.default_rng(N * 10 + EPS.index(eps))
    x, w = _rows(rng, 2500, N), np.concatenate([_bf16(rng, (2000, N), 112, 136), _rows(rng, 500, N)])
    want = _out(x, _recip(_sum(_squares(x)), N, eps), w)
    assert np.array_equal(_words(bind(R.RmsNormRow, N=N, EPS=eps), x, w).reshape(-1, N), want)


def test_a_nan_in_the_row_makes_every_output_the_f2fp_nan() -> None:
    rng = np.random.default_rng(2)
    x, w = _bf16(rng, (64, 5), 112, 136), _bf16(rng, (64, 5), 112, 136)
    x[np.arange(64), rng.integers(0, 5, 64)] = rng.choice([0x7F81, 0x7FC0, 0xFFFF, 0xFFC0], 64)
    assert (_words(bind(R.RmsNormRow, N=5, EPS=1e-5), x, w) == W.BF16_NAN_F2FP).all()


def test_the_parts_are_their_steps() -> None:
    """Each part on its own: a chunk's squares summed, G words summed, the reciprocal root of G sums, an output chunk."""
    rng = np.random.default_rng(3)
    x, w = _rows(rng, 2500, 4), _rows(rng, 2500, 4)
    sq = _squares(x)
    assert np.array_equal(_words(bind(R.RmsSumSqPart, C=4), x).reshape(-1), _sum(sq))
    s = np.stack(sq, 1)
    s[:300, 0] = rng.choice(F32_SPECIALS, 300)
    s[300:600, 3] = rng.integers(0, 1 << 32, 300, dtype=np.uint64).astype(np.uint32)
    assert np.array_equal(_words(bind(R.F32SumPart, G=4), s).reshape(-1), _sum(list(s.T)))
    for G in (1, 2, 4):
        got = _words(bind(R.RmsNormRecip, N=1000, G=G, EPS=1e-5), s[:, :G]).reshape(-1)
        assert np.array_equal(got, _recip(_sum(list(s[:, :G].T)), 1000, 1e-5)), G
    rs = np.concatenate([_recip(_sum(sq), 4, 1e-5)[:2000], rng.choice(F32_SPECIALS, 500)])
    assert np.array_equal(_words(bind(R.RmsNormOut, C=4), x, rs, w).reshape(-1, 4), _out(x, rs, w))


def test_the_two_level_form_sums_in_its_stated_grouping() -> None:
    """N = 24 as Phase 2's shape in small: chunks of C = 4, groups of 3 chunk sums, G = 2; each chunk's squares from +0, each
    group's chunk sums from +0, the G group sums from +0."""
    rng = np.random.default_rng(4)
    N, C, per = 24, 4, 3
    x, w = _rows(rng, 2000, N), _rows(rng, 2000, N)
    sq = _squares(x)
    chunks = [_sum(sq[i:i + C]) for i in range(0, N, C)]
    groups = [_sum(chunks[g:g + per]) for g in range(0, len(chunks), per)]
    rs = _recip(_sum(groups), N, 1e-5)
    got_chunks = [_words(bind(R.RmsSumSqPart, C=C), x[:, i:i + C]).reshape(-1) for i in range(0, N, C)]
    assert all(np.array_equal(a, b) for a, b in zip(got_chunks, chunks, strict=True))
    got_groups = [_words(bind(R.F32SumPart, G=per), np.stack(got_chunks[g:g + per], 1)).reshape(-1)
                  for g in range(0, len(chunks), per)]
    assert all(np.array_equal(a, b) for a, b in zip(got_groups, groups, strict=True))
    got_rs = _words(bind(R.RmsNormRecip, N=N, G=2, EPS=1e-5), np.stack(got_groups, 1)).reshape(-1)
    assert np.array_equal(got_rs, rs)
    out = np.concatenate([_words(bind(R.RmsNormOut, C=C), x[:, i:i + C], got_rs, w[:, i:i + C]).reshape(-1, C)
                          for i in range(0, N, C)], axis=1)
    assert np.array_equal(out, _out(x, rs, w))


@pytest.mark.parametrize("N", [1, 3, 8])
def test_one_chunk_and_one_sum_is_the_row(N: int) -> None:
    rng = np.random.default_rng(5 + N)
    x, w = _rows(rng, 2500, N), _rows(rng, 2500, N)
    rs = _words(bind(R.RmsNormRecip, N=N, G=1, EPS=1e-5), _words(bind(R.RmsSumSqPart, C=N), x).reshape(-1, 1)).reshape(-1)
    two = _words(bind(R.RmsNormOut, C=N), x, rs, w).reshape(-1, N)
    assert np.array_equal(two, _words(bind(R.RmsNormRow, N=N, EPS=1e-5), x, w).reshape(-1, N))


def test_sum_chunks_are_sum_parts() -> None:
    """Chunk i of `RmsSumSqChunks{C, K}` is `RmsSumSqPart{K}` of words iK to iK + K - 1; at K = C it is `RmsSumSqPart{C}`."""
    rng = np.random.default_rng(7)
    x = _rows(rng, 2000, 12)
    for K in (1, 3, 4, 12):
        got = _words(bind(R.RmsSumSqChunks, C=12, K=K), x).reshape(-1, 12 // K)
        want = np.stack([_words(bind(R.RmsSumSqPart, C=K), x[:, i:i + K]).reshape(-1) for i in range(0, 12, K)], 1)
        assert np.array_equal(got, want), K


@pytest.mark.parametrize("G", [2, 3])
def test_the_fused_output_unit_is_the_four_part_composition(G: int) -> None:
    """N = 24 in S = 6 chunk sums of 4 words, in G groups: each output chunk of `RmsNormOutSums` is `RmsNormOut`'s at
    `RmsNormRecip` of `F32SumPart`'s group sums, word for word, at an output chunk as wide as a sum's (C = 4) and twice as wide
    (C = 8). The chunk sums are the row's own, special or uniform words, or words that absorb and cancel (1, 2 and ±2^25), so
    that summing a group, or the group sums, in another order changes rs by far more than bf16's rounding hides."""
    rng = np.random.default_rng(8 + G)
    N, K, S = 24, 4, 6
    x, w = _rows(rng, 3000, N), _rows(rng, 3000, N)
    s = _words(bind(R.RmsSumSqChunks, C=N, K=K), x).reshape(-1, S)
    s[:200, 1] = rng.choice(F32_SPECIALS, 200)
    s[200:400, 4] = rng.integers(0, 1 << 32, 200, dtype=np.uint64).astype(np.uint32)
    s[1000:] = rng.choice(np.array([1, 2, 2 ** 25, -2 ** 25], np.float32).view(np.uint32), (2000, S))
    per = S // G
    groups = [_words(bind(R.F32SumPart, G=per), s[:, g:g + per]).reshape(-1) for g in range(0, S, per)]
    rs = _words(bind(R.RmsNormRecip, N=N, G=G, EPS=1e-5), np.stack(groups, 1)).reshape(-1)
    for C in (4, 8):
        fused = bind(R.RmsNormOutSums, C=C, S=S, G=G, N=N, EPS=1e-5)
        for i in range(0, N, C):
            want = _words(bind(R.RmsNormOut, C=C), x[:, i:i + C], rs, w[:, i:i + C]).reshape(-1, C)
            assert np.array_equal(_words(fused, x[:, i:i + C], w[:, i:i + C], s).reshape(-1, C), want), (C, i)


def test_a_row_of_activations_is_rmsnorm_within_bf16_rounding() -> None:
    """Against float64's ``w * x / sqrt(mean(x^2) + eps)``: two bf16 roundings and a few f32 ones, so within 2^-7 relative."""
    rng = np.random.default_rng(6)
    N = 16
    x, w = _bf16(rng, (3000, N), 112, 136), _bf16(rng, (3000, N), 120, 130)
    got = _f32(_words(bind(R.RmsNormRow, N=N, EPS=1e-5), x, w).reshape(-1, N).astype(np.uint16)).view(np.float32)
    xf, wf = _f32(x).view(np.float32).astype(np.float64), _f32(w).view(np.float32).astype(np.float64)
    exact = wf * xf / np.sqrt((xf ** 2).mean(axis=1, keepdims=True) + np.float32(1e-5))
    assert np.all(np.abs(got - exact) <= 2.0 ** -7 * np.abs(exact))


def test_outputs_fill_their_size_class_and_bad_statics_are_refused() -> None:
    S = SimpleNamespace
    assert R.RmsNormRow.signature(S(N=16384, EPS=1e-5))[1].n == 16384
    assert R.RmsNormOut.signature(S(C=128))[1].n == 128
    assert R.RmsNormOutSums.signature(S(C=128, S=128, G=2, N=16384, EPS=1e-5))[1].n == 128
    assert R.RmsSumSqChunks.signature(S(C=2048, K=128))[1].n == 16
    for d, s in ((R.RmsSumSqPart, S(C=128)), (R.F32SumPart, S(G=64)), (R.RmsNormRecip, S(N=16384, G=2, EPS=1e-5))):
        assert d.signature(s)[1] == R.F32
    for bad in (S(N=0, EPS=1e-5), S(N=True, EPS=1e-5), S(N=4, EPS=0.0), S(N=4, EPS=-1e-5), S(N=4, EPS=1e-50),
                S(N=4, EPS=1e39), S(N=4, EPS=1), S(N=4, EPS=float("nan"))):
        with pytest.raises(ValueError):
            R.RmsNormRow.signature(bad)
    for d, bad in ((R.RmsSumSqPart, S(C=0)), (R.F32SumPart, S(G=0)), (R.RmsNormRecip, S(N=4, G=0, EPS=1e-5)),
                   (R.RmsNormOut, S(C=-1)), (R.RmsSumSqChunks, S(C=12, K=5)), (R.RmsSumSqChunks, S(C=12, K=0)),
                   (R.RmsNormOutSums, S(C=4, S=6, G=4, N=24, EPS=1e-5)), (R.RmsNormOutSums, S(C=0, S=6, G=2, N=24, EPS=1e-5)),
                   (R.RmsNormOutSums, S(C=4, S=6, G=2, N=24, EPS=0.0))):
        with pytest.raises(ValueError):
            d.signature(bad)
