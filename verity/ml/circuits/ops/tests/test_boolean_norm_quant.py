"""`verity.ml.circuits.ops.boolean.norm_quant`: at the frozen shape's statics (S = 128 chunk sums in G = 2 groups, N = 16,384,
EPS = 1e-5) and parts of C words, each fused block is its unfused chain word for word, `RmsNormOutSums_v1` into
`RowAbsMax_v2` or `QuantChunkS8_v2`'s codes (and the toy's word Definitions, `RowAbsMax_v1` and `QuantChunkS8_v1`), on rows of
activations, special words and uniform words, under the part's own amax and under special and uniform amax words; the toy's
fused row, `RowAbsMax_v2` of the parts' amaxes, then `QuantScaleS8` and the parts' codes under it, is the unfused row's amax,
codes and scale; `QuantScaleS8` then `QuantCodeS8` a word is `QuantChunkS8_v2`; and what `RmsNormOutSums_v1` refuses each
refuses."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import quant as Q
from verity.ml.circuits.ops.boolean import norm_quant as NQ
from verity.ml.circuits.ops.boolean import quant as BQ
from verity.ml.circuits.ops.boolean import rmsnorm as RN
from verity.ml.circuits.types import Array

FROZEN = {"S": 128, "G": 2, "N": 16384, "EPS": 1e-5}
#: signed zeros, the smallest and largest subnormals, 1.0, the largest finite, infinities and NaNs, each of either sign
BF16_SPECIALS = np.array([m | s for m in (0x0000, 0x0001, 0x007F, 0x3F80, 0x7F7F, 0x7F80, 0x7F81, 0x7FC0)
                          for s in (0, 0x8000)], np.uint16)
F32_SPECIALS = np.array([m | s for m in (0x00000000, 0x00000001, 0x3F800000, 0x7F7FFFFF, 0x7F800000, 0x7FC00000)
                         for s in (0, 0x80000000)], np.uint32)


def _bf16(rng, shape, lo: int, hi: int) -> np.ndarray:
    """bf16 words of either sign with exponent fields in [lo, hi)."""
    return (rng.integers(0, 2, shape) << 15 | rng.integers(lo, hi, shape) << 7 | rng.integers(0, 128, shape)).astype(np.uint16)


def _inputs(rng, n: int, C: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` rows of (x, w, s): activations with chunk sums of a 16,384-word row's size, then rows with a special word in x or
    w or s, then uniform words."""
    k = n // 4
    x, w = _bf16(rng, (n, C), 112, 136), _bf16(rng, (n, C), 120, 130)
    s = (rng.integers(130, 140, (n, FROZEN["S"])).astype(np.uint32) << 23) | rng.integers(0, 1 << 23, (n, FROZEN["S"])).astype(
        np.uint32)
    r = np.arange(k)
    x[k + r, rng.integers(0, C, k)] = rng.choice(BF16_SPECIALS, k)
    w[k + r, rng.integers(0, C, k)] = rng.choice(BF16_SPECIALS, k)
    s[k + r, rng.integers(0, FROZEN["S"], k)] = rng.choice(F32_SPECIALS, k)
    x[3 * k:] = rng.integers(0, 1 << 16, (n - 3 * k, C))
    w[3 * k:] = rng.integers(0, 1 << 16, (n - 3 * k, C))
    s[3 * k:] = rng.integers(0, 1 << 32, (n - 3 * k, FROZEN["S"]), dtype=np.uint64).astype(np.uint32)
    return x, w, s


def _words(defn, *args) -> np.ndarray:
    return np.asarray(BS.evaluate_words(defn, *args)).astype(np.uint32)


@pytest.mark.slow
@pytest.mark.parametrize("C", [8, 3])
def test_each_fused_block_is_its_unfused_chain_at_the_frozen_statics(C: int) -> None:
    rng = np.random.default_rng(20 + C)
    x, w, s = _inputs(rng, 400, C)
    out = _words(bind(RN.RmsNormOutSums, C=C, **FROZEN), x, w, s).reshape(-1, C).astype(np.uint16)
    amax = _words(bind(BQ.RowAbsMax, N=C, RELU=False), out).reshape(-1)
    assert np.array_equal(amax, E.evaluate_batch(bind(Q.RowAbsMax, N=C, RELU=False), out).astype(np.uint32).reshape(-1))
    assert np.array_equal(_words(bind(NQ.RmsNormAbsMax, C=C, **FROZEN), x, w, s).reshape(-1), amax)
    amax = amax.astype(np.uint16)
    amax[100:200] = rng.choice(BF16_SPECIALS, 100)
    amax[200:300] = rng.integers(0, 1 << 16, 100)
    want = _words(bind(BQ.QuantChunkS8, N=C, RELU=False), out, amax).reshape(-1, C + 1)
    assert np.array_equal(E.evaluate_batch(bind(Q.QuantChunkS8, N=C, RELU=False), out, amax).astype(np.uint32), want)
    got = _words(bind(NQ.RmsNormQuantCodes, C=C, **FROZEN), x, w, s, amax).reshape(-1, C)
    assert np.array_equal(got, want[:, :C]), np.argwhere(got != want[:, :C])[:5]
    assert np.array_equal(_words(BQ.QuantScaleS8, amax).reshape(-1), want[:, C])


@pytest.mark.parametrize("P,M", [(4, 4), (3, 2)])
def test_the_toys_fused_row_is_the_unfused_rows_amax_codes_and_scale(P: int, M: int) -> None:
    """A row of M parts of P words, as the toy's ``norm`` fusion proves it: `RowAbsMax_v2{M, false}` of each part's
    `RmsNormAbsMax`, `QuantScaleS8` of that, and each part's `RmsNormQuantCodes` under it, against `RmsNormOutSums_v1` over
    the row into `RowAbsMax_v2` and `QuantChunkS8_v2` over the row."""
    rng = np.random.default_rng(40 + P)
    C = P * M
    x, w, s = _inputs(rng, 24, C)
    x[:4, :P] = BF16_SPECIALS[[0, 1, 10, 14]][:, None]
    out = _words(bind(RN.RmsNormOutSums, C=C, **FROZEN), x, w, s).reshape(-1, C).astype(np.uint16)
    amax = _words(bind(BQ.RowAbsMax, N=C, RELU=False), out).reshape(-1)
    want = _words(bind(BQ.QuantChunkS8, N=C, RELU=False), out, amax.astype(np.uint16)).reshape(-1, C + 1)
    parts = [slice(j * P, (j + 1) * P) for j in range(M)]
    partial = np.stack([_words(bind(NQ.RmsNormAbsMax, C=P, **FROZEN), x[:, p], w[:, p], s).reshape(-1) for p in parts], 1)
    row = _words(bind(BQ.RowAbsMax, N=M, RELU=False), partial.astype(np.uint16)).reshape(-1)
    assert np.array_equal(row, amax)
    codes = np.concatenate([_words(bind(NQ.RmsNormQuantCodes, C=P, **FROZEN), x[:, p], w[:, p], s, row.astype(np.uint16))
                            .reshape(-1, P) for p in parts], 1)
    assert np.array_equal(codes, want[:, :C])
    assert np.array_equal(_words(BQ.QuantScaleS8, row.astype(np.uint16)).reshape(-1), want[:, C])


def test_the_scale_then_a_code_a_word_is_the_chunk() -> None:
    """Every sign and exponent of amax, at the mantissa's ends and a random one, over words of every kind."""
    rng = np.random.default_rng(31)
    se = np.arange(512, dtype=np.uint16) << 7
    amax = np.concatenate([se, se | 0x7F, se | rng.integers(0, 0x80, 512).astype(np.uint16)])
    x = np.concatenate([_bf16(rng, (len(amax), 4), 112, 136), rng.integers(0, 1 << 16, (len(amax), 4)).astype(np.uint16)],
                       axis=1)
    x[:16, 0] = BF16_SPECIALS
    want = _words(bind(BQ.QuantChunkS8, N=8, RELU=False), x, amax).reshape(-1, 9)
    scale = _words(BQ.QuantScaleS8, amax).reshape(-1)
    assert np.array_equal(scale, want[:, 8])
    for j in range(8):
        assert np.array_equal(_words(BQ.QuantCodeS8, x[:, j], scale).reshape(-1), want[:, j]), j


def test_the_signatures_and_what_each_refuses() -> None:
    S = SimpleNamespace
    sig = NQ.RmsNormAbsMax.signature(S(C=2048, **FROZEN))
    assert [n for n, _ in sig[0]] == ["x", "w", "s"] and sig[0][0][1].n == 2048 and sig[0][2][1].n == 128 and sig[1] == RN.BF16
    sig = NQ.RmsNormQuantCodes.signature(S(C=512, **FROZEN))
    assert [n for n, _ in sig[0]] == ["x", "w", "s", "amax"] and sig[1] == Array(512, BQ.CODE)
    for bad in (S(C=4, S=6, G=4, N=24, EPS=1e-5), S(C=0, S=6, G=2, N=24, EPS=1e-5), S(C=4, S=6, G=2, N=24, EPS=0.0),
                S(C=4, S=6, G=2, N=0, EPS=1e-5)):
        for d in (NQ.RmsNormAbsMax, NQ.RmsNormQuantCodes):
            with pytest.raises(ValueError):
                d.signature(bad)
