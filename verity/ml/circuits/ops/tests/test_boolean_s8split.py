"""`verity.ml.circuits.ops.boolean.s8split`: each accumulator half is `GemmAccumulatorS8_v1{K, GEMM_SM90_S8}`'s 16 bits and
the exact sum; each sum half is the wrapping s32 sum's 16 bits; the epilogue is `S32ToF32Rn_v1`, two `F32Mul_v2` and
`F2fpBf16_v1` on the parts' sum; the split, each unit reading the 16-bit words the one before output, is
`ScaledMmS8Coordinate_v2{K, GEMM_SM90_S8}` at K = 512 and 1,024 (with a level of group sums), on uniform and extreme codes;
each is Boolean with 16 output bits; and a third half, a K that isn't a multiple of 32 below 2^17, or a G below 1 is refused."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import s8split as SS

#: signed zeros, the least subnormal, the largest finite, infinities, a quiet and a signalling NaN
F32_SPECIALS = np.array([0, 0x80000000, 1, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001], np.uint32)
#: 0, ±1, the s32 edges and their neighbours
S32_EDGES = np.array([0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x7FFFFFFE, 0x80000000, 0x80000001], np.uint32)


def _codes(rng, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform s8 codes; every fourth instance's from {-128, -127, 127} (the largest products), and the first instance all
    -128 (the largest sum)."""
    x = rng.integers(0, 256, (n, k), dtype=np.uint16).astype(np.uint8)
    w = rng.integers(0, 256, (n, k), dtype=np.uint16).astype(np.uint8)
    big = np.array([0x80, 0x81, 0x7F], np.uint8)
    x[::4], w[::4] = rng.choice(big, x[::4].shape), rng.choice(big, w[::4].shape)
    x[0], w[0] = 0x80, 0x80
    return x, w


def _scales(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    """An int8 linear's token and channel scales, uniform f32 words and special words."""
    pick = rng.integers(0, 3, (2, n))
    sx = np.choose(pick[0], [(np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32),
                             rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32), rng.choice(F32_SPECIALS, n)])
    sw = np.choose(pick[1], [(np.abs(rng.standard_normal(n)) * 0.02 / 127).astype(np.float32).view(np.uint32),
                             rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32), rng.choice(F32_SPECIALS, n)])
    return sx.astype(np.uint32), sw.astype(np.uint32)


def _words(target, *args) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *args)).astype(np.uint32).reshape(-1)


def _ev(defn, *args) -> np.ndarray:
    return np.asarray(E.evaluate_batch(defn, *args)).astype(np.uint32).reshape(-1)


def _halves(target, *args) -> list[np.ndarray]:
    return [_words(bind(target, HALF=h), *args) for h in (0, 1)]


def _join(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo | hi << np.uint32(16)


def _split(full: np.ndarray) -> list[np.ndarray]:
    return [(full >> np.uint32(16 * h)) & np.uint32(0xFFFF) for h in (0, 1)]


@pytest.mark.parametrize("K", [32, 256])
def test_an_accumulator_half_is_the_sm90_accumulators_16_bits_and_the_exact_sum(K: int) -> None:
    rng = np.random.default_rng(K)
    x, w = _codes(rng, 1500, K)
    want = _ev(bind(G.GemmAccumulatorS8, K=K, **G.GEMM_SM90_S8), x, w)
    exact = (x.view(np.int8).astype(np.int64) * w.view(np.int8)).sum(1)
    assert np.array_equal(want.view(np.int32), exact)
    got = _halves(bind(SS.S8AccumulatorPartHalf, K=K), x, w)
    assert all(np.array_equal(g, h) for g, h in zip(got, _split(want), strict=True))


@pytest.mark.parametrize("G_", [1, 3, 64])
def test_a_sum_half_is_the_wrapping_s32_sums_16_bits(G_: int) -> None:
    """Uniform words (whose sum wraps) and the s32 edges."""
    rng = np.random.default_rng(100 + G_)
    s = np.where(rng.integers(0, 3, (2000, G_)) == 0, rng.choice(S32_EDGES, (2000, G_)),
                 rng.integers(0, 1 << 32, (2000, G_), dtype=np.uint64).astype(np.uint32)).astype(np.uint32)
    want = (s.astype(np.uint64).sum(1) % (1 << 32)).astype(np.uint32)
    got = _halves(bind(SS.I32SumPartHalf, G=G_), s)
    assert all(np.array_equal(g, h) for g, h in zip(got, _split(want), strict=True))


@pytest.mark.parametrize("G_", [1, 2, 4])
def test_the_epilogue_scales_the_parts_sum_as_the_coordinate_does(G_: int) -> None:
    rng = np.random.default_rng(200 + G_)
    n = 3000
    s = np.where(rng.integers(0, 4, (n, G_)) == 0, rng.choice(S32_EDGES, (n, G_)),
                 rng.integers(-(1 << 22), 1 << 22, (n, G_)).astype(np.int32).view(np.uint32)).astype(np.uint32)
    sx, sw = _scales(rng, n)
    acc = (s.astype(np.uint64).sum(1) % (1 << 32)).astype(np.uint32)
    want = _ev(P.F2fpBf16, _ev(W.F32MulV2, _ev(W.F32MulV2, _ev(P.S32ToF32Rn, acc), sx), sw))
    assert np.array_equal(_words(bind(SS.ScaledMmS8Epilogue, G=G_), s, sx, sw), want)


@pytest.mark.parametrize("K,per", [(512, 0), (1024, 2)])
def test_the_split_is_the_unsplit_coordinate(K: int, per: int) -> None:
    """Chunks of 256 codes, each chunk's two accumulator halves, then (with ``per``) groups of ``per`` chunk sums summed by
    `I32SumPartHalf`, then the epilogue over what's left, every s32 read back from the halves the units before output."""
    rng = np.random.default_rng(300 + K)
    n = 400
    x, w = _codes(rng, n, K)
    sx, sw = _scales(rng, n)
    parts = [_join(*_halves(bind(SS.S8AccumulatorPartHalf, K=256), x[:, i:i + 256], w[:, i:i + 256]))
             for i in range(0, K, 256)]
    if per:
        parts = [_join(*_halves(bind(SS.I32SumPartHalf, G=per), np.stack(parts[g:g + per], 1)))
                 for g in range(0, len(parts), per)]
    got = _words(bind(SS.ScaledMmS8Epilogue, G=len(parts)), np.stack(parts, 1), sx, sw)
    assert np.array_equal(got, _words(bind(BG.ScaledMmS8Coordinate, K=K, **BG.GEMM_SM90_S8), x, sx, w, sw))


@pytest.mark.parametrize("target", [bind(SS.S8AccumulatorPartHalf, K=64, HALF=0), bind(SS.S8AccumulatorPartHalf, K=64, HALF=1),
                                    bind(SS.I32SumPartHalf, G=2, HALF=0), bind(SS.I32SumPartHalf, G=2, HALF=1),
                                    bind(SS.ScaledMmS8Epilogue, G=2)],
                         ids=lambda t: t.id)
def test_each_is_boolean_with_16_output_bits(target) -> None:
    assert BL.is_boolean(target)
    assert BL.elements(target.signature()[1]) == [16]


def test_a_third_half_a_k_off_the_step_or_past_the_exact_limit_and_no_sums_are_refused() -> None:
    for d, ok in ((SS.S8AccumulatorPartHalf, {"K": 64}), (SS.I32SumPartHalf, {"G": 2})):
        with pytest.raises(ValueError, match="HALF is 0"):
            bind(d, **ok, HALF=2).signature()
    for K in (0, 16, 48, G.S8_K_LIMIT, "256"):
        with pytest.raises(ValueError):
            bind(SS.S8AccumulatorPartHalf, K=K, HALF=0).signature()
    for g in (0, "2"):
        with pytest.raises(ValueError, match="G is the number"):
            bind(SS.I32SumPartHalf, G=g, HALF=0).signature()
        with pytest.raises(ValueError, match="G is the number"):
            bind(SS.ScaledMmS8Epilogue, G=g).signature()
