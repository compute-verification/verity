"""`registry.torch_words` and `registry.boolean_torch_words`: each word primitive against torch CPU (the int64 and uint8 arithmetic
and casts with wraparound, the int64 shifts by every amount from below 0 to past 63, int64 -> f32 around every rounding position, f32 -> int64 and -> uint8 at every exponent from below 1 to
past 2^63, f32 -> f16 at every exponent from below f16's subnormals to past 65504 around each rounding position, and every f16 word
to f32), the NaN and saturated words against the GPU's as the primitives define them, the order keys sorting values as torch's
radix select does and inverting; and each Boolean version against its word on special and random words."""
from __future__ import annotations

import itertools
import zlib

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.definitions.registry import boolean_torch_words as BW
from verity.definitions.registry import torch_words as W

M64 = (1 << 64) - 1
#: 0, ±1, the ends of int64, both sides of 2^24 (f32's last odd integer) and its ties, 2^53 + 1, and words that round up to 2^63
_I64 = [0, 1, -1, (1 << 63) - 1, -(1 << 63), (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + 3, (1 << 25) + 2, (1 << 25) + 6,
        (1 << 53) + 1, (1 << 63) - (1 << 39), (1 << 63) - (1 << 39) - 1, -((1 << 24) + 1), -((1 << 24) + 3), 255, 256, -256, -255]
#: ±0, the f32 subnormals' ends, the f16 subnormals' ends and their halves, 1, 65504, 65519.996, 65520 (rounds to infinity), the
#: largest finite f32, infinity and NaNs (quiet, signalling, all ones)
_F32 = [0x00000000, 0x00000001, 0x007FFFFF, 0x33800000, 0x33000000, 0x33000001, 0x387FC000, 0x38800000, 0x3F800000, 0x477FE000,
        0x477FEFFF, 0x477FF000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF]
F32_SPECIALS = np.array([w | s for w in _F32 for s in (0, 0x80000000)], np.uint32)


def _i64(values) -> np.ndarray:
    return np.array([v & M64 for v in values], np.uint64)


def _random_i64(rng, n: int) -> np.ndarray:
    """n int64 words of every bit length, both signs."""
    bits = rng.integers(0, 64, n)
    mag = np.array([int(rng.integers(0, 1 << 62)) >> (62 - min(b, 62)) if b else 0 for b in bits], dtype=object)
    sign = rng.integers(0, 2, n)
    return _i64([-int(m) if s else int(m) for m, s in zip(mag, sign)])


def _signed(w: np.ndarray) -> np.ndarray:
    return w.astype(np.uint64).view(np.int64)


def _f16_range(rng) -> np.ndarray:
    """Every exponent field from 101 (below half f16's smallest subnormal) to 143 (past 65504), each with the mantissa's low 14 bits
    (the rounding position of a normal f16 is bit 13, a subnormal's higher) at 0, 1, all ones, the tie and its neighbours, or one
    of 16 random patterns, under 32 random high parts; both signs."""
    e = np.arange(101, 144, dtype=np.uint32)[:, None, None] << np.uint32(23)
    hi = rng.integers(0, 1 << 9, 32, dtype=np.uint64).astype(np.uint32)[None, :, None] << np.uint32(14)
    lo = np.concatenate([np.array([0, 1, 0x3FFF, 0x1000, 0x0FFF, 0x1001, 0x2000, 0x1FFF, 0x2001], np.uint32),
                         rng.integers(0, 1 << 14, 16, dtype=np.uint64).astype(np.uint32)])
    w = (e | hi | lo[None, None, :]).reshape(-1)
    return np.concatenate([w, w | np.uint32(0x80000000)])


def _i64_range(rng) -> np.ndarray:
    """Every exponent field from 120 (below 1) to 195 (past 2^63), each under 64 significands (all zeros, all ones and random);
    both signs."""
    e = np.arange(120, 196, dtype=np.uint32)[:, None] << np.uint32(23)
    m = rng.integers(0, 1 << 23, (1, 64), dtype=np.uint64).astype(np.uint32)
    m[0, :2] = (0, 0x7FFFFF)
    w = (e | m).reshape(-1)
    return np.concatenate([w, w | np.uint32(0x80000000)])


def _nan32(w: np.ndarray) -> np.ndarray:
    return ((w & 0x7F800000) == 0x7F800000) & ((w & 0x7FFFFF) != 0)


def _same(got, want, what: str) -> None:
    got, want = np.asarray(got, np.uint64).reshape(-1), np.asarray(want, np.uint64).reshape(-1)
    bad = np.flatnonzero(got != want)
    assert got.shape == want.shape and not bad.size, f"{what}: {bad.size} differ, first {[(hex(got[i]), hex(want[i])) for i in bad[:5]]}"


# --- the words against torch CPU ------------------------------------------------------------------------------------------------

def test_the_int64_words_are_torchs() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    a = np.concatenate([_i64(_I64), _random_i64(rng, 4000)])
    b = np.concatenate([_i64(_I64[::-1]), _random_i64(rng, 4000)])
    ta, tb = torch.from_numpy(_signed(a).copy()), torch.from_numpy(_signed(b).copy())
    _same(E.evaluate_batch(W.I64Add, a, b), (ta + tb).numpy().view(np.uint64), "I64Add")
    _same(E.evaluate_batch(W.I64Le, a, b), (ta <= tb).numpy(), "I64Le")
    _same(E.evaluate_batch(W.I64And, a, b), (ta & tb).numpy().view(np.uint64), "I64And")
    c = rng.integers(0, 2, a.size, dtype=np.uint8)
    _same(E.evaluate_batch(W.SelectI64, c, a, b), torch.where(torch.from_numpy(c.astype(bool)), ta, tb).numpy().view(np.uint64),
          "SelectI64")
    _same(E.evaluate_batch(W.I64ToU8, a), ta.to(torch.uint8).numpy(), "I64ToU8")
    u = np.arange(256, dtype=np.uint8)
    _same(E.evaluate_batch(W.U8ToI64, u), torch.from_numpy(u).to(torch.int64).numpy().view(np.uint64), "U8ToI64")


def _shift_operands(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    """`_I64` and n random words, each shifted by every amount from -2 to 66, then n random words by random words."""
    a = np.concatenate([np.repeat(_i64(_I64), 69), _random_i64(rng, n)])
    b = np.concatenate([np.tile(_i64(range(-2, 67)), len(_I64)), _i64(rng.integers(-2, 67, n).tolist())])
    return np.concatenate([a, _random_i64(rng, n)]), np.concatenate([b, _random_i64(rng, n)])


def test_the_int64_shifts_are_torchs() -> None:
    """`<<` and `>>` of tensors and by a Python int, in torch's vector body and its scalar tail (the length is no multiple of 8):
    `<<` 0 for an amount below 0 or past 63, `>>` every bit the sign for one below 0 or past 62."""
    torch = pytest.importorskip("torch")
    a, b = _shift_operands(np.random.default_rng(6), 4001)
    ta, tb = torch.from_numpy(_signed(a).copy()), torch.from_numpy(_signed(b).copy())
    _same(E.evaluate_batch(W.I64Shl, a, b), (ta << tb).numpy().view(np.uint64), "I64Shl")
    _same(E.evaluate_batch(W.I64Shr, a, b), (ta >> tb).numpy().view(np.uint64), "I64Shr")
    for k in (0, 3, 8, 56, 63):
        _same(E.evaluate_batch(W.I64Shl, a, np.full(a.size, k, np.uint64)), (ta << k).numpy().view(np.uint64), f"I64Shl by {k}")
        _same(E.evaluate_batch(W.I64Shr, a, np.full(a.size, k, np.uint64)), (ta >> k).numpy().view(np.uint64), f"I64Shr by {k}")


def test_an_int64_rounds_to_f32_as_torch_rounds_it() -> None:
    """Ties to even at every bit length, both signs, and the words past 2^63 - 2^39 that round up to 2^63."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    ties = [s * ((((1 << 23) | int(rng.integers(0, 1 << 23))) << k) | (d << (k - 1)))
            for k in range(1, 40) for d in (0, 1) for s in (1, -1)]
    t = np.concatenate([_i64(_I64 + ties), _random_i64(rng, 20000)])
    want = torch.from_numpy(_signed(t).copy()).to(torch.float32).numpy().view(np.uint32)
    _same(E.evaluate_batch(W.I64ToF32Rn, t), want, "I64ToF32Rn")


def test_f32_truncates_to_int64_as_torch_cpu_below_2_63() -> None:
    """torch CPU's conversion to int64, and to uint8 (its low byte: c10 converts through int64), wherever x is no NaN and below
    2^63 in magnitude or -2^63; a NaN 0, from 2^63 up INT64_MAX and below -2^63 INT64_MIN, the GPU's saturation."""
    torch = pytest.importorskip("torch")
    x = np.concatenate([F32_SPECIALS, _i64_range(np.random.default_rng(4))])
    ok = (~_nan32(x) & (np.abs(x.view(np.float32).astype(np.float64)) < 2.0 ** 63)) | (x == 0xDF000000)
    got = np.asarray(E.evaluate_batch(W.F32ToI64Rz, x)).astype(np.uint64)
    t = torch.from_numpy(x[ok].view(np.float32).copy())
    _same(got[ok], t.to(torch.int64).numpy().view(np.uint64), "F32ToI64Rz")
    _same(got[ok] & np.uint64(0xFF), t.to(torch.uint8).numpy(), "F32ToI64Rz's low byte")
    nan, big = _nan32(x), ~_nan32(x) & ~ok
    _same(got[nan], np.zeros(int(nan.sum())), "F32ToI64Rz on NaNs")
    _same(got[big], [1 << 63 if w >> 31 else (1 << 63) - 1 for w in x[big].tolist()], "F32ToI64Rz past 2^63")


def test_f32_rounds_to_f16_as_torch_rounds_it_but_a_nan() -> None:
    """Every non-NaN word of `_f16_range` and the specials, in torch's vector body and in its scalar tail (the tensor's length is
    no multiple of 8); a NaN is 0x7FFF, which torch's CPU keeps the payload of in one path and not the other."""
    torch = pytest.importorskip("torch")
    x = np.concatenate([F32_SPECIALS, _f16_range(np.random.default_rng(2))])
    keep = x[~_nan32(x)]
    keep = np.concatenate([keep, keep[:5]])
    want = torch.from_numpy(keep.view(np.float32).copy()).to(torch.float16).numpy().view(np.uint16)
    _same(E.evaluate_batch(W.F32ToF16Rn, keep), want, "F32ToF16Rn")
    _same(E.evaluate_batch(W.F32ToF16Rn, x[_nan32(x)]), np.full(int(_nan32(x).sum()), W.F16_NAN), "F32ToF16Rn on NaNs")


def test_every_f16_widens_as_torch_widens_it_but_a_nan() -> None:
    torch = pytest.importorskip("torch")
    h = np.arange(1 << 16, dtype=np.uint16)
    nan = ((h & 0x7C00) == 0x7C00) & ((h & 0x3FF) != 0)
    want = torch.from_numpy(h[~nan].view(np.float16).copy()).to(torch.float32).numpy().view(np.uint32)
    _same(E.evaluate_batch(W.F16ToF32, h[~nan]), want, "F16ToF32")
    _same(E.evaluate_batch(W.F16ToF32, h[nan]), np.full(int(nan.sum()), W.F32_NAN), "F16ToF32 on NaNs")


def test_the_order_keys_sort_values_and_invert() -> None:
    """Sorted by key, the values never decrease, -0 comes before +0 and every NaN (key 0xFFFFFFFF) last; the inverse of each
    non-NaN key is its word, and of the NaN key the GPU's NaN."""
    rng = np.random.default_rng(3)
    x = np.concatenate([F32_SPECIALS, rng.integers(0, 1 << 32, 20000, dtype=np.uint64).astype(np.uint32)])
    k = E.evaluate_batch(W.F32OrderKey, x).astype(np.uint32)
    nan = _nan32(x)
    assert (k[nan] == W.M32).all() and (k[~nan] < W.M32).all()
    order = np.argsort(k[~nan], kind="stable")
    v, s = x[~nan][order].view(np.float32), x[~nan][order]
    assert (v[1:] >= v[:-1]).all()
    zero = (s & 0x7FFFFFFF) == 0
    assert not (zero[1:] & zero[:-1] & (s[:-1] == 0) & (s[1:] == 0x80000000)).any()
    _same(E.evaluate_batch(W.OrderKeyF32, k[~nan]), x[~nan], "OrderKeyF32 of F32OrderKey")
    _same(E.evaluate_batch(W.OrderKeyF32, np.array([W.M32], np.uint32)), [W.F32_NAN], "OrderKeyF32 of the NaN key")


# --- the Boolean versions against the words --------------------------------------------------------------------------------------

def _inputs(d, rng) -> list[np.ndarray]:
    n = 20000
    if d in (BW.I64Shl, BW.I64Shr):
        return list(_shift_operands(rng, n))
    if d in (BW.I64Add, BW.I64Le, BW.I64And, BW.SelectI64):
        grid = np.array(list(itertools.product(range(len(_I64)), repeat=2)), np.int64).T
        a, b = (np.concatenate([_i64(_I64)[g], _random_i64(rng, n)]) for g in grid)
        return [rng.integers(0, 2, a.size, dtype=np.uint8), a, b] if d is BW.SelectI64 else [a, b]
    if d in (BW.I64ToU8, BW.I64ToF32Rn):
        return [np.concatenate([_i64(_I64), _random_i64(rng, n)])]
    if d is BW.U8ToI64:
        return [np.arange(256, dtype=np.uint8)]
    if d is BW.F16ToF32:
        return [np.arange(1 << 16, dtype=np.uint16)]
    rand = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    if d is BW.F32ToF16Rn:
        extra = _f16_range(rng)
    elif d is BW.F32ToI64Rz:
        extra = _i64_range(rng)
    else:
        extra = np.array([W.M32, 0x80000000, 0x7FFFFFFF], np.uint32)
    return [np.concatenate([F32_SPECIALS, rand, extra])]


@pytest.mark.parametrize("d", [BW.I64Add, BW.I64Le, BW.I64Shl, BW.I64Shr, BW.I64And, BW.SelectI64, BW.I64ToU8, BW.U8ToI64,
                               BW.I64ToF32Rn, BW.F32ToI64Rz, BW.F32ToF16Rn, BW.F16ToF32, BW.F32OrderKey, BW.OrderKeyF32],
                         ids=lambda d: d.name)
def test_the_boolean_version_is_its_word(d) -> None:
    target = bind(d)
    word = TR.word_of(target)
    assert (word.name, word.version) == (d.name, 1)
    inputs = _inputs(d, np.random.default_rng(zlib.crc32(d.name.encode())))
    _same(BS.evaluate_words(target, *inputs), E.evaluate_batch(word, *inputs), target.id)
