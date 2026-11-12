"""`registry.fp8_ds_mla_upconvert.Fp8DsMlaUpconvertRow_v1`, the dense prefill's read-back of an `fp8_ds_mla` row, on vectors
worked by hand, on every E4M3 code against the exact reference, and its numpy kernel against the Definition."""
from __future__ import annotations

import struct
from fractions import Fraction

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate, self_check
from verity.definitions.registry import fp8_ds_mla_upconvert as M
from verity.definitions.silicon.cast import e4m3_to_f32_word

ONE, NEG_ONE, INF, NAN = 0x3F800000, 0xBF800000, 0x7F800000, 0x7FC00000

#: (E4M3 byte, f32 scale word, bf16 word the kernel writes, what it shows), each worked by hand
HAND = [
    (0x7E, ONE, 0x43E0, "largest code: 448 = 1.75 * 2^8"),
    (0xFE, ONE, 0xC3E0, "largest negative code: -448"),
    (0x01, ONE, 0x3B00, "smallest code: 2^-9, an E4M3 subnormal, a bf16 normal"),
    (0x81, ONE, 0xBB00, "smallest negative code: -2^-9"),
    (0x07, ONE, 0x3C60, "largest E4M3 subnormal: 7/8 * 2^-6"),
    (0x08, ONE, 0x3C80, "smallest E4M3 normal: 2^-6"),
    (0x00, ONE, 0x0000, "+0"),
    (0x80, ONE, 0x8000, "-0 keeps its sign through cvt.f32.f16 (HADD2.F32 -RZ + h)"),
    (0x00, NEG_ONE, 0x8000, "+0 * -1 = -0"),
    (0x80, NEG_ONE, 0x0000, "-0 * -1 = +0"),
    (0x7F, ONE, 0x7FFF, "NaN code"),
    (0xFF, ONE, 0x7FFF, "negative NaN code: F2FP writes 0x7FFF whatever the sign and payload"),
    (0x7F, 0x00000000, 0x7FFF, "NaN code times 0"),
    (0x3E, 0x3F808008, 0x3FE1, "scale 1 + 2^-8 + 2^-20, not a power of two: 1.75 * s = 1.7568359... exact in f32, RN to 1.7578125"),
    (0x38, NEG_ONE, 0xBF80, "negative scale: 1 * -1"),
    (0x38, 0x3F808000, 0x3F80, "tie to bf16: 1 + 2^-8 is halfway, to even (down)"),
    (0x38, 0x3F818000, 0x3F82, "tie to bf16: 1 + 2^-7 + 2^-8 is halfway, to even (up)"),
    (0x3C, 0x3FACAAAA, 0x4002, "double rounding: 1.5 * s = 2.0234375 - 2^-23 is an f32 tie, to 2.0234375, a bf16 tie, to "
                               "even 2.03125; rounding the exact product once would give 0x4001"),
    (0x08, 0x00800000, 0x0002, "subnormal product: 2^-6 * 2^-126 = 2^-132, kept (FMUL has no .FTZ)"),
    (0x01, 0x01000000, 0x0000, "subnormal product on a tie: 2^-9 * 2^-125 = 2^-134, half the least bf16 subnormal, to even 0"),
    (0x03, 0x01000000, 0x0002, "subnormal product on a tie: 1.5 * 2^-133, to even 2 * 2^-133"),
    (0x7E, 0x00000200, 0x0004, "subnormal scale 2^-140: 448 * 2^-140 = 3.5 * 2^-133, to even 4 * 2^-133"),
    (0x7E, 0x00400000, 0x0460, "subnormal scale 2^-127, normal product 1.75 * 2^-119 (DAZ would give 0)"),
    (0x7E, 0x7B800000, 0x7F80, "f32 overflow: 448 * 2^120 = 1.75 * 2^128 -> +inf"),
    (0x38, 0x7F7FFFFF, 0x7F80, "finite f32, bf16 overflow: the largest f32 rounds to +inf in bf16"),
    (0x00, INF, 0x7FFF, "0 * inf is NaN"),
    (0x38, NAN, 0x7FFF, "NaN scale"),
    (0xB8, INF, 0xFF80, "-1 * inf = -inf"),
]

SMALL = {"DN": 128, "DR": 2}


def _row(nope, scale, rope):
    return [int(v) for v in nope], [int(v) for v in scale], [int(v) for v in rope]


@pytest.mark.parametrize("b, s, want, why", HAND, ids=[h[3].split(":")[0] for h in HAND])
def test_hand_vectors(b, s, want, why):
    """Each vector: the exact reference, the Definition (every NoPE value of a DN = 128 row) and its numpy kernel."""
    assert M.reference_word(b, s) == want, why
    F = bind(M.Fp8DsMlaUpconvertRow, **SMALL)
    rope = [0x1234, 0xFFFF]
    out = evaluate(F, *_row([b] * 128, [s], rope))
    assert list(out) == [want] * 128 + rope, why
    k, _ = M.upconvert_rows(np.full((1, 128), b, np.uint8), np.array([[s]], np.uint32), np.array([rope], np.uint16))
    assert k.tolist() == [[want] * 128 + rope], why


SCALES = [ONE, NEG_ONE, 0x3F808008, 0x3FACAAAA, 0x3B800000, 0x00800000, 0x00000200, 0x00400000, 0x7B800000, 0x7F7FFFFF,
          0x00000000, 0x80000000, INF, NAN, 0x3E2AAAAB, 0x4C000000]


def test_every_code_against_the_exact_reference():
    """All 256 codes at each of `SCALES`, one scale per 128 values of a DN = 256 row, through the Definition."""
    F = bind(M.Fp8DsMlaUpconvertRow, DN=256, DR=1)
    for s0, s1 in zip(SCALES[0::2], SCALES[1::2]):
        nope = list(range(128)) + list(range(128, 256))
        out = evaluate(F, nope, [s0, s1], [0])
        want = [M.reference_word(b, s0) for b in range(128)] + [M.reference_word(b, s1) for b in range(128, 256)] + [0]
        assert list(out) == want, (hex(s0), hex(s1))
        nope = list(range(128, 256)) + list(range(128))
        out = evaluate(F, nope, [s0, s1], [0])
        want = [M.reference_word(b, s0) for b in range(128, 256)] + [M.reference_word(b, s1) for b in range(128)] + [0]
        assert list(out) == want, (hex(s0), hex(s1))


def test_the_f16_hop_is_exact():
    """`cvt.rn.f16x2.e4m3x2` then `cvt.f32.f16` is `E4m3ToF32` on every non-NaN code: each E4M3 value (subnormals from 2^-9)
    is an f16 normal or zero, kept with its sign."""
    for b in range(256):
        if b & 0x7F == 0x7F:
            continue
        v = M._e4m3_value(b)
        h = np.float16(-float(-v)) if b & 0x80 else np.float16(float(v))
        assert Fraction(float(h)) == v and np.signbit(h) == bool(b & 0x80), hex(b)
        f = np.float32(np.float16(-0.0) + h)
        assert struct.unpack("<I", struct.pack("<f", f))[0] == e4m3_to_f32_word(b), hex(b)


def test_split_rows_reads_the_656_byte_layout():
    DN, DR = M.FP8_DS_MLA_GLM53["DN"], M.FP8_DS_MLA_GLM53["DR"]
    assert M.row_bytes(DN, DR) == 656
    rng = np.random.default_rng(1)
    nope = rng.integers(0, 256, (3, DN), dtype=np.uint8)
    scale = rng.integers(0, 2**32, (3, DN // M.QT), dtype=np.uint32)
    rope = rng.integers(0, 2**16, (3, DR), dtype=np.uint16)
    rows = np.concatenate([nope, scale.astype("<u4").view(np.uint8), rope.astype("<u2").view(np.uint8)], axis=1)
    n, s, r = M.split_rows(rows, DN, DR)
    assert (n == nope).all() and (s == scale).all() and (r == rope).all()
    with pytest.raises(ValueError):
        M.split_rows(rows[:, :-2], DN, DR)


@pytest.mark.parametrize("statics", [M.FP8_DS_MLA_GLM53, SMALL])
def test_numpy_kernel_is_the_definition(statics):
    assert self_check(bind(M.Fp8DsMlaUpconvertRow, **statics), n=48).ok


@pytest.mark.parametrize("statics", [{"DN": 100, "DR": 64}, {"DN": 0, "DR": 64}, {"DN": 512, "DR": 0}])
def test_refuses_rows_off_the_grouping(statics):
    with pytest.raises(Exception, match="must"):
        bind(M.Fp8DsMlaUpconvertRow, **statics)
