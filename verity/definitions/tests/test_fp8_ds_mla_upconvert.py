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

SMALL = {"DN": 128, "DR": 2}


def _row(nope, scale, rope):
    return [int(v) for v in nope], [int(v) for v in scale], [int(v) for v in rope]


@pytest.mark.parametrize("b, s, want, why", M.HAND_VECTORS, ids=[h[3].split(":")[0] for h in M.HAND_VECTORS])
def test_hand_vectors(b, s, want, why):
    """Each vector: the exact reference, the Definition (every NoPE value of a DN = 128 row) and its numpy kernel."""
    assert M.reference_word(b, s) == want, why
    F = bind(M.Fp8DsMlaUpconvertRow, **SMALL)
    rope = [0x1234, 0xFFFF]
    out = evaluate(F, *_row([b] * 128, [s], rope))
    assert list(out) == [want] * 128 + rope, why
    k, _ = M.upconvert_rows(np.full((1, 128), b, np.uint8), np.array([[s]], np.uint32), np.array([rope], np.uint16))
    assert k.tolist() == [[want] * 128 + rope], why


def test_every_code_against_the_exact_reference():
    """All 256 codes at each of `EDGE_SCALES`, one scale per 128 values of a DN = 256 row, through the Definition."""
    F = bind(M.Fp8DsMlaUpconvertRow, DN=256, DR=1)
    for s0, s1 in zip(M.EDGE_SCALES[0::2], M.EDGE_SCALES[1::2]):
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
