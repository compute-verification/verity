"""The Hopper FP8 total step (`verity.ml.circuits.silicon.total_fp8`).

On finite operands the total step is the finite model (`silicon.tc_dot` on the same pipeline); the special encodings follow
the rules of the BF16 total step restricted to what E4M3 encodes (NaN bytes, no infinities).  The pinned pipeline parameters
are the ones measured on the H100 (replicated as
`verity/ml/circuits/silicon/captures/veritor-tc-fp8-2026-09-08.json`): one group of 32, 14-bit adder, floor -139.
"""

from __future__ import annotations

import random
import struct

import pytest

from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.silicon import HOPPER_E4M3_K32, tc_dot
from verity.ml.circuits.silicon.total import FP32_CANONICAL_NAN, HOPPER_BF16_WGMMA_K16
from verity.ml.circuits.silicon.total_fp8 import (
    HOPPER_E4M3_FAMILY_VERSION,
    HOPPER_E4M3_WGMMA_K32,
    e4m3_class,
    tc_dot_total_e4m3,
)

FP32_INF = 0x7F800000


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _finite_e4m3(rng: random.Random) -> int:
    while True:
        b = rng.randrange(256)
        if e4m3_class(b) != "nan":
            return b


def test_pipeline_is_the_pinned_one():
    p = HOPPER_E4M3_WGMMA_K32
    assert (p.name, p.arch, p.dtype, p.operand_bits) == ("hopper_e4m3_wgmma_k32", "sm_90", "e4m3", 8)
    assert (p.groups, p.width, p.zero_exponent) == ((32,), 14, -139)
    assert (p.groups, p.width, p.zero_exponent) == (HOPPER_E4M3_K32.groups, HOPPER_E4M3_K32.width, HOPPER_E4M3_K32.zero_exponent)
    assert p.k == 32
    assert HOPPER_E4M3_FAMILY_VERSION == "1"


def test_e4m3_classes():
    assert e4m3_class(0x7F) == e4m3_class(0xFF) == "nan"
    assert e4m3_class(0x00) == e4m3_class(0x80) == "zero"
    assert e4m3_class(0x01) == e4m3_class(0x7E) == e4m3_class(0xFE) == "finite"
    with pytest.raises(InvalidArtifact):
        e4m3_class(256)


def test_finite_operands_agree_with_the_finite_model():
    rng = random.Random(2026_09_08)
    for _ in range(2000):
        a = [_finite_e4m3(rng) for _ in range(32)]
        b = [_finite_e4m3(rng) for _ in range(32)]
        acc = _f32(rng.uniform(-1.0, 1.0) * 2.0 ** rng.randint(-20, 20))
        assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, acc, a, b) == tc_dot(HOPPER_E4M3_K32, acc, a, b)


def test_zero_products_still_pass_the_accumulator_through_the_adder():
    # the `accid` fact: the accumulator is truncated to the 14-bit adder like a product, so acc + 0 != acc for most words
    zeros = [0] * 32
    changed = sum(tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, acc, zeros, zeros) != acc
                  for acc in (_f32(1.0 + k / 4097.0) for k in range(1, 200)))
    assert changed > 150
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, _f32(1.0), zeros, zeros) == _f32(1.0)
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, 0, zeros, zeros) == 0
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, 0x80000000, zeros, zeros) == 0   # zero is always +0


def test_special_values():
    one = 0x38   # E4M3 1.0
    ones = [one] * 32
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, 0, ones, ones) == _f32(32.0)
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, 0, [0x7F] + ones[1:], ones) == FP32_CANONICAL_NAN
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, 0, ones, ones[:-1] + [0xFF]) == FP32_CANONICAL_NAN
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, 0x7FC00000, ones, ones) == FP32_CANONICAL_NAN
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, FP32_INF, ones, ones) == FP32_INF
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, FP32_INF | 0x80000000, ones, ones) == FP32_INF | 0x80000000
    # products are bounded (448^2 * 32 < 2^23), so a large finite accumulator stays finite and follows the finite model
    big = 0x7F7FFFFF
    assert tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, big, [0x7E] * 32, [0x7E] * 32) == tc_dot(HOPPER_E4M3_K32, big, [0x7E] * 32, [0x7E] * 32)


def test_wrong_pipeline_or_arity_fails_closed():
    with pytest.raises(InvalidArtifact):
        tc_dot_total_e4m3(HOPPER_BF16_WGMMA_K16, 0, [0] * 16, [0] * 16)
    with pytest.raises(InvalidArtifact):
        tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, 0, [0] * 31, [0] * 32)
