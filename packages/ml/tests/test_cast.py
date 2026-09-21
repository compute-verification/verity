"""The word casts of `verity_ml.tc.cast` against the measured `cvt.rn.bf16.f32` campaign and an exhaustive NaN sweep.

Ported from veritor ``tests/veritor/core/test_tc_nan.py`` (139-181) and ``verity_ir/tests/test_nan_conversion.py`` (41-64);
the numpy array forms were not carried over, so the sweeps are pure integer loops (the 2^32 sweep stays behind).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity_ml.tc.cast import (
    BF16_NAN_HARDWARE,
    BF16_NAN_SOFTWARE,
    bf16_is_nan,
    bf16_to_f32_word,
    f32_is_nan,
    f32_to_bf16_hw_word,
    f32_to_bf16_rn_word,
    f32_to_bf16_word,
)
from verity_ml.tc.total import cvt_rn_bf16_f32

MEASURED = sorted((Path(__file__).parent / "fixtures" / "tc-total-2026-09-07").glob("*/cvt_special.json"))

NAN_WORDS = [0x7FC00000, 0xFFC00000, 0x7FC00001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001, 0xFF800001, 0x7FBFFFFF,
             0xFFA00000, 0x7F808000, 0x7FC07FFF, 0x7FC08000]
FINITE_WORDS = [0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x3F800000, 0x3F808000, 0x3F818000, 0x3F808001,
                0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000]


def test_the_two_conversions_differ_exactly_on_nan() -> None:
    rng = random.Random(8)
    for x in NAN_WORDS + FINITE_WORDS + [rng.getrandbits(32) for _ in range(20000)]:
        hw, sw = f32_to_bf16_hw_word(x), f32_to_bf16_word(x)
        assert hw == cvt_rn_bf16_f32(x), hex(x)
        if f32_is_nan(x):
            assert (sw, hw) == (BF16_NAN_SOFTWARE, BF16_NAN_HARDWARE) == (0x7FC0, 0x7FFF), hex(x)
            assert bf16_is_nan(sw) and bf16_is_nan(hw)
        else:
            assert sw == hw, hex(x)
            assert not bf16_is_nan(hw)                       # a non-NaN word never rounds to a NaN (past-max -> Inf)
            assert not f32_is_nan(bf16_to_f32_word(hw))      # and widening preserves the class


def test_known_roundings() -> None:
    assert f32_to_bf16_word(0x3F808000) == 0x3F80          # tie -> even (down)
    assert f32_to_bf16_word(0x3F818000) == 0x3F82          # tie -> even (up)
    assert f32_to_bf16_word(0x3F808001) == 0x3F81          # above the tie
    assert f32_to_bf16_word(0x7F7FFFFF) == 0x7F80          # largest finite carries into an infinity
    assert f32_to_bf16_word(0xFF7FFFFF) == 0xFF80
    assert f32_to_bf16_word(0x7F800000) == 0x7F80          # infinities pass
    assert f32_to_bf16_word(0x007FFFFF) == 0x0080          # a subnormal rounds up into the smallest normal
    assert bf16_to_f32_word(0x3F80) == 0x3F800000
    assert f32_to_bf16_rn_word(0x7FC00000, nan=0x7E55) == 0x7E55  # the NaN word is a parameter


@pytest.mark.parametrize("path", MEASURED, ids=lambda p: p.parent.name)
def test_hw_cast_matches_every_measured_cvt_word(path: Path) -> None:
    """L6's `cvt.rn.bf16.f32` campaign: 4403 words per device, 84 of them NaN, 0 non-deterministic."""
    record = json.loads(path.read_text())
    assert "cvt.rn.bf16.f32" in record["instruction"]
    assert record["n_nondeterministic"] == 0
    words = record["words"]
    assert len(words) == record["n"] == 4403
    nan_words = 0
    for entry in words:
        x, measured = int(entry["x"], 16), int(entry["bf16"], 16)
        assert f32_to_bf16_hw_word(x) == measured, entry["name"]
        if f32_is_nan(x):
            nan_words += 1
            assert measured == BF16_NAN_HARDWARE, entry["name"]
            assert f32_to_bf16_word(x) == BF16_NAN_SOFTWARE, entry["name"]
        else:
            assert f32_to_bf16_word(x) == measured, entry["name"]
    assert nan_words == 84


def test_every_nan_payload_of_one_sign_and_a_band_of_finite_words() -> None:
    """All 2^23 - 1 positive NaN payloads exhaustively (the negative half differs only in the sign bit, which both casts
    discard on NaN), plus every word of a few exponent bands where the casts must agree."""
    for payload in range(1, 1 << 23):
        x = 0x7F800000 | payload
        assert f32_to_bf16_hw_word(x) == 0x7FFF and f32_to_bf16_word(x) == 0x7FC0
    for band in (0x00000000, 0x3F800000, 0x7F700000, 0xBF800000):
        for x in range(band, band + (1 << 16)):
            assert f32_to_bf16_hw_word(x) == f32_to_bf16_word(x) == f32_to_bf16_rn_word(x, 0), hex(x)
