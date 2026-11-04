"""The total step (`verity.ml.circuits.silicon.total`) against the A100 / RTX 4090 special-value campaign, the H100 campaign, and the vector files.

Ported from veritor ``tests/veritor/core/test_silicon_total.py``; the gate-set identity assertions became direct calls to
``tc_dot_total`` (the gate sets were not ported).  Fixtures under ``fixtures/`` (see its README).
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import struct

import pytest

from verity.ml.circuits.silicon.cast import f32_to_bf16_word
from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16, tc_dot
from verity.ml.circuits.silicon.total import (
    AMPERE_BF16_DOT16_TOTAL,
    FP32_CANONICAL_NAN,
    HOPPER_BF16_WGMMA_K16,
    Inf,
    NaN,
    bf16_product_total,
    cvt_rn_bf16_f32,
    fp32_value,
    tc_dot_total,
)

FIXTURES = Path(__file__).parent / "fixtures"
VECTORS = FIXTURES / "vectors"
DATA = FIXTURES / "tc-total-2026-09-07"

ONE, MAX, INF, NEG_INF, QNAN, SNAN = 0x3F80, 0x7F7F, 0x7F80, 0xFF80, 0x7FC0, 0x7F81
P = AMPERE_BF16_DOT16_TOTAL


def _row(fill: int, at: dict[int, int]) -> list[int]:
    r = [fill] * 16
    for k, w in at.items():
        r[k] = w
    return r


def test_registered_pipeline_is_the_ampere_finite_model() -> None:
    assert P is AMPERE_BF16_M16N8K16


def test_products() -> None:
    assert bf16_product_total(QNAN, ONE) == NaN()
    assert bf16_product_total(ONE, SNAN) == NaN()
    assert bf16_product_total(0xFFFF, INF) == NaN()
    assert bf16_product_total(INF, 0) == NaN()
    assert bf16_product_total(0x8000, NEG_INF) == NaN()
    assert bf16_product_total(INF, 0x0001) == Inf(False)  # subnormal is not a zero
    assert bf16_product_total(INF, 0x8001) == Inf(True)
    assert bf16_product_total(NEG_INF, NEG_INF) == Inf(False)
    assert bf16_product_total(INF, MAX) == Inf(False)
    assert bf16_product_total(ONE, ONE).magnitude != 0
    assert fp32_value(0x7FC00001) == NaN()
    assert fp32_value(0xFF800001) == NaN()
    assert fp32_value(0xFF800000) == Inf(True)


def test_special_cases_measured_on_a100_and_rtx4090() -> None:
    assert tc_dot_total(P, 0, [INF] * 16, [INF] * 16) == 0x7F800000
    assert tc_dot_total(P, 0, [INF] * 16, [NEG_INF] * 16) == 0xFF800000
    assert (
        tc_dot_total(P, 0, _row(0, {0: INF}), _row(ONE, {0: 0})) == FP32_CANONICAL_NAN
    )
    assert tc_dot_total(P, 0, _row(0, {0: INF}), _row(ONE, {0: 1})) == 0x7F800000
    assert (
        tc_dot_total(P, 0, _row(0, {0: INF, 1: NEG_INF}), [ONE] * 16)
        == FP32_CANONICAL_NAN
    )
    assert (
        tc_dot_total(P, 0, _row(0, {0: INF, 8: NEG_INF}), [ONE] * 16)
        == FP32_CANONICAL_NAN
    )
    assert tc_dot_total(P, 0xFFC00001, [0] * 16, [0] * 16) == FP32_CANONICAL_NAN
    assert (
        tc_dot_total(P, 0x7F800000, _row(0, {15: NEG_INF}), [ONE] * 16)
        == FP32_CANONICAL_NAN
    )
    assert tc_dot_total(P, 0x7F800000, [MAX | 0x8000] * 16, [MAX] * 16) == 0x7F800000
    # every NaN payload and sign collapses to the one canonical word
    for bit in range(7):
        for sign in (0, 1):
            w = (sign << 15) | (0xFF << 7) | (1 << bit)
            assert tc_dot_total(P, 0, _row(0, {3: w}), [ONE] * 16) == FP32_CANONICAL_NAN
    # zero results are +0
    assert tc_dot_total(P, 0x80000000, [0x8000] * 16, [ONE] * 16) == 0
    assert tc_dot_total(P, 0x80000000, [0x8000] * 16, [0x8000] * 16) == 0
    assert tc_dot_total(P, 0xBF800000, _row(0, {15: ONE}), [ONE] * 16) == 0


def test_saturated_infinity_is_an_ordinary_infinity_for_the_next_group() -> None:
    a = _row(0, {0: MAX})
    b = _row(ONE, {0: MAX})
    # group 1 saturates to +inf ...
    assert tc_dot_total(P, 0, a, b) == 0x7F800000
    # ... and meets -inf, NaN, +inf, or finite terms in group 2
    assert tc_dot_total(P, 0, _row(0, {0: MAX, 8: NEG_INF}), b) == FP32_CANONICAL_NAN
    assert tc_dot_total(P, 0, _row(0, {0: MAX, 8: QNAN}), b) == FP32_CANONICAL_NAN
    assert tc_dot_total(P, 0, _row(0, {0: MAX, 8: INF}), b) == 0x7F800000
    assert (
        tc_dot_total(
            P, 0, _row(0, {0: MAX, 8: MAX | 0x8000}), _row(ONE, {0: MAX, 8: MAX})
        )
        == 0x7F800000
    )
    # within one group an operand infinity dominates an overflowing finite sum
    assert tc_dot_total(P, 0, _row(0, {0: MAX, 1: NEG_INF}), b) == 0xFF800000
    # the finite model returned +inf here without examining group 2; the hardware says NaN
    old_a, old_b = [MAX] * 8 + [QNAN] * 8, [MAX] * 8 + [INF] * 8
    assert tc_dot(P, 0, old_a, old_b) == 0x7F800000
    assert tc_dot_total(P, 0, old_a, old_b) == FP32_CANONICAL_NAN


def test_agrees_with_the_finite_model_on_finite_operands() -> None:
    rng = random.Random(0x7C7)

    def bf16() -> int:
        return (
            (rng.randrange(2) << 15) | (rng.randint(0, 254) << 7) | rng.randrange(128)
        )

    for _ in range(3000):
        acc = (
            (rng.randrange(2) << 31)
            | (rng.randint(0, 254) << 23)
            | rng.randrange(1 << 23)
        )
        a = [bf16() for _ in range(16)]
        b = [bf16() for _ in range(16)]
        assert tc_dot_total(P, acc, a, b) == tc_dot(P, acc, a, b)


def test_total_on_every_class_and_shape_checks() -> None:
    for acc in (
        0,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0xFFFFFFFF,
        0x7F800001,
        0x00000001,
    ):
        for w in (0, 0x8000, 1, 0x7F, ONE, MAX, INF, NEG_INF, QNAN, SNAN, 0xFFFF):
            out = tc_dot_total(P, acc, [w] * 16, [ONE] * 16)
            assert 0 <= out < 1 << 32
    with pytest.raises(InvalidArtifact):
        tc_dot_total(P, 0, [ONE] * 15, [ONE] * 16)
    with pytest.raises(InvalidArtifact):
        tc_dot_total(P, 0, [1 << 16] + [ONE] * 15, [ONE] * 16)
    with pytest.raises(InvalidArtifact):
        tc_dot_total(P, 1 << 32, [ONE] * 16, [ONE] * 16)


def test_cvt_rn_bf16_f32() -> None:
    def pinned(x: int) -> int:
        return f32_to_bf16_word(x)

    for x in (
        0x3F800000,
        0x3F808000,
        0x3F818000,
        0x3F807FFF,
        0x00000001,
        0x00008001,
        0x7F7FFFFF,
        0x80000000,
    ):
        assert cvt_rn_bf16_f32(x) == pinned(x)
    assert cvt_rn_bf16_f32(0x7F7FFFFF) == 0x7F80  # rounds up to +inf
    assert cvt_rn_bf16_f32(0xFF800000) == 0xFF80
    for nan in (0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF, 0x7FA5A5A5):
        assert cvt_rn_bf16_f32(nan) == 0x7FFF
        assert pinned(nan) == 0x7FC0  # the pinned gate's (rejected-input) NaN differs


@pytest.mark.parametrize(
    "name", ["tc-ampere-bf16-total.json", "tc-ampere-bf16-total-hw.json"]
)
def test_vector_files(name: str) -> None:
    """Every evaluation of the @3 gate-set vector files through `tc_dot_total`; `tc_dot16` takes (acc, 16 a, 16 b),
    `tc_dot16_0` pins the accumulator to +0.  Source gates (`in`, `weight`) and out-of-width arguments carry `out: null`."""
    path = VECTORS / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    doc = json.loads(path.read_text())
    assert doc["gate_set"]["id"] == "veritor.tensor-core.ampere_bf16_m16n8k16@3"
    widths = {"tc_dot16": (32, *([16] * 32)), "tc_dot16_0": (16,) * 32}
    n = 0
    for e in doc["evaluations"]:
        w = widths.get(e["op"])
        if w is None or len(e["args"]) != len(w) or any(not 0 <= v < 1 << b for v, b in zip(e["args"], w, strict=True)):
            assert e["out"] is None
            continue
        args = e["args"]
        got = tc_dot_total(P, args[0], args[1:17], args[17:]) if e["op"] == "tc_dot16" else tc_dot_total(P, 0, args[:16], args[16:])
        assert got == e["out"], e
        for column in ("a100", "rtx4090"):
            if column in e:
                assert int(e[column], 16) == got, (column, e)
        n += 1
    assert n > 1000


def test_cvt_campaign_records() -> None:
    files = sorted(DATA.glob("*/cvt_special.json"))
    if not files:
        pytest.skip("campaign records not present")
    for path in files:
        doc = json.loads(path.read_text())
        assert doc["n_nondeterministic"] == 0
        for w in doc["words"]:
            assert cvt_rn_bf16_f32(int(w["x"], 16)) == int(w["bf16"], 16), (path, w)


def test_mma_campaign_records_target_words() -> None:
    files = sorted(DATA.glob("*/mma_special.reduced.json.gz"))
    if not files:
        pytest.skip("campaign records not present")
    for path in files:
        with gzip.open(path, "rt") as f:
            doc = json.load(f)
        assert doc["n_nondeterministic"] == 0
        for c in doc["cases"]:
            a = [int(c["a"][i : i + 4], 16) for i in range(0, 64, 4)]
            b = [int(c["b"][i : i + 4], 16) for i in range(0, 64, 4)]
            assert tc_dot_total(P, int(c["acc"], 16), a, b) == int(c["d"], 16), (
                path,
                c["name"],
            )
        for c in doc["pos"]:
            assert c["positions"] == 128 and len(c["d"]) == 1, c["name"]
            a = [int(c["a"][i : i + 4], 16) for i in range(0, 64, 4)]
            b = [int(c["b"][i : i + 4], 16) for i in range(0, 64, 4)]
            assert tc_dot_total(P, int(c["acc"], 16), a, b) == int(c["d"][0], 16), (
                path,
                c["name"],
            )


# --- Hopper: veritor.tensor-core.hopper_bf16_wgmma_k16@1 (veritor's H100 campaign, gpu/tc-hopper) -------

HP = HOPPER_BF16_WGMMA_K16
HDATA = FIXTURES / "tc-hopper-2026-09-07"


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bf16(x: float) -> int:
    return _f32(x) >> 16  # exact for the powers of two used here


def test_hopper_registered_family() -> None:
    assert HP.groups == (16,) and HP.width == 26 and HP.zero_exponent == -133 and HP.k == 16
    # same finite parameters as Hawkeye's H100 pipeline, now re-measured
    assert (HP.groups, HP.width, HP.zero_exponent) == (
        HOPPER_BF16_M16N8K16.groups,
        HOPPER_BF16_M16N8K16.width,
        HOPPER_BF16_M16N8K16.zero_exponent,
    )


def test_hopper_one_group_of_sixteen_versus_ampere_two_of_eight() -> None:
    """The neutral-subgroup measurement: +-2^20 cancelling inside k = 0..7 with a 2^-20 product in every other
    slot.  Hopper sums all sixteen in one group aligned to 2^20, so every small term is truncated (result 0);
    Ampere's first group cancels to 0 and its second group then adds the eight small terms exactly."""
    big, small = 0x4980, 0x3580  # bf16 2^20, 2^-20
    a = [small] * 16
    a[0], a[1] = big, big | 0x8000  # +2^20, -2^20
    b = [ONE] * 16
    assert tc_dot_total(HP, 0, a, b) == 0
    assert tc_dot_total(AMPERE_BF16_DOT16_TOTAL, 0, a, b) == _f32(8 * 2.0**-20)


def test_hopper_adder_keeps_26_bits() -> None:
    """`1*1 + 1*(-1) + 2^-c`: the residual survives for c <= 25 (26-bit adder) and is lost at c = 26."""
    for c, kept in ((25, True), (26, False)):
        up, fl = -(c // 2), -c + (c // 2)  # up + fl == -c
        a = [0] * 16
        b = [0] * 16
        a[0], b[0] = ONE, ONE
        a[1], b[1] = ONE, ONE | 0x8000
        a[2], b[2] = _bf16(2.0**up), _bf16(2.0**fl)
        got = tc_dot_total(HP, 0, a, b)
        assert (got == _f32(2.0**-c)) is kept, (c, hex(got))
    # Ampere keeps one bit fewer
    a = [0] * 16
    b = [0] * 16
    a[0], b[0], a[1], b[1] = ONE, ONE, ONE, ONE | 0x8000
    a[2], b[2] = _bf16(2.0**-12), _bf16(2.0**-13)
    assert tc_dot_total(HP, 0, a, b) == _f32(2.0**-25)
    assert tc_dot_total(AMPERE_BF16_DOT16_TOTAL, 0, a, b) != _f32(2.0**-25)


def test_hopper_special_values_follow_the_total_rules() -> None:
    assert tc_dot_total(HP, 0, _row(ONE, {3: QNAN}), [ONE] * 16) == FP32_CANONICAL_NAN
    assert tc_dot_total(HP, 0, _row(ONE, {3: INF}), [ONE] * 16) == 0x7F800000
    assert tc_dot_total(HP, 0, _row(ONE, {3: INF, 9: NEG_INF}), [ONE] * 16) == FP32_CANONICAL_NAN
    assert tc_dot_total(HP, 0, _row(ONE, {3: INF}), _row(ONE, {3: 0})) == FP32_CANONICAL_NAN
    assert tc_dot_total(HP, 0x80000000, [0x8000] * 16, [ONE] * 16) == 0  # zero is always +0


def test_hopper_campaign_records() -> None:
    """Every recorded word of the reduced H100 record reproduces (the full 307M-word check is the pod's
    ``conformance_hopper.json``, echoed in ``summary.json``)."""
    path = HDATA / "special.reduced.json.gz"
    if not path.exists():
        pytest.skip("campaign record not present")
    with gzip.open(path, "rt") as f:
        doc = json.load(f)
    assert doc["kernel_identity"]["sass_hgmma"] == ["HGMMA.64x128x16.F32.BF16"]
    assert doc["tiles_differing_between_reps"] == 0
    n = 0
    for fam, cases in doc["families"].items():
        for c in cases:
            a = [int(c["a"][i : i + 4], 16) for i in range(0, 64, 4)]
            b = [int(c["b"][i : i + 4], 16) for i in range(0, 64, 4)]
            assert tc_dot_total(HP, int(c["acc"], 16), a, b) == int(c["d"], 16), (fam, c)
            n += 1
    assert n == doc["records"]
    summary = json.loads((HDATA / "summary.json").read_text())
    assert summary["conformance"]["mismatches"] == 0
    assert summary["fit_grid"]["zero_mismatch"] == ["16:26:-133"]
