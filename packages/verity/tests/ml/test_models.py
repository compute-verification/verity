"""``verity.ml.tc.models``: the ``Model`` protocol, ``GroupSum`` (= ``Pipeline``) and ``BlockScaledAlignAdd``.

The block-scaled model is replayed on every RTX 5090 vector autoproof recorded (``fixtures/tc/autoproof-probe-vectors.json``:
1,601 nvfp4 ascending-K chains and 1,400 unscaled-FP4 dots) and must agree bit-for-bit; the refuted single-window
hypothesis (W=26) must *disagree* somewhere, or the fixture would not discriminate.  The accumulator rules that the
zero-accumulator vectors cannot see (sign of zero, the scale-exponent anchor, the accumulator window) are checked on
hand-computed cases taken from lane fp4-re's RTX 5090 evidence (log entries D16, D17, D18).
"""
from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import pytest

from verity.errors import InvalidArtifact
from verity.ml.tc import models, silicon
from verity.ml.tc.models import (
    ADA_E4M3_M16N8K32,
    ADA_E5M2_M16N8K32,
    AMPERE_BF16_M16N8K16,
    BLACKWELL_SM120_MXF4,
    BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED,
    BLACKWELL_SM120_NVF4,
    HOPPER_E4M3_K32,
    HOPPER_E5M2_K32,
    MODELS,
    PIPELINES,
    BlockScaledAlignAdd,
    GroupSum,
    Model,
    Pipeline,
    tc_dot,
    tc_dot_chain,
)
from verity.ml.tc.term import Term, e2m1_product, e5m2_product

ROOT = Path(__file__).resolve().parents[4]
VECTORS = ROOT / "fixtures" / "tc" / "autoproof-probe-vectors.json"


def _f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


# --- protocol and aliases ------------------------------------------------------------------------------------------------


def test_every_model_satisfies_the_protocol() -> None:
    assert MODELS, "the registry is empty"
    for name, m in MODELS.items():
        assert isinstance(m, Model), name
        assert m.name == name
        assert m.k > 0 and m.words_per_step >= m.k
        assert m.operand_bits in (4, 8, 16)
        assert m.dtype in ("bf16", "e4m3", "e5m2", "e2m1")
        assert isinstance(m.validated, str) and m.validated
        assert isinstance(m.product(0, 0), Term)


def test_pipeline_is_group_sum_and_nothing_moved() -> None:
    assert Pipeline is GroupSum
    assert silicon.Pipeline is GroupSum and silicon.Term is Term
    for name in ("AMPERE_BF16_M16N8K16", "HOPPER_BF16_M16N8K16", "HOPPER_E4M3_K32", "ADA_BF16_M16N8K16",
                 "ADA_E4M3_M16N8K32", "HAWKEYE_AMPERE_GROUPSUM_E4M3_V0"):
        assert getattr(silicon, name) is getattr(models, name) is PIPELINES[getattr(models, name).name]
    assert silicon.tc_dot is tc_dot and silicon.tc_dot_chain is tc_dot_chain
    assert set(PIPELINES) <= set(MODELS)


def test_group_sum_step_is_tc_dot() -> None:
    rng = random.Random(7)
    for m in (AMPERE_BF16_M16N8K16, HOPPER_E4M3_K32, ADA_E4M3_M16N8K32):
        for _ in range(200):
            acc = rng.randrange(1 << 32)
            if (acc >> 23) & 0xFF == 0xFF:
                continue
            a = [rng.randrange(1 << m.operand_bits) for _ in range(m.k)]
            b = [rng.randrange(1 << m.operand_bits) for _ in range(m.k)]
            try:
                want = tc_dot(m, acc, a, b)
            except InvalidArtifact:
                with pytest.raises(InvalidArtifact):
                    m.step(acc, a, b)
                continue
            assert m.step(acc, a, b) == want
            if (want >> 23) & 0xFF != 0xFF:  # a saturated (infinite) accumulator is outside the next step's domain
                assert tc_dot_chain(m, acc, a + a, b + b) == m.step(want, a, b)


def test_fp8_accumulator_is_truncated_to_the_adder_before_the_add() -> None:
    # 14-bit adder: an incoming accumulator with bits below its top 14 significand bits loses them when a nonzero
    # product joins the group -- and a group of all-zero products still truncates the accumulator (D10's Ada finding).
    acc = 0x3F800001  # 1 + 2^-23
    zeros = [0] * 32
    assert HOPPER_E4M3_K32.step(acc, zeros, zeros) == 0x3F800000
    assert ADA_E4M3_M16N8K32.step(acc, zeros, zeros) == 0x3F800000
    # BF16 Ampere (25-bit adder) keeps the whole accumulator through an all-zero group
    assert AMPERE_BF16_M16N8K16.step(acc, [0] * 16, [0] * 16) == acc


def test_e5m2_models_share_the_e4m3_pipeline() -> None:
    # Measured by lane tc-probe-fp8 on 2026-09-22 (RTX 4090 / H100, 0 mismatches over 25M+ elements each): the e5m2
    # instructions run the e4m3 pipeline with exact e5m2 products, so the models differ only in dtype/product.
    for e5, e4 in ((ADA_E5M2_M16N8K32, ADA_E4M3_M16N8K32), (HOPPER_E5M2_K32, HOPPER_E4M3_K32)):
        assert (e5.groups, e5.width, e5.zero_exponent) == (e4.groups, e4.width, e4.zero_exponent)
        assert e5.dtype == "e5m2" and e5.product(0x3C, 0x3C) == e5m2_product(0x3C, 0x3C)
        assert "HYPOTHESIS" not in e5.validated and "0 mismatches" in e5.validated
    # 1.0 * 1.0 summed 32 times from zero is exactly 32.0 under any 14-bit adder
    assert HOPPER_E5M2_K32.step(0, [0x3C] * 32, [0x3C] * 32) == 0x42000000
    with pytest.raises(InvalidArtifact):
        HOPPER_E5M2_K32.step(0, [0x7C] + [0] * 31, [0x3C] * 32)  # +inf operand is outside the domain


# --- BlockScaledAlignAdd ---------------------------------------------------------------------------------------------------


def _nibbles(s: str) -> list[int]:
    return [int(c, 16) for c in s]


def _bytes(s: str) -> list[int]:
    return [int(s[i : i + 2], 16) for i in range(0, len(s), 2)]


@pytest.fixture(scope="module")
def vectors() -> dict:
    return json.loads(VECTORS.read_text())


def test_fixture_provenance_is_recorded(vectors: dict) -> None:
    prov = vectors["provenance"]
    assert {p["source_path"].rsplit("/", 1)[-1] for p in prov} == {"probe3_1783788526.json", "probe2_1783788240.json"}
    for p in prov:
        assert p["source_repo"].endswith("/autoproof") and len(p["source_sha256"]) == 64 and p["imported_utc"]
    # The checked-in probe3 JSON scores autoproof's *model2* (w1=42/w2=36, no hot clip): 1,574,997 / 1,574,998, the
    # one span64 miss being the hot-group clip that model5 added; model5's full count lives only in the lockfile.  D12.
    assert prov[0]["total_dots"] == 1_574_998 and prov[0]["total_model2_mismatches"] == 1
    assert prov[0]["v0_hypothesis"]["verdict"] == "REFUTED" and prov[0]["v0_hypothesis"]["mismatches"] == 143_076


def test_nvfp4_model_replays_every_recorded_5090_vector(vectors: dict) -> None:
    m = BLACKWELL_SM120_NVF4
    cases = vectors["nvfp4_m16n8k64"]["cases"]
    assert len(cases) == 1601
    agree = 0
    lengths = set()
    for c in cases:
        a, b = _nibbles(c["a"]), _nibbles(c["b"])
        sa, sb = _bytes(c["sa"]), _bytes(c["sb"])
        lengths.add(len(a))
        got = m.chain(0, a, b, sa, sb)
        assert got == int(c["d"], 16), c
        agree += 1
    assert agree == 1601
    assert lengths == {64, 128, 256}


def test_nvfp4_step_accepts_codes_then_scales_as_one_operand_vector(vectors: dict) -> None:
    m = BLACKWELL_SM120_NVF4
    assert (m.k, m.block, m.scales_per_step, m.words_per_step) == (64, 16, 4, 68)
    for c in vectors["nvfp4_m16n8k64"]["cases"][:50]:
        a, b = _nibbles(c["a"]), _nibbles(c["b"])
        sa, sb = _bytes(c["sa"]), _bytes(c["sb"])
        acc = 0
        for s in range(len(a) // 64):
            acc = m.step(acc, a[s * 64 : (s + 1) * 64] + sa[s * 4 : (s + 1) * 4], b[s * 64 : (s + 1) * 64] + sb[s * 4 : (s + 1) * 4])
        assert acc == int(c["d"], 16)
    with pytest.raises(InvalidArtifact):
        m.step(0, [0] * 64, [0] * 64)  # scale bytes missing


def _bsaa(**kw) -> BlockScaledAlignAdd:
    base = dict(arch="sm_120", dtype="e2m1", operand_bits=4, scale="ue4m3", k_per_instruction=64, group=16,
                frac_bits=27, acc_window=36, grid_floor=-174, underflow_zero="positive", validated="test only")
    return BlockScaledAlignAdd(**{**base, **kw})


def test_unscaled_fp4_vectors_replay_under_the_same_transducer(vectors: dict) -> None:
    # PROBE-2: mma.sync m16n8k32 kind::f8f6f4 e2m1 without scales.  The same five-way align-add (scale None, unit
    # weight 2^-2) with k=32 in two groups of 16 reproduces all 1,400 recorded dots.
    unscaled = _bsaa(name="test_unscaled_fp4_m16n8k32", scale=None, k_per_instruction=32)
    cases = vectors["fp4_unscaled_m16n8k32"]["cases"]
    assert len(cases) == 1400
    for c in cases:
        a, b = _nibbles(c["a"]), _nibbles(c["b"])
        assert unscaled.chain(0, a, b) == int(c["d"], 16), c


def test_single_window_w26_hypothesis_is_refuted_by_the_fixture(vectors: dict) -> None:
    # autoproof's v0: one 26-bit window below the largest addend.  In this model's terms the closest instance is a
    # 26-bit accumulator window and a term anchor 26 bits below the leading unit weight minus the widest mantissa.
    v0 = _bsaa(name="test_v0_w26", frac_bits=26 - 19, acc_window=26)
    mismatches = 0
    for c in vectors["nvfp4_m16n8k64"]["cases"]:
        a, b = _nibbles(c["a"]), _nibbles(c["b"])
        sa, sb = _bytes(c["sa"]), _bytes(c["sb"])
        mismatches += v0.chain(0, a, b, sa, sb) != int(c["d"], 16)
    assert mismatches > 0, "the recorded vectors must discriminate the model from the refuted W=26 hypothesis"
    # ... and from a rule anchored one bit off in either direction
    for fb in (26, 28):
        off = _bsaa(name=f"test_frac{fb}", frac_bits=fb)
        assert any(off.chain(0, _nibbles(c["a"]), _nibbles(c["b"]), _bytes(c["sa"]), _bytes(c["sb"])) != int(c["d"], 16)
                   for c in vectors["nvfp4_m16n8k64"]["cases"]), fb


ONE = 0x38  # UE4M3 1.0


def test_hot_group_clip_is_what_the_anchor_gives_for_exponent_field_15() -> None:
    # autoproof's "hot group clip" (D12): a live group with both UE4M3 exponent fields at 15 clips every addend at
    # 2^-19.  Here that is simply the anchor: X = -2 + 5 + 5 = 8, lsb = 8 - 27 = -19.
    m = BLACKWELL_SM120_NVF4
    # Group 0 holds the deepest NVFP4 term: 0.5 * 0.5 under subnormal UE4M3 scales 2^-9 x 2^-9 (code 0x01) = 2^-20.
    # Group 1 holds +0.25 and -0.25 (exact cancellation, but a nonzero elementwise product).
    a = [0x1] + [0] * 15 + [0x1, 0x9] + [0] * 46
    b = [0x1] + [0] * 15 + [0x1, 0x1] + [0] * 46
    cold = m.step_scaled(0, a, b, [0x01, ONE, ONE, ONE], [0x01, ONE, ONE, ONE])
    assert _f32(cold) == 2.0**-20
    hot = m.step_scaled(0, a, b, [0x01, 0x7E, ONE, ONE], [0x01, 0x7E, ONE, ONE])
    assert hot == 0  # every addend truncated away: computed +0
    # one hot scale (X = -2 + 5 - 3 = 0, lsb = -27): the 2^-20 term survives
    assert _f32(m.step_scaled(0, a, b, [0x01, 0x7E, ONE, ONE], [0x01, ONE, ONE, ONE])) == 2.0**-20
    # hot scales over a group with no nonzero elementwise product do not participate in the anchor
    b_cold = [0x1] + [0] * 63
    assert _f32(m.step_scaled(0, a, b_cold, [0x01, 0x7E, ONE, ONE], [0x01, 0x7E, ONE, ONE])) == 2.0**-20
    # and the anchor is per firing: a hot firing followed by a cold one does not clip the second
    assert _f32(m.chain(0, a + a, b + b_cold, [0x01, 0x7E, ONE, ONE] * 2, [0x01, 0x7E, ONE, ONE] * 2)) == 2.0**-20
    # a nonzero hot group sum: 0.25 * 448 * 448 = 50176 and the 2^-20 term is below the anchor AND FP32 precision
    a_big = [0x1] + [0] * 15 + [0x1] + [0] * 47
    assert _f32(m.step_scaled(0, a_big, a_big, [0x01, 0x7E, ONE, ONE], [0x01, 0x7E, ONE, ONE])) == 50176.0


def _w(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_anchor_follows_the_scale_exponents_not_the_leading_mantissa_d17() -> None:
    # D17: model5 anchored a 42-bit window on the msb of the largest addend (1.0 -> lsb 2^-41); the hardware anchors
    # 27 bits below the unit weight of the leading participating group.  1.0 * 1.0 under UE4M3 1.0 (0x38 = 8 x 2^-3)
    # is the integer 4 * 8 * 8 = 256 at X = -2 - 3 - 3 = -8, so the grid is 2^-35 -- six binades coarser than model5's.
    m = BLACKWELL_SM120_NVF4
    a = [0x2] + [0] * 63  # +1.0
    ones = [ONE] * 4
    assert m.group_terms(a, a, ones, ones)[0] == (256, -8, True)
    # an opposite-sign accumulator below the grid is truncated away: 1.0 exactly (model5: 1 - 2^-36 -> RZ -> 1 - 2^-23)
    assert m.step_scaled(_w(-(2.0**-36)), a, a, ones, ones) == 0x3F80_0000
    assert m.step_scaled(_w(-(2.0**-41)), a, a, ones, ones) == 0x3F80_0000
    # on the grid: 1 - 2^-35 -> RZ -> the binary32 below 1.0
    assert m.step_scaled(_w(-(2.0**-35)), a, a, ones, ones) == 0x3F7F_FFFF
    # bits below the grid are dropped toward zero before the add: -(2^-30 + 2^-40) acts as -2^-30
    assert m.step_scaled(_w(-(2.0**-30 + 2.0**-40)), a, a, ones, ones) == 0x3F7F_FFFF
    # same-sign accumulators below the grid vanish too (1 + 2^-36 -> 1.0 under both rules: RZ hides the difference)
    assert m.step_scaled(_w(2.0**-36), a, a, ones, ones) == 0x3F80_0000
    # the widest nvfp4 group: 6*6 x 16 under 448 x 448 = 2304 quarters x 14 x 14 = 451,584 (19 bits) at X = 8, so the
    # grid is 2^-19 and a -2^-19 accumulator survives the add: 462,422,016 - 2^-19 -> RZ -> one ulp below the exact
    # sum.  model5 anchored on the addend's msb (2^26) with 42 bits: grid 2^-15, accumulator dropped, exact sum.
    six = [0x7] * 64
    big = m.step_scaled(0, six, six, [0x7E] * 4, [0x7E] * 4)
    assert _f32(big) == 64 * 36 * 448.0 * 448.0 == 2.0**20 * 441
    assert m.step_scaled(_w(-(2.0**-19)), six, six, [0x7E] * 4, [0x7E] * 4) == big - 1
    assert m.step_scaled(_w(-(2.0**-20)), six, six, [0x7E] * 4, [0x7E] * 4) == big


def test_accumulator_window_keeps_twelve_guard_bits() -> None:
    # The second anchor: lsb >= ilog2(acc) - 35.  A tiny product far below a big accumulator is dropped only when it
    # lies below the accumulator's 36-bit window; the term anchor alone would have kept it.
    m = BLACKWELL_SM120_NVF4
    acc = 0x3F80_0000  # 1.0, window down to 2^-35
    a = [0x1] + [0] * 63  # 0.5 * 0.5 = 0.25 under scales 2^-9 x 2^-9 (0x01): 2^-20 -- inside the window
    assert m.step_scaled(acc, a, a, [0x01] * 4, [0x01] * 4) == 0x3F80_0008  # 1 + 2^-20 exactly
    neg = [0x9] + [0] * 63
    assert m.step_scaled(acc, a, neg, [0x01] * 4, [0x01] * 4) == 0x3F7F_FFF0  # 1 - 2^-20 exactly (20 significant bits)
    # acc = 2^30: the window floor is 2^-5, the 2^-20 term is truncated away before the subtraction -> exact acc
    acc = 0x4E80_0000
    assert m.step_scaled(acc, a, neg, [0x01] * 4, [0x01] * 4) == acc
    # acc = 2^14: floor 2^-21 keeps the term; 2^14 - 2^-20 -> RZ -> the binary32 below 2^14
    acc = 0x4680_0000
    assert m.step_scaled(acc, a, neg, [0x01] * 4, [0x01] * 4) == 0x467F_FFFF


def test_participation_needs_a_nonzero_product_and_nonzero_scales() -> None:
    # A group anchors the grid when at least one elementwise product is nonzero AND both scale mantissas are nonzero,
    # even if its dot cancels to zero (fp4-5090 captures: the "nonzero-dot-only" rule fails 1,443 elements, the
    # "products-only" rule 3, "always" 12,729).
    m = BLACKWELL_SM120_NVF4
    tiny_acc = struct.unpack("<I", struct.pack("<f", -(2.0**-40)))[0]
    ones = [ONE] * 4
    cancel = [0x2, 0xA] + [0] * 62  # +1 * 1 and -1 * 1: dot 0, but live
    one_b = [0x2, 0x2] + [0] * 62
    assert m.step_scaled(tiny_acc, cancel, one_b, ones, ones) == 0  # live group: anchor -29 erases the -2^-40 acc -> +0
    assert m.step_scaled(tiny_acc, [0] * 64, one_b, ones, ones) == tiny_acc  # no nonzero product: acc kept exactly
    assert m.step_scaled(tiny_acc, cancel, one_b, [0x00] + ones[1:], ones) == tiny_acc  # zero A scale: no participation
    assert m.step_scaled(tiny_acc, cancel, one_b, ones, [0x00] + ones[1:]) == tiny_acc  # zero B scale: likewise
    # elementwise: a nonzero A nibble against a zero B nibble is not a nonzero product
    assert m.step_scaled(tiny_acc, [0x2] + [0] * 63, [0x0, 0x2] + [0] * 62, ones, ones) == tiny_acc
    # group_terms exposes (mant, unit weight, participates)
    assert m.group_terms(cancel, one_b, ones, ones)[0] == (0, -8, True)
    assert m.group_terms([0] * 64, one_b, ones, ones)[0] == (0, -8, False)
    assert m.group_terms([0x2] + [0] * 63, one_b, [0x7E] + ones[1:], ones)[0] == (4 * 14 * 8, -2 + 5 - 3, True)


def test_every_zero_result_is_plus_zero_d16() -> None:
    m = BLACKWELL_SM120_NVF4
    zeros, ones = [0] * 64, [ONE] * 4
    # D16: -0 accumulator with all-zero products comes out +0 on the RTX 5090 (model5 kept -0)
    assert m.step_scaled(0x8000_0000, zeros, zeros, ones, ones) == 0
    assert m.step_scaled(0x8000_0000, [0x8] * 64, [0x8] * 64, ones, ones) == 0  # -0 x -0 products
    assert m.step_scaled(0, [0x8] * 64, [0x8] * 64, ones, ones) == 0
    assert m.step_scaled(0x3F80_0000, zeros, zeros, ones, ones) == 0x3F80_0000  # a nonzero acc is untouched
    # exact cancellation of the accumulator by a product is +0
    a = [0x2] + [0] * 63  # +1
    b = [0xA] + [0] * 63  # -1
    assert m.step_scaled(0x3F80_0000, a, b, ones, ones) == 0
    # -0 accumulator plus products that cancel exactly: +0
    assert m.step_scaled(0x8000_0000, [0x2, 0xA] + [0] * 62, [0x2, 0x2] + [0] * 62, ones, ones) == 0
    # a -0 accumulator that is joined by a live group whose dot is zero: still +0
    assert m.step_scaled(0x8000_0000, [0x2, 0xA] + [0] * 62, [0x2, 0x2] + [0] * 62, ones, ones) == 0


def test_mxf4_scales_share_the_nvf4_adder_d18() -> None:
    m4, m8 = BLACKWELL_SM120_NVF4, BLACKWELL_SM120_MXF4
    assert (m8.k, m8.block, m8.scales_per_step, m8.words_per_step) == (64, 32, 2, 66)
    assert (m8.frac_bits, m8.acc_window) == (m4.frac_bits, m4.acc_window)
    # UE8M0 code 127 = 1.0 = UE4M3 0x38: identical unit weight (X = -2) and identical results on any nvfp4-representable
    # firing with block-constant scales
    rng = random.Random(11)
    for _ in range(300):
        a = [rng.randrange(16) for _ in range(64)]
        b = [rng.randrange(16) for _ in range(64)]
        acc = rng.choice([0, 0x3F80_0000, 0xC000_0000, 0x0080_0000, 0xB000_0000, 0x2F00_0000])
        assert m8.step_scaled(acc, a, b, [127, 127], [127, 127]) == m4.step_scaled(acc, a, b, [ONE] * 4, [ONE] * 4)
        # UE8M0 code 130 (8.0) on A == UE4M3 0x50 (8.0) on A
        assert m8.step_scaled(acc, a, b, [130, 130], [127, 127]) == m4.step_scaled(acc, a, b, [0x50] * 4, [ONE] * 4)
    # the anchor follows the raw exponent fields: codes (ca, cb) put the unit weight at ca + cb - 262 and the grid
    # at ca + cb - 289 (lane fp4-5090's fit: "X_raw - 33" with X_raw = ca + cb - 256)
    assert m8.group_terms([0x2] + [0] * 63, [0x2] + [0] * 63, [140, 127], [150, 127])[0] == (4 * 8 * 8, 140 + 150 - 262, True)
    a = [0x2] + [0] * 63
    term = 2.0**36  # 1.0 * 2^13 * 2^23 = 256 * 2^28; grid 2^1
    assert _f32(m8.step_scaled(_w(-2.0), a, a, [140, 127], [150, 127])) == term - 2.0**12  # 2^36 - 2 -> RZ
    assert _f32(m8.step_scaled(_w(-1.0), a, a, [140, 127], [150, 127])) == term  # -1 is below the grid: dropped


def test_underflow_sign_hypotheses_d13() -> None:
    # a nonzero mxfp4 sum entirely below the subnormal grid: the two instances differ only in the sign of that zero.
    # RTX 5090 (fp4-re run r20260922-215831-c3d8, ue8m0_underflow / ue8m0_scale_extremes): +0 -- the signed variant
    # misses 15,205 + 2,003 elements there, the pinned one none.
    tiny = [51] * 2  # UE8M0 code 51 = 2^-76: -0.5 * 0.5 * 2^-152 = -2^-154, below the grid but above grid_floor
    neg = [0x9] + [0] * 63  # -0.5
    pos = [0x1] + [0] * 63
    assert BLACKWELL_SM120_MXF4.group_terms(neg, pos, tiny, tiny)[0] == (-64, -160, True)
    assert BLACKWELL_SM120_MXF4.step_scaled(0, neg, pos, tiny, tiny) == 0
    assert BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED.step_scaled(0, neg, pos, tiny, tiny) == 0x8000_0000
    assert BLACKWELL_SM120_MXF4.step_scaled(0, pos, pos, tiny, tiny) == 0  # positive underflow is +0 either way
    # below grid_floor nothing survives the alignment at all: a computed zero, +0 under both
    assert BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED.step_scaled(0, neg, pos, [0, 0], [0, 0]) == 0
    # otherwise the two agree
    rng = random.Random(3)
    for _ in range(300):
        a = [rng.randrange(16) for _ in range(64)]
        b = [rng.randrange(16) for _ in range(64)]
        sa = [rng.randrange(100, 140) for _ in range(2)]
        sb = [rng.randrange(100, 140) for _ in range(2)]
        acc = rng.choice([0, 0x3F80_0000, 0xC000_0000, 0x0080_0000])
        assert BLACKWELL_SM120_MXF4.step_scaled(acc, a, b, sa, sb) == BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED.step_scaled(acc, a, b, sa, sb)


def test_subnormal_accumulator_anchors_at_its_exponent_field_d19() -> None:
    # The accumulator window rides the FP32 exponent *field*: a subnormal accumulator (field 0) anchors at -126 - 35 =
    # -161 whatever its true msb.  Terms below 2^-161 are dropped even though a true-msb window would keep them.
    m = BLACKWELL_SM120_MXF4
    acc = 0x0000_0100  # +2^-141 (msb -141; a true-msb window would reach 2^-176)
    one = [0x2] + [0] * 63  # 1.0 x 1.0 = 4 quarters -> mant 256 at X = ca + cb - 262
    neg = [0xA] + [0] * 63
    # codes (46, 46) -> X = -170: term 2^-162, just below the -161 anchor: dropped, so -2^-162 does not borrow
    assert m.step_scaled(acc, neg, one, [46, 46], [46, 46]) == acc
    # codes (50, 50) -> X = -162: term 2^-154, on the grid: 2^-141 - 2^-154 rounds toward zero to 0xFF
    assert m.step_scaled(acc, neg, one, [50, 50], [50, 50]) == 0x0000_00FF
    assert m.step_scaled(acc, one, one, [50, 50], [50, 50]) == acc  # +2^-154 is on the grid but RZ discards it
    # (a window anchored at the true msb, -141 - 35 = -176, would have kept the -2^-162 term and given 0xFF: that is
    # the rejected reading, 528 residual mismatches in fp4-re run r20260922-215831-c3d8)
    # the smallest normal accumulator anchors at -161 as well: -2^-148 (codes 53, 53 -> X = -156) is on the grid
    assert m.step_scaled(0x0080_0000, neg, one, [53, 53], [53, 53]) == 0x007F_FFFE
    assert m.step_scaled(0x0080_0000, neg, one, [46, 46], [46, 46]) == 0x0080_0000


def test_grid_floor_is_the_deepest_alignment_d19() -> None:
    # lsb never goes below 2^-174: bits of a term below that weight are truncated toward zero before the add.
    m = BLACKWELL_SM120_MXF4
    assert m.grid_floor == -174
    one = [0x2] + [0] * 63
    # leading block 1.0 x 1.0 at X = -150 (codes 56, 56): 256 * 2^-150 = 2^-142 = subnormal word 0x80; its anchor
    # -150 - 27 = -177 is floored to -174.  The second block holds -1.0 x 1.0 = -256 * 2^X: at X = -176 (codes 43, 43)
    # that is -2^-168, on the -174 grid, and 2^-142 - 2^-168 rounds toward zero to 0x7F; at X = -180 (codes 41, 41)
    # -2^-172 is on the grid too; at X = -183 (codes 40, 39) -2^-175 lies below the floor and vanishes.
    a = [0x2] + [0] * 31 + [0xA] + [0] * 31
    b = [0x2] + [0] * 31 + [0x2] + [0] * 31
    assert m.step_scaled(0, one, one, [56, 56], [56, 56]) == 0x0000_0080
    assert m.step_scaled(0, a, b, [56, 43], [56, 43]) == 0x0000_007F
    assert m.step_scaled(0, a, b, [56, 41], [56, 41]) == 0x0000_007F   # -2^-172: still on the -174 grid
    assert m.step_scaled(0, a, b, [56, 40], [56, 39]) == 0x0000_0080   # -2^-175: below the floor, gone
    # without the floor the anchor of a leading group at X = -150 would keep a -2^-176 term as well
    unfloored = _bsaa(name="test_no_floor", scale="ue8m0", grid_floor=-400)
    assert unfloored.step_scaled(0, a, b, [56, 40], [56, 39]) == 0x0000_007F


def test_block_scaled_domain_errors() -> None:
    m = BLACKWELL_SM120_NVF4
    with pytest.raises(InvalidArtifact):
        m.step_scaled(0x7F80_0000, [0] * 64, [0] * 64, [ONE] * 4, [ONE] * 4)  # inf accumulator
    with pytest.raises(InvalidArtifact):
        m.step_scaled(0, [0] * 64, [0] * 64, [0x7F] * 4, [ONE] * 4)  # UE4M3 NaN scale
    with pytest.raises(InvalidArtifact):
        m.step_scaled(0, [0] * 64, [0] * 64, [0x80] * 4, [ONE] * 4)  # UE4M3 padding bit
    with pytest.raises(InvalidArtifact):
        BLACKWELL_SM120_MXF4.step_scaled(0, [0] * 64, [0] * 64, [0xFF] * 2, [127] * 2)  # UE8M0 NaN scale
    with pytest.raises(InvalidArtifact):
        _bsaa(name="x", dtype="e4m3", operand_bits=8, scale=None)
    with pytest.raises(InvalidArtifact):
        _bsaa(name="x", k_per_instruction=40)
    with pytest.raises(InvalidArtifact):
        _bsaa(name="x", acc_window=0)
    with pytest.raises(InvalidArtifact):
        _bsaa(name="x", grid_floor=-140)
    # a sum leaving the binary32 range is outside the model (no silicon evidence yet): 2^127 acc + 6*6*2^127*2^0
    with pytest.raises(InvalidArtifact):
        BLACKWELL_SM120_MXF4.step_scaled(0x7F00_0000, [0x7] + [0] * 63, [0x7] + [0] * 63, [254, 127], [127, 127])
    assert m.product(0x7, 0x7) == e2m1_product(0x7, 0x7)
