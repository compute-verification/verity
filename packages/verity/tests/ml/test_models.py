"""``verity.ml.tc.models``: the ``Model`` protocol, ``GroupSum`` (= ``Pipeline``) and ``BlockScaledAlignAdd``.

The block-scaled model is replayed on every RTX 5090 vector autoproof recorded (``fixtures/tc/autoproof-probe-vectors.json``:
1,601 nvfp4 ascending-K chains and 1,400 unscaled-FP4 dots) and must agree bit-for-bit; the refuted single-window
hypothesis (W=26) must *disagree* somewhere, or the fixture would not discriminate.
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
    BLACKWELL_SM120_MXF4_UNDERFLOW_PLUS0,
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


def test_e5m2_hypotheses_share_the_e4m3_pipeline() -> None:
    for e5, e4 in ((ADA_E5M2_M16N8K32, ADA_E4M3_M16N8K32), (HOPPER_E5M2_K32, HOPPER_E4M3_K32)):
        assert (e5.groups, e5.width, e5.zero_exponent) == (e4.groups, e4.width, e4.zero_exponent)
        assert e5.dtype == "e5m2" and e5.product(0x3C, 0x3C) == e5m2_product(0x3C, 0x3C)
        assert "HYPOTHESIS" in e5.validated
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


def test_unscaled_fp4_vectors_replay_under_the_same_transducer(vectors: dict) -> None:
    # PROBE-2: mma.sync m16n8k32 kind::f8f6f4 e2m1 without scales.  The same five-way align-add (hot clip off,
    # scale None) with k=32 in two groups of 16 reproduces all 1,400 recorded dots.
    unscaled = BlockScaledAlignAdd(
        name="test_unscaled_fp4_m16n8k32", arch="sm_120", dtype="e2m1", operand_bits=4, scale=None,
        k_per_instruction=32, group=16, w1=42, w2=36, hot_clip=False, hot_clip_lsb=-19, underflow_zero="signed",
        validated="test only",
    )
    cases = vectors["fp4_unscaled_m16n8k32"]["cases"]
    assert len(cases) == 1400
    for c in cases:
        a, b = _nibbles(c["a"]), _nibbles(c["b"])
        assert unscaled.chain(0, a, b) == int(c["d"], 16), c


def test_single_window_w26_hypothesis_is_refuted_by_the_fixture(vectors: dict) -> None:
    # autoproof's v0: one 26-bit window below the largest addend, no accumulator window (w2 = w1), no hot clip.
    v0 = BlockScaledAlignAdd(
        name="test_v0_w26", arch="sm_120", dtype="e2m1", operand_bits=4, scale="ue4m3",
        k_per_instruction=64, group=16, w1=26, w2=26, hot_clip=False, hot_clip_lsb=-19, underflow_zero="signed",
        validated="refuted hypothesis (test only)",
    )
    mismatches = 0
    for c in vectors["nvfp4_m16n8k64"]["cases"]:
        a, b = _nibbles(c["a"]), _nibbles(c["b"])
        sa, sb = _bytes(c["sa"]), _bytes(c["sb"])
        mismatches += v0.chain(0, a, b, sa, sb) != int(c["d"], 16)
    assert mismatches > 0, "the recorded vectors must discriminate model5 from the refuted W=26 hypothesis"


def test_hot_group_clip_is_per_firing_and_needs_both_scales_hot() -> None:
    m = BLACKWELL_SM120_NVF4
    # Group 0 holds the deepest NVFP4 term: 0.5 * 0.5 under subnormal UE4M3 scales 2^-9 x 2^-9 (code 0x01) = 2^-20.
    # Group 1 holds +0.25 and -0.25 (exact cancellation, but a nonzero elementwise product): under scales 0x7E (448)
    # on BOTH sides it makes the firing "hot" and every addend is clipped at 2^-19 -- the 2^-20 term vanishes.
    a = [0x1] + [0] * 15 + [0x1, 0x9] + [0] * 46
    b = [0x1] + [0] * 15 + [0x1, 0x1] + [0] * 46
    one = 0x38
    cold = m.step_scaled(0, a, b, [0x01, one, one, one], [0x01, one, one, one])
    assert _f32(cold) == 2.0**-20
    hot = m.step_scaled(0, a, b, [0x01, 0x7E, one, one], [0x01, 0x7E, one, one])
    assert hot == 0  # every addend truncated away: computed +0
    half_hot = m.step_scaled(0, a, b, [0x01, 0x7E, one, one], [0x01, one, one, one])
    assert _f32(half_hot) == 2.0**-20  # one hot scale is not hot
    # hot scales over a group with no nonzero product do not trigger the clip
    b_cold = [0x1] + [0] * 63
    assert _f32(m.step_scaled(0, a, b_cold, [0x01, 0x7E, one, one], [0x01, 0x7E, one, one])) == 2.0**-20
    # and the clip is per firing: a hot firing followed by a cold one does not clip the second
    assert _f32(m.chain(0, a + a, b + b_cold, [0x01, 0x7E, one, one] * 2, [0x01, 0x7E, one, one] * 2)) == 2.0**-20
    # a nonzero hot group sum: 0.25 * 448 * 448 = 50176 and the 2^-20 term is below the clip AND FP32 precision
    a_big = [0x1] + [0] * 15 + [0x1] + [0] * 47
    assert _f32(m.step_scaled(0, a_big, a_big, [0x01, 0x7E, one, one], [0x01, 0x7E, one, one])) == 50176.0


def test_zero_handling_matches_the_lockfile() -> None:
    m = BLACKWELL_SM120_NVF4
    zeros, ones = [0] * 64, [0x38] * 4
    assert m.step_scaled(0x8000_0000, zeros, zeros, ones, ones) == 0x8000_0000  # nothing to fold: -0 stays -0
    assert m.step_scaled(0x3F80_0000, zeros, zeros, ones, ones) == 0x3F80_0000
    # exact cancellation of the accumulator by a product is +0
    a = [0x2] + [0] * 63  # +1
    b = [0xA] + [0] * 63  # -1
    assert m.step_scaled(0x3F80_0000, a, b, ones, ones) == 0
    # a nonzero mxfp4 sum entirely below the subnormal grid: the two hypotheses differ only in the sign of that zero
    tiny = [0x00] * 2  # UE8M0 code 0 = 2^-127
    neg = [0x9] + [0] * 63  # -0.5
    pos = [0x1] + [0] * 63
    assert BLACKWELL_SM120_MXF4.step_scaled(0, neg, pos, tiny, tiny) == 0x8000_0000
    assert BLACKWELL_SM120_MXF4_UNDERFLOW_PLUS0.step_scaled(0, neg, pos, tiny, tiny) == 0
    assert BLACKWELL_SM120_MXF4.step_scaled(0, pos, pos, tiny, tiny) == 0  # positive underflow is +0 either way
    # otherwise the two mxfp4 hypotheses agree
    rng = random.Random(3)
    for _ in range(300):
        a = [rng.randrange(16) for _ in range(64)]
        b = [rng.randrange(16) for _ in range(64)]
        sa = [rng.randrange(100, 140) for _ in range(2)]
        sb = [rng.randrange(100, 140) for _ in range(2)]
        acc = rng.choice([0, 0x3F80_0000, 0xC000_0000, 0x0080_0000])
        assert BLACKWELL_SM120_MXF4.step_scaled(acc, a, b, sa, sb) == BLACKWELL_SM120_MXF4_UNDERFLOW_PLUS0.step_scaled(acc, a, b, sa, sb)


def test_block_scaled_domain_errors() -> None:
    m = BLACKWELL_SM120_NVF4
    with pytest.raises(InvalidArtifact):
        m.step_scaled(0x7F80_0000, [0] * 64, [0] * 64, [0x38] * 4, [0x38] * 4)  # inf accumulator
    with pytest.raises(InvalidArtifact):
        m.step_scaled(0, [0] * 64, [0] * 64, [0x7F] * 4, [0x38] * 4)  # UE4M3 NaN scale
    with pytest.raises(InvalidArtifact):
        BlockScaledAlignAdd(name="x", arch="sm_120", dtype="e4m3", operand_bits=8, scale=None, k_per_instruction=64,
                            group=16, w1=42, w2=36, hot_clip=False, hot_clip_lsb=-19, underflow_zero="signed", validated="")
    with pytest.raises(InvalidArtifact):
        BlockScaledAlignAdd(name="x", arch="sm_120", dtype="e2m1", operand_bits=4, scale="ue8m0", k_per_instruction=64,
                            group=16, w1=42, w2=36, hot_clip=True, hot_clip_lsb=-19, underflow_zero="signed", validated="")
    assert m.product(0x7, 0x7) == e2m1_product(0x7, 0x7)
