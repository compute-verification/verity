"""The finite-domain step (`verity.ml.circuits.silicon.silicon`) against the RTX 4090 golden records and known answers.

Ported from veritor ``tests/veritor/core/test_silicon.py`` lines 1-208 (the gate-set and Hawkeye-C++ tests were left behind).
The golden vectors under ``fixtures/golden/`` were measured on an RTX 4090 (sm_89), one ``mma.sync`` per record.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.silicon import (
    ADA_BF16_M16N8K16,
    ADA_E4M3_M16N8K32,
    AMPERE_BF16_M16N8K16,
    HAWKEYE_AMPERE_GROUPSUM_E4M3_V0,
    HOPPER_BF16_M16N8K16,
    HOPPER_E4M3_K32,
    PIPELINES,
    Pipeline,
    tc_dot,
    tc_dot_chain,
)

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


def _load_golden(name: str) -> tuple[Pipeline, list[dict[str, str]]]:
    with open(GOLDEN_DIR / f"{name}.json") as f:
        data = json.load(f)
    return PIPELINES[data["pipeline"]], data["records"]


def _words(hex_text: str, bits: int) -> list[int]:
    width = bits // 4
    return [int(hex_text[i : i + width], 16) for i in range(0, len(hex_text), width)]


@pytest.mark.parametrize("name", ["ada_bf16_m16n8k16", "ada_e4m3_m16n8k32"])
def test_gpu_golden_vectors_reproduce_bit_exactly(name: str) -> None:
    pipeline, records = _load_golden(name)
    assert len(records) >= 300
    families = set()
    for record in records:
        families.add(record["family"])
        a = _words(record["a"], pipeline.operand_bits)
        b = _words(record["b"], pipeline.operand_bits)
        got = tc_dot(pipeline, int(record["acc"], 16), a, b)
        assert got == int(record["d"], 16), (record, f"{got:08x}")
    assert {
        "randn_zero_acc",
        "randn_random_acc",
        "subnormal",
        "cancellation",
    } <= families


# The ten known-answer tiles of openvm-tc-bench/crates/tc-dot-spec/src/golden.rs.
SPEC_GOLDEN = [
    ([0x38] * 16, [0x38] * 16, 0x00000000, 0x41800000),
    ([0x3C, 0x40] + [0] * 14, [0x38, 0xBC] + [0] * 14, 0x3E800000, 0xBFA00000),
    ([0x3C] + [0] * 15, [0x3C] + [0] * 15, 0x00000000, 0x40100000),
    ([0x38, 0x38] + [0] * 14, [0x38, 0xB8] + [0] * 14, 0x00000000, 0x00000000),
    ([0x03, 0x48] + [0] * 14, [0x38, 0x38] + [0] * 14, 0x00000000, 0x40803000),
    ([0x98] + [0] * 15, [0x18] + [0] * 15, 0x48000000, 0x48000000),
    ([0xA0] + [0] * 15, [0x18] + [0] * 15, 0x48000000, 0x47FFFFFF),
    ([0x3C] + [0] * 15, [0x38] + [0] * 15, 0x4B000000, 0x4B000001),
    (
        [0x51, 0x50, 0, 0, 0, 0, 0, 0, 0x04, 0, 0, 0, 0, 0, 0, 0],
        [0x50, 0xD0, 0, 0, 0, 0, 0, 0, 0x04, 0, 0, 0, 0, 0, 0, 0],
        0x00000000,
        0x41000040,
    ),
    ([0] * 16, [0] * 16, 0x80000000, 0x00000000),
]


def test_synthetic_v0_contract_matches_tc_dot_spec_goldens() -> None:
    for a, b, c, d in SPEC_GOLDEN:
        assert tc_dot(HAWKEYE_AMPERE_GROUPSUM_E4M3_V0, c, a, b) == d


def test_synthetic_v0_contract_differs_from_ada_fp8_silicon() -> None:
    """Three of the ten spec goldens depend on 24-bit alignment; Ada keeps 14 bits."""

    ada = [
        tc_dot(ADA_E4M3_M16N8K32, c, a + [0] * 16, b + [0] * 16)
        for a, b, c, _ in SPEC_GOLDEN
    ]
    spec = [d for *_, d in SPEC_GOLDEN]
    differing = [i for i, (x, y) in enumerate(zip(ada, spec, strict=True)) if x != y]
    assert differing == [6, 7, 8]
    assert ada[6] == 0x48000000 and ada[7] == 0x4B000000 and ada[8] == 0x41000000


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_bf16_pipeline_basic_values() -> None:
    one = 0x3F80
    assert tc_dot(ADA_BF16_M16N8K16, 0, [one] * 16, [one] * 16) == f32(16.0)
    assert tc_dot(ADA_BF16_M16N8K16, f32(-16.0), [one] * 16, [one] * 16) == 0
    # cancellation inside the first group truncates the small accumulator
    big, small = 0x4980, 0x3F80  # 2^20, 1
    a = [big, big] + [0] * 14
    b = [big, 0xC980] + [0] * 14  # +2^20, -2^20
    assert tc_dot(ADA_BF16_M16N8K16, f32(1.0), a, b) == 0
    # ... but a small product in the second group survives
    a2 = [big, big] + [0] * 6 + [small] + [0] * 7
    b2 = [big, 0xC980] + [0] * 6 + [small] + [0] * 7
    assert tc_dot(ADA_BF16_M16N8K16, 0, a2, b2) == f32(1.0)


def test_ada_fp8_has_two_groups_where_hopper_has_one() -> None:
    """Minimal tile separating Ada's (16, 16) FP8 grouping from Hopper's single group of 32.

    Products 0 and 1 cancel exactly (+2^16 - 2^16); product 16 is 1.0.  In one
    group of 32 the 1.0 is aligned to exponent 16 inside a 14-bit window and
    truncated away; on Ada it lands in the second group after the cancellation.
    Measured on the RTX 4090: Hawkeye's Hopper FP8 simulator disagreed with the
    hardware on 17.6% of random-tile outputs (the evidence of ``sm89.mma.m16n8k32.e4m3`` in
    ``verity.ml.circuits.silicon.instructions``).
    """

    a = [0x78, 0x78] + [0] * 14 + [0x38] + [0] * 15  # 256, 256, ..., 1.0 at k = 16
    b = [0x78, 0xF8] + [0] * 14 + [0x38] + [0] * 15  # 256, -256, ..., 1.0 at k = 16
    assert tc_dot(ADA_E4M3_M16N8K32, 0, a, b) == f32(1.0)
    assert tc_dot(HOPPER_E4M3_K32, 0, a, b) == 0
    # the GPU-measured record from the validation run on which the Hopper model first failed
    a_hex = "b08234bbb8a83aabc0beb2be9635bd35b19bc13b2eb93327ad31a8b8b5b636ba"
    b_hex = "3b382f3d1a352fb3b928ab3033b63cb62f402ead34bab221bbc0292c3039a7bb"
    assert (
        tc_dot(ADA_E4M3_M16N8K32, 0, _words(a_hex, 8), _words(b_hex, 8)) == 0xC0C9D800
    )
    assert tc_dot(HOPPER_E4M3_K32, 0, _words(a_hex, 8), _words(b_hex, 8)) == 0xC0C9D400


def test_bf16_group_overflow_saturates_and_is_sticky() -> None:
    """A group whose sum leaves the FP32 range becomes an infinity that later groups cannot undo.

    Measured on the RTX 4090 (gpu/tensor-core-semantics/probe_inf.py and the
    ``uniform_bits``/``mixed_magnitude``/``near_overflow`` families): 2^64*2^64
    in group 1 gives +inf even when group 2 subtracts the same amount, whereas
    the same cancellation inside one group gives exactly zero.
    """

    big, neg_big = 0x5F80, 0xDF80  # +-2^64 in BF16
    assert (
        tc_dot(ADA_BF16_M16N8K16, 0, [big] + [0] * 7 + [neg_big] + [0] * 7, [big] * 16)
        == 0x7F800000
    )
    assert tc_dot(ADA_BF16_M16N8K16, 0, [big, neg_big] + [0] * 14, [big] * 16) == 0
    # GPU-measured record that the unsaturated model got wrong (it produced -inf)
    a_hex = "44380e42c0f160f0fb492db220eda741544eb84c1ba72f43a68387b922cd16b2"
    b_hex = "358dfea91fc467f69c454637a0fc8e32ff20b9bc28cec926530b332c5b30ef0f"
    assert (
        tc_dot(ADA_BF16_M16N8K16, 0xB07E8ECC, _words(a_hex, 16), _words(b_hex, 16))
        == 0x7F800000
    )


def test_e4m3_pipeline_truncates_accumulator_to_14_bits() -> None:
    """Ada's FP8 path keeps 14 significand bits even for a pass-through accumulator."""

    zeros = [0] * 32
    acc = 0x3F80_0001  # 1 + 2^-23
    assert tc_dot(ADA_E4M3_M16N8K32, acc, zeros, zeros) == 0x3F80_0000
    acc = 0x3F80_1000  # 1 + 2^-11 : bit 12 of the mantissa survives (14-bit window)
    assert tc_dot(ADA_E4M3_M16N8K32, acc, zeros, zeros) == 0x3F80_1000
    acc = 0x3F80_0200  # 1 + 2^-14 : below the window
    assert tc_dot(ADA_E4M3_M16N8K32, acc, zeros, zeros) == 0x3F80_0000


def test_tc_dot_chain_is_sequential_fold() -> None:
    a = [0x3F80, 0x4000] * 16  # 1, 2 (K = 32 = two BF16 steps)
    b = [0x3F80] * 32
    step1 = tc_dot(ADA_BF16_M16N8K16, 0, a[:16], b[:16])
    step2 = tc_dot(ADA_BF16_M16N8K16, step1, a[16:], b[16:])
    assert tc_dot_chain(ADA_BF16_M16N8K16, 0, a, b) == step2 == f32(48.0)
    with pytest.raises(InvalidArtifact):
        tc_dot_chain(ADA_BF16_M16N8K16, 0, a[:20], b[:20])


def test_domain_errors() -> None:
    inf = 0x7F80
    with pytest.raises(InvalidArtifact):
        tc_dot(ADA_BF16_M16N8K16, 0, [inf] + [0] * 15, [0x3F80] + [0] * 15)
    with pytest.raises(InvalidArtifact):
        tc_dot(ADA_E4M3_M16N8K32, 0, [0x7F] + [0] * 31, [0] * 32)
    with pytest.raises(InvalidArtifact):
        tc_dot(ADA_BF16_M16N8K16, 0x7FC00000, [0] * 16, [0] * 16)
    with pytest.raises(InvalidArtifact):
        tc_dot(ADA_BF16_M16N8K16, 0, [0] * 15, [0] * 16)


def test_pipeline_registry() -> None:
    assert ADA_BF16_M16N8K16.k == 16 and ADA_E4M3_M16N8K32.k == 32
    assert ADA_BF16_M16N8K16.groups == AMPERE_BF16_M16N8K16.groups == (8, 8)
    assert HOPPER_BF16_M16N8K16.groups == (16,) and HOPPER_E4M3_K32.groups == (32,)
    assert set(PIPELINES) >= {"ada_bf16_m16n8k16", "ada_e4m3_m16n8k32"}
