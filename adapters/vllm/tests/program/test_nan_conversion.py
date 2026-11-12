"""The registered f32 -> bf16 conversions of the library and the production configuration's binding.

`F2fpBf16_v1` (NaN -> 0x7FFF, the measured `cvt.rn.bf16.f32` word) is the conversion B0/B1 bind; it is the same
function as `veritor.core.tc.f32_to_bf16_hw_word` (`veritor.tc-ampere-bf16@2`).  `F32ToBf16Rn_v1` (NaN -> 0x7FC0)
stays registered, byte for byte, as the software canonicalisation (`veritor.tc-ampere-bf16@1`); the production
program never binds it, so it is in neither B0/B1 program digest.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity.circuits.codec import encode_program, program_digest
from verity.circuits.defs import bind
from verity.circuits.program import Program
from verity.definitions.registry import b1
from verity.definitions.registry import prims as P

from verity.definitions.silicon.cast import f32_is_nan as is_nan_f32_word, f32_to_bf16_hw_word, f32_to_bf16_word

CAPTURES = Path(__file__).resolve().parents[4] / "verity/definitions/ops/tests/fixtures/tc-total-2026-09-07"
DEVICES = ("4090", "a100")

#: `Serve_v1{C=B0,LP=32,STEPS=7}`'s digest under the v1 codec, with the k16 step's id re-keyed to core's `AmpereBF16TcDot16_v2`
B0_DIGEST_TC_V2 = "9f74dc8e427111532cda0e1311eca470575ebfc18d544354d0a0d97e25667ec870d0eca262689eb4ec7f5fdbc3f8066b5425fe1da377c46a8809312f44e87865"

NAN_WORDS = [0x7FC00000, 0xFFC00000, 0x7FC00001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001, 0xFF800001, 0x7FBFFFFF,
             0xFFA00000, 0x7F808000, 0x7FC07FFF, 0x7FC08000]
FINITE_WORDS = [0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x3F800000, 0x3F808000, 0x3F818000, 0x3F808001,
                0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000]


def test_the_two_registered_conversions_are_the_two_family_versions():
    rng = random.Random(8)
    words = NAN_WORDS + FINITE_WORDS + [rng.getrandbits(32) for _ in range(20000)]
    for x in words:
        assert P.F2fpBf16.evaluate(x) == f32_to_bf16_hw_word(x), hex(x)
        assert P.F32ToBf16Rn.evaluate(x) == f32_to_bf16_word(x), hex(x)
        if is_nan_f32_word(x):
            assert (P.F32ToBf16Rn.evaluate(x), P.F2fpBf16.evaluate(x)) == (P.BF16_NAN_TORCH, P.BF16_NAN_F2FP) == (0x7FC0, 0x7FFF)
        else:
            assert P.F32ToBf16Rn.evaluate(x) == P.F2fpBf16.evaluate(x), hex(x)


@pytest.mark.parametrize("device", DEVICES)
def test_f2fp_matches_every_measured_cvt_word(device: str):
    record = json.loads((CAPTURES / device / "cvt_special.json").read_text())
    assert record["n_nondeterministic"] == 0 and len(record["words"]) == 4403
    nan = 0
    for entry in record["words"]:
        x, measured = int(entry["x"], 16), int(entry["bf16"], 16)
        assert P.F2fpBf16.evaluate(x) == measured, entry["name"]
        if is_nan_f32_word(x):
            nan += 1
            assert measured == 0x7FFF and P.F32ToBf16Rn.evaluate(x) == 0x7FC0
    assert nan == 84


def test_production_binds_the_hardware_conversion_and_the_b0_digest_is_unchanged():
    program = Program(bind(b1.Serve, C=b1.B0_CONFIG, LP=32, STEPS=7))
    desc = encode_program(program)
    ids = set(desc["definitions"])
    assert "F2fpBf16_v1" in ids
    assert "F32ToBf16Rn_v1" not in ids
    assert program.gates == 441_772_368
    assert program_digest(desc) == B0_DIGEST_TC_V2
    # the B1 program binds the same definitions (a one-position specialisation, built here for its definition set)
    small = Program(bind(b1.Serve, C=b1.B1_CONFIG, LP=2, STEPS=1))
    ids_b1 = set(encode_program(small)["definitions"])
    assert "F2fpBf16_v1" in ids_b1 and "F32ToBf16Rn_v1" not in ids_b1
