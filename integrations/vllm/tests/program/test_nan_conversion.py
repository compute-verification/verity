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

from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import b1
from verity_vllm.program.frontend import conformance as CF
from verity.ml.definitions.registry import prims as P

from verity.ml.circuits.silicon.cast import f32_is_nan as is_nan_f32_word, f32_to_bf16_hw_word, f32_to_bf16_word

REPO = Path(__file__).resolve().parents[2]
MEASURED = sorted((REPO / "docs/data/tc-total-2026-09-07").glob("*/cvt_special.json"))

#: `Serve_v1{C=B0,LP=32,STEPS=7}`: its digest under the v0 codec and the earlier builder, which the NaN record was written under (no
#: bound definition changes it); the builder fix B.1 and the v1 codec (the defaults) give the same program the digest
#: `2d483c89d14d215f…`, certified equivalent by the v0/v1 correspondence record. The NaN record keeps the digest it was written under.
B0_DIGEST = "03405bbdc7ec2825abd79aad723a2c75a417e8b3084567f89cf3341e31a5ab2d"
B0_DIGEST_V1 = "2d483c89d14d215f49305ab3db7b1dd6370bdecc4449f56ca36bd8c6ac4aae30"
#: the same program after the re-baseline re-key: only the k16 step's id moved (AmpereBF16TcDot16_v1 -> core's _v2), so the descriptor is
#: B0_DIGEST_V1's with that id substituted, byte for byte
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


@pytest.mark.parametrize("path", MEASURED, ids=lambda p: p.parent.name)
def test_f2fp_matches_every_measured_cvt_word(path: Path):
    record = json.loads(path.read_text())
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


def test_conformance_record_names_the_evidence():
    rec = CF.record()
    defs = rec["definitions"]
    assert defs["F2fpBf16_v1"]["status"] == "exact-model-tested"
    assert "cvt_special" in defs["F2fpBf16_v1"]["conformance"]
    assert defs["F32ToBf16Rn_v1"]["status"] == "exact-model-tested"
    nan = rec["nan_canonicalisation"]
    assert nan["production_conversion"] == "F2fpBf16_v1" and nan["nan_word"] == "0x7FFF"
    assert "cvt_special.json" in nan["source"]
    assert nan["interface_family"]["F2fpBf16_v1"].startswith("veritor.tc-ampere-bf16@2")
    assert "not bound by B0/B1" in nan["not_bound"]["F32ToBf16Rn_v1"]
    assert nan["program_digests_unchanged"]["B0"] == B0_DIGEST
    assert nan["recorded_traces_finite"]["B0"]["nan_words"] == 0 and nan["recorded_traces_finite"]["B1"]["nan_words"] == 0
    assert nan["source"].startswith("measured")


def test_scan_records_match_the_conformance_record():
    """The two store scans (`data/l8-nan-scan-2026-09-07/`) are what the record states."""
    root = Path(__file__).resolve().parents[2] / "data" / "l8-nan-scan-2026-09-07"
    stated = CF.NAN_CANONICALISATION["recorded_traces_finite"]
    for config in ("B0", "B1"):
        scan = json.loads((root / f"{config}.json").read_text())
        t = scan["totals"]
        assert t["bfloat16/leaves"] == stated[config]["bf16_leaves"]
        assert t["bfloat16/words"] == stated[config]["bf16_words"]
        assert t["bfloat16/nan_words"] == 0 and t["bfloat16/inf_words"] == 0
        assert scan["leaves_with_nan"] == 0 and scan["distinct_nan_words"] == {}
