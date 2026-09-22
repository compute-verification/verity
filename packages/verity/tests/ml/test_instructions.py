"""``verity.ml.tc.instructions``: the registry has exactly the agreed ids, and every status is backed the way it claims."""
from __future__ import annotations

import pytest

from verity.ml.tc import instructions
from verity.ml.tc.instructions import INSTRUCTIONS, Instruction, instructions_for
from verity.ml.tc.models import (
    ADA_BF16_M16N8K16,
    ADA_E4M3_M16N8K32,
    AMPERE_BF16_M16N8K16,
    BLACKWELL_SM120_NVF4,
    BLACKWELL_SM120_MXF4,
    HOPPER_E4M3_K32,
    MODELS,
    HOPPER_BF16_M16N8K16,
    ADA_E5M2_M16N8K32,
    HOPPER_E5M2_K32,
    Model,
)

EXPECTED_IDS = {
    "sm80.mma.m16n8k16.bf16",
    "sm89.mma.m16n8k16.bf16",
    "sm90.mma.m16n8k16.bf16",
    "sm89.mma.m16n8k32.e4m3",
    "sm89.mma.m16n8k32.e5m2",
    "sm90.wgmma.m64n8k32.e4m3",
    "sm90.wgmma.m64n8k32.e5m2",
    "sm90.mma.m16n8k32.e4m3",
    "sm120.mma.m16n8k16.bf16",
    "sm120.mma.m16n8k64.e2m1.nvf4",
    "sm120.mma.m16n8k64.e2m1.mxf4",
    "sm100.tcgen05.mma.e2m1.nvf4",
    "sm100.tcgen05.mma.e2m1.mxf4",
    "sm100.tcgen05.mma.bf16",
    "sm100.tcgen05.mma.e4m3",
}


def test_registry_has_exactly_the_agreed_ids() -> None:
    assert set(INSTRUCTIONS) == EXPECTED_IDS
    for ins_id, ins in INSTRUCTIONS.items():
        assert ins.id == ins_id
        assert ins.sm == ins_id.split(".")[0].replace("sm", "sm_")


def test_pinned_instructions_and_their_models() -> None:
    pinned = {i.id: i.model for i in INSTRUCTIONS.values() if i.status == "pinned"}
    assert pinned == {
        "sm80.mma.m16n8k16.bf16": AMPERE_BF16_M16N8K16,
        "sm89.mma.m16n8k16.bf16": ADA_BF16_M16N8K16,
        "sm89.mma.m16n8k32.e4m3": ADA_E4M3_M16N8K32,
        "sm89.mma.m16n8k32.e5m2": ADA_E5M2_M16N8K32,             # lane tc-probe-fp8 2026-09-22: 25.0M + 141k, 0 mismatches
        "sm90.mma.m16n8k16.bf16": HOPPER_BF16_M16N8K16,          # lane tc-probe-fp8 2026-09-22: 25.7M, 0 mismatches
        "sm90.wgmma.m64n8k32.e4m3": HOPPER_E4M3_K32,
        "sm90.wgmma.m64n8k32.e5m2": HOPPER_E5M2_K32,             # lane tc-probe-fp8 2026-09-22: 25.5M, 0 mismatches
        "sm120.mma.m16n8k16.bf16": HOPPER_BF16_M16N8K16,          # lane fp4-5090: the 5090 keeps Hopper's BF16 datapath
        "sm120.mma.m16n8k64.e2m1.nvf4": BLACKWELL_SM120_NVF4,      # lane fp4-re 2026-09-22: 8.3M, 0 mismatches, any FP32 acc
        "sm120.mma.m16n8k64.e2m1.mxf4": BLACKWELL_SM120_MXF4,      # lane fp4-re 2026-09-22: 7.4M, 0 mismatches (D13/D18/D19)
    }


def test_statuses_are_consistent_with_models_and_evidence() -> None:
    for ins in INSTRUCTIONS.values():
        if ins.status == "unprobed":
            assert ins.model is None, ins.id
        else:
            assert isinstance(ins.model, Model), ins.id
            assert ins.model.name in MODELS
            assert ins.model.dtype == ins.a_dtype == ins.b_dtype
            assert ins.shape is not None and ins.shape[2] == ins.model.k
        if ins.status == "pinned":
            assert ins.ptx is not None, f"{ins.id}: a pinned instruction has a recorded mnemonic"
            assert len(ins.evidence) >= 1 and "HYPOTHESIS" not in ins.model.validated
        if ins.status == "hypothesis":
            assert ins.evidence, f"{ins.id}: say what the hypothesis leans on"
        assert ins.acc_dtype == "f32"
        assert ins.scale in (None, "ue4m3", "ue8m0")
        if ins.a_dtype == "e2m1":
            assert ins.scale is not None
        else:
            assert ins.scale is None


def test_mnemonics_come_from_the_source_repos_or_are_none() -> None:
    assert INSTRUCTIONS["sm80.mma.m16n8k16.bf16"].ptx == "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
    assert INSTRUCTIONS["sm89.mma.m16n8k32.e4m3"].ptx == "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
    assert INSTRUCTIONS["sm90.wgmma.m64n8k32.e4m3"].ptx.startswith("wgmma.mma_async.sync.aligned.m64n")
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.nvf4"].ptx == (
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"
    )
    for ins_id in ("sm100.tcgen05.mma.e2m1.nvf4", "sm100.tcgen05.mma.e2m1.mxf4", "sm100.tcgen05.mma.bf16", "sm100.tcgen05.mma.e4m3"):
        ins = INSTRUCTIONS[ins_id]
        assert ins.status == "unprobed" and ins.model is None and ins.ptx is None and ins.shape is None


def test_block_scaled_ids_carry_their_scale_format() -> None:
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.nvf4"].scale == "ue4m3"
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.mxf4"].scale == "ue8m0"
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.mxf4"].status == "pinned"
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.mxf4"].ptx.endswith(".scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0")
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.mxf4"].model is BLACKWELL_SM120_MXF4
    assert INSTRUCTIONS["sm100.tcgen05.mma.e2m1.nvf4"].scale == "ue4m3"
    assert INSTRUCTIONS["sm100.tcgen05.mma.e2m1.mxf4"].scale == "ue8m0"


def test_instructions_for_filters_by_model() -> None:
    assert [i.id for i in instructions_for(HOPPER_E4M3_K32)] == ["sm90.wgmma.m64n8k32.e4m3"]
    assert instructions_for(BLACKWELL_SM120_NVF4)[0].id == "sm120.mma.m16n8k64.e2m1.nvf4"


def test_instruction_invariants_are_enforced() -> None:
    with pytest.raises(ValueError):
        Instruction(id="x", sm="sm_80", ptx=None, shape=None, a_dtype="bf16", b_dtype="bf16", acc_dtype="f32",
                    scale=None, model=None, status="pinned")
    with pytest.raises(ValueError):
        Instruction(id="x", sm="sm_80", ptx=None, shape=(16, 8, 16), a_dtype="e4m3", b_dtype="e4m3", acc_dtype="f32",
                    scale=None, model=AMPERE_BF16_M16N8K16, status="hypothesis")
    with pytest.raises(ValueError):
        Instruction(id="x", sm="sm_80", ptx=None, shape=(16, 8, 32), a_dtype="bf16", b_dtype="bf16", acc_dtype="f32",
                    scale=None, model=AMPERE_BF16_M16N8K16, status="hypothesis")
    assert instructions.Status.__args__ == ("pinned", "hypothesis", "unprobed")
