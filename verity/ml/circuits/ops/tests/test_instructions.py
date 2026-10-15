"""``verity.ml.circuits.silicon.instructions``: the registry has exactly the agreed ids, and every status is backed the way it claims."""
from __future__ import annotations

import pytest

from verity.ml.circuits.silicon import instructions
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS, Instruction, instructions_for
from verity.ml.circuits.silicon.models import (
    ADA_BF16_M16N8K16,
    ADA_E4M3_M16N8K32,
    AMPERE_BF16_M16N8K16,
    BLACKWELL_SM100_MXF4,
    BLACKWELL_SM100_NVF4,
    BLACKWELL_SM120_E4M3_M16N8K32,
    BLACKWELL_SM120_E5M2_M16N8K32,
    BLACKWELL_SM120_NVF4,
    BLACKWELL_SM120_MXF4,
    HOPPER_E4M3_K32,
    MODELS,
    HOPPER_BF16_M16N8K16,
    HOPPER_BF16_WGMMA_K16,
    ADA_E5M2_M16N8K32,
    HOPPER_E5M2_K32,
    BLACKWELL_SM120_S8_M16N8K32,
    Model,
)

EXPECTED_IDS = {
    "sm80.mma.m16n8k16.bf16",
    "sm89.mma.m16n8k16.bf16",
    "sm90.mma.m16n8k16.bf16",
    "sm90.wgmma.m64n8k16.bf16",
    "sm89.mma.m16n8k32.e4m3",
    "sm89.mma.m16n8k32.e5m2",
    "sm90.wgmma.m64n8k32.e4m3",
    "sm90.wgmma.m64n8k32.e5m2",
    "sm90.mma.m16n8k32.e4m3",
    "sm120.mma.m16n8k16.bf16",
    "sm120.mma.m16n8k32.e4m3",
    "sm120.mma.m16n8k32.e5m2",
    "sm120.mma.m16n8k64.e2m1.nvf4",
    "sm120.mma.m16n8k64.e2m1.mxf4",
    "sm120.mma.m16n8k32.s8",
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
        "sm90.wgmma.m64n8k16.bf16": HOPPER_BF16_WGMMA_K16,       # lane wgmma-bf16 2026-09-26: 2 x 26.4M (SS, RS) + chains, 0 mismatches
        "sm90.wgmma.m64n8k32.e4m3": HOPPER_E4M3_K32,
        "sm90.wgmma.m64n8k32.e5m2": HOPPER_E5M2_K32,             # lane tc-probe-fp8 2026-09-22: 25.5M, 0 mismatches
        "sm120.mma.m16n8k16.bf16": HOPPER_BF16_M16N8K16,          # lane fp4-5090: the 5090 keeps Hopper's BF16 datapath
        "sm120.mma.m16n8k32.e4m3": BLACKWELL_SM120_E4M3_M16N8K32,  # lane vllm-sm120-tc-gemm 2026-09-30: RTX PRO 6000, 25.0M, 0 mismatches
        "sm120.mma.m16n8k32.e5m2": BLACKWELL_SM120_E5M2_M16N8K32,  # the same lane's sweeps on two PRO 6000 dies: 25.0M + 6.4M, 0 mismatches
        "sm100.tcgen05.mma.e2m1.nvf4": BLACKWELL_SM100_NVF4,       # lane tcgen05 2026-09-23: B200, 16.5M, 0 mismatches, any FP32 acc
        "sm100.tcgen05.mma.e2m1.mxf4": BLACKWELL_SM100_MXF4,       # lane tcgen05 2026-09-23: B200, 12.9M, 0 mismatches (both kinds)
        "sm120.mma.m16n8k64.e2m1.nvf4": BLACKWELL_SM120_NVF4,      # lane fp4-proof 2026-09-22: 5090, 17.3M (fp4-re + fresh seed), 0 mismatches
        "sm120.mma.m16n8k64.e2m1.mxf4": BLACKWELL_SM120_MXF4,      # lane fp4-proof 2026-09-22: 5090, 14.4M, 0 mismatches
        "sm120.mma.m16n8k32.s8": BLACKWELL_SM120_S8_M16N8K32,      # lane compute-accounting 2026-10-15: RTX PRO 6000, 10.2M, 0 mismatches
    }
    # registry <= computed level (lane pin-bf16): the sm120 FP4 pair flipped to "pinned" only once benchmarks/tc_probe/trust.py
    # computed PINNED (P2 closed by fp4-proof's fresh-seed sweeps), and each points at its published dossier
    for iid in ("sm120.mma.m16n8k64.e2m1.nvf4", "sm120.mma.m16n8k64.e2m1.mxf4", "sm120.mma.m16n8k32.e4m3", "sm120.mma.m16n8k32.e5m2",
                "sm120.mma.m16n8k32.s8"):
        ins = INSTRUCTIONS[iid]
        assert ins.status == "pinned" and ins.dossier is not None and ins.dossier.startswith("art:")
    # measured on the PRO 6000 only: its entry names that anchor, as the e4m3 target does
    e5 = INSTRUCTIONS["sm120.mma.m16n8k32.e5m2"]
    assert e5.model is BLACKWELL_SM120_E5M2_M16N8K32 and e5.anchor_device == "NVIDIA RTX PRO 6000 Blackwell Server Edition"


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
        assert ins.acc_dtype == ("s32" if ins.a_dtype in ("s8", "u8") else "f32")
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
    # the tcgen05 mnemonics carry descriptors, not register lists: recorded as the PTX ISA spells the instruction plus the
    # descriptor fields the probe kernel set (benchmarks/tc_probe_fp4/tcgen05_fp4.cu)
    assert INSTRUCTIONS["sm100.tcgen05.mma.e2m1.nvf4"].ptx.startswith("tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X")
    assert INSTRUCTIONS["sm100.tcgen05.mma.e2m1.mxf4"].ptx.startswith("tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X")
    for ins_id in ("sm100.tcgen05.mma.bf16", "sm100.tcgen05.mma.e4m3"):
        ins = INSTRUCTIONS[ins_id]
        assert ins.status == "unprobed" and ins.model is None and ins.ptx is None and ins.shape is None


def test_block_scaled_ids_carry_their_scale_format() -> None:
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.nvf4"].scale == "ue4m3"
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.mxf4"].scale == "ue8m0"
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.mxf4"].status == "pinned"   # registry <= computed level (P2 closed by lane fp4-proof)
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.mxf4"].ptx.endswith(".scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0")
    assert INSTRUCTIONS["sm120.mma.m16n8k64.e2m1.mxf4"].model is BLACKWELL_SM120_MXF4
    assert INSTRUCTIONS["sm100.tcgen05.mma.e2m1.nvf4"].scale == "ue4m3"
    assert INSTRUCTIONS["sm100.tcgen05.mma.e2m1.mxf4"].scale == "ue8m0"
    assert INSTRUCTIONS["sm100.tcgen05.mma.e2m1.nvf4"].shape == INSTRUCTIONS["sm100.tcgen05.mma.e2m1.mxf4"].shape == (128, 8, 64)


def test_sm100_models_are_the_sm120_parameters_under_their_own_name() -> None:
    """Lane tcgen05's finding: the B200's tcgen05.mma computes the RTX 5090's words.  The registry keeps separate model objects
    (named for the architecture they were measured on); this pins that their numerics are identical field by field."""
    from dataclasses import fields

    for sm100, sm120 in ((BLACKWELL_SM100_NVF4, BLACKWELL_SM120_NVF4), (BLACKWELL_SM100_MXF4, BLACKWELL_SM120_MXF4)):
        assert type(sm100) is type(sm120)
        for f in fields(sm100):
            if f.name in ("name", "arch", "validated"):
                continue
            assert getattr(sm100, f.name) == getattr(sm120, f.name), f.name
        assert sm100.arch == "sm_100" and sm120.arch == "sm_120"


def test_instructions_for_filters_by_model() -> None:
    assert [i.id for i in instructions_for(HOPPER_E4M3_K32)] == ["sm90.wgmma.m64n8k32.e4m3"]
    assert [i.id for i in instructions_for(HOPPER_BF16_WGMMA_K16)] == ["sm90.wgmma.m64n8k16.bf16"]
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
