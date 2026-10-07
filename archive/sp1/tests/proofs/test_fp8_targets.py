"""The FP8 draft targets (``verity.ml.circuits.targets.target.FP8_HOPPER_WGMMA`` / ``FP8_ADA_MMA``) and their pinned vectors.

A draft target is a relation stated precisely enough to generate vectors against; these tests pin each clause of the
statement to the code that implements it (the model, the program builder, the family) and replay the recorded
``verity/ml/circuits/silicon/captures/vectors`` on the model.  The vectors are model replays, not hardware captures: their evidence is the
instruction pin the target names, and a semantic drift in the model, the program or the vector fails here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.silicon.models import ADA_E4M3_M16N8K32, HOPPER_E4M3_K32
from verity_sp1.proofs.families import make_tensor_core_gate_set
from verity_sp1.proofs.programs import gemm_accumulator_fp8
from verity.ml.circuits.targets.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, TARGETS, TC_VECTORS, Target

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists())
DRAFTS = (FP8_HOPPER_WGMMA, FP8_ADA_MMA)


def vector(name: str) -> dict:
    return json.loads((ROOT / TC_VECTORS / f"{name}.json").read_text())


def words(xs: list[str]) -> list[int]:
    return [int(x, 16) for x in xs]


def test_first_is_frozen_and_the_fp8_targets_are_drafts_over_pinned_instructions() -> None:
    assert FIRST.status == "frozen" and FIRST.instruction == "sm80.mma.m16n8k16.bf16"
    assert Target.__dataclass_fields__["status"].default == "frozen"
    for t in DRAFTS:
        assert t.status == "draft"
        inst = INSTRUCTIONS[t.instruction]
        assert inst.status == "pinned" and inst.model is t.pipeline
        assert (inst.a_dtype, inst.acc_dtype) == ("e4m3", "f32")
    assert set(TARGETS) == {FIRST.name, BF16_HOPPER.name, FP8_HOPPER_WGMMA.name, FP8_ADA_MMA.name, "nvfp4-sm120-mma-draft/2026-09-22",
                            "fp8-sm120-mma-draft/2026-09-30"}
    assert FP8_HOPPER_WGMMA.pipeline is HOPPER_E4M3_K32 and FP8_ADA_MMA.pipeline is ADA_E4M3_M16N8K32


def test_every_target_names_its_conformance_anchor_and_accumulates_in_fp32() -> None:
    """The overhead reference is the census line of the anchor device and operand dtype (outside core); core names the device."""

    anchors = {FIRST: "A100", BF16_HOPPER: "H100", FP8_HOPPER_WGMMA: "H100", FP8_ADA_MMA: "4090"}
    for t, device in anchors.items():
        assert device in t.anchor_device and INSTRUCTIONS[t.instruction].acc_dtype == "f32"


@pytest.mark.parametrize("t", DRAFTS, ids=lambda t: t.name)
def test_the_family_is_the_cast_free_tensor_core_gate_set_of_the_pipeline(t: Target) -> None:
    gate_set = make_tensor_core_gate_set(t.pipeline)
    assert f"{gate_set.name}@{gate_set.version}" == t.family
    assert gate_set.digest == t.family_digest
    assert sorted(g.name for g in gate_set) == ["in", "tc_dot32", "tc_dot32_0", "weight"], "no epilogue cast in the family"


def test_the_pipelines_are_the_14_bit_adders_with_floor_minus_139() -> None:
    for t in DRAFTS:
        assert (t.pipeline.dtype, t.pipeline.operand_bits, t.pipeline.width, t.pipeline.zero_exponent) == ("e4m3", 8, 14, -139)
    assert HOPPER_E4M3_K32.groups == (32,) and ADA_E4M3_M16N8K32.groups == (16, 16)
    assert "SEPARATE relation" in FP8_HOPPER_WGMMA.epilogue and "cvt.rn.satfinite.e4m3x2" in FP8_HOPPER_WGMMA.epilogue


def test_the_program_is_a_chain_of_tc_dot32_with_no_cast() -> None:
    sub = gemm_accumulator_fp8(96)
    assert [g.op for g in sub.gates] == ["tc_dot32_0", "tc_dot32", "tc_dot32"]
    assert sub.gates[0].args == tuple(("in", i) for i in [*range(32), *range(96, 128)])
    for s in (1, 2):
        expected = (("gate", s - 1), *(("in", 32 * s + i) for i in range(32)), *(("in", 96 + 32 * s + i) for i in range(32)))
        assert sub.gates[s].args == expected
    assert sub.outgoing == (2,) and len(sub.incoming) == 192 and set(sub.incoming) == {8}
    with pytest.raises(ValueError):
        gemm_accumulator_fp8(48)


@pytest.mark.parametrize("t", DRAFTS, ids=lambda t: t.name)
@pytest.mark.parametrize("which", [0, 1], ids=["k32", "k1536"])
def test_each_reference_vector_replays_on_the_model_and_names_the_program(t: Target, which: int) -> None:
    name = t.reference_vectors[which]
    rec = vector(name)
    assert rec["schema"] == "verity/tc-reference-vector/v0" and rec["name"] == name
    assert rec["target"] == t.name and rec["status"] == "draft" and rec["instruction"] == t.instruction
    assert rec["model"] == t.pipeline.name and rec["family"] == t.family and rec["family_digest"] == t.family_digest
    k, step_k = rec["k"], rec["step_k"]
    assert k == (32, 1536)[which] and step_k == t.pipeline.k == 32
    assert rec["program"] == f"GemmAccumulator<{k}>" and gemm_accumulator_fp8(k).digest == rec["program_digest"]
    a, b = words(rec["a"]), words(rec["b"])
    assert len(a) == len(b) == k and all((w & 0x7F) != 0x7F for w in a + b), "finite E4M3 only"
    assert any(w & 0x7F == 0 for w in a + b) and any(0 < (w & 0x7F) < 8 for w in a + b), "zeros and subnormals present"
    assert any(w & 0x7F == 0x7E for w in a + b), "the maximum 448 present"
    c = int(rec["c0"], 16)
    assert c == 0
    assert len(rec["steps"]) == k // step_k
    for s, step in enumerate(rec["steps"]):
        assert int(step["c_in"], 16) == c
        c = t.pipeline.step(c, a[32 * s : 32 * s + 32], b[32 * s : 32 * s + 32])
        assert c == int(step["c_out"], 16), f"{name}: step {s}"
    assert c == int(rec["y"], 16)
    assert (ROOT / TC_VECTORS / f"{name}.json").stat().st_size < 5_000_000


def test_the_vectors_are_regenerable_from_their_seed_and_recipe() -> None:
    """The recipe recorded in each vector (``generator``) reproduces its operands byte for byte."""

    def e4m3_byte(rng: random.Random) -> int:
        u = rng.random()
        sign = rng.randrange(2) << 7
        if u < 0.10:
            return sign
        if u < 0.25:
            return sign | rng.randrange(1, 8)
        if u < 0.30:
            return sign | 0x7E
        while True:
            w = rng.randrange(256)
            if (w & 0x7F) != 0x7F:
                return w

    for t in DRAFTS:
        for name in t.reference_vectors:
            rec = vector(name)
            rng = random.Random(rec["generator"]["seed"])
            k = rec["k"]
            assert [f"0x{e4m3_byte(rng):02X}" for _ in range(k)] == rec["a"], name
            assert [f"0x{e4m3_byte(rng):02X}" for _ in range(k)] == rec["b"], name


def test_the_same_operands_give_different_words_on_hopper_and_ada() -> None:
    """The two drafts are different functions: one group of 32 versus two of 16 (the K=1536 vectors share operands)."""
    h, a = vector(FP8_HOPPER_WGMMA.reference_vectors[1]), vector(FP8_ADA_MMA.reference_vectors[1])
    assert h["a"] == a["a"] and h["b"] == a["b"]
    differing = sum(1 for x, y in zip(h["steps"], a["steps"], strict=True) if x["c_out"] != y["c_out"])
    assert differing == 36 and h["y"] != a["y"]


def test_the_domain_is_finite_only() -> None:
    one = 0x38  # E4M3 1.0
    big = 0x7E  # 448, finite
    for t in DRAFTS:
        assert t.pipeline.step(0, [one] * 32, [one] * 32) == 0x42000000  # 32.0
        assert t.pipeline.step(0, [big] + [0] * 31, [big] + [0] * 31) == 0x48440000  # 200704.0
        for nan in (0x7F, 0xFF):
            with pytest.raises(InvalidArtifact):
                t.pipeline.step(0, [nan] + [one] * 31, [one] * 32)
        with pytest.raises(InvalidArtifact):
            t.pipeline.step(0x7F800000, [one] * 32, [one] * 32)
