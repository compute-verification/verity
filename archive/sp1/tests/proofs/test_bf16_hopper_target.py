"""The Hopper BF16 draft target (``verity.ml.circuits.targets.target.BF16_HOPPER``) and its pinned vectors.

FIRST's program and epilogue (``GemmCoordinate<K>``, ``f32_to_bf16``) with the steps on ``HOPPER_BF16_M16N8K16`` (one
group of 16, the 26-bit adder, floor -133) under the family ``veritor.tc-bf16.hopper_bf16_m16n8k16@1``.  The vectors
are model replays from a recorded seed and recipe, not hardware captures; their evidence is the pinned instruction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity.ml.circuits.silicon.cast import f32_to_bf16_word
from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.silicon.models import AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16
from verity_sp1.proofs.families import make_tc_ampere_bf16_gate_set, make_tc_bf16_gate_set
from verity_sp1.proofs.programs import gemm_coordinate
from verity.ml.circuits.targets.target import BF16_HOPPER, FIRST, TARGETS, TC_VECTORS

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists())


def vector(name: str) -> dict:
    return json.loads((ROOT / TC_VECTORS / f"{name}.json").read_text())


def words(xs: list[str]) -> list[int]:
    return [int(x, 16) for x in xs]


def test_the_target_is_a_draft_over_the_pinned_hopper_bf16_mma() -> None:
    t = BF16_HOPPER
    assert t.status == "draft" and t.name in TARGETS and TARGETS[t.name] is t
    inst = INSTRUCTIONS[t.instruction]
    assert t.instruction == "sm90.mma.m16n8k16.bf16" and inst.status == "pinned" and inst.model is t.pipeline
    assert (inst.a_dtype, inst.acc_dtype) == ("bf16", "f32")
    assert t.pipeline is HOPPER_BF16_M16N8K16
    assert (t.pipeline.groups, t.pipeline.width, t.pipeline.zero_exponent) == ((16,), 26, -133)
    assert (AMPERE_BF16_M16N8K16.groups, AMPERE_BF16_M16N8K16.width, AMPERE_BF16_M16N8K16.zero_exponent) == ((8, 8), 25, -132)


def test_program_domain_and_epilogue_are_firsts() -> None:
    """Same program and same cast as FIRST; only the family (the pipeline the steps run on) and the peak differ."""
    assert BF16_HOPPER.program == FIRST.program and BF16_HOPPER.domain == FIRST.domain
    assert BF16_HOPPER.statement_format == FIRST.statement_format == 2
    assert BF16_HOPPER.epilogue.startswith("f32_to_bf16: RN-even") and "0x7FC0" in BF16_HOPPER.epilogue
    assert BF16_HOPPER.family != FIRST.family and BF16_HOPPER.family_digest != FIRST.family_digest


def test_the_anchor_is_the_h100() -> None:
    assert BF16_HOPPER.anchor_device == "NVIDIA H100 SXM5 80GB"


def test_the_family_is_firsts_vocabulary_on_the_hopper_pipeline() -> None:
    gate_set = make_tc_bf16_gate_set(HOPPER_BF16_M16N8K16)
    assert f"{gate_set.name}@{gate_set.version}" == BF16_HOPPER.family
    assert gate_set.digest == BF16_HOPPER.family_digest
    ampere = make_tc_ampere_bf16_gate_set()
    assert sorted(g.name for g in gate_set) == sorted(g.name for g in ampere) == [
        "bf16_to_f32", "f32_to_bf16", "in", "tc_dot16", "tc_dot16_0", "weight",
    ]
    assert ampere.digest == FIRST.family_digest, "the Ampere family is untouched"
    # The same vocabulary: where the pipelines agree the gates agree; the k1536 vector below is where they do not.
    one = 0x3F80
    args_one = (0, *([one] * 16), *([one] * 16))
    assert gate_set["tc_dot16"].evaluate(args_one) == ampere["tc_dot16"].evaluate(args_one) == 0x41800000  # 16.0
    assert gate_set["tc_dot16"].evaluate(args_one) == HOPPER_BF16_M16N8K16.step(0, [one] * 16, [one] * 16)
    assert gate_set["f32_to_bf16"].evaluate((0x7FC00001,)) == 0x7FC0
    with pytest.raises(ValueError):
        make_tc_bf16_gate_set(INSTRUCTIONS["sm90.wgmma.m64n8k32.e4m3"].model)


@pytest.mark.parametrize("which", [0, 1], ids=["k16", "k1536"])
def test_each_reference_vector_replays_on_the_model_and_names_the_program(which: int) -> None:
    t = BF16_HOPPER
    name = t.reference_vectors[which]
    rec = vector(name)
    assert rec["schema"] == "verity/tc-reference-vector/v0" and rec["name"] == name
    assert rec["target"] == t.name and rec["status"] == "draft" and rec["instruction"] == t.instruction
    assert rec["model"] == t.pipeline.name and rec["family"] == t.family and rec["family_digest"] == t.family_digest
    assert rec["operand_format"] == "bf16"
    k, step_k = rec["k"], rec["step_k"]
    assert k == (16, 1536)[which] and step_k == t.pipeline.k == 16
    assert rec["program"] == f"GemmCoordinate<{k}>" and gemm_coordinate(k).digest == rec["program_digest"]
    a, b = words(rec["a"]), words(rec["b"])
    assert len(a) == len(b) == k and all((w & 0x7F80) != 0x7F80 for w in a + b), "finite BF16 only"
    assert any(w & 0x7FFF == 0 for w in a + b) and any(0 < (w & 0x7FFF) < 0x80 for w in a + b), "zeros and subnormals"
    c = int(rec["c0"], 16)
    assert c == 0 and len(rec["steps"]) == k // step_k
    for s, step in enumerate(rec["steps"]):
        assert int(step["c_in"], 16) == c
        c = t.pipeline.step(c, a[16 * s : 16 * s + 16], b[16 * s : 16 * s + 16])
        assert c == int(step["c_out"], 16), f"{name}: step {s}"
    assert c == int(rec["acc"], 16)
    assert f32_to_bf16_word(c) == int(rec["y"], 16), "the epilogue word"
    assert (ROOT / TC_VECTORS / f"{name}.json").stat().st_size < 1_000_000


def test_the_vectors_are_regenerable_from_their_seed_and_recipe() -> None:
    def bf16_word(rng: random.Random) -> int:
        u = rng.random()
        sign = rng.randrange(2) << 15
        if u < 0.10:
            return sign
        if u < 0.25:
            return sign | rng.randrange(1, 128)
        if u < 0.30:
            return sign | 0x3F80
        return sign | (rng.randrange(96, 160) << 7) | rng.randrange(128)

    for name in BF16_HOPPER.reference_vectors:
        rec = vector(name)
        rng = random.Random(rec["generator"]["seed"])
        k = rec["k"]
        assert [f"0x{bf16_word(rng):04X}" for _ in range(k)] == rec["a"], name
        assert [f"0x{bf16_word(rng):04X}" for _ in range(k)] == rec["b"], name


def test_the_k1536_vector_separates_hopper_from_ampere() -> None:
    """Same operands, different adders (26 bits / one group of 16 versus 25 bits / two groups of 8): the words differ."""
    rec = vector(BF16_HOPPER.reference_vectors[1])
    a, b = words(rec["a"]), words(rec["b"])
    ch = ca = 0
    differing = 0
    for s in range(96):
        ch = HOPPER_BF16_M16N8K16.step(ch, a[16 * s : 16 * s + 16], b[16 * s : 16 * s + 16])
        ca = AMPERE_BF16_M16N8K16.step(ca, a[16 * s : 16 * s + 16], b[16 * s : 16 * s + 16])
        differing += ch != ca
    assert differing == 42 and ch == int(rec["acc"], 16) and ch != ca


def test_the_domain_is_finite_only() -> None:
    one = 0x3F80
    assert BF16_HOPPER.pipeline.step(0, [one] * 16, [one] * 16) == 0x41800000  # 16.0
    for bad in (0x7F80, 0xFF80, 0x7FC0):
        with pytest.raises(InvalidArtifact):
            BF16_HOPPER.pipeline.step(0, [bad] + [one] * 15, [one] * 16)
    with pytest.raises(InvalidArtifact):
        BF16_HOPPER.pipeline.step(0x7F800000, [one] * 16, [one] * 16)
