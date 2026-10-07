"""The NVFP4 draft target (``verity.ml.circuits.targets.target.NVFP4_SM120``) and its pinned vectors.

Like ``test_fp8_targets.py``: each clause of the statement is pinned to the code that implements it (the model, the
program builder, the family) and the recorded ``verity/ml/circuits/silicon/captures/vectors`` replay on the model.  The vectors are model
replays, not hardware captures: their evidence is the instruction pin the target names (lane fp4-re's 0-mismatch
sweeps on the RTX 5090), and a semantic drift in the model, the program or the vector fails here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4, BlockScaledAlignAdd
from verity_sp1.proofs.families import make_tensor_core_gate_set
from verity_sp1.proofs.programs import gemm_accumulator_nvfp4
from verity.ml.circuits.targets.target import FP8_ADA_MMA, NVFP4_SM120, TARGETS, TC_VECTORS

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists())


def vector(name: str) -> dict:
    return json.loads((ROOT / TC_VECTORS / f"{name}.json").read_text())


def words(xs: list[str]) -> list[int]:
    return [int(x, 16) for x in xs]


def test_the_target_is_a_draft_over_the_pinned_nvfp4_instruction() -> None:
    t = NVFP4_SM120
    assert t.status == "draft" and TARGETS[t.name] is t
    inst = INSTRUCTIONS[t.instruction]
    assert t.instruction == "sm120.mma.m16n8k64.e2m1.nvf4" and inst.status == "pinned" and inst.model is t.pipeline
    assert (inst.a_dtype, inst.b_dtype, inst.acc_dtype, inst.scale) == ("e2m1", "e2m1", "f32", "ue4m3")
    assert isinstance(t.pipeline, BlockScaledAlignAdd) and t.pipeline is BLACKWELL_SM120_NVF4
    assert (t.pipeline.k, t.pipeline.group, t.pipeline.block, t.pipeline.words_per_step) == (64, 16, 16, 68)
    assert (t.pipeline.frac_bits, t.pipeline.acc_window, t.pipeline.grid_floor, t.pipeline.underflow_zero) == (27, 36, -174, "positive")
    assert "D20" in t.domain and "0x7F" in t.domain and "SEPARATE relation" in t.epilogue
    assert "+0" in t.transition_unit and "-174" in t.transition_unit and "36-bit" not in t.transition_unit


def test_the_anchor_is_the_rtx_5090() -> None:
    assert NVFP4_SM120.anchor_device == "NVIDIA GeForce RTX 5090"


def test_the_family_is_the_cast_free_tensor_core_gate_set_with_scale_bytes() -> None:
    gs = make_tensor_core_gate_set(NVFP4_SM120.pipeline)
    assert f"{gs.name}@{gs.version}" == NVFP4_SM120.family and gs.digest == NVFP4_SM120.family_digest
    assert sorted(g.name for g in gs) == ["in", "tc_dot64", "tc_dot64_0", "weight"]
    step = next(g for g in gs if g.name == "tc_dot64")
    assert step.arg_widths == (32, *([4] * 64 + [8] * 4) * 2)
    # the FP8 family is untouched by the generalisation (one operand = k codes when there are no scales)
    assert make_tensor_core_gate_set(FP8_ADA_MMA.pipeline).digest == FP8_ADA_MMA.family_digest


def test_the_program_is_a_chain_of_tc_dot64_over_codes_then_scales() -> None:
    sub = gemm_accumulator_nvfp4(128)
    assert [g.op for g in sub.gates] == ["tc_dot64_0", "tc_dot64"]
    a_codes, a_scales, b_codes, b_scales = range(0, 128), range(128, 136), range(136, 264), range(264, 272)
    step0 = [("in", i) for i in list(a_codes)[:64]] + [("in", i) for i in list(a_scales)[:4]] \
        + [("in", i) for i in list(b_codes)[:64]] + [("in", i) for i in list(b_scales)[:4]]
    assert list(sub.gates[0].args) == step0
    step1 = [("gate", 0)] + [("in", i) for i in list(a_codes)[64:]] + [("in", i) for i in list(a_scales)[4:]] \
        + [("in", i) for i in list(b_codes)[64:]] + [("in", i) for i in list(b_scales)[4:]]
    assert list(sub.gates[1].args) == step1
    assert sub.outgoing == (1,) and len(sub.incoming) == 272
    assert list(sub.incoming) == ([4] * 128 + [8] * 8) * 2
    with pytest.raises(ValueError):
        gemm_accumulator_nvfp4(96)


@pytest.mark.parametrize("which", [0, 1], ids=["k64", "k1536"])
def test_each_reference_vector_replays_on_the_model_and_names_the_program(which: int) -> None:
    t = NVFP4_SM120
    name = t.reference_vectors[which]
    rec = vector(name)
    assert rec["schema"] == "verity/tc-reference-vector/v0" and rec["name"] == name
    assert rec["target"] == t.name and rec["status"] == "draft" and rec["instruction"] == t.instruction
    assert rec["model"] == t.pipeline.name and rec["family"] == t.family and rec["family_digest"] == t.family_digest
    k, step_k = rec["k"], rec["step_k"]
    assert k == (64, 1536)[which] and step_k == t.pipeline.k == 64 and rec["block"] == 16
    assert rec["program"] == f"GemmAccumulator<{k}>" and gemm_accumulator_nvfp4(k).digest == rec["program_digest"]
    a, b = words(rec["a"]), words(rec["b"])
    sa, sb = words(rec["a_scales"]), words(rec["b_scales"])
    assert len(a) == len(b) == k and len(sa) == len(sb) == k // 16
    assert all(0 <= x < 16 for x in a + b) and all(0 <= s < 0x7F for s in sa + sb), "valid E2M1 codes and UE4M3 scales only"
    assert any(x & 7 == 0 for x in a + b) and any(x & 7 == 1 for x in a + b) and any(x & 7 == 7 for x in a + b)
    if which == 1:   # 192 scale bytes: the zero scale, the subnormal field and the top exponent all present
        assert any(s == 0 for s in sa + sb) and any(0 < s < 8 for s in sa + sb) and any(s >> 3 == 15 for s in sa + sb)
    c = int(rec["c0"], 16)
    assert c == 0 and len(rec["steps"]) == k // step_k
    for s, step in enumerate(rec["steps"]):
        assert int(step["c_in"], 16) == c
        aw = a[64 * s:64 * s + 64] + sa[4 * s:4 * s + 4]
        bw = b[64 * s:64 * s + 64] + sb[4 * s:4 * s + 4]
        c = t.pipeline.step(c, aw, bw)
        assert c == int(step["c_out"], 16), f"{name}: step {s}"
    assert c == int(rec["y"], 16)


def test_the_vectors_are_regenerable_from_their_seed_and_recipe() -> None:
    for which in (0, 1):
        rec = vector(NVFP4_SM120.reference_vectors[which])
        k, seed = rec["k"], rec["generator"]["seed"]
        rng = random.Random(seed)

        def code() -> int:
            u = rng.random()
            s = 8 if rng.random() < 0.5 else 0
            if u < 0.10:
                return s
            if u < 0.25:
                return s | 1
            if u < 0.30:
                return s | 7
            return rng.randrange(16)

        def scale() -> int:
            v = rng.random()
            if v < 0.05:
                return 0
            if v < 0.15:
                return rng.randrange(1, 8)
            if v < 0.20:
                return (15 << 3) | rng.randrange(7)
            return rng.randrange(8, 0x7F)

        a = [code() for _ in range(k)] + [scale() for _ in range(k // 16)]
        b = [code() for _ in range(k)] + [scale() for _ in range(k // 16)]
        assert words(rec["a"]) + words(rec["a_scales"]) == a and words(rec["b"]) + words(rec["b_scales"]) == b
