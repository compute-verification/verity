"""The frozen first target (``verity.ml.circuits.targets.target``) is exactly what the recorded vectors prove.

Each test pins one clause of the target's statement against the code that implements it and against the
conformance vectors a correct prover produced public values for; a semantic drift in any column fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.ml.circuits.silicon.cast import f32_to_bf16_hw_word, f32_to_bf16_word
from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot
from verity_sp1.proofs.families import make_tc_ampere_bf16_gate_set
from verity_sp1.proofs.programs import gemm_coordinate
from verity.ml.circuits.targets.target import FAMILY, FAMILY_DIGEST, FIRST, K16_VECTOR, K1536_VECTOR

FIXTURES = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists()) / "fixtures" / "typed-obligation-v0"


def vector(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def test_the_family_is_the_pinned_manifest() -> None:
    gate_set = make_tc_ampere_bf16_gate_set("1")
    assert f"{gate_set.name}@{gate_set.version}" == FAMILY == FIRST.family
    assert gate_set.digest == FAMILY_DIGEST == FIRST.family_digest
    assert sorted(gate.name for gate in gate_set) == ["bf16_to_f32", "f32_to_bf16", "in", "tc_dot16", "tc_dot16_0", "weight"]


def test_the_pipeline_is_ampere_two_groups_of_eight_25_bits_floor_minus_132() -> None:
    p = FIRST.pipeline
    assert p is AMPERE_BF16_M16N8K16
    assert (p.arch, p.dtype, p.operand_bits) == ("sm_80", "bf16", 16)
    assert p.groups == (8, 8)
    assert p.width == 25
    assert p.zero_exponent == -132


@pytest.mark.parametrize("name", [K16_VECTOR, K1536_VECTOR])
def test_each_reference_vector_is_one_gemm_coordinate_in_the_family(name: str) -> None:
    rec = vector(name)
    obj = rec["object"]
    assert obj["format"] == "typed-obligation/v0"
    assert obj["family"]["id"] == FAMILY and obj["family"]["digest"] == FAMILY_DIGEST
    assert rec["statement"]["format"] == FIRST.statement_format if isinstance(rec.get("statement"), dict) else True
    [sc] = obj["subcircuits"]
    k = sc["definition"]["static"]["K"]
    assert sc["definition"]["name"] == "GemmCoordinate" and k % 16 == 0
    ops = [g["op"] for g in sc["gates"]]
    # step 0 without an accumulator, then K/16 - 1 chained steps in ascending k, then the epilogue cast
    assert ops == ["tc_dot16_0", *["tc_dot16"] * (k // 16 - 1), "f32_to_bf16"]
    assert len(sc["in"]) == 2 * k and sc["out"] == [len(ops) - 1]
    # the same body verity_sp1.proofs.programs constructs
    assert gemm_coordinate(k).digest == sc["digest"]
    assert name in FIRST.reference_vectors


def test_the_chain_is_sequential_in_k_with_one_accumulator() -> None:
    sub = gemm_coordinate(48)
    steps = sub.gates[:-1]
    assert steps[0].op == "tc_dot16_0" and steps[0].args == tuple(("in", i) for i in [*range(16), *range(48, 64)])
    for s in (1, 2):
        expected = (("gate", s - 1), *(("in", 16 * s + i) for i in range(16)), *(("in", 48 + 16 * s + i) for i in range(16)))
        assert steps[s].op == "tc_dot16" and steps[s].args == expected
    assert sub.gates[-1].op == "f32_to_bf16" and sub.gates[-1].args == (("gate", 2),)


def test_the_domain_is_finite_only() -> None:
    one = 0x3F80  # BF16 1.0
    nan = 0x7FC0
    inf = 0x7F80
    finite = tc_dot(AMPERE_BF16_M16N8K16, 0, [one] * 16, [one] * 16)
    assert finite == 0x41800000  # 16.0
    for bad in (nan, inf):
        with pytest.raises(InvalidArtifact):
            tc_dot(AMPERE_BF16_M16N8K16, 0, [bad] + [one] * 15, [one] * 16)
    with pytest.raises(InvalidArtifact):
        tc_dot(AMPERE_BF16_M16N8K16, 0x7FC00000, [one] * 16, [one] * 16)


def test_the_epilogue_is_rn_even_with_the_software_nan_word() -> None:
    assert f32_to_bf16_word(0x41800000) == 0x4180                      # exact
    assert f32_to_bf16_word(0x3F808000) == 0x3F80                      # tie -> even (down)
    assert f32_to_bf16_word(0x3F818000) == 0x3F82                      # tie -> even (up)
    assert f32_to_bf16_word(0x7F7FFFFF) == 0x7F80                      # FP32 max rounds beyond BF16 -> +inf
    assert f32_to_bf16_word(0x7FC00001) == 0x7FC0                      # @1: software canonical NaN
    assert f32_to_bf16_hw_word(0x7FC00001) == 0x7FFF                   # @2: hardware word (unresolved, not the target)
