"""`check_identity`: a fully qualified id names one body. A body under a registered definition's id that is not the
registered body is `E_IDENTITY`, never first-wins (two bodies under one id in one program are refused at the Program's build:
`codec.IdentityConflict`). Pure `verity.circuits`: no torch."""

from __future__ import annotations

import pytest
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.types import Array
from verity_vllm.definitions.registry import b1
from verity_vllm.definitions.registry.prims import BF16, Bf16Add

from verity_torch.frontend.errors import FrontendError
from verity_torch.frontend.program_identity import check_identity


def _leaf(prim) -> CompositeDefinition:
    return CompositeDefinition("IdentityLeaf", 1, ("K",), lambda S: ((("x", Array(S.K, BF16)),), Array(S.K, BF16)),
                               lambda B, S, x: B.batch(prim, x, x, axes=(0, 0)), register=False)


def _both(first: CompositeDefinition, second: CompositeDefinition) -> Program:
    both = CompositeDefinition("IdentityBoth", 1, (), lambda S: ((("x", Array(4, BF16)),), Array(4, BF16)),
                               lambda B, S, x: B.call(bind(second, K=4), B.call(bind(first, K=4), x)), register=False)
    return Program(bind(both))


def test_two_definitions_with_one_body_under_one_id_are_one_definition():
    seen = check_identity(_both(_leaf(Bf16Add), _leaf(Bf16Add)))
    assert "IdentityLeaf_v1{K=4}" in seen and "IdentityBoth_v1" in seen


def test_another_body_under_a_registered_id_is_refused():
    not_gemm = CompositeDefinition("Gemm", 1, ("K", "N"),
                                   lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
                                   lambda B, S, x, w: B.batch(Bf16Add, w[0], w[1], axes=(0, 0)), register=False)
    program = Program(bind(not_gemm, K=4, N=4))
    with pytest.raises(FrontendError) as ei:
        check_identity(program)
    assert ei.value.code == "E_IDENTITY"
    assert "the authored body of Gemm_v1{K=4,N=4} differs from the registered definition's body" in ei.value.message
    assert "Gemm_v1{K=4,N=4}" in check_identity(program, check_registry=False)


def test_the_registered_definition_itself_passes():
    assert "Gemm_v1{K=32,N=4}" in check_identity(Program(bind(b1.Gemm, K=32, N=4)))
