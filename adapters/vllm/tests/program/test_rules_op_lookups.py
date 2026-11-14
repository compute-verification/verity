"""Three lookups of the vLLM rules answer "no target" or "no table" whatever the lookup raises:
`vllm_bindings.attention._vllm_op`, `vllm_moe._moe_C_op` and `vllm_bindings.pins.tuned_gemm_table_sha256`. A handler
that names the errors it expects lets every other one reach the rule lists' import or the build's record."""
from __future__ import annotations

import pytest

from verity_vllm.program.frontend.rules.vllm_bindings.attention import _vllm_op
from verity_vllm.program.frontend.rules.vllm_bindings.observations import _OBSERVED
from verity_vllm.program.frontend.rules.vllm_bindings.pins import (
    tuned_gemm_table_sha256,
)
from verity_vllm.program.frontend.rules.vllm_moe import _moe_C_op

NAMES = ["no_such_op", "", "a b", "__origin__", None, 3, 1.5, b"x", ("x",), ["x"], {"x": 1}]


@pytest.mark.parametrize("name", NAMES, ids=repr)
def test_a_vllm_op_lookup_that_raises_gives_no_targets(name):
    assert _vllm_op(name) == set()


@pytest.mark.parametrize("name", NAMES, ids=repr)
def test_a_moe_op_lookup_that_raises_gives_no_targets(name):
    assert _moe_C_op(name) == set()


class _Raises:
    """In the observed profile's place: its GEMM target raises `error`."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def gemm_target(self):
        raise self.error


@pytest.mark.parametrize("error", [RuntimeError("x"), AttributeError("x"), KeyError("x"), ValueError("x"), OSError("x")], ids=repr)
def test_a_target_that_raises_has_no_tuned_table(monkeypatch, error):
    monkeypatch.setitem(_OBSERVED, "target_profile", _Raises(error))
    assert tuned_gemm_table_sha256() is None


@pytest.mark.parametrize("made_up", [object(), {"compute_capability": [8, 9]}, "ada"], ids=["an object", "a dict", "a string"])
def test_an_observation_that_is_no_profile_has_no_tuned_table(monkeypatch, made_up):
    monkeypatch.setitem(_OBSERVED, "target_profile", made_up)
    assert tuned_gemm_table_sha256() is None
