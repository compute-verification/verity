"""The committer names the Program inputs its classes cover by binding and leaf width (`verity.ml.circuits.program.input_prim`), never by literal id."""
from __future__ import annotations

from verity.ml.circuits.program import REGISTRATION, RUN
from verity_vllm.commit.committer.input_classes import weights_inputs


def _inputs(names, binding=RUN):
    return weights_inputs({"names": names}, binding)


def test_weights_cover_one_input_per_registered_leaf_width():
    fp8 = [["w", "float8_e4m3fn", [4, 4], 16], ["s", "float32", [1], 4], ["n", "bfloat16", [4], 8]]
    assert _inputs(fp8) == ["Input16_v1", "Input32_v1", "Input8_v1"]
    assert _inputs([["n", "bfloat16", [4], 8]], REGISTRATION) == ["RegisteredInput16_v1"]
    assert _inputs([]) == ["Input16_v1"]
