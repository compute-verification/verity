"""program_compare reads a Program's input nodes by binding time: a `RegisteredInput<w>_v1` node is an input node like an `Input<w>_v1`
one, and only a run `Input64_v1` node of one leaf is a seed."""
from __future__ import annotations

import pytest

from verity.ml.circuits.codec import encode_program
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value
from verity_vllm.check.match import program_compare as C
from verity.ml.definitions.registry import prims as P

PARAMS = (("prompt", Array(1, Value(32))), ("seed", Array(1, Value(64))), ("k", Array(1, Value(64))), ("weights", Array(2, Value(16))))


def _body(B, S, prompt, seed, k, weights):
    return B.call(P.I32Add, prompt[0], prompt[0])


@pytest.mark.parametrize("compact", [False, True])
def test_registered_inputs_are_input_nodes_and_never_seeds(compact):
    fn = CompositeDefinition("CompareRegisteredToy", 1, (), lambda S: (PARAMS, Value(32)), _body, register=False)
    desc = encode_program(Program(bind(fn), registered=("k", "weights")))
    prog = C.Prog("toy", desc, {3: "w"}, tok_node=0, compact=compact)
    assert sorted(prog.inputs) == [0, 1, 2, 3]
    assert prog.inputs[2] == ("RegisteredInput64_v1", 1) and prog.inputs[3] == ("RegisteredInput16_v1", 2)
    assert prog.seed_nodes == {1} and prog.seed_index == {1: 0}
    assert [x["spec"] for x in prog.inst] == ["I32Add_v1"]
