"""difftest adapter for `dense/SquareF32@v1`: production = `aten.pow.Tensor_Scalar(x, 2)` on float32."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import one_input, pick

SEMANTIC_ID = "dense/SquareF32@v1"
SHAPES = (2304, 64, 1024)


def cases(rng, i: int) -> dict:
    return one_input(i, pick(SHAPES, i), "float32")


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [torch.pow(x, 2)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.square_f32 import SquareF32
    return bind(SquareF32, N=int(statics["N"]))
