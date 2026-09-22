"""difftest adapter for `dense/SquareBf16@v1`: production = `aten.pow.Tensor_Scalar(x.to(float32), 2)`."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import one_input, pick

SEMANTIC_ID = "dense/SquareBf16@v1"
SHAPES = (2304, 64, 1024)


def cases(rng, i: int) -> dict:
    return one_input(i, pick(SHAPES, i), "bfloat16")


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [torch.pow(x.to(torch.float32), 2)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.square_bf16 import SquareBf16
    return bind(SquareBf16, N=int(statics["N"]))
