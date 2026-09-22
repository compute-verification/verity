"""difftest adapter for `dense/NarrowF32ToBf16@v1`: production = `aten.to.dtype(x, bfloat16)` (`x.to(torch.bfloat16)`)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import one_input, pick

SEMANTIC_ID = "dense/NarrowF32ToBf16@v1"
SHAPES = (2304, 64, 1024)


def cases(rng, i: int) -> dict:
    return one_input(i, pick(SHAPES, i), "float32")


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [x.to(torch.bfloat16)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.narrow_f32_bf16 import NarrowF32ToBf16
    return bind(NarrowF32ToBf16, N=int(statics["N"]))
