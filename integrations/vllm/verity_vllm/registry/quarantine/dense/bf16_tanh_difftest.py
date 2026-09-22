"""difftest adapter for `dense/Bf16Tanh@v1`: production = `aten.tanh.default(x)` on bfloat16 (Gemma-2 `LogitsProcessor`)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import one_input, pick

SEMANTIC_ID = "dense/Bf16Tanh@v1"
SHAPES = (256000, 64, 2304)


def cases(rng, i: int) -> dict:
    return one_input(i, pick(SHAPES, i), "bfloat16")


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [torch.tanh(x)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.bf16_tanh import Bf16Tanh
    return bind(Bf16Tanh, N=int(statics["N"]))
