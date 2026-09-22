"""difftest adapter for `dense/Bf16MulScalar@v1`: production = `aten.mul.Tensor(x, C)` with a Python float C (Gemma-2
`LogitsProcessor`: `tanh(...) * 30.0`)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/Bf16MulScalar@v1"
SHAPES = ((256000, 30.0), (64, 30.0), (2304, 30.0), (64, 7.0))


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N, C = pick(SHAPES, i)
    return {"statics": {"N": N, "C": C}, "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind}], "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [torch.mul(x, float(statics["C"]))]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.bf16_mul_scalar import Bf16MulScalar
    return bind(Bf16MulScalar, N=int(statics["N"]), C=float(statics["C"]))
