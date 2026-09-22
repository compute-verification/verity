"""difftest adapter for `dense/Bf16DivScalar@v1`: production = `aten.div.Tensor(x, C)` with a Python float C (Gemma-2
`LogitsProcessor`: `logits / 30.0`); N cycles 256000 (V), 64, 2304."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/Bf16DivScalar@v1"
SHAPES = ((256000, 30.0), (64, 30.0), (2304, 30.0), (64, 7.0))


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N, C = pick(SHAPES, i)
    return {"statics": {"N": N, "C": C}, "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind}], "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [torch.div(x, float(statics["C"]))]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.bf16_div_scalar import Bf16DivScalar
    return bind(Bf16DivScalar, N=int(statics["N"]), C=float(statics["C"]))
