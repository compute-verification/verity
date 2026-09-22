"""difftest adapter for `dense/AddScalarF32@v1`: production = `aten.add.Tensor(x, C)` on float32 with a Python float C
(GemmaRMSNorm: `variance + 1e-6`, N = 1).  Also exercised at N = 64 and with C = 1.0."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/AddScalarF32@v1"
SHAPES = ((1, 1e-6), (64, 1e-6), (1, 1.0), (2304, 1e-6))


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N, C = pick(SHAPES, i)
    return {"statics": {"N": N, "C": C}, "inputs": [{"shape": [N], "dtype": "float32", "kind": kind}], "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [torch.add(x, float(statics["C"]))]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.add_scalar_f32 import AddScalarF32
    return bind(AddScalarF32, N=int(statics["N"]), C=float(statics["C"]))
