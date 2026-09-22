"""difftest adapter for `dense/AddScalarBf16@v1`: production = `aten.add.Tensor(w.to(float32), C)` (GemmaRMSNorm:
`self.weight.float() + 1.0`, N = H)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/AddScalarBf16@v1"
SHAPES = ((2304, 1.0), (64, 1.0), (1024, 1.0))


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N, C = pick(SHAPES, i)
    return {"statics": {"N": N, "C": C}, "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind}], "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    (w,) = inputs
    return [torch.add(w.to(torch.float32), float(statics["C"]))]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.add_scalar_bf16 import AddScalarBf16
    return bind(AddScalarBf16, N=int(statics["N"]), C=float(statics["C"]))
