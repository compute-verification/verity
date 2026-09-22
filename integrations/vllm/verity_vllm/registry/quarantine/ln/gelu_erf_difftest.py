"""difftest adapter for `ln/GeluErf@v1`: production = `torch.nn.functional.gelu(x)` (approximate='none') on a bf16 row on the
pod, N cycles 3072 (Pythia-160M I), 64, 1024."""
from __future__ import annotations

from verity_vllm.registry.quarantine.ln._difftest_common import kind_for, pick

SEMANTIC_ID = "ln/GeluErf@v1"
SHAPES = (3072, 64, 1024)


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N = pick(SHAPES, i)
    return {"statics": {"N": N}, "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind}], "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [torch.nn.functional.gelu(x)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.ln.gelu_erf import GeluErf
    return bind(GeluErf, N=int(statics["N"]))
