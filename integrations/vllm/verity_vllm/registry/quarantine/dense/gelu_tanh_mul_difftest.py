"""difftest adapter for `dense/GeluTanhMul@v1`: production = `torch.ops._C.gelu_tanh_and_mul(out, gu.view(1, 2I))` (vLLM
`GeluAndMul(approximate="tanh")`), I cycles 9216 (Gemma-2 I), 64, 1024."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/GeluTanhMul@v1"
SHAPES = (9216, 64, 1024)


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    I = pick(SHAPES, i)
    return {"statics": {"I": I}, "inputs": [{"shape": [2 * I], "dtype": "bfloat16", "kind": kind}], "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    import vllm._custom_ops  # noqa: F401  (loads _C_stable_libtorch and registers torch.ops._C)
    (gu,) = inputs
    I = int(statics["I"])
    out = torch.empty(1, I, dtype=torch.bfloat16, device=gu.device)
    torch.ops._C.gelu_tanh_and_mul(out, gu.view(1, 2 * I))
    return [out]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.gelu_tanh_mul import GeluTanhMul
    return bind(GeluTanhMul, I=int(statics["I"]))
