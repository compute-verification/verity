"""difftest adapter for `dense/Bf16MulScalarTensor@v1`: production = `aten.mul.Tensor(x.view(1, N), s.reshape(()))` — a bf16
tensor times a 0-d bf16 CUDA tensor (Gemma2Model's `embed * self.normalizer`)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/Bf16MulScalarTensor@v1"
SHAPES = (2304, 64, 1024)


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N = pick(SHAPES, i)
    return {"statics": {"N": N}, "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind},
                                            {"shape": [1], "dtype": "bfloat16", "kind": "random" if kind != "max" else "max"}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    x, s = inputs
    return [x.view(1, -1) * s.reshape(())]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.bf16_mul_scalar_tensor import Bf16MulScalarTensor
    return bind(Bf16MulScalarTensor, N=int(statics["N"]))
