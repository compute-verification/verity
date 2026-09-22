"""difftest adapter for `dense/MulVecF32@v1`: production = `aten.mul.Tensor(x.view(1, N), w)` on float32 with w `[N]` — the
`[M,N] * [N]` broadcast of GemmaRMSNorm's `x * (weight.float() + 1.0)`."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/MulVecF32@v1"
SHAPES = (2304, 64, 1024)


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N = pick(SHAPES, i)
    return {"statics": {"N": N}, "inputs": [{"shape": [N], "dtype": "float32", "kind": kind},
                                            {"shape": [N], "dtype": "float32", "kind": "random"}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    x, w = inputs
    return [x.view(1, -1) * w]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.mul_vec_f32 import MulVecF32
    return bind(MulVecF32, N=int(statics["N"]))
