"""difftest adapter for `dense/ScaleRowBf16@v1`: production = `aten.mul.Tensor(x.to(float32).view(1, N), s.view(1, 1))` — the
`[M,N] * [M,1]` broadcast of GemmaRMSNorm's `x * rsqrt(...)` on the widened input.  `s` is drawn `random` (a `zeros` s in the
zeros case); the edge kinds are on x."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/ScaleRowBf16@v1"
SHAPES = (2304, 64, 1024)


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N = pick(SHAPES, i)
    return {"statics": {"N": N}, "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind},
                                            {"shape": [1], "dtype": "float32", "kind": "zeros" if kind == "zeros" else "random"}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    x, s = inputs
    return [x.to(torch.float32).view(1, -1) * s.view(1, 1)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.scale_row_bf16 import ScaleRowBf16
    return bind(ScaleRowBf16, N=int(statics["N"]))
