"""difftest adapter for `dense/ScaleRowF32@v1`: production = `aten.mul.Tensor(x.view(1, N), s.view(1, 1))` on float32 — the
`[M,N] * [M,1]` broadcast of GemmaRMSNorm's `(x + residual) * rsqrt(...)`."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/ScaleRowF32@v1"
SHAPES = (2304, 64, 1024)


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N = pick(SHAPES, i)
    return {"statics": {"N": N}, "inputs": [{"shape": [N], "dtype": "float32", "kind": kind},
                                            {"shape": [1], "dtype": "float32", "kind": "zeros" if kind == "zeros" else "random"}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    x, s = inputs
    return [x.view(1, -1) * s.view(1, 1)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.scale_row_f32 import ScaleRowF32
    return bind(ScaleRowF32, N=int(statics["N"]))
