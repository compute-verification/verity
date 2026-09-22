"""difftest adapter for `dense/RsqrtF32@v1`: production = `aten.rsqrt.default(x)` on float32 (GemmaRMSNorm: N = 1 per row;
also run at N = 64 / 1024 for coverage of the MUFU.RSQ table and its subnormal / zero / negative (NaN) paths)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import one_input, pick

SEMANTIC_ID = "dense/RsqrtF32@v1"
SHAPES = (1, 64, 1024)


def cases(rng, i: int) -> dict:
    return one_input(i, pick(SHAPES, i), "float32")


def production(statics: dict, inputs: list) -> list:
    import torch
    (x,) = inputs
    return [torch.rsqrt(x)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.rsqrt_f32 import RsqrtF32
    return bind(RsqrtF32, N=int(statics["N"]))
