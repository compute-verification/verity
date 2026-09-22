"""difftest adapter for `dense/AddWidenedBf16@v1`: production = `aten.add.Tensor(x.to(float32), r.to(float32))`
(`ir.ops.fused_add_rms_norm`'s residual add)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import kind_for, pick

SEMANTIC_ID = "dense/AddWidenedBf16@v1"
SHAPES = (2304, 64, 1024)


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N = pick(SHAPES, i)
    return {"statics": {"N": N}, "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind},
                                            {"shape": [N], "dtype": "bfloat16", "kind": "random" if kind != "alternating" else "alternating"}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    x, r = inputs
    return [x.to(torch.float32) + r.to(torch.float32)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.dense.add_widened_bf16 import AddWidenedBf16
    return bind(AddWidenedBf16, N=int(statics["N"]))
