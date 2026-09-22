"""difftest adapter for `ln/LayerNormAten@v1`: production = `torch.nn.functional.layer_norm(x.view(1, N), (N,), w, b, eps)` in bf16
on the pod (the vectorized CUDA kernel; N a multiple of 4 keeps that path), N cycles 768 (Pythia-160M H), 64, 1024, 2304, 256;
`eps` = 1e-5 (Pythia's `layer_norm_eps`).  The weight / bias rows use the same edge kind as the row (a zero row with a random
weight would test little of the statistics path) except for `zeros` and `subnormal`, whose weight/bias are random so the
epilogue is exercised."""
from __future__ import annotations

from verity_vllm.registry.quarantine.ln._difftest_common import kind_for, pick

SEMANTIC_ID = "ln/LayerNormAten@v1"
SHAPES = (768, 64, 1024, 2304, 256)
EPS = 1e-5


def cases(rng, i: int) -> dict:
    kind = kind_for(i)
    N = pick(SHAPES, i)
    wk = "random" if kind in ("zeros", "subnormal") else kind
    return {"statics": {"N": N, "EPS": EPS},
            "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind}, {"shape": [N], "dtype": "bfloat16", "kind": wk},
                       {"shape": [N], "dtype": "bfloat16", "kind": wk}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    x, w, b = inputs
    N = int(statics["N"])
    y = torch.nn.functional.layer_norm(x.view(1, N), (N,), w, b, float(statics["EPS"]))
    return [y.view(N)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.ln.layer_norm_aten import LayerNormAten
    return bind(LayerNormAten, N=int(statics["N"]), EPS=float(statics["EPS"]))
