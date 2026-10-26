"""difftest adapter for `ln/LayerNormAten@v1`: production = `torch.nn.functional.layer_norm` on one bf16 row with bf16 weight and bias,
which dispatches `aten.layer_norm` -> `vectorized_layer_norm_kernel<c10::BFloat16, float>` for N a multiple of 4 (torch 2.13).

    verity-vllm properties-admission produce --adapter verity_vllm.program.layer_norm_difftest --n 60 --seed 7 --out DIR
    verity-vllm properties-admission check   --adapter verity_vllm.program.layer_norm_difftest --dir DIR --out quarantine.json

The row's kind cycles through every edge kind; weight and bias are random except every seventh case, where they take the row's kind
too.  N covers GPT-NeoX's hidden sizes (Pythia 160M 768, 410M 1024, 1B 2048, 2.8B 2560), a size where some of the 128 threads hold
no vector (64), and one where lanes hold 4 and 8 elements unevenly (2304).
"""
from __future__ import annotations

SEMANTIC_ID = "ln/LayerNormAten@v1"
KINDS = ("random", "zeros", "max", "subnormal", "alternating")
SHAPES = (768, 64, 1024, 2048, 2304, 2560)
EPS = 1e-5


def cases(rng, i: int) -> dict:
    kind = KINDS[i % len(KINDS)]
    N = SHAPES[(i // len(KINDS)) % len(SHAPES)]
    affine = kind if i % 7 == 6 else "random"
    return {"statics": {"N": N, "EPS": EPS},
            "inputs": [{"shape": [N], "dtype": "bfloat16", "kind": kind}, {"shape": [N], "dtype": "bfloat16", "kind": affine},
                       {"shape": [N], "dtype": "bfloat16", "kind": affine}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch

    x, w, b = inputs
    N = int(statics["N"])
    return [torch.nn.functional.layer_norm(x.view(1, N), (N,), w, b, float(statics["EPS"])).view(N)]


def spec(statics: dict):
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry.layer_norm import LayerNormAten

    return bind(LayerNormAten, N=int(statics["N"]), EPS=float(statics["EPS"]))
