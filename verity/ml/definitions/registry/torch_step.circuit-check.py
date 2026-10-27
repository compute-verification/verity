"""circuit-check's bindings of `verity.ml.definitions.registry.torch_step`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """F.linear's three GEMMs on two rows of a row GEMM at K 64, N 3, each on a route kind its role takes in the capture: the
    forward unsplit, dX split-K with bf16 partials, dW grid.z's aligned residue-first split; and the whole step at one layer,
    B 1 x S 256 (`TrainStep_v1` refuses a shape the cuBLAS capture doesn't cover, so no smaller one builds)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_step as TS

    return [
        bind(TS.LinearTorch, M=2, K=64, N=3, ROUTE={"kernel": "nosplit"}),
        bind(TS.LinearTorchDx, M=2, K=3, N=64, ROUTE={"kernel": "splitk", "SPLIT_K": 4, "partial": "bf16"}),
        bind(TS.LinearTorchDw, M=64, K=3, N=2,
             ROUTE={"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8, "tile_k": 16}),
        bind(TS.TrainStep, L=1, B=1, S=256),
    ]
