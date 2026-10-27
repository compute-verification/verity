"""circuit-check's bindings of `verity.ml.definitions.registry.torch_step`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """F.linear's three GEMMs at the smallest captured shape, M 256 against the k projection (K 576, N 192), and the whole step
    at one layer, B 1 x S 256 (`TrainStep_v1` refuses a shape the cuBLAS capture doesn't cover, so no smaller one builds)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_step as TS

    gemm = {"M": 256, "K": 576, "N": 192}
    return [
        bind(TS.LinearTorch, **gemm),
        bind(TS.LinearTorchDx, **gemm),
        bind(TS.LinearTorchDw, **gemm),
        bind(TS.TrainStep, L=1, B=1, S=256),
    ]
