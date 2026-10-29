"""circuit-check's bindings of `verity.ml.definitions.registry.torch_step`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """F.linear's three GEMMs on one row of a row GEMM over 64 terms, one output each: unsplit (the route of every forward and
    dW of the step at one layer, B 1 x S 256, and of three of its five dX shapes), and on a split route its role takes in the
    capture: dX split-K with bf16 partials, dW grid.z's aligned residue-first split. Each is under `targets.MAX_GATES`, which
    `targets.select` drops a binding over whenever its family has one under it, so every route `TrainStep_v1`'s calls take
    has a target its by-calls check relies on."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_step as TS

    nosplit = {"kernel": "nosplit"}
    return [
        bind(TS.LinearTorch, M=1, K=64, N=1, ROUTE=nosplit),
        bind(TS.LinearTorchDx, M=1, K=1, N=64, ROUTE=nosplit),
        bind(TS.LinearTorchDx, M=1, K=1, N=64, ROUTE={"kernel": "splitk", "SPLIT_K": 4, "partial": "bf16"}),
        bind(TS.LinearTorchDw, M=64, K=1, N=1, ROUTE=nosplit),
        bind(TS.LinearTorchDw, M=64, K=1, N=1,
             ROUTE={"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8, "tile_k": 16}),
    ]


def _by_calls_roots() -> list[Any]:
    """The whole step at one layer, B 1 x S 256, checked by its calls: lm_head's three GEMMs alone are past what circuit-check
    builds whole, and `TrainStep_v1` refuses a shape the cuBLAS capture doesn't cover, so no smaller one builds."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_step as TS

    return [bind(TS.TrainStep, L=1, B=1, S=256)]
