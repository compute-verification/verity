"""circuit-check's bindings of `verity.ml.definitions.registry.torch_clip`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The whole clip over a 3-element tensor (the strided path) and a 4-element one (the 4-vector path), which reaches every
    family at its smallest; a full lane's 32 squares; and a thread's sum of two chunk slots."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_clip as TC

    return [
        bind(TC.ClipGradNorm, SIZES=(3, 4)),
        bind(TC.Bf16SumSquares, J=32),
        bind(TC.NormThreadSum, K=2),
    ]
