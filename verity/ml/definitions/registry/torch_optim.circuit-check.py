"""circuit-check's bindings of `verity.ml.definitions.registry.torch_optim`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """AdamW's step at N = 1 under both lerp contractions (with and without the bf16 copy; at N = 4 each is past
    `targets.MAX_GATES`, so `select` would keep one), SGD's at N = 4 under both, and the step's host scalars (`AdamWScalars_v1`
    reaches its three parts)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_optim as TO

    return [
        bind(TO.AdamWForeach, N=1, LERP_FMA=1, BF16=1),
        bind(TO.AdamWForeach, N=1, LERP_FMA=0, BF16=1),
        bind(TO.AdamWForeach, N=1, LERP_FMA=1, BF16=0),
        bind(TO.SgdForeach, N=4, ALPHA_FMA=1),
        bind(TO.SgdForeach, N=4, ALPHA_FMA=0),
        bind(TO.AdamWScalars),
        bind(TO.TorchSgdScalar),
    ]
