"""circuit-check's bindings of `verity_vllm.program.registry.boolean_tp_allreduce`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The TP8 all-reduces on bits: FlashInfer's MNNVL one of 8 elements, one- and two-shot, and vLLM's custom one of 64
    elements, 1-stage and 2-stage (every rank owns one pack, so every START is reached)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_tp_allreduce as BA

    return [
        *[bind(BA.AllReduceMnnvl, WORLD=8, N=8, SHOT=s) for s in (1, 2)],
        *[bind(BA.AllReduceCustom, WORLD=8, SIZE=64, STAGE=s) for s in (1, 2)],
    ]
