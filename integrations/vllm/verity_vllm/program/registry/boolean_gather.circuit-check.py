"""circuit-check's bindings of `verity_vllm.program.registry.boolean_gather`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_gather_roots() -> list[Any]:
    """The Boolean embedding (`verity_vllm.program.registry.boolean_gather`) over gathers traced whole (V = 64), and a gather
    that is a tree of Calls with a partial last chunk (V = 1100); the bf16 widening (`verity.ml.circuits.ops.boolean.cast`) is static-free."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_gather as BGA

    return [
        bind(BGA.Embedding, V=64, H=16),
        bind(BGA.gather_bf16(1100)),
    ]
