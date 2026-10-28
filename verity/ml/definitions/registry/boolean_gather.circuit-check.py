"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_gather`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_gather_roots() -> list[Any]:
    """The Boolean embedding (`verity.ml.definitions.registry.boolean_gather`) over gathers traced whole (V = 64), and a gather
    that is a tree of Calls with a partial last chunk (V = 1100), bf16 and i32; the bf16 widening
    (`verity.ml.circuits.ops.boolean.cast`) is static-free."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_gather as BGA

    return [
        bind(BGA.Embedding, V=64, H=16),
        bind(BGA.gather_bf16(1100)),
        bind(BGA.gather_i32(1100)),
    ]
