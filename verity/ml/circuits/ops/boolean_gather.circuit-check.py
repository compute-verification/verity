"""circuit-check's bindings of `verity.ml.circuits.ops.boolean_gather`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_gather_roots() -> list[Any]:
    """The Boolean embedding (`verity.ml.circuits.ops.boolean_gather`) over gathers traced whole (V = 64), and a gather
    that is a tree of Calls with a partial last chunk (V = 1100); the bf16 widening (`verity.ml.circuits.silicon.definitions.boolean.cast`) is static-free."""
    from verity.primitives.circuits.defs import bind
    from verity.ml.circuits.ops import boolean_gather as BGA

    return [
        bind(BGA.Embedding, V=64, H=16),
        bind(BGA.gather_bf16(1100)),
    ]
