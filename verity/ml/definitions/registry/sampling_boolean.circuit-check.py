"""circuit-check's bindings of `verity.ml.definitions.registry.sampling_boolean`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Greedy token selection (`TokenSelect_v2`) at one, two and three levels of its tree, which reach every stage, leaf and
    merge shape (a full group, a remainder, the first group, the root group); the f32 argmax tree's stages at two and three
    levels reach its other shapes.  The argmax step is static-free."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import sampling_boolean as SB

    return [
        bind(SB.TokenSelect, V=50),
        bind(SB.TokenSelect, V=200),
        bind(SB.TokenSelect, V=4161),
        *SB.stages(200, SB.F32ArgmaxStage),
        *SB.stages(4161, SB.F32ArgmaxStage),
    ]
