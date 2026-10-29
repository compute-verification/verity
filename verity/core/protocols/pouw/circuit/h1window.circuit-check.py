"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.h1window`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """K = 64's template (`h1window.template`) over the zy unit at the smoke's 4 × 8 tile (`h1window.tile_unit`), its whole
    return a sink, and the window of two steps of two linears at that K (n = 16 and 8, slot 4) whose calls are each one
    batch of that template (2 and 1 tiles). None where `circuit.segment` is not on the tree, which is where `PouwH1Tile`
    is registered."""
    from verity.core.protocols.pouw.circuit import h1tile as H
    from verity.core.protocols.pouw.circuit import h1window as HW

    if H.SG is None:
        return []
    tile = (4, 8)
    window = HW.program((("q_proj", 64, 16), ("k_proj", 64, 8)),
                        tuple((lin, s) for s in range(2) for lin in ("q_proj", "k_proj")), 4, tile)
    return [HW.template(64, HW.tile_unit(64, tile)), window.fn]
