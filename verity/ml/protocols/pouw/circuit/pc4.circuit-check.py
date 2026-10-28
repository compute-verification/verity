"""circuit-check's bindings of `verity.ml.protocols.pouw.circuit.pc4`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """Pearl-C4's tile check: its unit templates at k = 64 with a one-word tile (small enough to lower whole), and its weight
    side, call side and whole check at one row a side; its hiding tiles and roots (``hidden``: the tile digest,
    `verity.ml.protocols.pouw.circuit.leaves`; their rows are C-Flock's ports) at the same shapes; and its roots with the row units cut
    along k (`verity.ml.protocols.pouw.circuit.rowk`, ``Shape.row_k``) in two segments at k = 128, plain and hiding."""
    from dataclasses import replace

    from verity.ml.protocols.pouw.circuit import pc4

    tile = pc4.Shape(1, 64, 1, tile=(1, 1))
    out = [*pc4.templates(64, tile.tile).values(), pc4.weight(tile), pc4.call(tile), pc4.check(tile)]
    h4 = replace(tile, hidden=True)
    out += [t for t in [*pc4.templates(64, h4.tile, True).values(), pc4.weight(h4), pc4.call(h4), pc4.check(h4)] if t not in out]
    s = pc4.Shape(1, 128, 1, tile=(1, 1), row_k=64)
    cut = [build(replace(s, hidden=h)) for h in (False, True) for build in (pc4.weight, pc4.call, pc4.check)]
    seen = {t.id for t in out}
    return out + [t for t in cut if t.id not in seen and not seen.add(t.id)]
