"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.pc8`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """Pearl-C's tile check as Pearl-C4's at k = 128 (one promotion window): its templates on the H100's and sm_120's
    records and its roots on sm_120's; its hiding templates and roots on sm_120's, the hiding tile under -h3
    (`Pc8TileHiddenH3`), and the served window's tile on its indexed side rows (`Pc8TileSide`); its hiding roots with the per-tile cap
    (``Shape.cap``, `verity.core.protocols.pouw.circuit.cap`) at k = 256 (two promotion windows) on sm_120's with
    one row a side; and its roots with the row units cut along k (`verity.core.protocols.pouw.circuit.rowk`,
    ``Shape.row_k``) in two segments at k = 256 on sm_120's, plain and hiding."""
    from dataclasses import replace

    from verity.core.protocols.pouw.circuit import pc8

    tile8 = pc8.Shape(1, 128, 1, tile=(1, 1), dev="sm120")
    out = [*pc8.templates(128, tile8.tile, "h100").values(), *pc8.templates(128, tile8.tile, "sm120").values(),
           pc8.weight(tile8), pc8.call(tile8), pc8.check(tile8)]
    h8 = replace(tile8, hidden=True)
    out += [t for t in [*pc8.templates(128, h8.tile, "sm120", True).values(), pc8.weight(h8), pc8.call(h8), pc8.check(h8)]
            if t not in out]
    out += [pc8.templates(128, h8.tile, "sm120", True, "h3")["tile"], pc8.side_tile(128, h8.tile, "sm120")]
    split = pc8.Shape(1, 256, 1, tile=(1, 1), dev="sm120", hidden=True, cap=True)
    out += [pc8.call(split), pc8.check(split)]
    s = pc8.Shape(1, 256, 1, tile=(1, 1), dev="sm120", row_k=128)
    cut = [build(replace(s, hidden=h)) for h in (False, True) for build in (pc8.weight, pc8.call, pc8.check)]
    seen = {t.id for t in out}
    return out + [t for t in cut if t.id not in seen and not seen.add(t.id)]
