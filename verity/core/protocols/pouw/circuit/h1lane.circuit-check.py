"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.h1lane`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """At a 4 × 16 leaf (two lanes, Q = 2) at K = 64 on the s8 accumulator on bits: `PouwH1LaneZy_v1` lane 1 over
    `PouwH1Tile_v1` and `PouwH1Tile_v3` (rows 0 .. 3, columns 2, 3, 6, 7, 10, 11, 14, 15: the lane's offsets in the leaf's
    layout and cells in its record) and `PouwH1Combine_v1` of the two lanes' partials; and `PouwH1Leaf_v1` over v1 at the
    one-lane 4 × 8 leaf, its seeds, its lane and a combine of one (two lanes would double a check of about 8 minutes a lane). None
    where `circuit.segment` is not on the tree, which is where `PouwH1Tile` is registered."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.core.protocols.pouw.circuit import h1lane as L
    from verity.core.protocols.pouw.circuit import h1tile as H

    if H.SG is None:
        return []
    exact = bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8)
    def tile(d: Any, c: int) -> Any:
        return bind(d, EXACT=exact, R=4, C=c, PAIRS=tuple((i, j) for i in range(4) for j in range(c)))

    tiles = [tile(d, 16) for d in (H.PouwH1Tile, H.PouwH1Tile3)]
    return [L.lane(t, 1) for t in tiles] + [L.combine(L.layout(tiles[0])), bind(L.PouwH1Leaf, TILE=tile(H.PouwH1Tile, 8))]
