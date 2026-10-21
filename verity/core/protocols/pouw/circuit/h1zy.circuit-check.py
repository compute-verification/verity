"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.h1zy`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwZyCheck_v1` at the smoke's 4 × 8 unit (32 `PouwZyDequant_v2` and `PouwZyCell_v2`, then 31 `PouwAndBit_v2`), and
    `PouwH1TileZy_v1` over `PouwH1Tile_v1` and `PouwH1Tile_v3` at their smallest tile, one lane (4 × 8) at K = 64 on the s8
    accumulator on bits (`h1tile`'s sidecar's), each the tile and that check. None where `circuit.segment` is not on the
    tree, which is where `PouwH1Tile` is registered. `PouwZyDequant`, `PouwZyCell` and `PouwAndBit` are catalog roots
    (static-free)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.core.protocols.pouw.circuit import h1tile as H
    from verity.core.protocols.pouw.circuit import h1zy as Z

    if H.SG is None:
        return []
    exact = bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8)
    pairs = tuple((i, j) for i in range(4) for j in range(8))
    return [bind(Z.PouwZyCheck, R=4, C=8)] + [
        Z.unit(bind(d, EXACT=exact, R=4, C=8, PAIRS=pairs)) for d in (H.PouwH1Tile, H.PouwH1Tile3)]
