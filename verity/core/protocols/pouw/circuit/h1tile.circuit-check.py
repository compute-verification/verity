"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.h1tile`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwH1Tile_v1` and `PouwH1Tile_v3` at their smallest tile, one lane (4 × 8) at K = 64 (KS = 2, one digit word), on
    the s8 accumulator on bits: the seed, 13 noise blocks, the lane's four fx1 sets, and z (v1: 32 exact dots; v3: 32
    `PouwOneStream2` of the folded O' and D_1, each through two `PouwWrapSeg` of its row's X and its column's
    `PouwWrapBits`, then two `PouwPeelSeg` of its row's X, dither bits and `PouwXWrap` bits and its column's B and coin
    bits). They grow linearly in lanes and in KS. None where `circuit.segment` (E2E-120) is not on the tree, which is where
    `PouwH1Tile` is registered. `PouwU129`, the one-stream recombinations, the wrap pieces and the peel pieces are catalog
    roots (primitives and static-free composites)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.core.protocols.pouw.circuit import h1tile as H

    if H.SG is None:
        return []
    exact = bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8)
    pairs = tuple((i, j) for i in range(4) for j in range(8))
    return [bind(d, EXACT=exact, R=4, C=8, PAIRS=pairs) for d in (H.PouwH1Tile, H.PouwH1Tile3)]
