"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.h1tile`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwH1Tile_v1` at its smallest tile, one lane (4 × 8) at K = 64 (KS = 2, one digit word), on the s8 accumulator on bits:
    the seed, 13 noise blocks, 32 exact dots and the lane's four fx1 sets. It grows linearly in lanes and in KS. None where
    `circuit.segment` (E2E-120) is not on the tree, which is where `PouwH1Tile` is registered. `PouwU129` is a catalog root
    (a primitive and a static-free composite)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.core.protocols.pouw.circuit import h1tile as H

    if H.SG is None:
        return []
    exact = bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8)
    return [bind(H.PouwH1Tile, EXACT=exact, R=4, C=8, PAIRS=tuple((i, j) for i in range(4) for j in range(8)))]
