"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.lead`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwLeadTile_v1` at a one-step call (K = 21: step 0 only, the fold over one step's coins and no tag), at a ragged
    tile of a two-step call (K = 49, the tile at row 16 and column 8 holding 1 row and 5 columns of a 17 × 13 call: padding
    rows and columns, a later step's tag lane, a one-tag walk), at block 0 of a 33-step call (K = 917: 32 steps, no fold)
    and at its block 1 (one step after the committed words before it, the fold over all 33 steps' coins and a 32-tag walk,
    reading A·B back from the last word). The pieces, `PouwLeadStep0Row`, `PouwLeadStep0Mma`, `PouwLeadStepRow` and
    `PouwLeadStepMma` are catalog roots (primitives and static-free composites)."""
    from verity.core.protocols.pouw.circuit import lead as L
    from verity.ml.circuits.defs import bind

    shapes = ((21, 0, 0, 16, 8, 0), (49, 16, 8, 1, 5, 0), (917, 0, 0, 16, 8, 0), (917, 0, 0, 16, 8, 1))
    return [bind(L.PouwLeadTile, K=K, R0=R0, C0=C0, ROWS=ROWS, COLS=COLS, BLK=BLK) for K, R0, C0, ROWS, COLS, BLK in shapes]
