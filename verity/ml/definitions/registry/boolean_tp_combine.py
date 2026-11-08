"""`registry.tp_combine`'s combine units as Boolean Definitions by σ (`verity.ml.definitions.boolean.sigma_lift`): the next version of
each, its word body with each Call the Boolean version of its Definition (`boolean_tp_allreduce`'s coordinates and vectors,
`Bf16AddF2fp`'s), over the bits each word operand occupies, so its word view is that Definition at the same bindings.

    AllReduceResidualMnnvlCoordinate_v2{WORLD,SHOT}       (col: Array<WORLD, BF16>, res: BF16) -> BF16
    AllReduceMultimemRow_v2{WORLD,N,ROWS,ROW,TREES,FTZ}   (parts: Array<WORLD, Array<N, BF16>>) -> Array<N, BF16>
    AllReduceCustomRow_v2{WORLD,N,ROWS,ROW,STAGE}         (parts: Array<WORLD, Array<N, BF16>>) -> Array<N, BF16>
"""
from __future__ import annotations

from verity.ml.definitions.boolean import sigma_lift
from verity.ml.definitions.registry import tp_combine as TC

__all__ = ["AllReduceCustomRow", "AllReduceMultimemRow", "AllReduceResidualMnnvlCoordinate"]

AllReduceResidualMnnvlCoordinate = sigma_lift(TC.AllReduceResidualMnnvlCoordinate, 2)
AllReduceMultimemRow = sigma_lift(TC.AllReduceMultimemRow, 2)
AllReduceCustomRow = sigma_lift(TC.AllReduceCustomRow, 2)
