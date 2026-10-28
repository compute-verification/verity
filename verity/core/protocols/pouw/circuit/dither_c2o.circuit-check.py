"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.dither_c2o`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwDitherServedC2O_v1` over the form's block of 8 cells at K = 32 (one `PouwDitherOutDot0_v2` a cell), at
    BYTE-CELL's window, bit 9 alone; and over 2 cells at K = 64 (then one `PouwDitherOutDot_v2`) and over the byte of bits
    8 .. 15, since 8 cells at K = 64 pass the whole-Definition lowering budget. Each window bit is checked by `dither_c2t`'s
    `PouwDitherBitEq_v2` and `PouwDitherBitEqAnd_v2`. It grows linearly in CELLS · K / 32 and in CELLS · W, which larger
    shapes are priced by."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import dither_c2o as O

    return [bind(O.PouwDitherServedC2O, K=k, CELLS=c, R=r, W=w) for k, c, r, w in ((32, 8, 9, 1), (64, 2, 9, 1), (64, 2, 8, 8))]
