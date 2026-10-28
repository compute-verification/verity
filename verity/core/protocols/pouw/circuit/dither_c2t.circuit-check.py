"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.dither_c2t`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwDitherFillC2T_v1` over the form's block of 8 cells at L = 32 (one `PouwSegDot0_v2` a cell), at BYTE-CELL's
    window, bit 9 alone, and over the byte of bits 8 .. 15; and over 2 cells at L = 64 (then one `PouwSegDot_v2`), since 8
    cells at L = 64 pass the whole-Definition lowering budget. Each operand is formed by `PouwI8Add_v2`, the first window
    bit checked by `PouwDitherBitEq_v2` and the rest by `PouwDitherBitEqAnd_v2`. It grows linearly in CELLS · L / 32 and
    in CELLS · W, which larger shapes are priced by."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import dither_c2t as C

    return [bind(C.PouwDitherFillC2T, L=L, CELLS=c, R=r, W=w) for L, c, r, w in ((32, 8, 9, 1), (32, 8, 8, 8), (64, 2, 9, 1))]
