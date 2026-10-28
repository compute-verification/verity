"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.dither_amax`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwAmaxBlock_v1` at W = 8, C2-O's 8-cell blocks of one row (eight `PouwAmaxFold_v2` and `PouwAmaxEq_v2`), and
    `PouwAmaxRow_v1` at Llama 3.1 8B's I = 14,336 in those blocks (N = 1,792: N - 1 `PouwAmaxMax_v2` and `PouwAmaxEq_v2`)
    and at a toy N = 64. `PouwSiluMulBf16_v2`, the gate block's h, is bound alone."""
    from verity.core.protocols.pouw.circuit import dither_amax as AM

    return [AM.gate_definition(8), AM.row_definition(14336, 8), AM.row_definition(512, 8), AM.PouwSiluMulBf16]
