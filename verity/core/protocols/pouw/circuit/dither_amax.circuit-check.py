"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.dither_amax`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwDitherAmax_v1` at K = 32 (one `PouwDitherOutDot0_v2` a dot) and K = 64 (then one `PouwDitherOutDot_v2`), at Llama
    3.1 8B's I = 14,336 and at a toy I = 64: its two dots, two `PouwZyDequant_v2`, one `PouwSiluMulBf16_v2` and
    `PouwAmaxAt_v2`. The served K = 4,096 is 128 steps a dot, past the whole-Definition lowering budget, and grows linearly
    in K / 32, which it is priced by. `PouwAmaxBound_v2`, the gate block's bound, is bound alone."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import dither_amax as AM

    return [bind(AM.PouwDitherAmax, K=k, I=i) for k, i in ((32, 14336), (64, 64))] + [AM.PouwAmaxBound]
