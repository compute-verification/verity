"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.dither`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwDitherBlock_v1` at its smallest segments: L = 32 (one `PouwSegDot0_v2`) and L = 64 and 96 (then one and two
    `PouwSegDot_v2`), each first (`PouwDitherFirst_v2`, no t0) and later (`PouwDitherStep_v2`), each operand formed by
    `PouwI8Add_v2`. It grows linearly in L / 32, which the filler's L = 4,096 and 8,192 are priced by. `PouwDitherStep_v2`
    and `PouwDitherFirst_v2` are catalog roots (static-free composites)."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import dither as D

    return [bind(D.PouwDitherBlock, L=L, FIRST=first) for L in (32, 64, 96) for first in (True, False)]
