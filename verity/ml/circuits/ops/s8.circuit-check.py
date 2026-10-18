"""circuit-check's bindings of `verity.ml.circuits.ops.s8`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """The plain arm on both steps at K = 64, the egemm-shaped edited arm (T every 32, two fresh tail words) as a coordinate and
    over N = 3 rows, an incoming accumulator, and both limb splits at their widest intervals."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import prims as P
    from verity.ml.circuits.ops import s8 as S8

    return [
        bind(S8.S8Dot, **S8.SMOKE_PLAIN),
        bind(S8.S8Dot, **S8.SMOKE_PLAIN | {"DOT": P.BlackwellU8S8ImmaDot32, "A": (0, 255), "B": (-128, 127)}),
        bind(S8.S8Dot, **S8.SMOKE_PLAIN | {"T0": "input", "C": (-(1 << 30), 1 << 30)}),
        bind(S8.S8Dot, **S8.SMOKE_EDITED),
        bind(S8.S8Matmul, N=3, **S8.SMOKE_EDITED),
        bind(S8.LimbSplit, FORM="bytes", N=3, Y=(-(127 * 65793), 127 * 65793)),
        bind(S8.LimbSplit, FORM="byteclamp", N=3, Y=(-65_152, 65_151)),
        S8.S32LowByte,
        S8.S32ByteCarry,
        S8.S32ClampS8,
        S8.S32ClampRest,
    ]
