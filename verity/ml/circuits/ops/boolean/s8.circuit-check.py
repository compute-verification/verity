"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.s8`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Both Boolean IMMA steps, the family on bits (the plain arm at K = 64 and the egemm-shaped edited arm: T every 32, two
    fresh tail words), and the limb splits on bits at the bindings `ops.s8`'s sidecar checks, with their four steps."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import s8 as S8
    from verity.ml.circuits.ops.boolean import s8 as BS8

    step = BS8.BlackwellS8ImmaDot32
    return [
        bind(step),
        bind(BS8.BlackwellU8S8ImmaDot32),
        bind(BS8.S8Dot, **S8.SMOKE_PLAIN | {"DOT": step}),
        bind(BS8.S8Dot, **S8.SMOKE_EDITED | {"DOT": step, "TDOT": step}),
        bind(BS8.LimbSplit, FORM="bytes", N=3, Y=(-127 * 65793, 127 * 65793)),
        bind(BS8.LimbSplit, FORM="byteclamp", N=3, Y=(-65152, 65151)),
        bind(BS8.S32LowByte),
        bind(BS8.S32ByteCarry),
        bind(BS8.S32ClampS8),
        bind(BS8.S32ClampRest),
    ]
