"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.s8`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Both Boolean IMMA steps, and the family on bits: the plain arm at K = 64 and the egemm-shaped edited arm (T every 32,
    two fresh tail words)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import s8 as S8
    from verity.ml.circuits.ops.boolean import s8 as BS8

    step = BS8.BlackwellS8ImmaDot32
    return [
        bind(step),
        bind(BS8.BlackwellU8S8ImmaDot32),
        bind(BS8.S8Dot, **S8.SMOKE_PLAIN | {"DOT": step}),
        bind(BS8.S8Dot, **S8.SMOKE_EDITED | {"DOT": step, "TDOT": step}),
    ]
