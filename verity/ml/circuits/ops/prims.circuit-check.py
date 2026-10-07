"""circuit-check's bindings of `verity.ml.circuits.ops.prims`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    from verity.ml.circuits.ops import prims as P

    return [
        P.BlackwellE4m3QmmaDot32,
        P.BlackwellE5m2QmmaDot32,
        P.BlackwellNvf4OmmaDot64,
        P.BlackwellMxf4OmmaDot64,
        P.F32ToE2m1Sat,
        P.E4m3ToF32,
    ]
