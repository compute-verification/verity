"""circuit-check's bindings of `verity.ml.kernels.tc`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    from verity.ml.kernels import tc as K

    return [
        K.HOPPER_E4M3_STEP,
        K.HOPPER_E4M3_DOT.bind(K=128),
        K.ADA_E4M3_STEP,
        K.ADA_E4M3_DOT.bind(K=128),
        K.SM120_NVF4_STEP,
        K.SM120_NVF4_DOT.bind(K=256),
        K.SM120_MXF4_STEP,
        K.SM120_MXF4_DOT.bind(K=256),
    ]
