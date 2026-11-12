"""circuit-check's bindings of `verity.definitions.registry.boolean_torch_words`' Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_torch_words as BW

    return [bind(d) for d in (BW.I64Add, BW.I64Le, BW.I64Shl, BW.I64Shr, BW.I64And, BW.SelectI64, BW.I64ToU8, BW.U8ToI64,
                              BW.I64ToF32Rn, BW.F32ToI64Rz, BW.F32ToF16Rn, BW.F16ToF32, BW.F32OrderKey, BW.OrderKeyF32)]
