"""circuit-check's bindings of `verity_vllm.definitions.registry.boolean_gate_ops`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import boolean_gate_ops as BG

    return [("F32ToU32RzByte_v2", lambda: bind(BG.F32ToU32RzByteV2)), ("Ue8m0CeilByte_v2", lambda: bind(BG.Ue8m0CeilByteV2)),
            ("I32Mul_v2", lambda: bind(BG.I32MulV2)), ("I32Div_v2", lambda: bind(BG.I32DivV2)), ("I32Rem_v2", lambda: bind(BG.I32RemV2)),
            ("MufuLg2Mantissa_v2", lambda: bind(BG.MufuLg2MantissaV2))]
