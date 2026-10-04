"""circuit-check's bindings of `verity_vllm.program.registry.act`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """`silu_and_mul`'s served row, `SiluMul_v2`, at I = 8: the word view of `boolean_silu`'s `SiluMul_v4`."""
    from verity.ir.defs import bind
    from verity_vllm.program.registry import act

    return [
        bind(act.SiluMul, I=8),
    ]
