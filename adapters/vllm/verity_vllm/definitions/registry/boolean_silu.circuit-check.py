"""circuit-check's bindings of `verity_vllm.definitions.registry.boolean_silu`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """`silu_and_mul`'s row at I = 8, `SiluMul_v3`; `SiluMulBf16_v3` is static-free."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import boolean_silu as BSL

    return [
        bind(BSL.SiluMul, I=8),
    ]
