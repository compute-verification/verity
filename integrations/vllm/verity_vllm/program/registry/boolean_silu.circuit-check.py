"""circuit-check's bindings of `verity_vllm.program.registry.boolean_silu`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """`silu_and_mul`'s rows at I = 8: `SiluMul_v3` and `SiluMul_v4` (v4's word view is the served `SiluMul_v2` of
    `registry.act`); `SiluMulBf16_v3` is static-free."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_silu as BSL

    return [
        bind(BSL.SiluMul, I=8),
        bind(BSL.SiluMulExpf, I=8),
    ]
