"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dsv4_moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """DeepSeek-V4's routers at 8 experts, x2.5, and the clamped SwiGLU row at I = 4, limit 10, on bits (their word views are
    `registry.dsv4_moe`'s); their pieces with statics are reached through them.  `MufuLg2Mantissa_v2`, `Ex2Approx_v2`,
    `Dsv4HashWeight_v2`, `Dsv4SqrtSoftplus_v2` and their pieces are static-free."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dsv4_moe as BD

    return [
        bind(BD.MoeRouterHash, E=8, SCALE=2.5),
        bind(BD.MoeRouterSqrtSoftplus, E=8, SCALE=2.5),
        bind(BD.SiluMulClamp, I=4, LIMIT=10.0),
    ]
