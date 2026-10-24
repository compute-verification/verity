"""circuit-check's bindings of `verity_vllm.program.registry.dsv4_moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """DeepSeek-V4's routers at 8 experts, x2.5 (the hash router's six table ids and the Triton router's six slots), and the
    clamped SwiGLU row at I = 4, limit 10 (its element `SiluMulClampBf16_v1{LIMIT=10.0}` with it); the word views of
    `boolean_dsv4_moe`'s.  `MufuLg2Mantissa_v1`, `Ex2Approx_v1`, `Dsv4HashWeight_v1` and `Dsv4SqrtSoftplus_v1` are static-free."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsv4_moe as D

    return [
        bind(D.MoeRouterHash, E=8, SCALE=2.5),
        bind(D.MoeRouterSqrtSoftplus, E=8, SCALE=2.5),
        bind(D.SiluMulClamp, I=4, LIMIT=10.0),
    ]
