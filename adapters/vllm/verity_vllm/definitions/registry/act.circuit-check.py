"""circuit-check's bindings of `verity_vllm.definitions.registry.act`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _boolean_roots() -> list[Any]:
    """`silu_and_mul`'s served row at I = 8: `SiluMul_v2` and its Boolean form `SiluMul_v4`; `SiluMulBf16_v4` is static-free."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import act

    return [
        bind(act.SiluMul, I=8),
        bind(act.SiluMulExpf, I=8),
    ]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import act

    return [
        ("GeluErf_v1", lambda: bind(act.GeluErf, N=8)),
    ]
