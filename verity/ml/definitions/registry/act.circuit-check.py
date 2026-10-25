"""circuit-check's bindings of `verity.ml.definitions.registry.act`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _boolean_roots() -> list[Any]:
    """`silu_and_mul`'s served row, `SiluMul_v2`, at I = 8: the word view of `boolean_silu`'s `SiluMul_v4`."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import act

    return [
        bind(act.SiluMul, I=8),
    ]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import act

    return [
        ("GeluErf_v1", lambda: bind(act.GeluErf, N=8)),
    ]
