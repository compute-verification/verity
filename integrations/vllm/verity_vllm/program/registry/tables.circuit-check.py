"""circuit-check's bindings of `verity_vllm.program.registry.tables`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import tables

    return [
        # the Definitions that read a registered table (`gelu_tanh_bf16`, `tanh_rn`) through `TableRead16x1_v1`, `TableRead27x2_v1`
        ("GeluTanhMul_v3", lambda: bind(tables.GeluTanhMulV3, I=16)),
        ("TanhF32Rn_v2", lambda: bind(tables.TanhF32RnV2)),
    ]
