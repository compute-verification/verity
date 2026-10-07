"""circuit-check's bindings of `verity_vllm.program.registry.nvfp4`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import BlackwellNvf4OmmaDot64
    from verity_vllm.program.registry import nvfp4

    return [
        ("QuantLinearNvfp4_v1", lambda: bind(nvfp4.QuantLinearNvfp4, K=64, N=2, DOT=BlackwellNvf4OmmaDot64)),
    ]
