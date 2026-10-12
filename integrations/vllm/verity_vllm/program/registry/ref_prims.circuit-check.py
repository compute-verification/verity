"""circuit-check's bindings of `verity_vllm.program.registry.ref_prims`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import ref_prims

    return [
        ("RoPEGptJ_v1", lambda: bind(ref_prims.RoPEGptJ, NHEADS=2, D=16)),
        ("RoPEInterleavedRef_v1", lambda: bind(ref_prims.RoPEInterleavedRef, D=16, DT="bf16")),
    ]
