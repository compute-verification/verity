"""circuit-check's bindings of `verity_vllm.definitions.registry.hopper`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`verity.circuits.tiny`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.circuits.defs import bind
    from verity_vllm.definitions.registry import hopper
    from verity.circuits.tiny import TINY

    return [
        ("ServeHopper_v1", lambda: bind(hopper.ServeHopper, C=TINY, LP=3, STEPS=2)),
    ]
