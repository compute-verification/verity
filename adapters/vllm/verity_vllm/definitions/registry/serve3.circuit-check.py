"""circuit-check's bindings of `verity_vllm.definitions.registry.serve3`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`verity.circuits.tiny`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity_vllm.definitions.registry import serve3
    from verity.circuits.tiny import TINY

    return [
        ("Serve_v3", lambda: serve3.serve_v3({**TINY, "EOS_ID": 7}, 3, 2)),
    ]
