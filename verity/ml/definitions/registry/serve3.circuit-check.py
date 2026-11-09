"""circuit-check's bindings of `verity.ml.definitions.registry.serve3`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`verity.ml.definitions.tiny`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.definitions.registry import serve3
    from verity.ml.definitions.tiny import TINY

    return [
        ("Serve_v3", lambda: serve3.serve_v3({**TINY, "EOS_ID": 7}, 3, 2)),
    ]
