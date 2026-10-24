"""circuit-check's bindings of `verity_vllm.program.registry.boolean_io_shell`' Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definition at a small layout (3 ids under a 1,000-entry vocabulary) and at a served one (16 ids under a
    128,256-entry vocabulary, 6-character fields)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_io_shell as BX

    return [
        ("IngressPrompt_v2:1", lambda: bind(BX.IngressPromptV2, MODEL="m", LP=3, V=1000)),
        ("IngressPrompt_v2:2", lambda: bind(BX.IngressPromptV2, MODEL="served-model", LP=16, V=128256)),
    ]
