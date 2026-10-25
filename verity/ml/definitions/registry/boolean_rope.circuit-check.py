"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_rope`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _boolean_roots() -> list[Any]:
    """vLLM's Boolean RoPE at the kernel rows' `RoPE_v1` binding, which reaches its head and halves, and vLLM's GPT-J RoPE on
    bits at two heads of 8."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_rope

    return [
        bind(boolean_rope.RoPE, NHEADS=2, D=8),
        bind(boolean_rope.RoPEGptJ, NHEADS=2, D=8),
    ]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_rope

    return [
        ("RoPEGptJ_v2", lambda: bind(boolean_rope.RoPEGptJV2, NHEADS=2, D=16)),
    ]
