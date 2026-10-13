"""circuit-check's bindings of `verity_vllm.program.registry.dsv4_indices`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definition at DeepSeek-V4-Pro's statics (`dsv4_indices.DSV4_INDICES`): ratio 4's row of 1152 and ratio 128's
    widest, 8320."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsv4_indices as X

    S = X.DSV4_INDICES
    return [
        ("Dsv4CombineTopkSwa_v1", lambda: bind(X.Dsv4CombineTopkSwa, **S["c4"])),
        ("Dsv4CombineTopkSwa_v1", lambda: bind(X.Dsv4CombineTopkSwa, **S["c128"])),
    ]
