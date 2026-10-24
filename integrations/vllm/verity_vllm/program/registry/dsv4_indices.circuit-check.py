"""circuit-check's bindings of `verity_vllm.program.registry.dsv4_indices`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definition at DeepSeek-V4-Pro's statics (`dsv4_indices.DSV4_INDICES`): ratio 4's row of 1152, and ratio
    128's narrowest (`active_topk_width` 128, a row of 256), as the twin's self-check binds it.  Ratio 128's widest row (8320)
    is not bound: each slot spells its own 32-bit literals, and each `Const32[...]_v1` a body reaches is a target of its own
    (8,333 there)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsv4_indices as X

    S = X.DSV4_INDICES
    return [
        ("Dsv4CombineTopkSwa_v1:1", lambda: bind(X.Dsv4CombineTopkSwa, **S["c4"])),
        ("Dsv4CombineTopkSwa_v1:2", lambda: bind(X.Dsv4CombineTopkSwa, TOPK=128, RATIO=128, WINDOW=128, WIDTH=256)),
    ]
