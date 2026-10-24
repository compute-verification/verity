"""circuit-check's bindings of `verity_vllm.program.registry.dsv4_decode_indices`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The row at DeepSeek-V4-Pro's statics (`dsv4_decode_indices.DSV4_DECODE_INDICES`: top-k 1024, block size 64, the block
    table's row at max_model_len 1,048,576 and 4096), which reach its slot and length, and at a small one (top-k 8, block size 4,
    a row of 8)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsv4_decode_indices as X

    S = X.DSV4_DECODE_INDICES
    return [
        ("Dsv4GlobalTopk_v1:1", lambda: bind(X.Dsv4GlobalTopk, **S["c4"])),
        ("Dsv4GlobalTopk_v1:2", lambda: bind(X.Dsv4GlobalTopk, **S["c4-4096"])),
        ("Dsv4GlobalTopk_v1:3", lambda: bind(X.Dsv4GlobalTopk, TOPK=8, BS=4, NB=8)),
    ]
