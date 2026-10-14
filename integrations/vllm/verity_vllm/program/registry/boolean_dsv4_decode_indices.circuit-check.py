"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dsv4_decode_indices`' Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The row at the served block size over the inventory's row of 16 blocks with a top-k of 16, and at a small one (top-k 8,
    block size 4, a row of 8); the served slot (a row of 4096 blocks, over `targets.MAX_GATES`, so checked when named) and the
    served length (top-k 1024).  The served row is TOPK Calls of the served slot, the word Definition's
    (`dsv4_decode_indices.circuit-check.py`)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dsv4_decode_indices as BX

    return [
        ("Dsv4GlobalTopk_v2", lambda: bind(BX.Dsv4GlobalTopkV2, TOPK=16, BS=64, NB=16)),
        ("Dsv4GlobalTopk_v2", lambda: bind(BX.Dsv4GlobalTopkV2, TOPK=8, BS=4, NB=8)),
        ("Dsv4GlobalTopkSlot_v2", lambda: bind(BX.Dsv4GlobalTopkSlotV2, BS=64, NB=4096)),
        ("Dsv4GlobalTopkLen_v2", lambda: bind(BX.Dsv4GlobalTopkLenV2, TOPK=1024)),
    ]
