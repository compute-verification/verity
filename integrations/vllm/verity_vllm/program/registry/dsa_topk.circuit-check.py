"""circuit-check's bindings of `verity_vllm.program.registry.dsa_topk`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The relation at GLM-5.3's first row past TOPK (2049 keys, TOPK 2048: one primitive over the row), and the gather at small
    statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa_topk

    return [
        ("DsaTopKOrder_v1", lambda: bind(dsa_topk.DsaTopKOrder, N=2049, TOPK=2048)),
        ("DsaGatherRows_v1", lambda: bind(dsa_topk.DsaGatherRows, N=5, TOPK=4, W=3)),
    ]
