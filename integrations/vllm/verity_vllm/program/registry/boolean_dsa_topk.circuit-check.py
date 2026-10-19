"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dsa_topk`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The relation over 20 keys selecting 16 (b = 5 index bits, 27 high bits required zero; reaches `DsaTopKOneHot_v1` and
    `DsaTopKVerdict_v1`), and the gather at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dsa_topk

    return [
        ("DsaTopKOrder_v2", lambda: bind(boolean_dsa_topk.DsaTopKOrderV2, N=20, TOPK=16)),
        ("DsaGatherRows_v2", lambda: bind(boolean_dsa_topk.DsaGatherRowsV2, N=5, TOPK=4, W=3)),
    ]
