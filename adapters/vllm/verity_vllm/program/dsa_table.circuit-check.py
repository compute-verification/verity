"""circuit-check's bindings of `verity_vllm.program.dsa_table`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The index at the concentrator's small statics shaped as GLM-5.3's (an odd N, K = N - 1) and with picks (N 33, K 8), and the
    reads of 8 rows of 3 words from 9, a long row's shape (TOPK below V)."""
    from verity.circuits.defs import bind
    from verity_vllm.program import dsa_table

    return [
        ("DsaTopKIndex_v1", lambda: bind(dsa_table.DsaTopKIndex, N=9, TOPK=8)),
        ("DsaTopKIndex_v1-pick", lambda: bind(dsa_table.DsaTopKIndex, N=33, TOPK=8)),
        ("DsaHistoryReads_v1", lambda: bind(dsa_table.DsaHistoryReads, V=9, TOPK=8, H=3)),
    ]
