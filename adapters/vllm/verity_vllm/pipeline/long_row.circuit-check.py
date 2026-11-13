"""circuit-check's bindings of `verity_vllm.pipeline.long_row`'s Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`DsaTopKReadRowsCut_v1` whole at small statics: three blocks of 4 keys (19 units of the mask's cut, 3 positions and the
    merge), one chunk, two slots and two reads of a 16-row table of one word."""
    from verity.circuits.defs import bind
    from verity.commitments.gates.embedding import read
    from verity_vllm.pipeline import long_row as LR

    return [("DsaTopKReadRowsCut_v1", lambda: bind(LR.DsaTopKReadRowsCut, N=12, TOPK=2, BLOCK=4, READ=read(16, 1)))]


def _by_calls_roots() -> list[Any]:
    """`DsaTopKReadRowsCut_v1` at the served statics (`long_row.served`), checked by its calls: its 4,213 units are each past
    nothing a proof block holds, and together past what circuit-check builds whole."""
    from verity_vllm.pipeline import long_row as LR

    return [LR.served()]
