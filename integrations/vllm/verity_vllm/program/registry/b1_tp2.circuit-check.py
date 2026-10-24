"""circuit-check's bindings of `verity_vllm.program.registry.b1_tp2`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from circuit_check.targets import TINY_NOBIAS
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import b1_tp2

    tp2 = {**TINY_NOBIAS, "BIAS": True, "TP": 2}
    return [
        ("ServeTP2_v1", lambda: b1_tp2.serve_tp2(3, 2, tp2)),
        ("AllReduce_v2", lambda: b1_tp2.all_reduce(4, 8)),
        ("AllGather_v1", lambda: b1_tp2.all_gather(4, 8)),
        ("EmbeddingShard_v1", lambda: bind(b1_tp2.EmbeddingShard, VS=16, H=16, START=16)),
    ]
