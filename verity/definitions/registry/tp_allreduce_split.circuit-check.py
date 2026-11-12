"""circuit-check's bindings of `verity.definitions.registry.tp_allreduce_split`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`RmsNormMnnvl_v1`'s scale at `tp_allreduce`'s model statics (N = 512 over a 2-CTA cluster one-shot, the + EPS unfused),
    and its apply unit of 64 elements at each shot."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import tp_allreduce_split as TS

    return [
        ("RmsNormMnnvlSumSq_v1", lambda: bind(TS.RmsNormMnnvlSumSq, N=512, SHOT=1, CLUSTER=2, EPS=1e-5, FMA=False)),
        ("RmsNormMnnvlApply_v1/1", lambda: bind(TS.RmsNormMnnvlApply, G=64, SHOT=1)),
        ("RmsNormMnnvlApply_v1/2", lambda: bind(TS.RmsNormMnnvlApply, G=64, SHOT=2)),
    ]
