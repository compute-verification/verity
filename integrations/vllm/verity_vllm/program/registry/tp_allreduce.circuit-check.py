"""circuit-check's bindings of `verity_vllm.program.registry.tp_allreduce`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The word Definitions at WORLD 8: the MNNVL all-reduce of 8 elements two-shot, its fused norm at N = 512 over a 2-CTA
    cluster (one warp per CTA) one-shot with the + EPS unfused, vLLM's custom all-reduce of 64 elements 2-stage (every
    rank owns one pack), and torch's multimem all-reduce of 128 elements (16 per owner) at a chain for owners 0-3 and a balanced
    tree for 4-7."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import tp_allreduce as TA

    trees = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
    return [
        ("AllReduceMnnvl_v1", lambda: bind(TA.AllReduceMnnvl, WORLD=8, N=8, SHOT=2)),
        ("AllReduceRmsNormMnnvl_v1",
         lambda: bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=512, SHOT=1, CLUSTER=2, EPS=1e-5, FMA=False)),
        ("AllReduceCustom_v1", lambda: bind(TA.AllReduceCustom, WORLD=8, SIZE=64, STAGE=2)),
        ("AllReduceMultimem_v1", lambda: bind(TA.AllReduceMultimem, WORLD=8, SIZE=128, TREES=trees, FTZ=False)),
    ]
