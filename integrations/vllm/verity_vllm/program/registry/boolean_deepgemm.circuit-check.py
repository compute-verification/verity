"""circuit-check's bindings of `verity_vllm.program.registry.boolean_deepgemm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_deepgemm as BDG

    return [
        ("Fp8GroupQuantUe8m0_v2", lambda: bind(BDG.Fp8GroupQuantUe8m0V2, K=64, G=32)),
        ("DeepGemmFp8Linear_v2", lambda: bind(BDG.DeepGemmFp8LinearV2, K=128, N=2)),
        ("FlashInferFp8Linear_v2", lambda: bind(BDG.FlashInferFp8LinearV2, K=128, N=2)),
        # two groups, one whole weight block and a partial one
        ("ScaledMmFp8BlockGrouped_v2", lambda: bind(BDG.ScaledMmFp8BlockGroupedV2, H=2, K=64, N=34, G=32)),
    ]
