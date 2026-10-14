"""circuit-check's bindings of `verity_vllm.program.registry.deepgemm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import deepgemm

    return [
        # the sm_90 block-FP8 linear's two branches over two k blocks and two weight blocks
        ("DeepGemmFp8Linear_v1", lambda: bind(deepgemm.DeepGemmFp8Linear, K=256, N=256)),
        ("FlashInferFp8Linear_v1", lambda: bind(deepgemm.FlashInferFp8Linear, K=256, N=256)),
        # a partial last weight block (GLM-5.3's fused_qkv_a_proj, N = 2624 over 21 scale rows)
        ("DeepGemmFp8Linear_v1/ragged", lambda: bind(deepgemm.DeepGemmFp8Linear, K=256, N=192)),
        # DeepSeek-V4's wo_a einsum: two local groups, a partial last weight block
        ("ScaledMmFp8BlockGrouped_v1", lambda: bind(deepgemm.ScaledMmFp8BlockGrouped, H=2, K=256, N=192, G=128)),
    ]
