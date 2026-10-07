"""circuit-check's bindings of `verity_vllm.program.registry.fp8_moe_experts`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32
    from verity_vllm.program.registry import fp8_moe_experts

    return [
        # the expert-indexed pair forms the derive rule and the fold bind: Qwen3-235B-A22B's 128 experts at the served w2 coordinate,
        # and pairs over a whole and a partial weight block
        ("MoeExpertFp8CoordinateW_v1", lambda: bind(fp8_moe_experts.MoeExpertFp8CoordinateW, E=128, K=192, G=64, KB=32, SWAP=False,
                                                    DOT=BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8Gemm_v1", lambda: bind(fp8_moe_experts.MoeExpertFp8Gemm, E=4, K=128, N=80, G=64, KB=32, SWAP=False,
                                             DOT=BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8GemmW_v1", lambda: bind(fp8_moe_experts.MoeExpertFp8GemmW, E=4, K=64, N=48, G=32, KB=32, SWAP=False,
                                              DOT=BlackwellE4m3QmmaDot32)),
    ]
