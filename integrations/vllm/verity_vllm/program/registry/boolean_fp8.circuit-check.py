"""circuit-check's bindings of `verity_vllm.program.registry.boolean_fp8`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import boolean_fp8

    return [
        ("QuantLinearFp8_v2", lambda: bind(boolean_fp8.QuantLinearFp8V2, K=256, N=2, BIAS=True)),
        ("ScaledMmFp8_v2", lambda: bind(boolean_fp8.ScaledMmFp8V2, K=32, N=2, BIAS=False)),
        ("Fp8GroupQuant_v2", lambda: bind(boolean_fp8.Fp8GroupQuantV2, K=64, G=32)),
        ("ScaledMmFp8Block_v2", lambda: bind(boolean_fp8.ScaledMmFp8BlockV2, K=32, N=32, G=32)),
        ("ScaledMmFp8BlockSharedScale_v2", lambda: bind(boolean_fp8.ScaledMmFp8BlockSharedScaleV2, K=64, N=32, G=32)),
        ("ScaledMmFp8NoSwap_v2", lambda: bind(boolean_fp8.ScaledMmFp8NoSwapV2, K=64, N=2, BIAS=True,
                                              DOT=BG.BlackwellE4m3QmmaDot32)),
        ("ScaledMmFp8Block_v4", lambda: bind(boolean_fp8.ScaledMmFp8BlockV4, K=32, N=32, G=32, DOT=BG.BlackwellE4m3QmmaDot32)),
        ("ScaledMmFp8BlockSharedScale_v4", lambda: bind(boolean_fp8.ScaledMmFp8BlockSharedScaleV4, K=64, N=32, G=32,
                                                        DOT=BG.BlackwellE4m3QmmaDot32)),
    ]
