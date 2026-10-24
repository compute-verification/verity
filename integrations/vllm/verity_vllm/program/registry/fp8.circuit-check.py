"""circuit-check's bindings of `verity_vllm.program.registry.fp8`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from circuit_check.targets import TINY_FP8
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32
    from verity_vllm.program.registry import fp8

    return [
        ("ServeFp8_v1", lambda: bind(fp8.ServeFp8, C=TINY_FP8, LP=2, STEPS=1)),
        ("QuantLinearFp8_v1", lambda: bind(fp8.QuantLinearFp8, K=256, N=2, BIAS=True)),
        ("ScaledMmFp8NoSwap_v1", lambda: bind(fp8.ScaledMmFp8NoSwap, K=64, N=2, BIAS=True, DOT=BlackwellE4m3QmmaDot32)),
        ("ScaledMmFp8BlockSharedScale_v1", lambda: bind(fp8.ScaledMmFp8BlockSharedScale, K=256, N=256, G=128)),
        # a partial last weight block (GLM-5.3's fused_qkv_a_proj, N = 2624 over 21 scale rows)
        ("ScaledMmFp8Block_v1/ragged", lambda: bind(fp8.ScaledMmFp8Block, K=256, N=192, G=128)),
        ("ScaledMmFp8BlockSharedScale_v1/ragged", lambda: bind(fp8.ScaledMmFp8BlockSharedScale, K=256, N=192, G=128)),
        # the blockwise linear over a GEMM target's step (cc 12.0's), at the v1 and v2 forms' shapes
        ("ScaledMmFp8Block_v3", lambda: bind(fp8.ScaledMmFp8BlockV3, K=256, N=192, G=128, DOT=BlackwellE4m3QmmaDot32)),
        ("ScaledMmFp8BlockSharedScale_v3", lambda: bind(fp8.ScaledMmFp8BlockSharedScaleV3, K=256, N=192, G=128,
                                                        DOT=BlackwellE4m3QmmaDot32)),
        # the activation quant at fused_moe's group (fp8.QUANT_GROUPS)
        ("Fp8GroupQuant_v1/g64", lambda: bind(fp8.Fp8GroupQuant, K=128, G=64)),
    ]
