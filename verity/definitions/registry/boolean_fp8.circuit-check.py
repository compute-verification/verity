"""circuit-check's bindings of `verity.definitions.registry.boolean_fp8`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: The block-scaled FP8 GEMM on bits at GLM-5.3's served widths at TP8 (the census's: N = 4 at K = 6,144, 2,048 and 256, N = 8 at
#: K = 1,536, G = 128), checked at these spots (`circuit_check.ranges`; ``circuit-check --ranges ScaledMmFp8BlockSharedScale_v2
#: ScaledMmFp8BlockCoordinateGivenScale_v2``).  At N < G its body is the K/G tile scale products and N coordinates
#: (`ScaledMmFp8BlockCoordinateGivenScale_v2{K,G}`) reading them; from K = 1,536 at N = 8 and K = 2,048 at N = 4 it is over the
#: partition's gate budget, and is partitioned by those callees (`circuit_check.checks.by_callees`), so the coordinate is held at
#: the same K, where each is within the budget and partitioned itself.  A family has one rule, so each spans K by the step that
#: reaches every served K from 256
RANGES = {
    "ScaledMmFp8BlockSharedScale_v2{K=256..6144 step 256,N=4..8 step 4,G=128}": (
        "K=256,N=4", "K=6144,N=4", "K=2048,N=4", "K=1536,N=8",
    ),
    "ScaledMmFp8BlockCoordinateGivenScale_v2{K=256..6144 step 256,G=128}": (
        "K=256", "K=6144", "K=2048", "K=1536",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.circuits.defs import bind
    from verity.circuits.ops.boolean import gemm as BG
    from verity.definitions.registry import boolean_fp8

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
