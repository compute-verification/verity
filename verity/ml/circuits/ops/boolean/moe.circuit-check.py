"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The select over 3, 128 and 256 experts, the coordinates on both steps over 3 experts at K = 32 and over Qwen3-30B-A3B's
    128 at K = 16, and the GEMMs over 3 experts at K = 16, N = 3."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.ops.boolean import moe as BM

    return [
        bind(BM.select_bf16(3)),
        bind(BM.select_bf16(128)),
        bind(BM.select_bf16(256)),
        bind(BM.MoePairCoordinate, E=3, K=32, **BG.GEMM_AMPERE),
        bind(BM.MoePairCoordinate, E=3, K=32, **BG.GEMM_HOPPER),
        bind(BM.MoePairCoordinate, E=128, K=16, **BG.GEMM_HOPPER),
        bind(BM.MoePairCoordinateW, E=3, K=32, **BG.GEMM_HOPPER),
        bind(BM.MoePairCoordinateW, E=128, K=16, **BG.GEMM_HOPPER),
        bind(BM.MoePairGemm, E=3, K=16, N=3, **BG.GEMM_HOPPER),
        bind(BM.MoePairGemmW, E=3, K=16, N=3, **BG.GEMM_HOPPER),
    ]
