"""circuit-check's bindings of `verity.ml.definitions.registry.marlin_moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics, the schedule, the align and the w2 pair coordinate also at DeepSeek-V4-Pro's
    served shapes on one TP8 rank (w13 at one token: TK 128, TN 64, two CTAs per SM of 132; one token's 6 pairs over 384 experts
    in blocks of 8; w2's K = 384 over its 384 experts)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import marlin_moe as MM

    D = MM.DOT
    return [
        ("E2m1x2Codes_v1", lambda: bind(MM.E2m1x2Codes)),
        ("MoeExpertE2m1Row_v1", lambda: bind(MM.MoeExpertE2m1Row, E=3, K=64)),
        ("MarlinMxfp4Group_v1", lambda: bind(MM.MarlinMxfp4Group)),
        ("MarlinMxfp4Step_v1", lambda: bind(MM.MarlinMxfp4Step, DOT=D)),
        ("MarlinMxfp4StepFromZero_v1", lambda: bind(MM.MarlinMxfp4StepFromZero, DOT=D)),
        ("MarlinMoeSchedule_v1", lambda: bind(MM.MarlinMoeSchedule, KT=4, NT=3, GRID=7)),
        ("MarlinMoeSchedule_v1/served-w13", lambda: bind(MM.MarlinMoeSchedule, KT=56, NT=12, GRID=264)),
        ("MoeAlignRanks_v1", lambda: bind(MM.MoeAlignRanks, E=3, P=30, BM=4)),
        ("MoeAlignRanks_v1/served-decode", lambda: bind(MM.MoeAlignRanks, E=384, P=6, BM=8)),
        ("MoeAlignBlocks_v1", lambda: bind(MM.MoeAlignBlocks, P=30, BM=4)),
        ("MoeAlignBlocks_v1/served-decode", lambda: bind(MM.MoeAlignBlocks, P=6, BM=8)),
        ("MarlinMxfp4Coordinate_v1", lambda: bind(MM.MarlinMxfp4Coordinate, K=256, TK=128, DOT=D)),
        ("MarlinMxfp4Coordinate_v1/tk64", lambda: bind(MM.MarlinMxfp4Coordinate, K=128, TK=64, DOT=D)),
        ("MarlinMxfp4CoordinateW_v1", lambda: bind(MM.MarlinMxfp4CoordinateW, K=128, TK=64, DOT=D)),
        ("MarlinMoeExpertCoordinate_v1", lambda: bind(MM.MarlinMoeExpertCoordinate, E=3, K=256, TK=128, DOT=D)),
        ("MarlinMoeExpertCoordinateW_v1/served-w2", lambda: bind(MM.MarlinMoeExpertCoordinateW, E=384, K=384, TK=128, DOT=D)),
        ("MarlinMoeExpertGemm_v1", lambda: bind(MM.MarlinMoeExpertGemm, E=3, K=256, N=16, TK=128, TN=8, GRID=5, DOT=D)),
        ("MarlinMoeExpertGemmW_v1", lambda: bind(MM.MarlinMoeExpertGemmW, E=3, K=256, N=16, TK=128, TN=8, GRID=5, DOT=D)),
    ]
