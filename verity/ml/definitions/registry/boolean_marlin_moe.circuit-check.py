"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_marlin_moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics on the H200's k16 step: the steps and their halves cut at x's words, the
    coordinates at TK 64 (over 150k gates already, so the suite keeps a family's smallest binding only: `targets.select`), and the
    pair forms over 3 experts, which reach the gathers and the schedule."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import boolean_marlin_moe as BM

    D = BG.HopperBF16WgmmaDot16
    return [
        ("MarlinMxfp4Group_v2", lambda: bind(BM.MarlinMxfp4GroupV2)),
        ("MarlinMxfp4Step_v2", lambda: bind(BM.MarlinMxfp4StepV2, DOT=D)),
        ("MarlinMxfp4StepFromZero_v2", lambda: bind(BM.MarlinMxfp4StepFromZeroV2, DOT=D)),
        ("MarlinMxfp4StepActivation_v1", lambda: bind(BM.MarlinMxfp4StepActivation, DOT=D)),
        ("MarlinMxfp4StepColumn_v1", lambda: bind(BM.MarlinMxfp4StepColumn, DOT=D)),
        ("MarlinMxfp4StepFromZeroActivation_v1", lambda: bind(BM.MarlinMxfp4StepFromZeroActivation, DOT=D)),
        ("MarlinMxfp4StepFromZeroColumn_v1", lambda: bind(BM.MarlinMxfp4StepFromZeroColumn, DOT=D)),
        ("MarlinMoeSchedule_v2", lambda: bind(BM.MarlinMoeScheduleV2, KT=4, NT=3, GRID=7)),
        ("MarlinMoeSchedule_v2/grid132", lambda: bind(BM.MarlinMoeScheduleV2, KT=5, NT=2, GRID=132)),
        ("MarlinMxfp4Coordinate_v2", lambda: bind(BM.MarlinMxfp4CoordinateV2, K=128, TK=64, DOT=D)),
        ("MarlinMxfp4CoordinateW_v2", lambda: bind(BM.MarlinMxfp4CoordinateWV2, K=128, TK=64, DOT=D)),
        ("MoeExpertE2m1Row_v2", lambda: bind(BM.MoeExpertE2m1RowV2, E=3, K=64)),
        ("MoeExpertE8m0Row_v2", lambda: bind(BM.MoeExpertE8m0RowV2, E=3, K=4)),
        ("MarlinMoeExpertCoordinate_v2", lambda: bind(BM.MarlinMoeExpertCoordinateV2, E=3, K=128, TK=64, DOT=D)),
        ("MarlinMoeExpertCoordinateW_v2", lambda: bind(BM.MarlinMoeExpertCoordinateWV2, E=3, K=128, TK=64, DOT=D)),
        ("MarlinMoeExpertColumn_v1", lambda: bind(BM.MarlinMoeExpertColumn, E=3, K=128, TK=64, DOT=D)),
        ("MarlinMoeExpertColumnW_v1", lambda: bind(BM.MarlinMoeExpertColumnW, E=3, K=128, TK=64, DOT=D)),
        ("MarlinMoeExpertGemm_v2", lambda: bind(BM.MarlinMoeExpertGemmV2, E=3, K=128, N=4, TK=64, TN=2, GRID=5, DOT=D)),
        ("MarlinMoeExpertGemmW_v2", lambda: bind(BM.MarlinMoeExpertGemmWV2, E=3, K=128, N=4, TK=64, TN=2, GRID=5, DOT=D)),
    ]
