"""circuit-check's bindings of `verity_vllm.program.registry.boolean_fp8_moe`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import boolean_fp8_moe

    return [
        ("MoeExpertFp8BlockCoordinate_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=32, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockCoordinate_v2/swap", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=32,
                                                             SWAP=True)),
        ("MoeExpertFp8BlockCoordinate_v2/kb64", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=64,
                                                             SWAP=False)),
        ("MoeExpertFp8BlockCoordinateW_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateWV2, K=64, G=32, KB=32, SWAP=True)),
        ("MoeExpertFp8BlockGemm_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmV2, K=32, N=40, G=32, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockGemmW_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmWV2, K=32, N=40, G=32, KB=32, SWAP=True)),
        # the same with the e4m3 step as a static (cc 12.0's): two tiles per scale block over a whole and a partial weight block; the
        # coordinate at the `_v2` coordinate's shape, small enough for C-Flock to lower whole (its AND count pinned)
        ("MoeExpertFp8BlockCoordinate_v4", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV4, K=64, G=32, KB=32, SWAP=False,
                                                        DOT=BG.BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8BlockCoordinateW_v4", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateWV4, K=128, G=64, KB=32, SWAP=False,
                                                         DOT=BG.BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8BlockGemm_v4", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmV4, K=32, N=40, G=32, KB=32, SWAP=False,
                                                  DOT=BG.BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8BlockGemmW_v4", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmWV4, K=32, N=40, G=32, KB=32, SWAP=False,
                                                   DOT=BG.BlackwellE4m3QmmaDot32)),
        # the expert-indexed pair forms over 3 experts
        ("MoeExpertFp8Gemm_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8GemmV2, E=3, K=32, N=40, G=32, KB=32, SWAP=False,
                                             DOT=BG.BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8GemmW_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8GemmWV2, E=3, K=32, N=40, G=32, KB=32, SWAP=False,
                                              DOT=BG.BlackwellE4m3QmmaDot32)),
        ("SiluMulFp8BlockQuantResult_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantResultV2)),
        ("SiluMulFp8BlockQuantGroup_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantGroupV2, G=4)),
        ("SiluMulFp8BlockQuant_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantV2, I=8, G=4)),
    ]
