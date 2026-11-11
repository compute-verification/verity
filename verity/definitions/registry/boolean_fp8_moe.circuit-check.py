"""circuit-check's bindings of `verity.definitions.registry.boolean_fp8_moe`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.circuits.defs import bind
    from verity.definitions.ops.boolean import gemm as BG
    from verity.commitments.gates.embedding import read
    from verity.definitions.registry import boolean_fp8_moe

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
        # coordinate at the `_v2` coordinate's shape, small enough for Flock to lower whole (its AND count pinned)
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
        # at cc 9.0's MMAv2 step (the small-batch H200 tiles, with and without SWAP_AB): the coordinates at the `_v2` coordinate's
        # shape and at one served K tile (KB = G = 128), the GEMMs over a whole and a partial weight block, a pair form over 3 experts
        ("MoeExpertFp8BlockCoordinate_v4/mmav2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV4, K=64, G=32, KB=32,
                                                              SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8BlockCoordinateW_v4/mmav2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateWV4, K=128, G=128, KB=128,
                                                               SWAP=True, DOT=BG.HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8BlockGemm_v4/mmav2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmV4, K=32, N=40, G=32, KB=32, SWAP=True,
                                                        DOT=BG.HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8BlockGemmW_v4/mmav2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmWV4, K=32, N=40, G=32, KB=32, SWAP=False,
                                                         DOT=BG.HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8GemmW_v2/mmav2", lambda: bind(boolean_fp8_moe.MoeExpertFp8GemmWV2, E=3, K=32, N=40, G=32, KB=32, SWAP=True,
                                                    DOT=BG.HopperE4m3MmaV2Dot32)),
        # each GEMM over the rows it reads itself, two experts (a one-level read): w13's one 32-byte row at the MMAv2 step, w2's
        # two under SWAP_AB at the wgmma step; the served units repeat these parts at their widths
        ("MoeExpertFp8ReadGemm_v1", lambda: bind(boolean_fp8_moe.MoeExpertFp8ReadGemmV1, READ=read(2, 16), K=32, N=1, G=32, KB=32,
                                                 SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8ReadGemmW_v1", lambda: bind(boolean_fp8_moe.MoeExpertFp8ReadGemmWV1, READ=read(2, 32), K=32, N=2, G=32, KB=32,
                                                  SWAP=True, DOT=BG.HopperE4m3QgmmaDot32)),
        # a w13 coordinate cut along K, over 3 experts: a middle chunk (an f32 in and out) and a last one (the bf16 word)
        ("MoeExpertFp8CoordinateChunk_v2/mmav2", lambda: bind(boolean_fp8_moe.MoeExpertFp8CoordinateChunkV2, E=3, K=64, G=32, KB=32,
                                                              SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32, FIRST=False, LAST=False)),
        ("MoeExpertFp8CoordinateChunk_v2/last", lambda: bind(boolean_fp8_moe.MoeExpertFp8CoordinateChunkV2, E=3, K=64, G=64, KB=32,
                                                             SWAP=True, DOT=BG.HopperE4m3MmaV2Dot32, FIRST=False, LAST=True)),
        ("SiluMulFp8BlockQuantResult_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantResultV2)),
        ("SiluMulFp8BlockQuantGroup_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantGroupV2, G=4)),
        ("SiluMulFp8BlockQuant_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantV2, I=8, G=4)),
    ]
