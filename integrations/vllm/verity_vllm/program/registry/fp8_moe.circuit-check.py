"""circuit-check's bindings of `verity_vllm.program.registry.fp8_moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32
    from verity_vllm.program.registry import fp8_moe

    return [
        # fused_moe_kernel's block FP8 expert GEMMs: GLM-5.3's served coordinates (w13 K = 6144, w2 K = 256 at TP8), and small ones
        # that reach two tiles per scale block, two dot steps per tile, SWAP_AB and a partial last weight block
        ("MoeExpertFp8BlockCoordinate_v1", lambda: bind(fp8_moe.MoeExpertFp8BlockCoordinate, K=6144, G=128, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockCoordinateW_v1", lambda: bind(fp8_moe.MoeExpertFp8BlockCoordinateW, K=256, G=128, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockGemm_v1", lambda: bind(fp8_moe.MoeExpertFp8BlockGemm, K=128, N=80, G=64, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockGemmW_v1", lambda: bind(fp8_moe.MoeExpertFp8BlockGemmW, K=64, N=48, G=32, KB=32, SWAP=True)),
        # the same with the e4m3 step as a static (cc 12.0's): Qwen3-235B-A22B-FP8's served coordinates at TP8 (w13 K = 4096, w2 K = 192,
        # G = 64), and two tiles per scale block over a whole and a partial weight block
        ("MoeExpertFp8BlockCoordinate_v3", lambda: bind(fp8_moe.MoeExpertFp8BlockCoordinateV3, K=4096, G=64, KB=32, SWAP=False,
                                                        DOT=BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8BlockCoordinateW_v3", lambda: bind(fp8_moe.MoeExpertFp8BlockCoordinateWV3, K=192, G=64, KB=32, SWAP=False,
                                                         DOT=BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8BlockGemm_v3", lambda: bind(fp8_moe.MoeExpertFp8BlockGemmV3, K=128, N=80, G=64, KB=32, SWAP=False,
                                                  DOT=BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8BlockGemmW_v3", lambda: bind(fp8_moe.MoeExpertFp8BlockGemmWV3, K=192, N=72, G=64, KB=32, SWAP=False,
                                                   DOT=BlackwellE4m3QmmaDot32)),
        # silu_and_mul_per_block_quant between the expert launches: one element, GLM-5.3's served group (G = 128), and rows of
        # two groups; on bits at G = 4 (each element about 19.6k ANDs)
        ("SiluMulFp8BlockQuantResult_v1", lambda: bind(fp8_moe.SiluMulFp8BlockQuantResult)),
        ("SiluMulFp8BlockQuantGroup_v1", lambda: bind(fp8_moe.SiluMulFp8BlockQuantGroup, G=128)),
        ("SiluMulFp8BlockQuant_v1", lambda: bind(fp8_moe.SiluMulFp8BlockQuant, I=16, G=8)),
    ]
