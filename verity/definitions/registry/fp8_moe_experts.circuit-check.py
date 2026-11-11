"""circuit-check's bindings of `verity.definitions.registry.fp8_moe_experts`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


#: GLM-5.3's served (SWAP, DOT) on one TP8 rank of the H200, by the tuned tile vLLM takes (`enable_swap_ab`, `fp8_moe.sm90_dot`):
#: 8x32 below 4 tokens, 8x64 at 4 to 6, 16x64 and 16x128 at 7 to 768, 64x128 past 768
SERVED_GLM53_H200 = (("nonswap-mmav2", False, "HopperE4m3MmaV2Dot32"), ("swap-mmav2", True, "HopperE4m3MmaV2Dot32"),
                     ("swap-qgmma", True, "HopperE4m3QgmmaDot32"), ("nonswap-qgmma", False, "HopperE4m3QgmmaDot32"))


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics.  Then GLM-5.3's served expert GEMMs (``served-``): the w13 launch
    (K = 6144, N = 512) and the routed w2 launch (K = 256, N = 6144) of one TP8 rank's 256 experts at each served (SWAP, DOT)
    (`SERVED_GLM53_H200`), which verity-vllm's `test_moe_fp8_served` holds to the compiler's tile rule.  Each is over
    `targets.MAX_GATES`, and its ``digests`` pin keeps it whole in ``--all``: its expert stack is read lazily
    (`targets.LAZY_OPERANDS`), and its gathers resolve only the expert a vector selects."""
    from verity.circuits.defs import bind
    from verity.definitions.ops.prims import BlackwellE4m3QmmaDot32, HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32
    from verity.definitions.registry import fp8_moe_experts

    return [
        # GLM-5.3's served expert GEMMs, one entry a (SWAP, DOT) of `SERVED_GLM53_H200`: literal labels, so each is its own
        # cache unit (`circuit_check.cache._model_list`)
        ("MoeExpertFp8Gemm_v1/served-nonswap-mmav2",
         lambda: bind(fp8_moe_experts.MoeExpertFp8Gemm, E=256, K=6144, N=512, G=128, KB=128, SWAP=False, DOT=HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8Gemm_v1/served-swap-mmav2",
         lambda: bind(fp8_moe_experts.MoeExpertFp8Gemm, E=256, K=6144, N=512, G=128, KB=128, SWAP=True, DOT=HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8Gemm_v1/served-swap-qgmma",
         lambda: bind(fp8_moe_experts.MoeExpertFp8Gemm, E=256, K=6144, N=512, G=128, KB=128, SWAP=True, DOT=HopperE4m3QgmmaDot32)),
        ("MoeExpertFp8Gemm_v1/served-nonswap-qgmma",
         lambda: bind(fp8_moe_experts.MoeExpertFp8Gemm, E=256, K=6144, N=512, G=128, KB=128, SWAP=False, DOT=HopperE4m3QgmmaDot32)),
        ("MoeExpertFp8GemmW_v1/served-nonswap-mmav2",
         lambda: bind(fp8_moe_experts.MoeExpertFp8GemmW, E=256, K=256, N=6144, G=128, KB=128, SWAP=False, DOT=HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8GemmW_v1/served-swap-mmav2",
         lambda: bind(fp8_moe_experts.MoeExpertFp8GemmW, E=256, K=256, N=6144, G=128, KB=128, SWAP=True, DOT=HopperE4m3MmaV2Dot32)),
        ("MoeExpertFp8GemmW_v1/served-swap-qgmma",
         lambda: bind(fp8_moe_experts.MoeExpertFp8GemmW, E=256, K=256, N=6144, G=128, KB=128, SWAP=True, DOT=HopperE4m3QgmmaDot32)),
        ("MoeExpertFp8GemmW_v1/served-nonswap-qgmma",
         lambda: bind(fp8_moe_experts.MoeExpertFp8GemmW, E=256, K=256, N=6144, G=128, KB=128, SWAP=False, DOT=HopperE4m3QgmmaDot32)),
        # the expert-indexed pair forms the derive rule and the fold bind: Qwen3-235B-A22B's 128 experts at the served w2 coordinate,
        # and pairs over a whole and a partial weight block
        ("MoeExpertFp8CoordinateW_v1", lambda: bind(fp8_moe_experts.MoeExpertFp8CoordinateW, E=128, K=192, G=64, KB=32, SWAP=False,
                                                    DOT=BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8Gemm_v1", lambda: bind(fp8_moe_experts.MoeExpertFp8Gemm, E=4, K=128, N=80, G=64, KB=32, SWAP=False,
                                             DOT=BlackwellE4m3QmmaDot32)),
        ("MoeExpertFp8GemmW_v1", lambda: bind(fp8_moe_experts.MoeExpertFp8GemmW, E=4, K=64, N=48, G=32, KB=32, SWAP=False,
                                              DOT=BlackwellE4m3QmmaDot32)),
        # Hopper's wgmma k32 step on the H200's tuned tiles (BLOCK_SIZE_K 128, SWAP_AB): GLM-5.3's 256 experts at a TP8 rank's w2
        # coordinate as rank 0's log serves it, and pairs over two K tiles and a partial weight block
        ("MoeExpertFp8CoordinateW_v1/hopper", lambda: bind(fp8_moe_experts.MoeExpertFp8CoordinateW, E=256, K=256, G=128, KB=128, SWAP=True,
                                                           DOT=HopperE4m3QgmmaDot32)),
        ("MoeExpertFp8Gemm_v1/hopper", lambda: bind(fp8_moe_experts.MoeExpertFp8Gemm, E=4, K=256, N=80, G=128, KB=128, SWAP=True,
                                                    DOT=HopperE4m3QgmmaDot32)),
        ("MoeExpertFp8GemmW_v1/hopper", lambda: bind(fp8_moe_experts.MoeExpertFp8GemmW, E=4, K=256, N=48, G=128, KB=128, SWAP=True,
                                                     DOT=HopperE4m3QgmmaDot32)),
        # a w13 coordinate cut along K: a middle chunk over two scale blocks, and GLM-5.3's served decode chunk (two K tiles of a
        # TP8 rank's K = 6144 coordinate at the MMAv2 step)
        ("MoeExpertFp8CoordinateChunk_v1", lambda: bind(fp8_moe_experts.MoeExpertFp8CoordinateChunk, E=4, K=128, G=64, KB=32,
                                                        SWAP=False, DOT=BlackwellE4m3QmmaDot32, FIRST=False, LAST=False)),
        ("MoeExpertFp8CoordinateChunk_v1/hopper", lambda: bind(fp8_moe_experts.MoeExpertFp8CoordinateChunk, E=256, K=256, G=128,
                                                               KB=128, SWAP=False, DOT=HopperE4m3MmaV2Dot32, FIRST=False, LAST=False)),
    ]
