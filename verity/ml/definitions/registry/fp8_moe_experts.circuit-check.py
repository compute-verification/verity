"""circuit-check's bindings of `verity.ml.definitions.registry.fp8_moe_experts`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32, HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32
    from verity.ml.definitions.registry import fp8_moe_experts

    return [
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
