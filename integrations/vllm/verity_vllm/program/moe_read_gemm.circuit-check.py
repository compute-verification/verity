"""circuit-check's bindings of `verity_vllm.program.moe_read_gemm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _moe_read_gemm_roots() -> list[Any]:
    """Each GEMM over the rows it reads at its smallest binding: two experts (a one-level read), one 32-byte e4m3 row for w13
    (H = 16 words) at cc 9.0's MMAv2 step, and two for w2 (H = 32) with SWAP_AB at its wgmma step. The served units repeat
    these parts at their widths, each checked at its smallest binding in its own module's bindings (`gates.embedding`,
    `registry.boolean_fp8_moe`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program import moe_read_gemm as RG

    return [bind(RG.MoeExpertFp8ReadGemm, V=2, K=32, N=1, G=32, KB=32, SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32),
            bind(RG.MoeExpertFp8ReadGemmW, V=2, K=32, N=2, G=32, KB=32, SWAP=True, DOT=BG.HopperE4m3QgmmaDot32)]
