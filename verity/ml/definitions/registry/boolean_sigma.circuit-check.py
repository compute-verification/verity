"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_sigma`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Each σ Definition with statics at the smallest statics its word body takes: b1's GEMM at one k16 step and two rows
    (reaching its coordinate and dot), the collectives at two ranks, MNNVL's fused
    norm one-shot over one warp and its two parts (the all-reduce and residual add of two elements, the norm as the fused
    one's), mHC's post + pre layer at hidden 64 (two splits, one thread a column), the Inductor norms at
    one warp a row over one 64-column chunk, FA3's split decode at two splits of a 20-token context (the second partial) on the
    H200's k16 step, and the MoE router, expert GEMMs and align at 2 to 4 experts.  The static-free ones (`MoeAlignBefore_v2`,
    `RouterArgmaxStep_v2`, `InductorSiluMulElem_v2`, ...) are catalog roots already, and each binding here reaches the σ
    Definitions its body Calls (`Fa3AttentionHeadSplit_v2`, `MoeExpertCoordinate_v3`, the align steps)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import boolean_attention as BA
    from verity.ml.definitions.registry import boolean_sigma as SG

    D = BG.HopperBF16WgmmaDot16
    norm = {"N": 64, "EPS": 1e-5, "XBLOCK": 1, "R0_BLOCK": 64, "NUM_WARPS": 1}
    return [
        bind(SG.ResidualAdd, N=4),
        bind(SG.Gemm, K=16, N=2),
        bind(SG.AllGather, WORLD=2, N=2),
        bind(SG.EmbeddingShard, VS=16, H=2, START=16),
        bind(SG.AllReduceRmsNormMnnvl, WORLD=2, N=256, SHOT=1, CLUSTER=1, EPS=1e-5, FMA=False),
        bind(SG.AllReduceResidualMnnvl, WORLD=2, N=2, SHOT=1),
        bind(SG.RmsNormMnnvl, N=256, SHOT=1, CLUSTER=1, EPS=1e-5, FMA=False),
        bind(SG.MhcFusedPostPre, HC=2, H=64, NS=2, TH=8, TN=4, REPEAT=2, EPS=1e-6, HC_EPS=1e-6),
        *[bind(d, **norm) for d in (SG.InductorRMSNorm, SG.InductorAddRMSNorm, SG.InductorAdd2RMSNorm, SG.InductorAdd2RMSNormFinal)],
        bind(SG.InductorSiluMul, I=1),
        bind(SG.InductorRoPE, NHEADS=1, D=4),
        bind(SG.InductorQKNormRoPE, NHEADS=1, D=64, EPS=1e-6, XBLOCK=1, R0_BLOCK=64, NUM_WARPS=1),
        bind(SG.Fa3AttentionSplit, T=20, NS=2, NH=1, KVH=1, D=16, BN=16, DOT=D, INV=BA.Fa3InvSum, MASKED_FROM=1),
        bind(SG.MoeRouterTopK, E=4, TOPK=2, VPT=4),
        bind(SG.MoeRouterTopKNorm, E=4, TOPK=2, VPT=4),
        bind(SG.MoeExpertGemm, E=2, K=16, N=2, DOT=D),
        bind(SG.MoeExpertGemmW, E=2, K=16, N=2, DOT=D),
        bind(SG.MoeAlignRanks, E=3, P=4, BM=2),
        bind(SG.MoeAlignBlocks, P=4, BM=2),
    ]
