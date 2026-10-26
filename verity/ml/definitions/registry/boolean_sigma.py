"""Word composites as Boolean Definitions by σ (`verity.ml.definitions.boolean.sigma_lift`): the next version of each, its word body
at the same statics with each Call the Boolean version of its Definition, over the bits each word operand occupies, so its word
view is that composite at the same bindings (a function-valued static, `DOT` or `INV`, binds a Boolean Definition and is read as
its word).

    ResidualAdd_v2{N}, AllGather_v2{WORLD,N}, EmbeddingShard_v2{VS,H,START}, AllReduceRmsNormMnnvl_v2{WORLD,N,SHOT,CLUSTER,EPS,FMA}
    AllReduceResidualMnnvl_v2{WORLD,N,SHOT}, RmsNormMnnvl_v2{N,SHOT,CLUSTER,EPS,FMA}   (the fused one's parts)
    DotBf16_v4{K}, GemmCoordinate_v4{K}, Gemm_v4{K,N}   (σ of b1's v1, the Ampere accepted kind; v2's is ops.boolean.gemm's v3)
    MhcFusedPostPre_v2{HC,H,NS,TH,TN,REPEAT,EPS,HC_EPS}
    InductorRMSNorm_v2, InductorAddRMSNorm_v2, InductorAdd2RMSNorm_v2, InductorAdd2RMSNormFinal_v2   {N,EPS,XBLOCK,R0_BLOCK,NUM_WARPS}
    InductorSiluMulElem_v2, InductorSiluMul_v2{I}, InductorRoPEHead_v2{D}, InductorRoPE_v2{NHEADS,D}
    InductorQKNormRoPEHead_v2{D,EPS,XBLOCK,R0_BLOCK,NUM_WARPS}, InductorQKNormRoPE_v2{NHEADS,D,EPS,XBLOCK,R0_BLOCK,NUM_WARPS}
    Fa3AttentionHeadSplit_v2{T,NS,D,BN,DOT,INV,MASKED_FROM}, Fa3AttentionSplit_v2{T,NS,NH,KVH,D,BN,DOT,INV,MASKED_FROM}
    RouterArgmaxStep_v2, MoeRouterTopK_v2{E,TOPK,VPT}, MoeRouterTopKNorm_v2{E,TOPK,VPT}
    MoeExpertRow_v2{E,K}, MoeExpertCoordinate_v3{E,K,DOT}, MoeExpertCoordinateW_v3{E,K,DOT}
    MoeExpertGemm_v3{E,K,N,DOT}, MoeExpertGemmW_v3{E,K,N,DOT}
    MoeAlignRanks_v2{E,P,BM}, MoeAlignBlocks_v2{P,BM} and the align steps they Call, each _v2

Each is the served row's word Definition with no Boolean version of its own whose body's callees all have one: the collectives
and plumbing of the TP rows, the GEMM of the rows on the Ampere accepted kinds (no declared target, the registered-values serve
among them), DeepSeek-V4's mHC layer and MoE align, and the served-compiled rows' Inductor kernels, FA3's split decode and the MoE
router and expert GEMMs.  A σ body is built from the Boolean versions registered when it is built, so the
word body and its callees' Boolean versions fix it; circuit-check pins each one's lowering counts.
"""
from __future__ import annotations

from verity.ml.definitions.boolean import sigma_lift
from verity.ml.definitions.registry import b1, b1_tp2, inductor as I, marlin_moe as MM, mhc, moe, tp_allreduce as TA

__all__ = ["AllGather", "AllReduceResidualMnnvl", "AllReduceRmsNormMnnvl", "DotBf16", "EmbeddingShard", "Fa3AttentionHeadSplit",
           "Fa3AttentionSplit", "Gemm", "GemmCoordinate", "InductorAdd2RMSNorm", "InductorAdd2RMSNormFinal", "InductorAddRMSNorm",
           "InductorQKNormRoPE", "InductorQKNormRoPEHead", "InductorRMSNorm", "InductorRoPE", "InductorRoPEHead",
           "InductorSiluMul", "InductorSiluMulElem", "MhcFusedPostPre", "MoeAlignBefore", "MoeAlignBelow", "MoeAlignBlockOf",
           "MoeAlignBlocks", "MoeAlignPar", "MoeAlignRank", "MoeAlignRankStart", "MoeAlignRanks", "MoeAlignStart",
           "MoeExpertCoordinate", "MoeExpertCoordinateW", "MoeExpertGemm", "MoeExpertGemmW", "MoeExpertRow", "MoeRouterTopK",
           "MoeRouterTopKNorm", "ResidualAdd", "RmsNormMnnvl", "RouterArgmaxStep"]

ResidualAdd = sigma_lift(b1.ResidualAdd, 2)
DotBf16 = sigma_lift(b1.DotBf16, 4)
GemmCoordinate = sigma_lift(b1.GemmCoordinate, 4)
Gemm = sigma_lift(b1.Gemm, 4)
AllGather = sigma_lift(b1_tp2.AllGather, 2)
EmbeddingShard = sigma_lift(b1_tp2.EmbeddingShard, 2)
AllReduceRmsNormMnnvl = sigma_lift(TA.AllReduceRmsNormMnnvl, 2)
AllReduceResidualMnnvl = sigma_lift(TA.AllReduceResidualMnnvl, 2)
RmsNormMnnvl = sigma_lift(TA.RmsNormMnnvl, 2)
MhcFusedPostPre = sigma_lift(mhc.MhcFusedPostPre, 2)

InductorRMSNorm = sigma_lift(I.InductorRMSNorm, 2)
InductorAddRMSNorm = sigma_lift(I.InductorAddRMSNorm, 2)
InductorAdd2RMSNorm = sigma_lift(I.InductorAdd2RMSNorm, 2)
InductorAdd2RMSNormFinal = sigma_lift(I.InductorAdd2RMSNormFinal, 2)
InductorSiluMulElem = sigma_lift(I.InductorSiluMulElem, 2)
InductorSiluMul = sigma_lift(I.InductorSiluMul, 2)
InductorRoPEHead = sigma_lift(I.InductorRoPEHead, 2)
InductorRoPE = sigma_lift(I.InductorRoPE, 2)
InductorQKNormRoPEHead = sigma_lift(I.InductorQKNormRoPEHead, 2)
InductorQKNormRoPE = sigma_lift(I.InductorQKNormRoPE, 2)
Fa3AttentionHeadSplit = sigma_lift(I.Fa3AttentionHeadSplit, 2)
Fa3AttentionSplit = sigma_lift(I.Fa3AttentionSplit, 2)

RouterArgmaxStep = sigma_lift(moe.RouterArgmaxStep, 2)
MoeRouterTopK = sigma_lift(moe.MoeRouterTopK, 2)
MoeRouterTopKNorm = sigma_lift(moe.MoeRouterTopKNorm, 2)
MoeExpertRow = sigma_lift(moe.MoeExpertRow, 2)
MoeExpertCoordinate = sigma_lift(moe.MoeExpertCoordinateV2, 3)
MoeExpertCoordinateW = sigma_lift(moe.MoeExpertCoordinateWV2, 3)
MoeExpertGemm = sigma_lift(moe.MoeExpertGemmV2, 3)
MoeExpertGemmW = sigma_lift(moe.MoeExpertGemmWV2, 3)

MoeAlignBefore = sigma_lift(MM.MoeAlignBefore, 2)
MoeAlignRank = sigma_lift(MM.MoeAlignRank, 2)
MoeAlignStart = sigma_lift(MM.MoeAlignStart, 2)
MoeAlignRankStart = sigma_lift(MM.MoeAlignRankStart, 2)
MoeAlignRanks = sigma_lift(MM.MoeAlignRanks, 2)
MoeAlignBelow = sigma_lift(MM.MoeAlignBelow, 2)
MoeAlignBlockOf = sigma_lift(MM.MoeAlignBlockOf, 2)
MoeAlignPar = sigma_lift(MM.MoeAlignPar, 2)
MoeAlignBlocks = sigma_lift(MM.MoeAlignBlocks, 2)
