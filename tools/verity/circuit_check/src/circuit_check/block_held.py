"""The Definitions circuit-check checks whole whose Flock block (``block``, `checks.block_range`) is past C-Flock's proved
range of 2^27 bits and that the report holds rather than fails (`checks.summarize`), each with why: none stages it whole
as one Flock unit, or it is served and serving stages it in units that fit, or its calls are to move to a chunked form.

The report reads this list, no target's check: an edit to it reruns no target.  It only shrinks: `MAX` is its length, a
test holds it there, a held Definition whose block fits (a Call its module cuts, `checks.cut_of`: each unit of the cut) fails
``block/stale-hold``, and one the whole catalog no longer checks fails ``block/absent-hold``, so a fix deletes its entry and
lowers `MAX` in the same change."""
from __future__ import annotations

_GLM53_TREES = ",".join(["((((((0,(1,2)),3),5),6),4),7)"] * 8)

TP_ROW = ("a GLM-5.3 TP8 all-reduce route call (tp_allreduce.circuit-check.py's served roots) staged as its whole row, past the "
          "block from one row of 6,144: its half-row column blocks (serving_templates.COLS) and tp_combine's chunks each fit, "
          "and a chunked staging of every such call is to come")
GEMM_SLICED = ("a served GEMM row whole over its N weight rows: serving stages it as its N = 1 slices "
               "(serving_templates.SLICES), which fit, a sixth of the block at K = 6,144")
UNCUT = ("a served Call with no cut in serving_templates, past the block whole even at these statics: no partition may name it "
         "until its chunked form, which is to come")
MOE = ("the FP8 MoE expert GEMM's whole coordinate over every expert's K weights (benchmarks/one_stage/a9.py's unit), or its "
       "row: serving stages its K-chunks (serving_templates.KCHUNKS) or per-expert blocks (serving_templates.EXPERTS) instead")
MOE_GEMM = ("one of GLM-5.3's served FP8 MoE expert GEMMs whole (fp8_moe_experts.circuit-check.py's served roots): serving "
            "stages its K-chunks (serving_templates.KCHUNKS) or per-expert blocks (serving_templates.EXPERTS), never the GEMM")
TRAINING = "a training step's row, not a Call of a served Program nor PoUW's: no window stages it whole"
INSIDE = ("neither a Call of a served Program (targets.call_families), a serving cut (serving_templates) nor PoUW's: a "
          "subcircuit proved inside its callers' units, or a whole step's composition, which no window stages whole")

#: Definition id -> why it is held
HELD: dict[str, str] = {
    **{f"AllReduceMnnvl_v1{{WORLD=8,N={6144 * rows},SHOT=1}}": TP_ROW for rows in range(1, 11)},
    **{f"AllReduceMnnvl_v1{{WORLD=8,N={6144 * rows},SHOT=2}}": TP_ROW for rows in range(1, 43)},
    **{f"AllReduceMultimem_v1{{WORLD=8,SIZE={6144 * rows},TREES=({_GLM53_TREES}),FTZ=False}}": TP_ROW
       for rows in (1, 41, 85, 86, 256)},
    **dict.fromkeys([
        "CublasSplitKGemmF32Out_v1{K=6144,N=256,SPLIT_K=2,DOT=HopperBF16WgmmaDot16_v1}",
        "CublasSplitKGemmF32Out_v1{K=6144,N=256,SPLIT_K=3,DOT=HopperBF16WgmmaDot16_v1}",
        "CublasSplitKGemmF32Out_v1{K=6144,N=256,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v1}",
        "CublasSplitKGemm_v1{K=6144,N=160,SPLIT_K=2,DOT=HopperBF16WgmmaDot16_v1}",
        "CublasSplitKGemm_v1{K=6144,N=160,SPLIT_K=3,DOT=HopperBF16WgmmaDot16_v1}",
        "CublasSplitKGemm_v1{K=6144,N=160,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v1}",
        "Gemm_v2{K=6144,N=160,DOT=HopperBF16WgmmaDot16_v1}",
        "GemmF32Out_v1{K=1536,N=160,DOT=HopperBF16WgmmaDot16_v1}",
        "GemmF32Out_v1{K=1536,N=256,DOT=HopperBF16WgmmaDot16_v1}",
        "GemmF32Out_v1{K=2048,N=160,DOT=HopperBF16WgmmaDot16_v1}",
        "GemmF32Out_v1{K=2048,N=256,DOT=HopperBF16WgmmaDot16_v1}",
        "GemmF32Out_v1{K=3072,N=160,DOT=HopperBF16WgmmaDot16_v1}",
        "GemmF32Out_v1{K=3072,N=256,DOT=HopperBF16WgmmaDot16_v1}",
        "GemmF32Out_v1{K=6144,N=256,DOT=HopperBF16WgmmaDot16_v1}",
    ], GEMM_SLICED),
    **dict.fromkeys([
        "Dsv4MlaCompressC4_v1{ROWS=8,D=512,QB=64,EPS=1e-06}",
        "Dsv4QNormRopeKvInsert_v1{NH=16,NHP=64,DN=448,DR=64,QB=64,EPS=1e-06}",
        "FlashMlaPrefill_v1{NH=1,NP=1,D=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922}",
        "GatherI32x129280_v1",
        "MhcPostPreTf32_v1{HC=4,H=1024,NS=4,TN=64,REPEAT=20,EPS=1e-06,HC_EPS=1e-06}",
        "MhcPost_v1{HC=4,N=1024}",
        "MhcPreBroadcast_v1{HC=4,H=7168,NS=28,TN=64,REPEAT=20,EPS=1e-06,HC_EPS=1e-06}",
        "RMSNormCudaVector8_v1{N=7168,EPS=1e-06,BLOCK=256}",
        "RMSNormCudaVector8_v1{N=7168,EPS=1e-06,BLOCK=896}",
    ], UNCUT),
    **dict.fromkeys([
        "MoeExpertFp8Coordinate_v1{E=256,K=6144,G=128,KB=128,SWAP=False,DOT=HopperE4m3MmaV2Dot32_v1}",
        "MoeExpertFp8Coordinate_v1{E=256,K=6144,G=128,KB=128,SWAP=False,DOT=HopperE4m3QgmmaDot32_v1}",
        "MoeExpertFp8Coordinate_v1{E=256,K=6144,G=128,KB=128,SWAP=True,DOT=HopperE4m3MmaV2Dot32_v1}",
        "MoeExpertFp8Coordinate_v1{E=256,K=6144,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1}",
        "MoeExpertFp8Row_v1{E=256,K=6144}",
    ], MOE),
    **{f"MoeExpertFp8Gemm{w}_v1{{E=256,{kn},G=128,KB=128,SWAP={swap},DOT={dot}}}": MOE_GEMM
       for w, kn in (("", "K=6144,N=512"), ("W", "K=256,N=6144")) for swap in (False, True)
       for dot in ("HopperE4m3MmaV2Dot32_v1", "HopperE4m3QgmmaDot32_v1")},
    **dict.fromkeys(["LogSoftmaxNllBackwardRow_v1{V=12288}", "LogSoftmaxRow_v1{V=12288}"], TRAINING),
    **dict.fromkeys([
        "Fa2AttentionSplit_v1{T=300,NS=2,NH=4,KVH=2,D=64,BN=256,DOT=HopperBF16WgmmaDot16_v1,INV=Fa2InvSum_v1,MASKED_FROM=1}",
        "Fa3AttentionSplit_v1{T=300,NS=2,NH=4,KVH=2,D=64,BN=192,DOT=HopperBF16WgmmaDot16_v1,INV=Fa3InvSum_v1,MASKED_FROM=1}",
        "FlashMlaDecodeHead_v1{SPLITS=(0,1,3),D=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922}",
        "FlashMlaPrefillHead_v1{NP=1,D=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922}",
        "FlashMlaPrefillHead_v1{NP=2,D=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922}",
        "FlashMlaPrefillPair_v1{D=512,BN=64,FIRST=False,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922}",
        "FlashMlaPrefillPair_v1{D=512,BN=64,FIRST=True,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922}",
        "FlashMlaSplit_v1{NB=2,D=512,BN=64,PARTIAL=False,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922}",
        "FlashMlaSplit_v1{NB=2,D=512,BN=64,PARTIAL=True,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.04419417382415922}",
        "LayerPostFp8_v1{C={bbdad8e23e0c}}",
        "LayerPreFp8_v1{C={bbdad8e23e0c},L0=True}",
        "MhcFusedSplit_v1{HC=4,N=256,OUT=24,TH=256}",
        "MhcPreBigFuseBroadcast_v1{HC=4,H=7168,NS=28,TN=64,REPEAT=20,EPS=1e-06,HC_EPS=1e-06}",
        "MhcPreBigFuse_v1{HC=4,H=7168,NS=4,TN=64,REPEAT=20,EPS=1e-06,HC_EPS=1e-06}",
        "MhcPreNormBroadcast_v1{HC=4,H=7168,TN=64,EPS=1e-06}",
        "MhcPreNorm_v1{HC=4,H=7168,TN=64,EPS=1e-06}",
        "MhcPrenormTf32_v1{K=4096,NS=4,OUT=24}",
        "MhcPrenormTf32_v1{K=7168,NS=28,OUT=24}",
        "ScaledMmFp8_v1{K=256,N=512,BIAS=False}",
        "ScaledMmFp8_v1{K=256,N=512,BIAS=True}",
        "ServeFp8_v1{C={bbdad8e23e0c},LP=2,STEPS=1}",
        "ServeSpec_v1{C={31c87bbeabe0},CD={4284b55c5ab3},LP=3,K=2,A=(2,2)}",
        "TableRead27x2_v1",
        "TanhF32Rn_v2",
        "TopPMaskWordx64_v1",
        "TopPMask_v1{V=64}",
    ], INSIDE),
}
MAX = 121
