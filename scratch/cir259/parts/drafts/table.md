| retired type | what `parse()` says to stage instead |
|---|---|
| `moe-gemmw` | stage census:MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"}} in its place (its own Definition) |
| `moe-chunk` | stage census:MoeExpertFp8CoordinateChunk_v1{E=256,K=512,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"},FIRST=false,LAST=false} in its place (its own Definition) |
| `moe-w13` | stage census:MoeExpertFp8BlockGemm_v3{K=6144,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"}} or census:MoeExpertFp8BlockGemm_v3{K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}} in its place (the census's whole Definition, its swapped variant or the decode one) |
| `dsa-krow` | stage census:DsaIndexerKRow_v1{D=128,ROPE=64,EPS={"f64":"0x1.0c6f7a0b5ed8dp-20"}} in its place (the census's Definition) |
| `dsa-logits` | stage census:DsaMqaLogits_v3{N=1,H=32,D=128} in its place (the census's whole Definition) |
| `moe-sum` | stage census:MoeSum_v1{TOPK=8,H=96} in its place (the census's Definition) |
| `silu-mul` | stage census:SiluMul_v2{I=256} in its place (the census's Definition) |
| `rope` | stage census:RoPEHeadGptJ_v2{D=64} in its place (the census's Definition) |
| `dsa-topk` | stage census:DsaTopKShortRow_v1{TOPK=2048} in its place (the census's Definition) |
| `s8-linear` | stage s8-coord in its place (its block, the W8A8 coordinate) |
| `router` | stage router-row or router-slots in its place (the router's Boolean version, its row form or its slots form) |
| `dsa-q` | stage dsa-q-row in its place (the indexer query's Boolean version) |
| `rmsnorm-cuda` | stage rmsnorm-v8 in its place (the fused RMSNorm the census serves) |
| `sparse-mla-head` | stage census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},SCALE={"f64":"0x1.0000000000000p-4"}} and census:SparseMlaHeadEnd_v1{DV=512,INV={"fn":"Fa3InvSum_v1"}} in its place (the census's cut, its key block (the Boolean version, whose reads are in gates: the census's own reads in lookup slots, which `verify --zk` refuses) and its end block) |
| `token-select` | stage census:GreedyChunk_v1{N=16384,FIRST=false} and census:GreedyCombine_v1{V=154880,N=16384} in its place (the census's cut, a chunk and the combine) |
| `moe-w2` | nothing stages in its place (the census does the down projection as moe-gemmw's Definition) |
| `rmsnorm-triton` | nothing stages in its place (the census has no Definition of it) |
| `silu-quant` | nothing stages in its place (the census has no Definition of it) |
