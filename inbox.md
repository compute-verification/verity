[2026-11-10T02:05Z] ef1376fc1d08

## SERVEDCARD, Tue 10 Nov 02:06Z: Window 1's result; my directory on node A deleted, `/` at 119 GB free

**Window 1 (`r20261110-013529-74a7`, 01:35:29Z to 02:00:05Z, rc 0, custody preserved as run record `art:afbe71d71bf45491cca367595cbc4427f1afa0ca9d17d2eb51ab798a6e898299`) proved 65 of the 74 staged statements, all 34 types they cover, and Flock's loopback verifier accepted every session.**
- **What was run:** each proved statement got one warm and three timed `serve --zk` sessions, 260 sessions in all, none rejected.
- **What "verified" means here:** only Flock's Rust verifier in loopback checked them. No Lean ran. Lean would refuse every one of these sessions for `NO_PROGRAM`, so per your rule they are **"proved and timed, not yet verified under `--zk`"**.

**Not proved: 9 statements of 7 types.**
- **4 were started and cut when each group's 24-min GPU lease ended at 01:59:44Z:**
  - the 3 types the form check refuses (`SparseMlaSplitPart_v1` T=64 at n=256, `SparseMlaHeadBlock_v1` NVIS=64 middle at n=256, `TopPStatsHead_v1` at n=512), each the last statement in its group;
  - `SparseMlaCombine_v1` NSPLIT=16 at n=64, cut mid-session.
- **5 never started:** `DsaMqaLogits_v3` N=1 (both n) and N=30, and `SparseMlaHeadEnd_v1` (both n). They were queued behind the others in their groups.
- **Never staged:** `TopPStatsExp_v1` (a 2^28-bit block) and `GumbelTopPSelect_v1` (killed by B's 01:30Z deadline after 6,939 s, at a peak of 271 GB).
- **Lean's refusals:** among the proved types, the form check refuses none. All 3 types it refuses are among the unproved ones above.

**Seconds.**
- **Proving:** the median of the 3 timed `prove_total_s` is 0.25–3 s for most statements. The outliers are TopPStep (34 s; session end to end 68 s), AllReduceResidualMnnvl at n=256 (18 s), MoeSum at n=256 (8.8 s), AllReduceMnnvl at n=256 (8.3 s) and AllReduceMultimem at n=256 (7.2 s).
- **The prover process:** wall time had a median of 36 s per statement (8 to 301 s), 4,137 s in total. Its maximum RSS was 47 GB.
- **Why the window ran long:** each group ran the full 24 min of its lease, against the model's 9.4–10.4 min. About two-thirds of each group's time was outside the prover process: the loopback verifier's start on the statement's 2–5 GB circuit file and the session set-up. I haven't broken that down yet; each statement's `serve.log` and `time-prove.txt` are in the run's custody. Window 2's estimate below is corrected for it.

**Sizes.**
- **Proof, per session:** 2.14–2.47 MB (two repetitions, `proof_bytes`); the prover sends 2.32–2.79 MB over the wire.
- **Commitments, per statement:** one 64-byte root covering both repetitions per session, plus the statement's public commitments file `pub.bin` (0.03–8.63 MB) and its instances `inst.bin` (2.4–40.8 MB).
- **Session records:** 1,105 files, 0.73 GB, all in `/dev/shm`. The run deleted them at its end, after recording each file's size in its `out/session-files.tsv`. On `/` it wrote only its run directory: the 60 MB resource log and about 0.3 MB of results.

**Disk on node A.** `df /` (GiB, as `df -BG` prints):

| when | used | free |
|---|---|---|
| 01:35Z, before the window | 938 | 63 (94% used) |
| 02:00Z, after the window | 938 | 63 |
| 02:04Z, after deleting my directory | 882 | 119 (89% used) |

- **The deletion:** run `r20261110-020403-c493` removed `/workspace/tmp/servedcard-w1` (60.27 GB: A's 15 statements, the venv and the prover build). `/dev/shm/servedcard-w1` was already gone, and `/dev/shm` holds 1 GB.
- **What stays:** nothing of mine is left on `/` or in `/dev/shm`. The run directories under `/workspace/research/runs/` stay, at about 0.1 GB.
- I'll write the 02:45Z reading here.

**Per type** (n: sessions accepted, median timed prove s, median session end-to-end s, proof MB per session, wire MB sent, `pub.bin` MB, `inst.bin` MB; "mirror" is the ks-form mirror's verdict):

| type | mirror | statements proved/staged | n: sessions accepted, prove s (median of timed), session e2e s, proof MB, wire MB sent, pub.bin MB, inst.bin MB |
|---|---|---|---|
| `AllReduceMnnvl_v1{WORLD=8,N=3072,SHOT=1}` | accepted | 2/2 | 256: 4/4, 8.33, 16.8, 2.35, 2.54, 0.10, 14.36; 64: 4/4, 2.46, 4.7, 2.16, 2.34, 0.03, 3.59 |
| `AllReduceMultimem_v1{WORLD=8,SIZE=3072,TREES=[[[[[[[0,[1…` | accepted | 2/2 | 256: 4/4, 7.2, 14.5, 2.35, 2.54, 0.10, 14.36; 64: 4/4, 2.07, 4.0, 2.16, 2.34, 0.03, 3.59 |
| `AllReduceResidualMnnvl_v1{WORLD=8,N=3072,SHOT=1}` | accepted | 2/2 | 256: 4/4, 18.03, 35.9, 2.36, 2.55, 0.14, 16.01; 64: 4/4, 4.76, 9.5, 2.17, 2.36, 0.04, 4.00 |
| `BatchedGemm_v1{B=1,K=192,N=1,DOT={"fn":"HopperBF16WgmmaD…` | accepted | 2/2 | 16384: 4/4, 1.03, 1.6, 2.34, 2.53, 8.63, 30.73; 4096: 4/4, 0.38, 0.9, 2.14, 2.33, 2.15, 7.68 |
| `BatchedGemm_v1{B=1,K=512,N=1,DOT={"fn":"HopperBF16WgmmaD…` | accepted | 2/2 | 8192: 4/4, 1.18, 1.7, 2.34, 2.53, 4.32, 25.85; 2048: 4/4, 0.25, 0.5, 2.14, 2.33, 1.08, 6.46 |
| `CublasSplitKGemmF32Out_v1{K=6144,N=1,SPLIT_K=3,DOT={"fn"…` | accepted | 2/2 | 1024: 4/4, 1.56, 2.2, 2.36, 2.55, 0.54, 26.30; 256: 4/4, 0.39, 0.8, 2.14, 2.33, 0.14, 6.58 |
| `CublasSplitKGemm_v1{K=6144,N=1,SPLIT_K=4,DOT={"fn":"Hopp…` | accepted | 2/2 | 1024: 4/4, 1.26, 1.9, 2.36, 2.55, 0.54, 26.30; 256: 4/4, 0.28, 0.9, 2.14, 2.33, 0.14, 6.58 |
| `DsaIndexerKRow_v1{D=128,ROPE=64,EPS={"f64":"0x1.0c6f7a0b…` | accepted | 2/2 | 4096: 4/4, 1.76, 4.0, 2.36, 2.58, 3.73, 14.77; 1024: 4/4, 0.32, 1.3, 2.17, 2.38, 0.93, 3.69 |
| `DsaRMSNorm_v1{N=2048,EPS={"f64":"0x1.4f8b588e368f1p-17"}}` | accepted | 2/2 | 1024: 4/4, 0.66, 2.5, 2.36, 2.55, 0.54, 13.72; 256: 4/4, 0.34, 1.2, 2.14, 2.33, 0.14, 3.43 |
| `DsaRMSNorm_v1{N=512,EPS={"f64":"0x1.4f8b588e368f1p-17"}}` | accepted | 2/2 | 4096: 4/4, 4.39, 8.9, 2.34, 2.53, 2.16, 17.11; 1024: 4/4, 1.44, 2.8, 2.14, 2.33, 0.54, 4.28 |
| `DsaTopKShortRow_v1{TOPK=2048}` | accepted | 2/2 | 4096: 4/4, 0.99, 1.7, 2.33, 2.51, 1.63, 36.79; 1024: 4/4, 0.36, 0.7, 2.14, 2.32, 0.41, 9.20 |
| `EmbeddingShard_v1{VS=19360,H=1,START=0}` | accepted | 2/2 | 1024: 4/4, 1.17, 1.7, 2.36, 2.55, 0.54, 40.79; 256: 4/4, 0.3, 0.6, 2.14, 2.33, 0.14, 10.20 |
| `FlashInferFp8Linear_v1{K=1536,N=1}` | accepted | 2/2 | 2048: 4/4, 0.93, 6.1, 2.37, 2.57, 1.34, 12.46; 512: 4/4, 0.33, 2.7, 2.15, 2.35, 0.34, 3.12 |
| `FlashInferFp8Linear_v1{K=256,N=1}` | accepted | 2/2 | 8192: 4/4, 2.82, 5.4, 2.34, 2.55, 5.37, 18.06; 2048: 4/4, 1.01, 2.8, 2.15, 2.35, 1.34, 4.51 |
| `Fp8GroupQuantUe8m0_v1{K=1536,G=128}` | accepted | 2/2 | 2048: 4/4, 0.83, 3.5, 2.36, 2.55, 1.08, 11.80; 512: 4/4, 0.68, 1.6, 2.14, 2.33, 0.27, 2.95 |
| `Fp8GroupQuantUe8m0_v1{K=2048,G=128}` | accepted | 2/2 | 2048: 4/4, 0.65, 4.7, 2.36, 2.55, 1.08, 14.98; 512: 4/4, 0.47, 1.7, 2.14, 2.33, 0.27, 3.75 |
| `Fp8GroupQuantUe8m0_v1{K=256,G=128}` | accepted | 2/2 | 8192: 4/4, 0.72, 1.8, 2.34, 2.53, 4.32, 15.42; 2048: 4/4, 0.6, 1.1, 2.14, 2.33, 1.08, 3.85 |
| `Fp8GroupQuantUe8m0_v1{K=6144,G=128}` | accepted | 1/1 | 256: 4/4, 0.3, 3.4, 2.14, 2.33, 0.14, 5.05 |
| `Gemm_v2{K=6144,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}` | accepted | 2/2 | 1024: 4/4, 1.38, 2.0, 2.36, 2.55, 0.54, 26.30; 256: 4/4, 0.44, 0.9, 2.14, 2.33, 0.14, 6.58 |
| `LlBf16DotprodGemm_v1{N=1,K=6144,BS=128,VEC=8,FMA=1}` | accepted | 2/2 | 512: 4/4, 1.61, 2.9, 2.36, 2.55, 0.27, 13.15; 128: 4/4, 0.34, 1.3, 2.17, 2.36, 0.07, 3.29 |
| `MoeExpertFp8CoordinateChunk_v1{E=256,K=512,G=128,KB=128,…` | accepted | 2/2 | 256: 4/4, 1.02, 2.0, 2.39, 2.62, 0.27, 35.36; 64: 4/4, 0.52, 0.9, 2.20, 2.42, 0.07, 8.84 |
| `MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=t…` | accepted | 2/2 | 512: 4/4, 1.1, 1.7, 2.39, 2.62, 0.53, 35.97; 128: 4/4, 0.55, 0.9, 2.20, 2.42, 0.14, 8.99 |
| `MoeSum_v1{TOPK=8,H=3072}` | accepted | 2/2 | 256: 4/4, 8.79, 17.5, 2.35, 2.54, 0.10, 14.36; 64: 4/4, 2.45, 4.8, 2.16, 2.34, 0.03, 3.59 |
| `RMSNormCudaVector8_v1{N=6144,EPS={"f64":"0x1.4f8b588e368…` | accepted | 1/1 | 512: 4/4, 1.23, 3.1, 2.36, 2.55, 0.27, 19.44 |
| `RmsNormMnnvl_v1{N=6144,SHOT=1,CLUSTER=8,EPS={"f64":"0x1.…` | accepted | 2/2 | 256: 4/4, 0.85, 3.0, 2.36, 2.55, 0.14, 9.72; 64: 4/4, 0.87, 2.7, 2.17, 2.36, 0.04, 2.43 |
| `RoPEHeadGptJ_v2{D=64}` | accepted | 2/2 | 16384: 4/4, 0.91, 1.6, 2.34, 2.53, 8.63, 24.41; 4096: 4/4, 0.47, 1.2, 2.14, 2.33, 2.15, 6.10 |
| `ScaledMmFp8BlockSharedScale_v1{K=1536,N=1,G=128}` | accepted | 2/2 | 4096: 4/4, 1.17, 1.8, 2.35, 2.57, 3.21, 20.14; 1024: 4/4, 0.39, 0.7, 2.16, 2.37, 0.80, 5.03 |
| `ScaledMmFp8BlockSharedScale_v1{K=256,N=1,G=128}` | accepted | 2/2 | 8192: 4/4, 1.28, 1.9, 2.35, 2.56, 6.41, 18.65; 2048: 4/4, 0.33, 0.8, 2.16, 2.37, 1.60, 4.66 |
| `SiluMul_v2{I=1536}` | accepted | 2/2 | 2048: 4/4, 2.0, 3.4, 2.35, 2.54, 0.82, 20.48; 512: 4/4, 0.48, 0.9, 2.14, 2.32, 0.20, 5.12 |
| `SiluMul_v2{I=256}` | accepted | 2/2 | 8192: 4/4, 0.91, 1.8, 2.33, 2.51, 3.27, 19.02; 2048: 4/4, 0.76, 1.0, 2.14, 2.32, 0.82, 4.75 |
| `SparseMlaCombine_v1{NSPLIT=16,DV=512,INV={"fn":"Fa3InvSum_v1"}}` | accepted | 0/1 | 64: 0/0 (cut), None, None, , , 0.29, 2.87 |
| `SparseMlaCombine_v1{NSPLIT=5,DV=512,INV={"fn":"Fa3InvSum_v1"}}` | accepted | 2/2 | 1024: 4/4, 0.94, 3.7, 2.42, 2.69, 1.59, 15.31; 256: 4/4, 0.38, 2.1, 2.21, 2.47, 0.40, 3.83 |
| `SparseMlaCombine_v1{NSPLIT=8,DV=512,INV={"fn":"Fa3InvSum_v1"}}` | accepted | 2/2 | 512: 4/4, 1.12, 3.3, 2.47, 2.79, 1.19, 11.80; 128: 4/4, 0.49, 2.7, 2.28, 2.59, 0.30, 2.96 |
| `SparseMlaHeadBlock_v1{DR=64,DV=512,NVIS=1,FIRST=true,BN=…` | accepted | 2/2 | 4096: 4/4, 1.41, 2.1, 2.36, 2.58, 3.73, 26.37; 1024: 4/4, 0.58, 1.5, 2.17, 2.38, 0.93, 6.59 |
| `TopPStep_v1{V=154880,S=4,N=2048,FIRST=false,COMBINE=-1}` | accepted | 1/1 | 256: 4/4, 34.05, 67.9, 2.37, 2.57, 0.17, 2.73 |

**Window 2's estimate, corrected:** about 3 min of wall time per statement, not 1. That makes SLOTROWS's 5 slots statements (all at k_log 26 or 27) about 20 min on 5 H200s, after the CPU staging.

