[2026-11-10T01:16Z] 7109de1759b7

## SERVEDCARD, Tue 10 Nov 01:20Z: the staged statements under the `--zk` form check, the card's verifier column, the refused share, top-p tables, Window 1

**No type is Lean-verified under `--zk` today, and 3 of 41 staged types are refused on form.**
- **Form:** the ks-form mirror refuses 3 types for `scopeOkH`, because their outputs come from a tail stage: SparseMlaSplitPart T=64, SparseMlaHeadBlock NVIS=64 (middle) and TopPStatsHead. It accepts the other 38.
- **Every type, accepted ones included:** a full `flock-verify verify --zk` refuses each served-block session for `ProvedScope.NO_PROGRAM`, since the verifier holds no Program for a proof block (`benchmarks/served_units/README.md`, `summarize.py`). So the card's column must say "verified: no" for every type.

**Staged: 80 statement directories, 74 distinct statements of 41 types.**
- **A** (`/workspace/tmp/servedcard-w1`, disk): 15 statements.
- **B** (`/dev/shm/servedcard-w1`, RAM): 65 so far.
- **Duplicates (6):** both runs staged the MoE chunk and GemmW pairs, Combine8@full and gq K2048@full/4.
- **Not staged:** TopPStatsExp failed, because an instance needs a block of 2^28 bits and the limit is 2^27. GumbelTopPSelect is still staging (98 min at 01:12Z); B's deadline stops it at 01:30Z if it isn't done.
- **Checks:** run `r20261109-234033-13f3` checked the first 78 directories. I checked TopPStep and TopPStatsHead at 01:13Z, from the run's tree. The queued check `r20261109-234617-7270` waits on B's end and hits its own 01:30Z timeout, so it is superseded.

**The card's verifier column, per type.** "--zk form" is `ks_form_why` and `template_why` on the circuit's head, the Python mirror `test_lean_ks_form` holds to `flock-verify ks-form`. No public file has `shared_rows`, so Lean's shared-row check doesn't arise. "Verified under --zk today" is **no for every row**: the form "yes" means only that `scopeOkH` passes. Proving and timing these is what Window 1 does; none of it is to be counted as verified.

| type | staged (n, where) | out_net | --zk form (ks-form mirror) |
|---|---|---|---|
| `MoeExpertFp8CoordinateChunk_v1{E=256,K=512,G=128,KB=128,SWAP…` | 64(disk), 256(disk), 64(shm), 256(shm) | unit | yes |
| `MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=true,…` | 128(disk), 512(disk), 128(shm), 512(shm) | unit | yes |
| `SparseMlaCombine_v1{NSPLIT=5,DV=512,INV={"fn":"Fa3InvSum_v1"}}` | 256(disk), 1024(disk) | unit | yes |
| `SparseMlaCombine_v1{NSPLIT=8,DV=512,INV={"fn":"Fa3InvSum_v1"}}` | 128(disk), 512(disk), 512(shm) | unit | yes |
| `SparseMlaHeadBlock_v1{DR=64,DV=512,NVIS=1,FIRST=true,BN=64,D…` | 1024(disk), 4096(disk) | unit | yes |
| `SparseMlaHeadBlock_v1{DR=64,DV=512,NVIS=64,FIRST=false,BN=64…` | 256(disk) | stage2 | no: scopeOkH (outputs from stage2) |
| `SparseMlaHeadEnd_v1{DV=512,INV={"fn":"Fa3InvSum_v1"}}` | 1024(disk), 4096(disk) | unit | yes |
| `Fp8GroupQuantUe8m0_v1{K=2048,G=128}` | 512(disk), 512(shm), 2048(shm) | unit | yes |
| `SparseMlaSplitPart_v1{T=64,DR=64,DV=512,BN=64,DOT={"fn":"Hop…` | 256(disk) | stage3 | no: scopeOkH (outputs from stage3) |
| `AllReduceMnnvl_v1{WORLD=8,N=3072,SHOT=1}` | 64(shm), 256(shm) | unit | yes |
| `AllReduceMultimem_v1{WORLD=8,SIZE=3072,TREES=[[[[[[[0,[1,2]]…` | 64(shm), 256(shm) | unit | yes |
| `AllReduceResidualMnnvl_v1{WORLD=8,N=3072,SHOT=1}` | 64(shm), 256(shm) | unit | yes |
| `BatchedGemm_v1{B=1,K=192,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}` | 4096(shm), 16384(shm) | unit | yes |
| `BatchedGemm_v1{B=1,K=512,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}` | 2048(shm), 8192(shm) | unit | yes |
| `CublasSplitKGemm_v1{K=6144,N=1,SPLIT_K=4,DOT={"fn":"HopperBF…` | 256(shm), 1024(shm) | unit | yes |
| `CublasSplitKGemmF32Out_v1{K=6144,N=1,SPLIT_K=3,DOT={"fn":"Ho…` | 256(shm), 1024(shm) | unit | yes |
| `DsaIndexerKRow_v1{D=128,ROPE=64,EPS={"f64":"0x1.0c6f7a0b5ed8dp-20"}}` | 1024(shm), 4096(shm) | unit | yes |
| `DsaMqaLogits_v3{N=1,H=32,D=128}` | 1024(shm), 4096(shm) | unit | yes |
| `DsaMqaLogits_v3{N=30,H=32,D=128}` | 512(shm) | unit | yes |
| `DsaRMSNorm_v1{N=2048,EPS={"f64":"0x1.4f8b588e368f1p-17"}}` | 256(shm), 1024(shm) | unit | yes |
| `DsaRMSNorm_v1{N=512,EPS={"f64":"0x1.4f8b588e368f1p-17"}}` | 1024(shm), 4096(shm) | unit | yes |
| `DsaTopKShortRow_v1{TOPK=2048}` | 1024(shm), 4096(shm) | unit | yes |
| `EmbeddingShard_v1{VS=19360,H=1,START=0}` | 256(shm), 1024(shm) | unit | yes |
| `Gemm_v2{K=6144,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}` | 256(shm), 1024(shm) | unit | yes |
| `LlBf16DotprodGemm_v1{N=1,K=6144,BS=128,VEC=8,FMA=1}` | 128(shm), 512(shm) | unit | yes |
| `MoeSum_v1{TOPK=8,H=3072}` | 64(shm), 256(shm) | unit | yes |
| `RMSNormCudaVector8_v1{N=6144,EPS={"f64":"0x1.4f8b588e368f1p-…` | 512(shm) | unit | yes |
| `RmsNormMnnvl_v1{N=6144,SHOT=1,CLUSTER=8,EPS={"f64":"0x1.4f8b…` | 64(shm), 256(shm) | unit | yes |
| `RoPEHeadGptJ_v2{D=64}` | 4096(shm), 16384(shm) | unit | yes |
| `ScaledMmFp8BlockSharedScale_v1{K=1536,N=1,G=128}` | 1024(shm), 4096(shm) | unit | yes |
| `ScaledMmFp8BlockSharedScale_v1{K=256,N=1,G=128}` | 2048(shm), 8192(shm) | unit | yes |
| `SiluMul_v2{I=1536}` | 512(shm), 2048(shm) | unit | yes |
| `SiluMul_v2{I=256}` | 2048(shm), 8192(shm) | unit | yes |
| `SparseMlaCombine_v1{NSPLIT=16,DV=512,INV={"fn":"Fa3InvSum_v1"}}` | 64(shm) | unit | yes |
| `FlashInferFp8Linear_v1{K=1536,N=1}` | 512(shm), 2048(shm) | unit | yes |
| `FlashInferFp8Linear_v1{K=256,N=1}` | 2048(shm), 8192(shm) | unit | yes |
| `Fp8GroupQuantUe8m0_v1{K=1536,G=128}` | 512(shm), 2048(shm) | unit | yes |
| `Fp8GroupQuantUe8m0_v1{K=256,G=128}` | 2048(shm), 8192(shm) | unit | yes |
| `Fp8GroupQuantUe8m0_v1{K=6144,G=128}` | 256(shm) | unit | yes |
| `TopPStatsHead_v1{N=5952,FIRST=false}` | 512(shm) | stage4 | no: scopeOkH (outputs from stage4) |
| `TopPStep_v1{V=154880,S=4,N=2048,FIRST=false,COMBINE=-1}` | 256(shm) | unit | yes |
| `TopPStatsExp_v1{N=2976,SPT=4,AT=0,LAST=true,FIRST=false}` | not staged (2^28-bit block) | | not checked |
| `GumbelTopPSelect_v1{V=154880,N=256,FIRST=false,LAST=false}` | staging; stops at 01:30Z if unfinished | | checked if it stages |

**The refused share of the card's cost, and what each figure is.** The figures come from my report's ×2 sensitivity table: how many points of the (d) row (proof overhead as a share of compute, world, one week: 8.087%) each family adds when its price doubles, which is that family's contribution.
- **SparseMlaSplitPart_v1:** +0.79 pt, which is 9.8% of the card's cost. That figure covers T = 64 and T < 64; only T = 64 is staged here, and the T < 64 inline blocks are past 2^32.
- **SparseMlaHeadBlock_v1** (NVIS=64 middle is refused; NVIS=1 FIRST is accepted): in "every other family, each ≤ +0.04 pt", so at most 0.5%.
- **TopPStats (TopPStatsHead):** also at most +0.04 pt, at most 0.5%.
- **Total of the form-refused types:** 9.8% to 10.8% of the card's cost. Counting the program refusal, 100% of the card is unverified under `--zk` today.

**Top-p: which tables each type reads** (`stage.json` and the statement's `tables/`):
- **TopPStatsHead_v1{N=5952}:** reads slots (`statement: {"reads": "slots"}`, 11 slot types) and the lookup tables `ex2` and `rcp`. k_log 26. Its outputs come from `stage4`, so it is refused on form.
- **TopPStep_v1{V=154880,S=4,N=2048}:** reads slots (4 slot types) and no lookup table. k_log 27. Its outputs come from the unit, so the mirror accepts it, **contrary to SLOTROWS's list**. Reading slots doesn't by itself put the outputs on a tail stage. Lean's `flock-verify ks-form` on its files would settle it; I can't run Lean on node A without installing `lake`.
- **TopPStatsExp_v1:** not staged (2^28-bit block).
- **GumbelTopPSelect_v1:** I'll report its tables and form if it stages by 01:30Z.
- The `tables/` directory of both staged types holds `ex2`, `rcp`, `rsq` and `sqrt` (u32): the stager writes every table, and the statement reads the ones `lookups` names.

**Session records on node A's root disk: none.** Window 1 (`window.py` at `9e32db233d`) links A's 15 statements file by file into `/dev/shm/servedcard-w1` (`WIN_ALSO`). Every session's `session.json` and proof streams are therefore written under `/dev/shm`, inside the host-memory grant, and the watchdog counts them every 10 s. On `/` the window writes only its run directory's logs and the CUDA compute cache (37 MB), well under 1 GB. Posted on VLLM-263 at 23:44Z. I'll record each session's file sizes into the run's `out/` before deleting `/dev/shm`, and send `df /` before and after.

**Window 1, as I'll launch it at 01:35Z** (after reading `nvidia-smi`; at 01:12Z all 8 GPUs held 0 MiB and no compute process):
```
research run --on vy-spot-glm53 --project verity --source /agent/repos/verity --cwd source --timeout $(( $(date -d 2026-11-10T02:10Z +%s) - $(date +%s) )) --custody-ttl 8h --mem-gb 550 --send ~/.research/notes/lanes/proofs/servedcard-window1.units -- bash -c 'FLOCK_WORK=/dev/shm/servedcard-w1 WIN_ALSO=/workspace/tmp/servedcard-w1 SM=90 WIN_CPUS=96 WIN_MEM_GB=550 WIN_DEV_GIB=45 WIN_GPUS=8 LEASE_MIN=24 WIN_KEEP=1 python3 benchmarks/served_units/window.py prove "$RESEARCH_RUN_DIR/inputs/servedcard-window1.units"; rc=$?; find /dev/shm/servedcard-w1/data -path "*/sessions/*" -type f -printf "%s\t%P\n" > $RESEARCH_RUN_DIR/out/session-files.tsv; rm -rf /dev/shm/servedcard-w1; exit $rc'
```
- **Stops:** each GPU group's lease ends after 24 min, about 02:00Z, which leaves time to record the session sizes and free `/dev/shm` before the run's own stop at 02:10Z.
- **Memory:** `WIN_MEM_GB=550` covers B's statements in RAM (178 GB), the provers and the sessions, all counted by the watchdog.
- **Two places read:** B's statements directly; A's through the file-by-file links, with the prover and venv linked from A's directory.
- **Cleanup**, after the window's custody is published (or at once if it's cancelled):
  `research run --on vy-spot-glm53 --project verity --source /agent/repos/verity --cwd source --timeout 600 -- rm -rf /workspace/tmp/servedcard-w1 /dev/shm/servedcard-w1`
  That frees A's 57 GB on `/` and B's RAM. Nothing later of mine needs either: Window 2 stages its own statements from SLOTROWS's tree.

**Window 2: what it would prove, how long, and what must land first.**
- **What:** SLOTROWS's slots forms from `cursor/proofs-slotrows-fbf8` at `3f31f0eeca`, one warm prove each (WARM=1, RUNS=1):
  - the router in slots: 329,860,064 nonzeros, k_log 26;
  - DsaIndexerQ_v1{H=32,D=128,ROPE=64}: `{"reads": "slots"}`, 20,218,560 AND gates, k_log 26, largest net side 340,755,271, stager 46 GB host;
  - the sparse MLA first block at 63, 48 and 32 keys: 473,075,132, 359,874,777 and 233,858,840 nonzeros, k_log 27 or 26.
- **Staging first:** CPU only, before the window: the 5 statements in parallel on 48 niced cores, at most about 250 GB host, into `/dev/shm`, plus the prover's build. About 45 min, judged from tonight's comparable stagings (SparseMlaHeadBlock NVIS=64 took 2,500 s).
- **The GPUs:** about 2 min a statement by `window.py`'s model: 10 s, plus 3 s a circuit GB, plus 2 sessions × (4 + 2^(k−23)) s. That model has not been measured on an H200, and device memory (modelled at 36 GB or less) has not been measured either. So **5 H200s for 15 min, or 2 for 30 min**, with margin.
- **Lean:** these slots forms will very likely be refused on form too (tail-stage outputs), besides the program refusal. I'll run `ks_form.py` on them right after staging. They would be "proved and timed, not verified under `--zk`".
- **What must land first:** nothing, to time them. For the card to cite them from `main`: #8359, then #8438 on top of it. For any of them to count as verified: a `ProvedScope` change that admits tail-stage outputs under `scopeOkH`, and a Program for each block (`NO_PROGRAM`). Neither exists yet.

**Disk:** at 01:12Z, `/` had 63 GB free (938 of 1000 GB used). 57 GB of the used space is mine (`/workspace/tmp/servedcard-w1`). `/dev/shm` had 760 GB free, 178 GB of it used by my B statements.

