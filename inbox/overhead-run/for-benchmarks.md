Moves: none since it was first written (Thu 12 Nov 18:46Z). A line marked **(R)** can still change when the CPU rehearsal `r20261112-184230-99b2` ends (about 21:30Z). If anything moves, this line says what and when; the run ids are added at the launch.

# The overhead run on node A tonight, for benchmarks' Lean leg (PRF-502)

SERVEDCARD (proofs) writes this file. The run proves 70 statements of 36 block types, 98.19% of the ranked proving time:
- the seven types first, each at a full statement;
- every other type at a full statement, by share;
- then a quarter statement of each type except the two head blocks.

The job scripts and data files are in the notes, `lanes/proofs/overhead-run/` (`overhead.units`, `balance.tsv`, `link.txt`, `stage-job.sh`, `prove-job.sh`, `launch.sh`, `lean-shard.sh`).

## The tree

- **Commit `0803d36697868861ec37c71fd656789911c719a4`**, branch `cursor/proofs-overhead-7c41`, cut from `origin/main` `9ddae97ba2`. **(R)**: a fix the rehearsal needs in the tree would give a new commit.
  - It is `main` plus one change: `benchmarks/served_units/prove_time.py` gives a statement a time only when its prover exited 0, every session it filed was accepted, and it filed every timed session the run asked for.
  - Nothing that stages or proves changed.
- **The staging key at that commit is `aade0df3ae6e7661`** (`window.py`'s `key()`, computed locally). A statement's directory is `data/NAME-aade0df3ae6e7661`, where NAME is the kept name below.
- **To restage a statement**, run `stage.py UNIT N DIR` as `prove_time.sh`'s stage step does:
  - the tree's `benchmarks/served_units/stage.py`;
  - `PYTHONPATH=REPO:REPO/verity/protocols/flock/prover/python:REPO/tools/verity/circuit_check/src:REPO/adapters/vllm`;
  - a Python 3.12 venv with `numpy` and `blake3`.
  The circuit depends only on the commit. The values (`pub.bin`, `inst.bin`) are seeded by the unit's name.

## The launch lines

Both jobs are launched ahead of their rows and sleep until their start, holding nothing.

- **Staging**, Fri 13 Nov 01:00Z to 04:25Z on cores 48-95 at nice 19:
  - the start check waits up to 30 min for those cores to be at most 30% busy, else exits 9 having started nothing;
  - every staging is killed at 04:25Z;
  - at most 600 GB of host memory, `/dev/shm` counted.

  ```
  research run --on vy-spot-glm53 --project verity --source /tmp/sc/ovh --cwd source --as proofs --campaign research-priorities-v1 --send overhead-run/stage-job.sh --send overhead-run/overhead.units --send overhead-run/link.txt --env UNITS=overhead.units --env RAM=/dev/shm/proofs-overhead --env DISK=/workspace/jobs/proofs-overhead --env DEV=gpu --env SM=90 --env NOT_BEFORE=1794531600 --env START_BUSY_PCT=30 --env START_WAIT_S=1800 --env WIN_CPUS=48 --env WIN_NICE=19 --env WIN_MEM_GB=600 --env WIN_DISK_GB=560 --env WIN_FREE_GB=100 --env WIN_UNTIL=2026-11-13T04:25Z --no-sampler --custody-ttl 13h --timeout 36997 -- taskset -c 48-95 nice -n 19 bash -c exec\ bash\ \"\$RESEARCH_RUN_DIR/inputs/stage-job.sh\"
  ```

- **Proving**, Fri 04:30Z to 06:00Z on all eight cards:
  - it waits up to 10 min for the staging's marker and eight idle cards;
  - each group leases one card to 05:57Z;
  - it is stopped at 05:58:30Z and killed by 05:59:30Z;
  - `/dev/shm/proofs-overhead` is removed at the end.

  Each statement gets one warm session, then four timed ones (`RUNS=4`), against its own loopback verifier on two cores. The prover runs with `FC_WITNESS_TIMES=1`.

  ```
  research run --on vy-spot-glm53 --project verity --source /tmp/sc/ovh --cwd source --as proofs --campaign research-priorities-v1 --send overhead-run/prove-job.sh --send overhead-run/overhead.units --send overhead-run/balance.tsv --env UNITS=overhead.units --env RAM=/dev/shm/proofs-overhead --env DEV=gpu --env NOT_BEFORE=1794544200 --env KILL_AT=1794549600 --env START_WAIT_S=600 --env WIN_CPUS=96 --env WIN_NICE=0 --env WIN_MEM_GB=1700 --env WIN_DEV_GIB=135 --env VCPUS=2 --env WARM=1 --env RUNS=4 --env WIN_SECONDS=balance.tsv --env FC_WITNESS_TIMES=1 --no-sampler --custody-ttl 14h --timeout 42397 -- bash -c exec\ bash\ \"\$RESEARCH_RUN_DIR/inputs/prove-job.sh\"
  ```

## The statements, group by group, as each group's `config.txt` will have them

`window.py prove` runs eight groups, one prover at a time on one leased card each. A group's statements are `prove_time.sh`'s `PT_UNITS`, in the order below. Its `out/gpuG/out/config.txt` reads `units=` followed by them space-separated, `UNIT@N`, which is `overhead.units`' unit with the size of that line.

Which physical card a group leased is in `out/gpuG/out/lease-gpu.txt` (`nvidia-smi -L` under its lease).

These lists are fixed now, because the groups are balanced by `balance.tsv` (window.py's `WIN_SECONDS`). The file holds estimates, not measurements: 300 s a full statement, 150 s a quarter, and 330 and 320 s for the two head blocks so they start first on two cards. **(R)**: the lists change only if:
- a statement fails to stage by 04:25Z (it is skipped, and the groups after it in the balancing order shift);
- fewer than eight cards are idle at 04:40Z (fewer groups);
- the commit changes.

`out/gpuG/out/config.txt` is then the record. On node A it is at `/workspace/research/runs/<proving run>/out/gpuG/out/config.txt` from about 04:31Z.

**Group 0** (`out/gpu0/out/config.txt`; 9 statements, about 33 min by the balance estimate):

1. `census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},SCALE={"f64":"0x1.0000000000000p-4"}}@full`, kept as `census-sparsemlaheadblock-v2-dr-64-dv-512-nvis-64-first-false-bn-64-dot-fn-hoppe-f11dcb8477d0-nfull` (unit 1 of `overhead.units`)
2. `census:AllReduceMultimem_v2{WORLD=8,SIZE=96,TREES=[[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7]],FTZ=false}@full`, kept as `census-allreducemultimem-v2-world-8-size-96-trees-0-1-2-3-5-6-4-7-0-1-2-3-5-6-4--92be95d7b94f-nfull` (unit 16 of `overhead.units`)
3. `census:EmbeddingShard_v2{VS=19360,H=1,START=116160}@full`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-116160-nfull` (unit 24 of `overhead.units`)
4. `census:Fp8GroupQuantUe8m0_v2{K=2048,G=128}@full`, kept as `census-fp8groupquantue8m0-v2-k-2048-g-128-nfull` (unit 32 of `overhead.units`)
5. `census:ScaledMmFp8BlockSharedScale_v2{K=6144,N=4,G=128}@full/4`, kept as `census-scaledmmfp8blocksharedscale-v2-k-6144-n-4-g-128-nfull-4` (unit 40 of `overhead.units`)
6. `census:CublasSplitKGemm_v2{K=6144,N=1,SPLIT_K=4,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full/4`, kept as `census-cublassplitkgemm-v2-k-6144-n-1-split-k-4-dot-fn-hopperbf16wgmmadot16-v2-nfull-4` (unit 44 of `overhead.units`)
7. `census:EmbeddingShard_v2{VS=19360,H=1,START=19360}@full/4`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-19360-nfull-4` (unit 52 of `overhead.units`)
8. `census:AllReduceResidualMnnvl_v2{WORLD=8,N=96,SHOT=1}@full/4`, kept as `census-allreduceresidualmnnvl-v2-world-8-n-96-shot-1-nfull-4` (unit 60 of `overhead.units`)
9. `census:MoeRouterSigmoidBiasSlots_v2{E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}@full/4`, kept as `census-moeroutersigmoidbiasslots-v2-e-256-topk-8-scale-f64-0x1.4000000000000p-1-kernel-block-nfull-4` (unit 68 of `overhead.units`)


**Group 1** (`out/gpu1/out/config.txt`; 9 statements, about 33 min by the balance estimate):

1. `census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=true,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},SCALE={"f64":"0x1.0000000000000p-4"}}@full`, kept as `census-sparsemlaheadblock-v2-dr-64-dv-512-nvis-64-first-true-bn-64-dot-fn-hopper-82f79443762c-nfull` (unit 2 of `overhead.units`)
2. `census:BatchedGemm_v2{B=1,K=192,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full`, kept as `census-batchedgemm-v2-b-1-k-192-n-1-dot-fn-hopperbf16wgmmadot16-v2-nfull` (unit 15 of `overhead.units`)
3. `census:EmbeddingShard_v2{VS=19360,H=1,START=96800}@full`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-96800-nfull` (unit 23 of `overhead.units`)
4. `census:ResidualAdd_v2{N=384}@full`, kept as `census-residualadd-v2-n-384-nfull` (unit 31 of `overhead.units`)
5. `census:MoeExpertFp8ReadGemmW_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=4096}"},K=256,N=32,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v2"}}@full/4`, kept as `census-moeexpertfp8readgemmw-v1-read-fn-embeddingread-v1-v-256-h-4096-k-256-n-32-3cb804caecb8-nfull-4` (unit 39 of `overhead.units`)
6. `census:LlBf16DotprodGemm_v2{N=1,K=6144,BS=128,VEC=8,FMA=1}@full/4`, kept as `census-llbf16dotprodgemm-v2-n-1-k-6144-bs-128-vec-8-fma-1-nfull-4` (unit 43 of `overhead.units`)
7. `census:EmbeddingShard_v2{VS=19360,H=1,START=0}@full/4`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-0-nfull-4` (unit 51 of `overhead.units`)
8. `census:Fp8GroupQuantUe8m0_v2{K=6144,G=128}@full/4`, kept as `census-fp8groupquantue8m0-v2-k-6144-g-128-nfull-4` (unit 59 of `overhead.units`)
9. `census:DsaMqaLogits_v4{N=1,H=32,D=128}@full/4`, kept as `census-dsamqalogits-v4-n-1-h-32-d-128-nfull-4` (unit 67 of `overhead.units`)


**Group 2** (`out/gpu2/out/config.txt`; 9 statements, about 35 min by the balance estimate):

1. `census:SparseMlaCombine_v2{NSPLIT=5,DV=64,INV={"fn":"Fa3InvSum_v2"}}@full`, kept as `census-sparsemlacombine-v2-nsplit-5-dv-64-inv-fn-fa3invsum-v2-nfull` (unit 3 of `overhead.units`)
2. `census:ScaledMmFp8BlockSharedScale_v2{K=6144,N=4,G=128}@full`, kept as `census-scaledmmfp8blocksharedscale-v2-k-6144-n-4-g-128-nfull` (unit 9 of `overhead.units`)
3. `census:ScaledMmFp8BlockSharedScale_v2{K=256,N=4,G=128}@full`, kept as `census-scaledmmfp8blocksharedscale-v2-k-256-n-4-g-128-nfull` (unit 17 of `overhead.units`)
4. `census:EmbeddingShard_v2{VS=19360,H=1,START=135520}@full`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-135520-nfull` (unit 25 of `overhead.units`)
5. `census:DsaMqaLogits_v4{N=1,H=32,D=128}@full`, kept as `census-dsamqalogits-v4-n-1-h-32-d-128-nfull` (unit 33 of `overhead.units`)
6. `census:Gemm_v3{K=6144,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full/4`, kept as `census-gemm-v3-k-6144-n-1-dot-fn-hopperbf16wgmmadot16-v2-nfull-4` (unit 45 of `overhead.units`)
7. `census:EmbeddingShard_v2{VS=19360,H=1,START=38720}@full/4`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-38720-nfull-4` (unit 53 of `overhead.units`)
8. `census:ScaledMmFp8BlockSharedScale_v2{K=1536,N=8,G=128}@full/4`, kept as `census-scaledmmfp8blocksharedscale-v2-k-1536-n-8-g-128-nfull-4` (unit 61 of `overhead.units`)
9. `census:SparseMlaHeadEnd_v2{DV=512,INV={"fn":"Fa3InvSum_v2"}}@full/4`, kept as `census-sparsemlaheadend-v2-dv-512-inv-fn-fa3invsum-v2-nfull-4` (unit 69 of `overhead.units`)


**Group 3** (`out/gpu3/out/config.txt`; 9 statements, about 35 min by the balance estimate):

1. `census:MoeExpertFp8ReadGemm_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=3072}"},K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v2"}}@full`, kept as `census-moeexpertfp8readgemm-v1-read-fn-embeddingread-v1-v-256-h-3072-k-6144-n-1--c7d7156ac83b-nfull` (unit 4 of `overhead.units`)
2. `census:ScaledMmFp8BlockSharedScale_v2{K=2048,N=4,G=128}@full`, kept as `census-scaledmmfp8blocksharedscale-v2-k-2048-n-4-g-128-nfull` (unit 10 of `overhead.units`)
3. `census:EmbeddingShard_v2{VS=19360,H=1,START=0}@full`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-0-nfull` (unit 18 of `overhead.units`)
4. `census:Fp8GroupQuantUe8m0_v2{K=6144,G=128}@full`, kept as `census-fp8groupquantue8m0-v2-k-6144-g-128-nfull` (unit 26 of `overhead.units`)
5. `census:MoeRouterSigmoidBiasSlots_v2{E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}@full`, kept as `census-moeroutersigmoidbiasslots-v2-e-256-topk-8-scale-f64-0x1.4000000000000p-1-kernel-block-nfull` (unit 34 of `overhead.units`)
6. `census:BatchedGemm_v2{B=1,K=192,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full/4`, kept as `census-batchedgemm-v2-b-1-k-192-n-1-dot-fn-hopperbf16wgmmadot16-v2-nfull-4` (unit 46 of `overhead.units`)
7. `census:EmbeddingShard_v2{VS=19360,H=1,START=58080}@full/4`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-58080-nfull-4` (unit 54 of `overhead.units`)
8. `census:RmsNormMnnvl_v2{N=6144,SHOT=1,CLUSTER=8,EPS={"f64":"0x1.4f8b588e368f1p-17"},FMA=true}@full/4`, kept as `census-rmsnormmnnvl-v2-n-6144-shot-1-cluster-8-eps-f64-0x1.4f8b588e368f1p-17-fma-true-nfull-4` (unit 62 of `overhead.units`)
9. `census:DsaRMSNorm_v2{N=2048,EPS={"f64":"0x1.4f8b588e368f1p-17"}}@full/4`, kept as `census-dsarmsnorm-v2-n-2048-eps-f64-0x1.4f8b588e368f1p-17-nfull-4` (unit 70 of `overhead.units`)


**Group 4** (`out/gpu4/out/config.txt`; 8 statements, about 32 min by the balance estimate):

1. `census:MoeExpertFp8ReadGemmW_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=4096}"},K=256,N=32,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v2"}}@full`, kept as `census-moeexpertfp8readgemmw-v1-read-fn-embeddingread-v1-v-256-h-4096-k-256-n-32-3cb804caecb8-nfull` (unit 5 of `overhead.units`)
2. `census:CublasSplitKGemmF32Out_v2{K=6144,N=1,SPLIT_K=3,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full`, kept as `census-cublassplitkgemmf32out-v2-k-6144-n-1-split-k-3-dot-fn-hopperbf16wgmmadot16-v2-nfull` (unit 11 of `overhead.units`)
3. `census:EmbeddingShard_v2{VS=19360,H=1,START=19360}@full`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-19360-nfull` (unit 19 of `overhead.units`)
4. `census:AllReduceResidualMnnvl_v2{WORLD=8,N=96,SHOT=1}@full`, kept as `census-allreduceresidualmnnvl-v2-world-8-n-96-shot-1-nfull` (unit 27 of `overhead.units`)
5. `census:SparseMlaHeadEnd_v2{DV=512,INV={"fn":"Fa3InvSum_v2"}}@full`, kept as `census-sparsemlaheadend-v2-dv-512-inv-fn-fa3invsum-v2-nfull` (unit 35 of `overhead.units`)
6. `census:EmbeddingRead_v1{V=256,H=4}@full/4`, kept as `census-embeddingread-v1-v-256-h-4-nfull-4` (unit 47 of `overhead.units`)
7. `census:EmbeddingShard_v2{VS=19360,H=1,START=77440}@full/4`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-77440-nfull-4` (unit 55 of `overhead.units`)
8. `census:RMSNormFusedCudaVector8_v2{N=6144,EPS={"f64":"0x1.4f8b588e368f1p-17"},BLOCK=256,RSQRT={"fn":"RsqrtApprox_v2"}}@full/4`, kept as `census-rmsnormfusedcudavector8-v2-n-6144-eps-f64-0x1.4f8b588e368f1p-17-block-256-15c597d772e7-nfull-4` (unit 63 of `overhead.units`)


**Group 5** (`out/gpu5/out/config.txt`; 8 statements, about 32 min by the balance estimate):

1. `census:EmbeddingRead_v1{V=256,H=4}@full`, kept as `census-embeddingread-v1-v-256-h-4-nfull` (unit 6 of `overhead.units`)
2. `census:LlBf16DotprodGemm_v2{N=1,K=6144,BS=128,VEC=8,FMA=1}@full`, kept as `census-llbf16dotprodgemm-v2-n-1-k-6144-bs-128-vec-8-fma-1-nfull` (unit 12 of `overhead.units`)
3. `census:EmbeddingShard_v2{VS=19360,H=1,START=38720}@full`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-38720-nfull` (unit 20 of `overhead.units`)
4. `census:ScaledMmFp8BlockSharedScale_v2{K=1536,N=8,G=128}@full`, kept as `census-scaledmmfp8blocksharedscale-v2-k-1536-n-8-g-128-nfull` (unit 28 of `overhead.units`)
5. `census:DsaRMSNorm_v2{N=2048,EPS={"f64":"0x1.4f8b588e368f1p-17"}}@full`, kept as `census-dsarmsnorm-v2-n-2048-eps-f64-0x1.4f8b588e368f1p-17-nfull` (unit 36 of `overhead.units`)
6. `census:AllReduceMultimem_v2{WORLD=8,SIZE=96,TREES=[[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7],[[[[[[0,[1,2]],3],5],6],4],7]],FTZ=false}@full/4`, kept as `census-allreducemultimem-v2-world-8-size-96-trees-0-1-2-3-5-6-4-7-0-1-2-3-5-6-4--92be95d7b94f-nfull-4` (unit 48 of `overhead.units`)
7. `census:EmbeddingShard_v2{VS=19360,H=1,START=96800}@full/4`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-96800-nfull-4` (unit 56 of `overhead.units`)
8. `census:EmbeddingRead_v1{V=256,H=96}@full/4`, kept as `census-embeddingread-v1-v-256-h-96-nfull-4` (unit 64 of `overhead.units`)


**Group 6** (`out/gpu6/out/config.txt`; 9 statements, about 32 min by the balance estimate):

1. `census:EmbeddingRead_v1{V=256,H=96}@full`, kept as `census-embeddingread-v1-v-256-h-96-nfull` (unit 7 of `overhead.units`)
2. `census:CublasSplitKGemm_v2{K=6144,N=1,SPLIT_K=4,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full`, kept as `census-cublassplitkgemm-v2-k-6144-n-1-split-k-4-dot-fn-hopperbf16wgmmadot16-v2-nfull` (unit 13 of `overhead.units`)
3. `census:EmbeddingShard_v2{VS=19360,H=1,START=58080}@full`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-58080-nfull` (unit 21 of `overhead.units`)
4. `census:RmsNormMnnvl_v2{N=6144,SHOT=1,CLUSTER=8,EPS={"f64":"0x1.4f8b588e368f1p-17"},FMA=true}@full`, kept as `census-rmsnormmnnvl-v2-n-6144-shot-1-cluster-8-eps-f64-0x1.4f8b588e368f1p-17-fma-true-nfull` (unit 29 of `overhead.units`)
5. `census:MoeExpertFp8ReadGemm_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=3072}"},K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v2"}}@full/4`, kept as `census-moeexpertfp8readgemm-v1-read-fn-embeddingread-v1-v-256-h-3072-k-6144-n-1--c7d7156ac83b-nfull-4` (unit 37 of `overhead.units`)
6. `census:ScaledMmFp8BlockSharedScale_v2{K=2048,N=4,G=128}@full/4`, kept as `census-scaledmmfp8blocksharedscale-v2-k-2048-n-4-g-128-nfull-4` (unit 41 of `overhead.units`)
7. `census:SparseMlaCombine_v2{NSPLIT=5,DV=64,INV={"fn":"Fa3InvSum_v2"}}@full/4`, kept as `census-sparsemlacombine-v2-nsplit-5-dv-64-inv-fn-fa3invsum-v2-nfull-4` (unit 49 of `overhead.units`)
8. `census:EmbeddingShard_v2{VS=19360,H=1,START=116160}@full/4`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-116160-nfull-4` (unit 57 of `overhead.units`)
9. `census:ResidualAdd_v2{N=384}@full/4`, kept as `census-residualadd-v2-n-384-nfull-4` (unit 65 of `overhead.units`)


**Group 7** (`out/gpu7/out/config.txt`; 9 statements, about 32 min by the balance estimate):

1. `census:BatchedGemm_v2{B=1,K=512,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full`, kept as `census-batchedgemm-v2-b-1-k-512-n-1-dot-fn-hopperbf16wgmmadot16-v2-nfull` (unit 8 of `overhead.units`)
2. `census:Gemm_v3{K=6144,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full`, kept as `census-gemm-v3-k-6144-n-1-dot-fn-hopperbf16wgmmadot16-v2-nfull` (unit 14 of `overhead.units`)
3. `census:EmbeddingShard_v2{VS=19360,H=1,START=77440}@full`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-77440-nfull` (unit 22 of `overhead.units`)
4. `census:RMSNormFusedCudaVector8_v2{N=6144,EPS={"f64":"0x1.4f8b588e368f1p-17"},BLOCK=256,RSQRT={"fn":"RsqrtApprox_v2"}}@full`, kept as `census-rmsnormfusedcudavector8-v2-n-6144-eps-f64-0x1.4f8b588e368f1p-17-block-256-15c597d772e7-nfull` (unit 30 of `overhead.units`)
5. `census:BatchedGemm_v2{B=1,K=512,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full/4`, kept as `census-batchedgemm-v2-b-1-k-512-n-1-dot-fn-hopperbf16wgmmadot16-v2-nfull-4` (unit 38 of `overhead.units`)
6. `census:CublasSplitKGemmF32Out_v2{K=6144,N=1,SPLIT_K=3,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}@full/4`, kept as `census-cublassplitkgemmf32out-v2-k-6144-n-1-split-k-3-dot-fn-hopperbf16wgmmadot16-v2-nfull-4` (unit 42 of `overhead.units`)
7. `census:ScaledMmFp8BlockSharedScale_v2{K=256,N=4,G=128}@full/4`, kept as `census-scaledmmfp8blocksharedscale-v2-k-256-n-4-g-128-nfull-4` (unit 50 of `overhead.units`)
8. `census:EmbeddingShard_v2{VS=19360,H=1,START=135520}@full/4`, kept as `census-embeddingshard-v2-vs-19360-h-1-start-135520-nfull-4` (unit 58 of `overhead.units`)
9. `census:Fp8GroupQuantUe8m0_v2{K=2048,G=128}@full/4`, kept as `census-fp8groupquantue8m0-v2-k-2048-g-128-nfull-4` (unit 66 of `overhead.units`)

The fused MoE type `MoeExpertFp8ReadGemmW_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=4096}"},K=256,N=32,...}` is a `CompositeDefinition`: `MoeExpertFp8BlockGemmW_v4{K, N, G, KB, SWAP, DOT}` over the rows its `READ` opens (`verity/definitions/registry/boolean_fp8_moe.py`). Its full statement is first on group 4, its quarter fifth on group 1.

## Circuit pins (`circuit_sha512`)

- **The two head blocks**, linked from statements staged on Tue 10 Nov, so certain now [read on node A at 18:43Z: `pin.txt`]:
  - `SparseMlaHeadBlock_v2{...,NVIS=64,FIRST=false,...}@full`: `4b31e7caba0113987350c49cb19e8a5d18558a517aa3297986cd5f9732e4f3f9eb589b79c47a3da24870ba151250eee25fef7a2872acebc14c0e2185acb6b0b5`. Its `circuit.txt` is 12,191,652,715 bytes and its `pub.bin` 335,406 bytes.
  - `SparseMlaHeadBlock_v2{...,NVIS=64,FIRST=true,...}@full`: `be5bca5c30515dc443efd8889063517410e610d334806931acd7d6348e10ae9cda1b7091e5b6385bc7d403d01fcf511fa1d389ddd486e63f4688e86395ca646a`. Its `circuit.txt` is 11,667,342,102 bytes and its `pub.bin` 235,788 bytes.
  - The staging job links each only if its circuit file's SHA-512 equals its `stage.json`'s `circuit_sha512` and its `pin.txt`.
  - Kernels restaged the FIRST=false block under its v2 name from `main` at `62bf29346f76` and got the same pin. For FIRST=true the equality is inferred.
- **Every other statement**: unknown until it is staged at this commit. Each pin will be in two places:
  - the staging run's `out/stage/NAME/stage.json` (`circuit_sha512`), written as each staging ends, from about 01:15Z;
  - the proving run's `out/kept/NAME/pin.txt`.
  Hold each restaged circuit to the kept `pin.txt`.

## Where the kept sessions and public files will be

- **On node A, while the proving run runs**: `/workspace/research/runs/<proving run>/out/kept/NAME/`. `window.py` copies a statement's files there as its session set ends (it checks every 2 s):
  - `pub.bin`, `stage.json`, `pin.txt` (and `partition.json` or `tables/` where a statement has them);
  - `sessions/index.jsonl`;
  - for each session, `sessions/<id>/session.json`, `circuit.rep0.bin`, `circuit.rep1.bin` and `verifier.json`;
  - `exit`, the prover's exit status (a statement finished only on 0).
  `circuit.txt` and `inst.bin` are not kept.
- **The statements themselves** (`circuit.txt`, `pub.bin`, `inst.bin`, `stage.json`, `pin.txt`) are in `/dev/shm/proofs-overhead/data/NAME-aade0df3ae6e7661/` on node A from their staging until the proving run ends.
- **In the store**: the proving run's custody uploads all of its `out/`, `kept/` included, when the run ends (about 06:05Z): `research fetch <proving run>`. The staging run's `out/` (each statement's `stage.json` and staging log) is uploaded when it ends (about 04:30Z).
- **The timers of every type**, beside the kept files, under `out/gpuG/out/gpu/NAME/`:
  - `prove.tsv`: the prover's `LIVE` lines, one a session, with `witness_s`, `prove_total_s`, `e2e_s` and the `t.*` buckets;
  - `prove.err`: the prover's `WITNESS` lines, its host build's parts with `units_pack`, `HostSlots::pack_units`' own seconds, printed on the device path whose unit net returns the outputs;
  - `smi.tsv`: the card's utilization, memory, SM clock and power, sampled through the statement's sessions;
  - `start`, `done`, `exit`.
  Each group's `summary.json` holds every statement's medians and `gpu_util_mean_pct` over its timed sessions. The run's `out/prove.tsv` and `out/seconds.tsv` collect them.

## `program-word.json`

It is the Program of the word Definition that `stage.json` names, written from `stage.json` alone, as kernels' `steady.sh` `program_word` writes it:

```python
import hashlib, json, sys
from pathlib import Path
from verity.circuits.codec import descriptor_chunks
from verity.circuits.program import Program
from verity_vllm.pipeline.serving_rows import template
d = Path(sys.argv[1]); word = json.loads((d / "stage.json").read_text())["word"]
with open(d / "program-word.json", "wb") as f:
    for c in descriptor_chunks(Program(template(word).fn)):
        f.write(c)
desc = json.loads((d / "program-word.json").read_text())
assert desc.get("program_fn") == word and "annotations" not in desc, (desc.get("program_fn"), word)
```

`lean_verify.py` compares the Program's `program_fn` with `stage.json`'s `word` by name only, and holds the Program to nothing in the circuit. So an accept is labelled "accepted on the provisional unit cover, units as stated, the Program not held to the circuit".

## The head blocks' `stage.json` names v1, as every Boolean version's does

- A census statement's `stage.json` has three name fields:
  - `census`: the name it was staged under, here `SparseMlaHeadBlock_v2{...}`;
  - `definition`: the Definition staged, the Boolean version `SparseMlaHeadBlock_v2{...}`;
  - `word`: `template(name).word`, the census's own word, which for the v2 name is `SparseMlaHeadBlock_v1{...}` [run: `serving_rows.template` at this commit].
- So `word` names v1 while the circuit is v2's, as for every Boolean version.
- The recipe above writes the v1 word's Program, whose `program_fn` equals `word`. `lean_verify.py` checks only that name equality.
- Both head blocks on node A were staged under their **v2** census names [run on node A]. Their tags are `census-41f74af0767c73b3` (FIRST=false) and `census-0ff329cb767f45b2` (FIRST=true), the SHA-256 of each v2 name.
- So a restaging under the v2 name at this commit has the same values and the same `pub.bin`. Kernels' restaging of FIRST=false from `main` had the same tag and pin [relayed].
- Even so, verify against the kept `pub.bin`, the public file the sessions were proved on.

## Table reads need `flock-read-tmpl` while staging

- Four statements are of the `read` layout:
  - `census:EmbeddingRead_v1{V=256,H=4}` at full and full/4;
  - `census:EmbeddingRead_v1{V=256,H=96}` at full and full/4.
- `chunk_rows.statement` makes such a statement structural, and `stage.py` then stages it on the verifier's own read template. That template comes from `flock-read-tmpl`, looked for in this order: `read_tmpl.EXE`, else this tree's `verity/protocols/flock/verifier/.lake/build/bin/flock-read-tmpl`, else the one on `PATH`.
- To build it: `lake build flock-read-tmpl` in `verity/protocols/flock/verifier`, toolchain `leanprover/lean4:v4.34.0`. It requires only the three base packages, by path (`verity/circuits/base`, `verity/circuits/lean`, `verity/commitments/lean`).
- Of the 36 types, no other is structural [run: `chunk_rows.statement` on each of the 36 at this commit]. The fused MoE types (`fused` layout) stage without it.
- The staging job builds it from a copy of those four packages and puts it on `PATH` **(R: the rehearsal is the first run of that step)**.

## Checks my `lean-shard.sh` makes, which a Lean leg may want

(Not launched: the Lean leg is benchmarks'.) For each statement, in the units file's order, it:
1. restages it at the run's commit;
2. holds the circuit's SHA-512 to the kept `stage.json`'s `circuit_sha512` and to `pin.txt`, and stops that statement on a mismatch;
3. writes `program-word.json` from the restaged `stage.json`;
4. builds the verifier's tree from the session directory (copied, because Python 3.12's `rglob` descends into no symlinked directory), the restaged `circuit.txt` and `stage.json`, the kept `pub.bin` and `program-word.json`;
5. runs `verity bench prove lean-verify --repeat 1`.

It verifies one kept session of every statement (the last by its directory's name) before any second session of any statement. It records each verdict, label, wall seconds and peak memory in `lean.tsv`, and each restaging's seconds, peak and pin in `restage.tsv`.
