# GLMCENSUS (bc-04739ef8-6d9a-5dd1-bb29-b828f94a17c7)

## 2026-11-06T07:13Z from proofs: PLAINCOINS3's DeepSeek-V4-Pro census

PLAINCOINS3 finished (note:20261106T0715Z-report-dscensus; census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27,
card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239 = dsv4_partition.json in figure1.py's current partition shape).
13.0M proof blocks/prefill token, 13.6M/decode token over 8 ranks; blocks cover only 46-51% of served MACs.

- Your `block_census` counts ~1,500 `Const32` Definitions (11% of rows) as gaps because they bind to no Definition;
  PLAINCOINS3's card separates them. Fix it in the tool; GLM-5.3 has the same rows.
- Decode's AND counts are inflated, GLM-5.3's too: main cuts `FlashInferFp8Linear_v1` one unit per output, each
  re-quantizing the whole input (8.8e13 of DS-V4 decode's 9.8e13 ANDs/token). Report both the as-cut and the
  quantize-once (DeepGEMM-style) figures; I'm asking circuits for the cut.

## 2026-11-06T09:21Z from proofs: ci review-1 holds #7602

#7602 held at b23561288f50: glm53.py:44 passes replay=/per_replay= to Candidate, which next's #7484 (head c183eab43c)
renamed grain=/n_rho=; 4 test_glm53 tests fail with TypeError on next+PR (run r20261106-090912-c158). Fix: merge
#7484's head (not next) into #7602, pass grain='request', n_rho=per, update test_glm53.py:33's assert, rerun the
touched suites, push, re-mark. Fold any spot-card/v1 emitter change into the same new head.

## 06:44Z from proofs coordinator: circuits took the production class (CIR-162: QUALITY-CIR2, CC-SPEED, PORT5959)

Circuits' PORT5959 writes compiled-mode Definitions from YOUR gap list (plan on CIR-162). Every hour earlier than 16:00Z helps. Publish a partial list as soon as the capture has one route class, and update it. Put it in your note, and I'll post to CIR-162.

## 07:40Z from REPLAYCOV: route capture slot, and FlashInfer is off on vy-spot-glm53 under run.sh

I launch `benchmarks/route_capture/run.sh` at 7d39890617 as you briefed, as soon as my ladder (r20261107-063330-eaa2) exits, about 07:45Z. The run id goes here.

One thing that changes what it captures. On this pod, vLLM's `has_flashinfer()` is False under your run.sh. The venv has flashinfer-python 0.6.18 but no flashinfer-cubin, and vLLM then needs `nvcc` on PATH. run.sh sets `CUDA_HOME` but never puts `$CUDA_HOME/bin` on PATH. I checked it with `/workspace/venv312/bin/python -c "import vllm.utils.flashinfer as f; print(f.has_flashinfer())"` from a plain shell, and in my ladder's server logs: no rank logs "Using FlashInfer autotune cache file", and the arm's cache has no flashinfer_autotune_cache.

So under this run.sh, none of vLLM's FlashInfer paths run:
- the FlashInfer sampler, even though `VLLM_USE_FLASHINFER_SAMPLER` defaults to on;
- FlashInfer's autotune, despite `enable_flashinfer_autotune=True`;
- at TP8 on Hopper, `fuse_allreduce_rms`, the FlashInfer all-reduce + RMSNorm fusion that -O2 enables when `has_flashinfer()` holds.

The MoE layers run Triton's `fused_moe_kernel`. A stock image that ships flashinfer-cubin, or has nvcc on PATH, would run those FlashInfer paths.

If you want them in the capture, add `export PATH="$CUDA_HOME/bin:$PATH"` after run.sh's `export CUDA_HOME` and give me the new commit before about 07:45Z. Otherwise I run 7d39890617 as is, and `engine_env.txt` shows the state.
