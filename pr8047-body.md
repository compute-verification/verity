Lane circuits
Refs CIR-191
Goal proofs/weight-security-bridge

This PR is stacked on #8017 (`RMSNormFusedCudaVector8_v3`) and lands after it. Until #8017 lands, this diff includes its commits.

Served fused RMSNorm now binds `RMSNormFusedCudaVector8_v3`. It computes the same words as v1, but its sum of squares is stated as batches, so struct_lower lays it out as a few calls instead of about 12,300 nodes. #8017 holds v3 bit-equal to v1 at the served statics and on random and adversarial rows, and has its circuit-check.

## What binds v3

- The unsuffixed `cuda_rmsnorm.RMSNormFusedCudaVector8` is now v3, and v1 is `RMSNormFusedCudaVector8V1`. Every served site that binds the fused norm by name therefore binds v3:
  - `targets.rmsnorm_fused_spec`, which the Build's vocabularies, the fold's `FusedAddRMSNorm` and the predictor all ask;
  - the `rmsnorm_fused_cuda` vocabulary;
  - the fold's norm pattern.

  Their id strings now say v3.
- v1 keeps its hand-made Boolean v2 (`boolean_norms`), whose word view is now `RMSNormFusedCudaVector8V1`.
- v3's Boolean version is a σ lift, `RMSNormFusedCudaVector8_v4`, in `boolean_sigma`. Its three helpers are σ-lifted too: `RMSNormPairSumCuda_v2`, `RMSNormSumStepCuda_v2` and `RMSNormSumInOrderCuda_v2`. `boolean_version(v3)` finds v4.
- `boolean_sigma.circuit-check.py` binds v4 at N=536 BLOCK=33. That one setting exercises every batch:
  - two vectors per thread, with a third for some threads;
  - a full warp and a partial one;
  - two warp aggregates.
- v3 has a twin (`RMSNormFusedCudaVector8BatchedTwin`) and GLM-5.3 rows that use v1's row function.
- The capture map's `rmsnorm-cuda` Program cites v3 and v4. The H200 capture checks the numpy rows. Those compute v1's words, which are also v3's.
- spot_proved's staged fused-norm blocks and its synthetic block name v3, and so does m1_unit's example.
- `derive_neutrality.json` is re-pinned for the GLM-5.3 fused-norm stand-in. Its Program digest moves, and its gates (73,732) and nodes are unchanged.

## What stays on v1

- **The warp-cut Flock template `rmsnorm-fused-cuda-vec8`.** Its row now names `RMSNormFusedCudaVector8V1` explicitly. v3's unit netlist has the same gates in a different order: at N=256 BLOCK=64 the text has the same length and a different digest. Moving the template to v3 would need new pins derived at N=6144 under `FLOCK_IR_SLOW`. Nothing served picks this template by family.
- **The recorded cards.** `sampling_plan/glm53_blocks.json` and the spot_figures cards are censuses and measurements of past Builds, which bound v1.
- **`TP8_B1_FAMILIES`** in the replay rows test. It records the families one past Commit failed on.

## The rows of record move: same gates, a new binding

Every row of record in `manifests/records.json` that serves the fused norm now binds v3, so its pinned Program digests move. This PR re-pins them. Without that, the after-landing neutrality check would fail on main.

- **How:** `verity-vllm neutrality ROW --tier T --update` runs on a CPU node with the pinned vLLM, outside `check`, at this branch's head.
- **Heavy rows:** each re-pinned row's traced Build then gets `verity-vllm predict compare`. The row is re-switched (`records switch`) only when that comparison is equal on every rank, with `pinned_from.build` and `aot.trace` set to the traced Build's `art:` id. Each row keeps its `record`.
- **Fast rows:** the fast rows go in as neutrality wrote them.

| row | tier | what moves | state |
| --- | --- | --- | --- |
| GLM-5.3 b1 greedy | heavy | workload and request Programs on all 8 ranks; step unchanged | re-pinned (r20261108-085056-40e5), AOT equal on 8 ranks (r20261108-091211-a664), re-switched |
| Qwen3-0.6B RTX PRO 6000 TP1 | fast | step, workload and request | re-pinned (r20261108-091426-d7df) |
| Qwen3-14B RTX PRO 6000 TP2 | fast | step, workload and request on both ranks | re-pinned (r20261108-091446-fb0a) |
| DeepSeek-V4-Pro b1 | heavy | 8 ranks moved | re-pin running |
| GLM-5.3 b8 greedy | heavy | | re-pin running |
| GLM-5.3 b8 stochastic | heavy | | re-pin running |
| GLM-5.3 b32 | heavy | | re-pin running |

Both Qwen rows reproduce their old pins at this branch's base, `8c7321cd7c` (r20261108-092937-6239 and r20261108-092957-6f20 pass). So the rebind alone moves them.

**Llama-3.2-1B H200 (fast) is not re-pinned here.** It is stale on main independently of this PR. At the base and at the head alike:
- its step Program builds `066d1950…` against the pinned `806aac0c…`;
- its request derive refuses with `numerics_unregistered`.

This PR is marked only once every heavy row above is re-pinned and re-switched.

## circuit-check

`boolean_sigma.circuit-check.py`, with `--update-pins`:

```text
unchecked definition RMSNormFusedCudaVector8_v4{N=536,EPS=1e-05,BLOCK=33}  failures=0 unchecked=1
unchecked definition RMSNormPairSumCuda_v2  failures=0 unchecked=1 q_call=1u/32b
unchecked definition RMSNormSumStepCuda_v2  failures=0 unchecked=1 q_call=1u/32b
unchecked definition RMSNormSumInOrderCuda_v2{K=2}  failures=0 unchecked=1 q_call=1u/32b
unchecked definition RMSNormSumInOrderCuda_v2{K=4}  failures=0 unchecked=1 q_call=1u/32b
```

What these lines mean:
- **v4:** its word view matches v3 on 1,024 vectors with 0 mismatches. It is unchecked only because its partition (13.9 M gates) is over the checker's 5 M budget.
- **The helpers:** they were unchecked only for "no pin". This PR commits their pins.
- **The run's one failure is not v4's.** It is `GemmCoordinate_v4{K=16}: Undecided ... give --as-call or --no-as-call`, a callee of main's existing `bind(SG.Gemm, K=16, N=2)` that the sidecar mode can't decide.
- **Pins left out:** `--update-pins` also pinned 11 targets that are unpinned on main. They are not mine and are left out.

`cuda_rmsnorm.circuit-check.py`, both versions at every served static:

```text
ok  definition RMSNormFusedCudaVector8_v1{N=6144,EPS=1e-05,BLOCK=1024}  failures=0 unchecked=0 q_call=55298u/1572928b
ok  definition RMSNormFusedCudaVector8_v1{N=6144,EPS=1e-05,BLOCK=256}   failures=0 unchecked=0 q_call=55298u/1572928b
ok  definition RMSNormFusedCudaVector8_v1{N=7168,EPS=1e-06,BLOCK=1024}  failures=0 unchecked=0 q_call=64514u/1835072b
ok  definition RMSNormFusedCudaVector8_v1{N=7168,EPS=1e-06,BLOCK=256}   failures=0 unchecked=0 q_call=64514u/1835072b
ok  definition RMSNormFusedCudaVector8_v3{N=6144,EPS=1e-05,BLOCK=1024}  failures=0 unchecked=0 q_call=53740u/1523072b
ok  definition RMSNormFusedCudaVector8_v3{N=6144,EPS=1e-05,BLOCK=256}   failures=0 unchecked=0 q_call=53500u/1515392b
ok  definition RMSNormFusedCudaVector8_v3{N=7168,EPS=1e-06,BLOCK=1024}  failures=0 unchecked=0 q_call=62696u/1776896b
ok  definition RMSNormFusedCudaVector8_v3{N=7168,EPS=1e-06,BLOCK=256}   failures=0 unchecked=0 q_call=62460u/1769344b
circuit-check: 19 target(s), 0 failure(s), 7 unchecked (warnings, which never fail) on 7 Definition(s)
```

## Tests

The affected suites ran in quick mode, at `4c3044172d`:

| suite | passed |
| --- | --- |
| verity/ml/definitions | 453 |
| repository | 264 |
| benchmarks/one_stage | 470 |
| benchmarks/spot_proved | 37 |
| benchmarks/sampling_plan | 52 |
| verity-flock | 1,376 |
| circuit_check | 297 |
| verity-numerical | 640 |

verity-vllm passed 13,219 and failed 2 tests, both now fixed:
- P11 refused `_RMSNormFusedCudaVector8V3Twin`. The twin is now named by its job.
- The fused-norm stand-in's pinned Program moved, and it is re-pinned.

At the head before the re-pins, `e5c19917f0`: verity-vllm passed 13,221 (486 skipped), and verity/ml/definitions passed 453.

After the re-pins, locally: the records tests (`test_circuit_db`, `test_neutrality`, `test_done_source`, `test_predict`) pass.

