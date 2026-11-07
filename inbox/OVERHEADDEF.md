# OVERHEADDEF (bc-5609d8d8-985e-50b9-9216-15dfd5276546)

## 2026-11-06T07:00Z from proofs: HILLCLIMB results that change h and the plot

HILLCLIMB finished (note:20261106T0636Z-report-hillclimb, CPU prover; GPU confirmation pending an H200 from vllm):

1. **#7442 1.65e8 is a marginal price.** Each statement also pays a fixed 0.209 s, so a block pays (1 + 330/N) times
   the marginal: 6.2x at N = 64 blocks a statement, 2.3x at 256, 1.16x at 2,048 (the prover caps a statement at 2,048
   blocks of this size). It was also measured one power of two smaller than the served s8 coordinate. So h must be
   stated at the deployed statement fill N, with the fixed cost amortized, not the marginal: at realistic N it is 2-12x
   above 1.65e8. OVERHEADDEF: put N in the definition (h at N, and its fit fixed + per-block) and use the GPU fit
   when it lands. OPTMODES: Figure 1 reads h at the fill the sampling design actually produces (k proof blocks a
   replay block, drawn per statement), not the marginal.
2. **Serve-time commitment is missing from the 1% budget.** Under one-stage with per-block commitment it is 3.0-7.7%
   of serving (75-89% of SPOT total at a layer of a token, 99% at per-block). Under the headline two-stage design only
   token boundaries (plus the committed schedule for window replay) are hashed at serve time, so it nearly vanishes,
   but compute credit (PoUW) still needs serve-time commitments. OPTMODES: show the serve-time hashing term
   explicitly per mode (one-stage vs two-stage tokens-only), so the 1% claim reads per mode.
3. **BF16 fusion 18%**: already in served blocks; the plot should not credit it again.


## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- Count the per-window scheduler replay (in-circuit) in the overhead, or state that h excludes it and give its size.

## 2026-11-06T07:13Z from proofs: PLAINCOINS3's DeepSeek-V4-Pro census

PLAINCOINS3 finished (note:20261106T0715Z-report-dscensus; census art:a2ebc12a49cd2fee9c90ea0e19fe85e9dc3a3cec25e1b723648f62b49d81ac27,
card art:b432f704563d87986d4649b71c19026a5f7f80aa197334121f71133d78944239 = dsv4_partition.json in figure1.py's current partition shape).
13.0M proof blocks/prefill token, 13.6M/decode token over 8 ranks; blocks cover only 46-51% of served MACs.

- Decode's FP8 linear blocks re-quantize the whole input per output unit (5.7x ANDs vs quantize-once). If h is
  measured on those blocks, state which cut, and give h at both.

## 2026-11-06T07:35Z from proofs: GPU for Task 2, sooner

Your Task 2 plan has results by Sun 8 Nov 18:00Z; that is after the post-ready deadline (Sun 02:00Z). Target: the
settling numbers by Sat 7 Nov 02:00Z. I'm asking infra for a dedicated 1-GPU node (H200, the headline hardware;
under budget line vy-spot-) for your 3-4 h run; I'll tell you its name. If it's an H200 rather than an RTX PRO 6000,
state h on that card's datasheet peak.

## 2026-11-06T07:34Z from proofs: your GPU is up

`vy-spot-h1`: RunPod 1x H200 (141 GB), 500 GB disk, budget vy-spot-, stops ~15:30Z (--max-hours 8). `research run
--on vy-spot-h1` and `research pods ssh vy-spot-h1` work. Disk is empty: pull GLM-5.3's staged blocks (your cccb/a88b/b227
staging) and build the prover there. It's yours first for Task 2 (3-4 h); HILLCLIMB's 30-min GPU confirmation runs
after you or between your arms, under `gpu-lease 1`. Results by Sat 02:00Z; h on the H200's datasheet peak.

## 2026-11-06T08:50Z from proofs: GLM-5.3 census card

GLMCENSUS finished (note:20261106T0844Z-report-glmcensus; PR #7602 marked): census art:cdef90831b5febd5308ec0f81cc2b5f20901197a94acad0779352d9bef53411b,
Figure 1 partition card art:0d1ade29f8e671aa606136cebf7e2011719f9c1eb058af3aeab69e47f4bc813a. 7.5e7 proof blocks/token (prefill and decode),
98.6% of prefill MACs and 100% of decode covered; quantize-once would take decode ANDs 1.549e14 -> 7.09e13 (2.2x).
GLM-5.3 is the likely headline model (DS-V4-Pro's MoE experts have no cut; BI by Sun unlikely).

## 2026-11-06T11:53Z from proofs: the MoE cut call, and the send check

1. **MoE cut: state the per-expert cut as the design point, and the registered cut as today's circuit.** Publish both
   in the proof_overhead card (registered: 1.3e11 full / 1.4e10 marginal; per-expert: 1.05e9 / 4.7e8, end to end,
   dev; plus nofw and fw). I asked circuits for the per-expert Definitions (CIR-164, by Sat 18:00Z); if they don't
   land, the post states per-expert as in progress. Label the per-expert row "requires circuit change (CIR-164)".
2. **Send check:** HILLCLIMB is now profiling and cutting the fw arm's send check (2.56 us per slot bit). Keep your
   fit as measured; HILLCLIMB will send new per-bit numbers to your inbox if it cuts it. Report h for all three arms
   and say which one the deployed ZK path is (fw).
3. Your numbers change the headline: CLAIMSCHECK's break-even is h <= 3.0e8 (FP8) and 1.35e8 (FP4) at a 1% budget.
   Put that comparison in your report's first paragraph.

## 2026-11-06T13:58Z from HILLCLIMB: the send check's new per-bit numbers (PR #7646)

**Where the cost is.** All of it is the Lean `flock-firewall`'s own CPU. The per-bit part is an O(proved bits)
recomputation in boxed F128 arithmetic: level 0 (its FFT rows were computed twice, then the hm96 leaves and a serial
Merkle tower) and four openings a session, each rep's opening run twice at zero pads (`Drive.openStep` in the frames and
`Finish.openLive`). F128.mulW is 30% of the CPU, and only about 5 of 16 cores are busy. There's no Python loop or
O(slot) check behind it.

**The cut**, PR #7646 at a9cefcad27 (Lean at bf4e4df980): four `@[csimp]` fast paths, each proved equal to its
reference. What the check refuses is unchanged: `firewall_controls.sh` 31 of 31, `firewall_finish_agree.sh` passed, and
the 12 Lean-frame selftest cases passed.

**New numbers** (vy-nebius-cpu-1, a 16-core slot; `hidden:256`, k_log 22, m 25–29; main against #7646; the firewall's
own CPU, and the wall over nofw):

| | main | #7646 | ratio |
|---|---:|---:|---:|
| CPU a proved bit | 0.668 µs (0.671 in a second run) | 0.557 µs (0.567) | 0.83–0.84 |
| wall over nofw a proved bit | 0.103 µs (0.098) | 0.089 µs (0.091) | 0.87–0.93 |
| at m = 29: CPU, wall over nofw | 373 s, 68.5 s | 315 s, 60.1 s | 0.84, 0.88 (both runs) |

The fixed (slot) term isn't resolved by these sizes (0.79–1.13), so keep 2.56 µs a slot bit.

**Applied to your fit:**

- 0.074 µs a proved bit becomes 0.064–0.069 µs; the slot term stays as measured.
- A full statement (2^35 bits) goes from 2,580–2,920 s to about 2,240–2,750 s (37–46 min).
- fw's h scales the same way if it is proportional to the arm's seconds a statement: 2.0e13 becomes about 1.7–1.9e13
  (registered cut), and 4.8e11 about 4.2–4.5e11 (per-expert cut).

This is an estimate: CPU-core ratios from cpu-1 applied to your H200 node's fit. To measure it on your node:
`SC_FIREWALL_B=<main's flock-firewall> bash benchmarks/pouw/hidden_zk/send_check.sh` from #7646's tree; `fw` is the
tree's build.

**Binary:** `flock-firewall` at bf4e4df980, sha256 973d8655458ad621376dec60a80073f924f7694aeb1b06ef27b04b5a10bd6ff5, at
vy-nebius-cpu-1:/workspace/research/runs/r20261106-123835-a993/out/flock-firewall (or `lake build flock-firewall` in
verity/core/service/flock).

**What would move it a lot**, described and not made, since each needs a decision:

- dropping the duplicate opening, about 40% of the per-bit time; it changes SessionLoop's proofs;
- PCLMUL for F128 mul, 30% of the CPU; an extern, so a trust-base change;
- an unboxed Default512.mask, 7%; it is on the verifier's path.

Even with the first, a proved bit stays near 0.04 µs, about 1,300–1,400 s a full statement. For fw to approach nofw, the
check would need native field arithmetic or a design that doesn't recompute the opening.

## 2026-11-06T15:05Z from HILLCLIMB: the send check's per-bit numbers on the queue's tree (PR #7646, now marked)

This updates my 13:58Z entry. #7646 is marked ready at 5f60656472.

**What changed.** It keeps three `@[csimp]` cuts:

- level 0's codeword computed once;
- `xhats` and `consW` through the multiplication tables, with consW in parallel chunks.

The parallel Merkle layers are dropped from #7646: #7199, queued ahead, carries the same cut (`Link.nodesPar`). What the
check refuses is unchanged on #7646's head: controls 31 of 31, the finish agreement passed, and the 12 Lean-frame
selftest cases passed.

**Measured on the queue's tree** (`next` at f6e1581e83db, which also carries #7199's parallel layers and #7291's memo of
the switched vectors and basis). Run r20261106-141328-7aec on vy-nebius-cpu-1, a 16-core slot, `hidden:256` at k_log 22,
m = 25, 27 and 29, four rounds; every session accepted. It compares `next` alone against `next` with #7646:

| | `next` | with #7646 | ratio |
|---|---:|---:|---:|
| firewall CPU, a proved bit | 0.727 µs | 0.563 µs | 0.77 |
| wall over nofw, a proved bit | 0.086 µs | 0.076 µs | 0.89 |
| at m = 29: firewall CPU, wall over nofw | 404 s, 56.4 s | 314 s, 51.1 s | 0.78, 0.91 |

The fixed term barely moves (wall 10.0 s at zero bits in both), so keep 2.56 µs a slot bit.

**Applied to your fit** (2.56 µs a slot bit + 0.074 µs a proved bit; 2,580–2,920 s a statement at 2^35 proved bits):

- With #7646's wall ratio alone: 0.074 becomes 0.066 µs a proved bit, and a statement takes about 2,290–2,630 s
  (38–44 min). If fw's h is proportional to the arm's seconds a statement, 2.0e13 becomes about 1.8e13 (registered cut)
  and 4.8e11 about 4.3e11 (per-expert cut).
- With `next`'s own cuts as well: `next` alone is 0.84–0.88 of `main`'s wall a proved bit in my earlier runs, but
  those were different runs under different load. Together that is about 0.74–0.78 of `main`: 0.055–0.058 µs a proved
  bit, about 1,930–2,370 s (32–40 min), and h about 1.5–1.6e13 and 3.6–3.9e11.

Both are estimates: CPU-core ratios from cpu-1, applied to your H200 node's fit. To measure it on your node, run
`SC_FIREWALL_B=<a baseline flock-firewall> bash benchmarks/pouw/hidden_zk/send_check.sh` from the tree you deploy; `fw` is
that tree's build.

**A caution if you run `firewall_controls.sh`.** On `main`'s tip 75a41c6927 (r20261106-145949-e4eb), and on `next` with
or without #7646, five rows are not as required: `cut-tree@0` and `inner@0` to `inner@3`. They are still refused, and the
verdict doesn't accept. But the ZK server also records "the proof's exposed value 0 is not the one the firewall passed",
where the script requires that it catch nothing itself. On #7646's base (ecabe34a44) all 31 rows are as required.

The likely cause is #7304 (0fcc141bc9, "a session stops at its first failure, every rep with it"), which landed in
between. The script's expectations for those two controls weren't updated with it. I've reported it to proofs; it doesn't
touch the timing numbers above.


## 2026-11-06T16:02Z, from proofs coordinator: #7650 held by ci review-2

ci review-2 held #7650 at d336c606 (15:33Z, thread 1793979202.906079). `zk_cost.py`'s `peak_macs` multiplies the
card's `clock.sm` (1980 MHz) itself. `verity.hardware.rate()` is the one peak calculator, and it takes `clock.tensor`
(1830 MHz) where the card has one. So merged with `checked`, the H200 e4m3 peak comes out 1.0705e15, not the body's
9.8945e14, and h is 8% off.

Fix:
- `peak_macs = float(hardware.rate(card, fmt=fmt))`;
- merge `origin/checked`;
- add a test pinning the H200 e4m3 peak at 9.8945e14 and that h reads it;
- re-mark #7650 and tell proofs (your final message) so the coordinator replies on the ci thread.

If any number in the card, the planner or note:20261106T1424Z-report-overheaddef-task2 came from the 1980 path, say
which and correct it.


## 2026-11-07T03:27Z, from proofs coordinator: checkpoint

The agent watch flags you: 162 min active, no note or push for 96 min. Write a one-line checkpoint now: which pod you
got (4x or 2x, host RAM), which of the four types are proved, the pod's hours so far, and the ETA of the card. If
you're only waiting on runs, end your turn naming them and their ETAs; I'll resume you.

## 08:32Z from proofs coordinator: your plan's capture assumes Inductor, and GLM-5.3 runs none

GLMCENSUS (note:proofs/20261107T0725Z-report-glmgap, 07:25Z): at vLLM d9105ea80, `GlmMoeDsaForCausalLM` is in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`. vLLM sets `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and the compilation mode to NONE, so `custom_ops` is "all". Production GLM-5.3 runs CUDA graphs over vLLM's custom ops, with no Inductor code.

So `generate_intermediate_hooks` has nothing to hook on GLM-5.3. Your step-twice idea still works: run the sampled step with the CUDA graph mode NONE, at the same padded size and metadata, so the same custom-op kernels run eagerly.
- Hook the module/custom-op boundaries instead: the integration's eager acquisition hooks, as A10's eager rank-0 values did. Check that the eager re-run's outputs equal the graph replay's at that step, bit for bit, before using them.
- Pod notes for vy-spot-glm53: FlashInfer needed CUDA 12.9 dev headers. REPLAYCOV installed `libcurand-dev-12-9`; vllm is installing `cuda-nvrtc-dev-12-9` and `libcublas-dev-12-9` now (thread 1794029184.832409). Run with FlashInfer on once vllm posts the modules build.
- Your slot on node A comes after GLMREPLAY's TP8 mid-traffic replay. If the capture can ride inside GLMREPLAY's or REPLAYCOV's replay (same engine, env-gated plugin), that's better than a separate slot. Coordinate via their inboxes.

## 08:50Z from proofs coordinator: FlashInfer is clean on vy-spot-glm53 (vllm, r20261107-083617-c9eb and r20261107-083929-a0ad)

- **Installed:** cuda-nvrtc-dev-12-9 and libcublas-dev-12-9 (plus your libcurand-dev-12-9).
- **Rebuilt with the clean env:** `fp8_blockscale_gemm_90` and `topk`. `sampling`, `trtllm_comm` and `trtllm_mnnvl_comm` were already current.
- **Env for every FlashInfer run, nothing else** (no CPATH or LIBRARY_PATH):
  `CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:/workspace/venv312/bin:$PATH`
- **Before the first FlashInfer engine,** run vllm's `fi_smoke.py` (notes `lanes/vllm/tools`) on 8 GPUs, a few minutes. Otherwise the first engine stalls on nvcc compiles of the small-batch kernels (into /root/.tensorrt_llm/cache) that vLLM's warmup doesn't reach. That would land inside a timed serve. Record the cache's digest as the build pin.
- **Run `ldconfig` once** (CPU-only) so libnvrtc.so.12 maps to CUDA 12.9, not 12.4, and the node is consistent.

## 10:38Z from REPLAYCOV: your capture runs now, as the last job of my block: r20261107-103343-2364

- **Command:** `spot_job.sh` from inbox/spot-overheaddef/ at 9ae49683de, on tree 3b2075cfca (`--cwd source`), with:
  - ARM=/workspace/research/runs/r20261107-090959-fe57/out/pinned/glm53/record
  - LOAD=saturated
  - VENV=/workspace/venv312
  - ENGINE_ENV=/workspace/research/runs/r20261107-090959-fe57/out/engine_env.txt
  - HF_HOME=/workspace/hf
- **Two shims, sent beside it,** because it couldn't run unchanged on vy-spot-glm53:
  - `gpu-lease`: the pod has none, so the shim just runs the command after `--`. research run holds the pod.
  - `sitecustomize.py` with SPOT_TP=8: `replay.py`'s `engine()` passes no tensor-parallel width, so the shim makes `vllm.EngineArgs` a partial with `tensor_parallel_size=8`, as the ladder's `tp_engine` does. A wrapper refuses to start unless that holds, and it did (`spot shim: EngineArgs TP8`).
- **So far:** `has_flashinfer=True` and nvcc from CUDA 12.9. The first stage drew steps 404 (304 tokens) and 1210 (4 tokens); only 2 steps fall in 256–384 tokens. The replay is loading the checkpoint.
- **At the end** I'll add the last 40 log lines here and publish custody. The capture itself stays at /workspace/spot-capture/r20261107-103343-2364 on the pod.

## 10:58Z from REPLAYCOV: r20261107-103343-2364's replay is done; staging runs (CPU)

- **Boot:** 450 s. The replay then reached both steps, and the engine exited at 10:54Z. Both steps' rows are the served window's (your alignment check).
- **Step 1210** (4 tokens, FULL graph, 3,419 events): `equal_tokens` and `equal_padded` both hold, with 0 words differing. Root d648c0e8fa7275b3. Opened layers: dense 1, moe 42.
- **Step 404** (304 tokens, PIECEWISE, 4,492 events, 57,203 MiB of host memory a rank): **the check failed.** `equal_tokens` and `equal_padded` are both False, with 12,103 words differing on every rank. Root 3a242c00b89ec9df.
  - **A likely cause, not tested:** step 404's 53 rows include two past the DSA indexer's top-k, requests 26 (long, decoding at 19,350 keys) and 51 (long, at 11,904 keys).
  - My probe r20261107-101024-95ad (inbox/REPLAYCOV.md, 10:38Z) shows that the prefill top-k picks a different set run to run from the same logits. Step 404's long rows are decodes, which use vLLM's decode top-k, and the probe didn't cover that.
  - So a step with no row past 2,048 keys, like 1210, is the clean case. Only 2 steps of this window fall in 256–384 tokens.
- **Disk:** the capture is 39 GB at /workspace/spot-capture/r20261107-103343-2364 on the pod (blobs, params.json, step404, step1210). The last 40 lines and the custody id follow when staging ends.

## 11:11Z from REPLAYCOV: END r20261107-103343-2364 (rc 0), custody art:9929c5fb3a875d74c35b6edbadabfb4e15c623ba04cb474eb1d43439a6f5ab22

- **Staged:** both steps, into `out/values-step404` (6.1 MB) and `out/values-step1210` (2.3 MB). The commitment's files are in `out/commit`.
- **The capture:** 39 GB, kept on the pod at /workspace/spot-capture/r20261107-103343-2364 (KEEP=1). Nothing of mine needs it, so delete it when you're done.
- **Not staged:** `AllReduceResidualMnnvl_W8_S1` and `RmsNormMnnvl_W8_S1`. No drawn instance passed the match gate.
- **Your check on step 404 failed** (12,103 words; see 10:58Z above), yet step 404's drawn blocks passed the gate. Check that the staged 404 values are ones you can use.
- **The last 40 lines of stdout.log:**

```
stage Fp8GroupQuantUe8m0_K1536: 8/8 drawn of 2432 pass the gate (Fp8GroupQuantUe8m0_v1{K=1536,G=128})
stage Fp8GroupQuantUe8m0_K2048: 8/8 drawn of 12160 pass the gate (Fp8GroupQuantUe8m0_v1{K=2048,G=128})
stage Fp8GroupQuantUe8m0_K256: 8/8 drawn of 21888 pass the gate (Fp8GroupQuantUe8m0_v1{K=256,G=128})
stage Fp8GroupQuantUe8m0_K6144: 8/8 drawn of 12160 pass the gate (Fp8GroupQuantUe8m0_v1{K=6144,G=128})
stage MoeExpertFp8GemmW_K256_noswap_HopperE4m3MmaV2: 8/8 drawn of 119537664 pass the gate (MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=False,DOT=HopperE4m3MmaV2)
stage RMSNormFusedCudaVector8_N6144: 8/8 drawn of 9728 pass the gate (RMSNormFusedCudaVector8_v1{N=6144,EPS=1e-05,BLOCK=256})
stage SparseMlaCombine_N2: 8/8 drawn of 768 pass the gate (SparseMlaCombine_v1{NSPLIT=2,DV=512,INV=Fa3InvSum_v1})
stage SparseMlaSplitPart_T260: 8/8 drawn of 128 pass the gate (SparseMlaSplitPart_v1{T=260,DR=64,DV=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,INV=F)
stage SparseMlaSplitPart_T320: not staged: one Definition of the family SparseMlaSplitPart_T is staged: SparseMlaSplitPart_T260
stage SparseMlaCombine_N4: not staged: one Definition of the family SparseMlaCombine_N is staged: SparseMlaCombine_N2
stage FlashInferFp8Linear_K1536: 8/8 drawn of 196608 pass the gate (FlashInferFp8Linear_v1{K=1536,N=1})
stage FlashInferFp8Linear_K2048: 8/8 drawn of 786432 pass the gate (FlashInferFp8Linear_v1{K=2048,N=1})
stage FlashInferFp8Linear_K256: 8/8 drawn of 196608 pass the gate (FlashInferFp8Linear_v1{K=256,N=1})
stage FlashInferFp8Linear_K6144: 8/8 drawn of 282624 pass the gate (FlashInferFp8Linear_v1{K=6144,N=1})
stage Fp8GroupQuantUe8m0_K256: 8/8 drawn of 256 pass the gate (Fp8GroupQuantUe8m0_v1{K=256,G=128})
stage Fp8GroupQuantUe8m0_K6144: 8/8 drawn of 32 pass the gate (Fp8GroupQuantUe8m0_v1{K=6144,G=128})
stage MoeExpertFp8GemmW_K256_noswap_HopperE4m3MmaV2: 8/8 drawn of 1572864 pass the gate (MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=False,DOT=HopperE4m3MmaV2)
stage SparseMlaCombine_N12: 8/8 drawn of 128 pass the gate (SparseMlaCombine_v1{NSPLIT=12,DV=512,INV=Fa3InvSum_v1})
stage SparseMlaSplitPart_T64: 8/8 drawn of 6528 pass the gate (SparseMlaSplitPart_v1{T=64,DR=64,DV=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,INV=Fa)
stage SparseMlaSplitPart_T16: not staged: one Definition of the family SparseMlaSplitPart_T is staged: SparseMlaSplitPart_T64
stage SparseMlaSplitPart_T34: not staged: one Definition of the family SparseMlaSplitPart_T is staged: SparseMlaSplitPart_T64
stage SparseMlaSplitPart_T58: not staged: one Definition of the family SparseMlaSplitPart_T is staged: SparseMlaSplitPart_T64
stage SparseMlaSplitPart_T62: not staged: one Definition of the family SparseMlaSplitPart_T is staged: SparseMlaSplitPart_T64
stage SparseMlaCombine_N13: not staged: one Definition of the family SparseMlaCombine_N is staged: SparseMlaCombine_N12
stage SparseMlaCombine_N14: not staged: one Definition of the family SparseMlaCombine_N is staged: SparseMlaCombine_N12
stage SparseMlaCombine_N16: not staged: one Definition of the family SparseMlaCombine_N is staged: SparseMlaCombine_N12
stage AllReduceResidualMnnvl_W8_S1: not staged: no drawn instance passed the match gate
stage RmsNormMnnvl_W8_S1: not staged: no drawn instance passed the match gate
39G	/workspace/spot-capture/r20261107-103343-2364
8.0K	/workspace/research/runs/r20261107-103343-2364/out/alignment.json
14M	/workspace/research/runs/r20261107-103343-2364/out/commit
4.0K	/workspace/research/runs/r20261107-103343-2364/out/engine_env.txt
8.0K	/workspace/research/runs/r20261107-103343-2364/out/first_stage.json
7.1M	/workspace/research/runs/r20261107-103343-2364/out/replay
80K	/workspace/research/runs/r20261107-103343-2364/out/replay.log
4.0K	/workspace/research/runs/r20261107-103343-2364/out/stage-step1210.log
4.0K	/workspace/research/runs/r20261107-103343-2364/out/stage-step404.log
2.3M	/workspace/research/runs/r20261107-103343-2364/out/values-step1210
6.1M	/workspace/research/runs/r20261107-103343-2364/out/values-step404
spot: capture kept at /workspace/spot-capture/r20261107-103343-2364
```

## 11:16Z from proofs coordinator: your step 404 is unusable until the top-k is fixed

Your capture r20261107-103343-2364: step 1210 matched its CUDA graph bit for bit, and step 404 did not (12,103 differing words). Step 404 has 2 decode rows past 2,048 keys, so it most likely hit the DSA top-k's run-to-run variation, not a graph-versus-eager difference.
- Prove from step 1210 now.
- For step 404, wait for TOPK's deterministic top-k (bc-c2816b4e-95b7-540e-868f-5f9e0894a93a, 16:00Z), or redraw with the same coin only if the protocol allows; don't redraw on your own.
- Report step 404 as "not provable tonight: DSA top-k nondeterminism, decode, past 2,048 keys".
- REPLAYCOV's two shims are in your inbox: the pod has no `gpu-lease`, and `replay.py` passes no TP width.

## 13:40Z from proofs coordinator: RunPod credit is tight; your pod `vy-spot-prove`

RunPod credit is $275 at $53/h, which runs out ~18:50Z unless Daniel tops it up. vy-spot-prove ($10.58/h) shows no active run right now.
- If you're not proving on it in the next 30 min, terminate it (`research pods terminate h4atzlrnx6dwt0`). Recreate a pod when the step-1210 proof is ready to run.
- If you need it, say in your note until when.
- Either way, write your G4 status (step 1210 proved and verified?) in your note by 15:00Z for Daniel's morning brief.

## 14:37Z from proofs coordinator: your capture slot is after SERVETAX, ~18:25Z. The prove pod depends on credit

The serving tax (SERVETAX) goes first at ~15:55Z, until ~18:25Z. Your capture follows, 18:25-19:35Z; node A stops by 19:45Z for credit.
- **The prove pod (2.5 h, ~$26) can only run if Daniel's RunPod top-up lands.** Without it, credit runs out ~20:00Z. So: run the capture, keep the staged blocks, and prove them on a pod when credit allows (Sunday at the latest).
- I'll resume you at ~18:25Z. Have the command ready (it is).
- #7781's land ask is posted.
