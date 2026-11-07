# SERVETAX inbox

## 2026-11-06T16:06Z, from proofs coordinator

- vllm (thread 1793981012.698519): the vy-dc-h200-* machines are ONE H200 each, so GLM-5.3 TP8 runs only on INF-553's
  8 x H200 when it comes up. h200-3 is busy (Pearl+ serves 16:30Z, seal ~19:00Z).
- Order proofs set on INF-553's node: (1) vllm's tp_replay + tp_rerun (~1.5 h); (2) your compiled baseline and eager
  arms, shared with #7631's tick point (coordinate with vllm so one engine setup serves both); (3) VLLM-228's six
  sessions (~2.5 h); (4) your remaining arms. Plan for about 2 h of node time in slot (2) and the rest in slot (4).
- proofs asked vllm for one vy-dc-h200 (not h200-3) for your Moonlight iteration. Don't launch on any vy-dc-h200 until
  this inbox names the machine vllm approved.


## 2026-11-06T16:08Z, from proofs coordinator: H200 approved

vllm approved (thread 1793981012.698519, 16:05Z): SERVETAX may use **vy-dc-h200** (the original, vllm's standby) for
about 4 h today, with `--max-hours`, **Moonlight-16B only**. Deadline Fri 22:00Z. Never vy-dc-h200-3 or vy-dc-h200-2. If
it's stopped, infra starts it on vllm's vy-dc- line. `benchmarks/replay_determinism` (#7616) has a Moonlight fetch with
retention keep: reuse it.


## 2026-11-06T16:15Z, from proofs coordinator: vy-dc-h200 is up

infra (16:13Z): vy-dc-h200 is up on vllm's vy-dc- line, deadline Fri 22:00Z, for SERVETAX, Moonlight-16B only. Use
`--max-hours` so your runs end before 21:55Z.


## 2026-11-06T21:00Z, from proofs coordinator: no 8 x H200 today

infra (thread 1793995241.992249): no 8x H200 or 8x B200 on either RunPod cloud since 07:03Z, and Nebius refused 97 of 101
H200 starts. Infra's Nebius loops keep trying (8x H200 to Sat 02:00Z, and 8x B200), and it rings proofs if one comes
up. So take your fallback:
- finish Moonlight on vy-dc-h200 (to 22:00Z);
- deliver Moonlight's measured table plus a clearly labelled per-token scaling to GLM-5.3, saying what an 8x node would
  change;
- keep the GLM-5.3 harness ready to run within minutes if a node appears before Sat 02:00Z.
Don't rent anything yourself.

## 06:42Z from proofs coordinator: your next task after the batch-invariance arm (Daniel's demo, top priority)

When the batch-invariance table is in, do **GLM-5.3's production tax at TP8** on the second 8x H200, `vy-spot-glm53b`. A retry loop is grabbing it; I'll put its status here.

- **The serve:** `vllm serve` with V1 defaults (compiled, chunked prefill, prefix caching, async scheduling).
- **The workload:** ShareGPT + multi-turn + 8-32k prompts, at three loads set from the measured capacity.
- **The arms:**
  - stock compiled;
  - + Inductor pin;
  - + step record;
  - + tokens-only commit;
  - + a KV-pool snapshot at replay-block edges, taken consistently (pause at a step boundary). Use RESTORE's (bc-d43dc609-c280-5995-93bc-97e6fd8235e6) snapshot code when it's pushed (`cursor/proofs-restore-*`), else the pause-and-hash variant of your KV-edge prototype.
- **Report:** tok/s, TTFT and ITL p50/p99, the tax against stock with CIs, and busy GPU-seconds.

If node B never comes, do one load (medium) on node A after REPLAYCOV's block. Ask me for the slot.

Use Glossary terms: replay block, boundary commitment; never "window".

## 12:04Z from proofs coordinator: your GLM-5.3 production tax goes on node A at ~16:30Z

Node B never came (200 RunPod tries). Node A frees after vllm's queue (~15:55Z) and TOPK's ~25-min rerun. Your slot is ~16:30Z to 20:00Z (lease 22:26Z).
- Prepare now (CPU only):
  - the arms per your 06:42Z inbox (stock compiled; + step record; + tokens-only commit; + a consistent KV snapshot at edges via RESTORE's #7744 code, using the GPU hash kernel, not GLMREPLAY's slow per-block digest);
  - prod.sh's workload (#7766), at three loads from the measured capacity (vllm measured 5.57 req/s with the same setup);
  - FlashInfer on, env as above.
- Report tok/s, TTFT and ITL p50/p99, the tax against stock with CIs, the per-edge pause, and busy GPU-s.
- I'll resume you at ~16:20Z to launch. Have the command ready in your note.

## 14:28Z from proofs coordinator: node A hard stop at 19:45Z (RunPod credit)

Infra: credit runs out ~20:00Z, so vy-spot-glm53 must stop by 19:45Z, unless more credit lands. Your run starts after OVERHEADDEF's 30-min capture, ~16:25Z.
- Set `BUDGET_MIN` so everything ends by **19:30Z**: about 175 min from 16:25Z, minus setup. Cut reps first, never arms or loads, as you planned.
- Write partial results as each arm ends, so a stop at 19:35Z loses nothing finished.

## 14:37Z from proofs coordinator: order changed. You go FIRST after vllm (~15:55Z), and must end by 18:25Z

OVERHEADDEF's capture needs ~70 min, not 30, and RunPod credit ends ~20:00Z. The serving tax is the core deliverable, so it goes first.
- Launch when vllm's queue ends (~15:55Z); I'll resume you.
- Set `BUDGET_MIN` so you finish by **18:25Z** (~150 min including setup). Cut reps before arms or loads.
- OVERHEADDEF's capture follows you (18:25-19:35Z).

## 15:31Z SERVETAX: START r20261107-152819-1122 on vy-spot-glm53 (GLM-5.3 TP8 serving tax)

- It launched at 15:28Z; the GPUs were held at 15:29Z. It runs #7776's head 767ead1fea with fe57's plan (sha256 7cb8c2f3…, checked on the node), FlashInfer on (CUDA 12.9, `LDCONFIG=1`).
  - fi_smoke passed: 279 cases on all 8 ranks.
  - The `compiled` engine is loading.
- **Arms:** `compiled`, `record`, `tokens-only`, `boundary`.
  - The boundary arm takes an edge each 30 s of a load's serving. Each rank hashes its KV pool with the `sm_90` row-table kernels, and rank 0 copies its pool.
- **Loads:** light 1.4, medium 3.3, saturated 8.4 req/s, 60 s of arrivals each.
- **Cut: reps, 3 to 2 for every arm, so it ends by 18:00Z.**
  - `BUDGET_MIN` alone would have cut only the last arm, `boundary`.
  - `BUDGET_MIN=135` stays as a safety net; it bounds the arms to ~17:44Z.
  - `--timeout 8700` is a hard stop at ~17:53Z.
- **ETA ~17:30Z to 17:45Z.** The node is free for OVERHEADDEF from my END.
- **Partial results** go in note:proofs/20261107T1300Z-checkpoint-servetax-glm as each arm ends.

## 16:00Z SERVETAX: RESTART as r20261107-155724-3f53 (loads from this plan's own capacity, 1.67 req/s); ETA ~17:35Z, hard stop 17:52Z

- **I cancelled r20261107-152819-1122 at 15:56Z after its `compiled` arm.** The 5.57 req/s was vllm's probe of a different workload: the trace corpus's plan, 1,200 requests in one burst, greedy.
  - fe57's plan has a capacity of 1.67 req/s (475 out tok/s), and `compiled` reproduced it: 1.62 req/s, 463 tok/s at 8.4 req/s offered.
  - So the old loads were 84%, 198% and 503% of capacity. Medium and saturated measured the same overloaded regime, and there was no light load.
  - The old `compiled` arm (2 reps) stays in that run's record.
- **New loads:** light 0.42, medium 1.0, saturated 2.5 req/s (25%, 60%, 150% of 1.67), with 60 s of arrivals each. That makes a set of three loads short enough for **REPS=3 on every arm**, the planned number, by ~17:35Z.
  - `BUDGET_MIN=108` from the GPU hold at 15:57:45Z bounds the arms to ~17:46Z.
  - `--timeout 6900` is the hard stop at 17:52Z.
- **Found: `research cancel` doesn't reach vLLM's engine.** The engine core and its 8 TP workers were orphaned in their own session and kept 133 GB on every GPU.
  - I checked that they were mine (their environment names the run dir) and stopped them by PID with SIGTERM. All 8 GPUs went to 0 MiB before the relaunch.
  - I'll check for orphans again at END. A friction note for @infra follows.
- 16:14Z (proofs): ci review-1: #7776 landed at 75cad307bf with bookkeeping. benchmarks/replay_ladder/README.md:70 'production-like' trips the retired term 'production' (tests/test_retired_terms.py; say serving or proving). The line came from #7766's 09c14fe242. Fix it in your next PR (e.g. with the GLM tax results).

## 17:10Z SERVETAX: END r20261107-155724-3f53 (rc 0, 17:05:05Z). Every arm's tax is within noise; the boundary commitment pauses 0.98 s an edge with all 8 roots equal

- **Node A is free from 17:05Z**, 55 min early. All 8 GPUs are at 0 MiB and no vLLM process is left.
  - It ran all 4 arms with 3 reps each, 67 min from the GPU hold. Nothing was cut.
  - The cancelled first launch, r20261107-152819-1122 (15:28–15:56Z), used 28 more GPU-minutes. Together that is ~1.6 h, ~$68.
- **The loads** are 0.42, 1.0 and 2.5 req/s: 25%, 60% and 150% of this plan's capacity of 1.67 req/s. There are about 23, 55 and 141 requests a rep, so a p99 is close to the slowest request.
- **Tax** is against `compiled`, in out tok/s, paired by rep with a 95% t-interval over 3 reps (rep r sends the same requests in every arm). Below capacity, tok/s follows the offered load, so the light and medium taxes show in the latency columns and busy GPU-s.

| load | arm | out tok/s | TTFT p50 / p99 (ms) | ITL p50 / p99 (ms) | tax (tok/s) | busy GPU-s (per 1k tok) |
|---|---|---:|---:|---:|---:|---:|
| light | compiled | 100.3 | 88 / 3,728 | 17.6 / 35.1 | — | 386 (64.6) |
| light | record | 100.4 | 85 / 4,043 | 16.9 / 38.4 | −0.1% [−6.4, +6.3] | 387 (64.7) |
| light | tokens-only | 98.1 | 85 / 4,020 | 17.8 / 35.2 | +2.3% [−2.1, +6.6] | 388 (65.9) |
| light | boundary | 99.4 | 81 / 3,817 | 17.2 / 34.7 | +1.0% [−5.6, +7.5] | 375 (63.3) |
| medium | compiled | 189.5 | 207 / 6,702 | 49.0 / 188 | — | 591 (41.8) |
| medium | record | 188.8 | 213 / 6,702 | 49.0 / 193 | +0.4% [−2.4, +3.1] | 590 (42.0) |
| medium | tokens-only | 190.1 | 210 / 6,691 | 50.2 / 182 | −0.3% [−8.0, +7.5] | 590 (41.7) |
| medium | boundary | 189.9 | 213 / 7,680 | 47.9 / 217 | −0.2% [−1.5, +1.1] | 578 (40.9) |
| saturated | compiled | 298.2 | 4,379 / 20,960 | 107 / 3,098 | — | 1,115 (26.8) |
| saturated | record | 299.3 | 4,345 / 20,910 | 107 / 3,092 | −0.3% [−3.2, +2.5] | 1,113 (26.7) |
| saturated | tokens-only | 301.2 | 4,359 / 20,790 | 106 / 3,082 | −1.0% [−2.3, +0.3] | 1,106 (26.5) |
| saturated | boundary | 298.4 | 4,118 / 20,220 | 106 / 3,047 | −0.0% [−2.5, +2.4] | 1,087 (26.2) |

- **The boundary commitment.** There were 21 edges, each 30 s of a load's serving, none in the warm-up.
  - **Pause:** p50 / p99 / max 0.98 / 1.06 / 1.06 s. Under 1 s plus the copy, as targeted; GLMREPLAY's digest took 27 s.
  - **Its parts at p50, on the slowest rank:**
    - hash kernels 318 ms for each rank's 37.26 GB (range 250–346 ms);
    - rank 0's copy to pinned host memory 680 ms (55 GB/s);
    - runner rows 0.14 s;
    - scheduler state 15 ms.
  - **Drain before it:** p50 0.10 s. One edge per medium and saturated rep drained ~3 s, waiting for an in-flight step.
  - **Roots:** all 8 ranks' KV roots were equal at all 21 edges. Each edge hashed 298 GB and copied 37 GB.
- **The H200 hash kernels on their first run:** no failure.
  - `sm_90` built from source on the node at the engine's first step, taking `hash_build_s` 20.3 s. That fell in the warm-up, which no load counts.
  - Ranks 1–7 recorded 17.9 s each. Rank 0 recorded 0.28 s, most likely because it was pinning its host buffer meanwhile and then loaded the built kernels.
- **Caveats:**
  - Busy GPU-s is NVML utilization, which counts NCCL's spin-waits as busy. So it reads the pauses, about 3% of a load's wall at one edge each 30 s, as idle. That is why `boundary` shows −2% per 1k tokens (medium [−3.2, −1.1]); it is not a compute saving.
  - At saturation the pauses alone would cost about 3% of throughput. Three reps don't resolve that: the measured tax is −0.0% [−2.5, +2.4].
  - Every engine saw the same FlashInfer caches, unchanged from before the first arm to after the last.
- **Details** (per-period figures, the unpaired CIs, the edges by load) are in note:proofs/20261107T1300Z-checkpoint-servetax-glm.
- **Also:**
  - #7776's fix (75cad307bf): a terminated arm now kills its server's group within the runner's 20 s grace, so a cancel doesn't leave vLLM on the GPUs.
  - #7795 fixes the retired term ci flagged; it is marked ready.
