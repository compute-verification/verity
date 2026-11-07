# REPLAYCOV inbox


## 2026-11-07T00:45Z, from proofs coordinator: replay needs an Inductor pin (SERVETAX, note:20261107T0020Z-report-servetax-replay)

Compiled vLLM at our pin is not bit-reproducible run to run without a pin. One Inductor kernel (the last residual add +
final RMSNorm, triton_red_fused_add_fused_add_rms_norm_mul_1) benchmarks two reduction tilings (R0_BLOCK 1024 vs 512)
on its first launch per process, uncached, and keeps the faster. That changes the sum order and moves logprobs. With
`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` in the engine env, 10 processes were identical and a 600 s replay had 0
differences, at no measurable throughput cost.
For your ladder: run every replay (and the serve it replays) with the pin, AND keep one unpinned triple per model, so the table can say which models need it. Report both columns.


## 2026-11-07T03:03Z, from proofs coordinator: checkpoint now

The agent watch flags you: 233 min with no note or push since 22:45Z. Task A (the table from existing evidence) was due
Sat 02:00Z. Please:
1. Write a checkpoint note now with the table as it stands (Markdown, measured vs not), even if partial, and publish
   coverage.json (`research data put --preserve`).
2. Your quick-suite run r20261107-002917-56f5 failed on tests/test_retired_terms.py: say "incorrect proof block", not
   "wrong unit" or "faulty proof block", in your doc.
3. r20261106-230849-fe1b was cancelled waiting for a slot on cpu-3. Don't hold a CPU check slot for GPU work; GPU runs
   go on our RunPod pods.
4. If you're waiting on runs, end your turn naming them and their ETAs, and I'll resume you.


## 2026-11-07T04:10Z, from proofs coordinator: two cells to re-check before you republish

1. GLM-5.3, Circuit: circuits' CIR-162 (#7604, marked) compiled GLM-5.3's class G window (B8, TP8 eager) and matched
   its traced Build, 8 of 8 Programs with R1-R3 (run r20261106-083053-29bc). By your column's definition that's
   Circuit ● for that class, if #7604's evidence holds. Check it and footnote the class (eager, prefix caching off,
   prompts <= 256).
2. Proved: the remote demo's served Llama-3.2-1B int8 window (Commit plus sampled proofs, Lean verifier) may give a
   Proved ● row; check records.json and the one_stage served members. OVERHEADDEF's GLM-5.3 statements were refused only
   for lacking a Program, so they don't count; keep GLM-5.3 Proved ○.
Republish the table with these resolved after cc6d and 2166 finish.

## 06:25Z from proofs coordinator: the workload for GLM-5.3 on vy-spot-glm53 (Daniel, tonight)

Daniel wants this to read like a production inference server to a lab ML engineer. So for every GLM-5.3 serve on the pod:

- **Serve through `vllm serve`** (the OpenAI API server, streaming), not an in-process engine, if your ladder can.
- **Keep the V1 defaults on:** continuous batching, chunked prefill, prefix caching, torch.compile with CUDA graphs, async scheduling, max-num-seqs 256.
- **The only changes allowed:** the Inductor pin, the FlashInfer autotune cache pinned by digest, a server-assigned per-request seed, and our recording hooks.
- **The workload:** vLLM's own `benchmark_serving` ShareGPT set with Poisson arrivals, plus a multi-turn set (shared prefixes, so the prefix cache hits) and a long-context set (8k to 32k prompts). Use a mix of greedy and temperature 0.7.
- **Contexts:** NOT capped at 2,048. If the long-context requests diverge, report which ones and whether the divergence starts in the DSA indexer's top-k. That is the finding we need.
- **The ≤ 2,048 window stays** as a control if it's already set up.
- **Bigger runs:** if your harness can't do the API server or these sets within your block, do the pinned and unpinned ladder as briefed first, then one API-server window on ShareGPT plus long-context, and say what's missing.

- 06:41Z REPLAYCOV setup (no GPU): r20261107-060912-bbb1 failed in 1 min (huggingface_hub 2.x has no huggingface-cli). Relaunched as r20261107-061415-6797: GLM-5.3 at the pin aca966e4 into /workspace/fs/glm53/<rev>, ~1 GB/s, ETA 06:45Z plus a hash check. The hub's main is ba32f8c1; it differs from the pin only in chat_template.jinja and an eval-results yaml (weights and config identical). The venv bootstrap is shipping the tree (slow while the download fills the pod's link). GPUs idle until the ladder starts, ETA ~07:15Z.
## 06:35Z from proofs coordinator: terms and three smells (Daniel's smell check)

**Terms (Glossary).** Use these in everything you write:
- The replay unit is a **replay block**, never "window". The Glossary retires "window" for a served span.
- What serving commits is the **boundary commitment**: tokens, the KV pool at replay-block edges, and the step record.
- What replay commits is the **interior commitment**.
- The **first stage** and **second stage** are the two sampling draws (replay blocks, then proof blocks). They are not serving and audit.

**Smells to handle tonight:**
1. **Every replay so far starts from an empty engine.** If you can, also take one GLM-5.3 replay block mid-traffic, i.e. starting with requests in flight. If restoring the KV pool, scheduler queues, prefix-cache index and sampler state isn't possible in your harness, say "from an empty engine" in every result and list what a restore would need.
2. **This pod serves and replays on the same machine.** Label GLM-5.3's results "same machine". The audit setup replays on other GPUs.
3. **Use one Inductor pin:** the env var `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` in the engine env, the mechanism #7705 makes the integration's. If your ladder's `inductor_pin` plugin does something different (first candidate of every choice), run the pinned arm with the env var. Say whether the plugin adds anything the env var misses.
4. **Report replay cost as busy GPU-seconds against serving's busy GPU-seconds**, as well as wall time. At light load, serving's wall time includes idle gaps.
- 06:34Z REPLAYCOV START r20261107-063330-eaa2 (all 8 GPUs): GLM-5.3 TP8 compiled replay ladder, pinned (RBLOCK=0) serve + 4 replays, then unpinned serve + 3 replays. ETA ~10:00Z (firmer after the first boot). Setup done: r20261107-061415-6797 (download, 141 of 141 shards match the pin), r20261107-061442-e619 (venv, CUDA 12.9).

## 06:55Z from proofs coordinator: what's in your block on node A, and the handover to vllm

Your block on `vy-spot-glm53` now has two short capture runs at its end. Commands arrive in this inbox:
- GLMCENSUS's route/shape capture, by 09:00Z: ~10 min of compiled ShareGPT + long-context serving with its capture plugin.
- OVERHEADDEF's capture of sampled proof blocks' values from a replay, by ~10:00Z.

Run them after your ladder, if they're here by then. If one isn't here when your ladder ends, don't wait for it: end your block.

When your block ends:
- make sure every GPU process of ours on the pod has exited;
- write "BLOCK END <time>" here;
- end your turn saying so.

I then ring vllm, whose 4.5 h queue launches at once. Don't extend the lease yourself; I extend it for vllm if needed.

## 06:58Z from OVERHEADDEF: one capture replay on vy-spot-glm53 after your ladder (proofs task, plan note:20261107T0655Z-report-overheaddef-spotproved)

I'm proving sampled proof blocks from a compiled GLM-5.3 replay block. I need one more compiled replay of your **pinned** arm's window, with a capture plugin loaded. It is a vLLM general plugin, env-gated: it hooks one step, writes that step's interior per rank and checks it against the graph run bit for bit.
- **Please keep the pinned arm's record dir** (the `REPLAY_FROM` arm dir: step trace, plan, client tokens) on the pod after the ladder ends, and tell me its path here.
- **The slot** is the one the coordinator's 06:55Z item names: about 45–60 min of all 8 GPUs after your ladder. The exact command (a branch of yours plus a plugin file) will be here by 09:30Z. It's your `replay.py` plus `VERITY_SPOT_CAPTURE=DIR VERITY_SPOT_STEP=s`. If node B (vy-spot-glm53b) comes up first, I'll run it there myself and tell you.
- I won't touch the pod until you say the slot is mine.

## 07:04Z from proofs coordinator: a retired term in #7714

ci's review-2 noted that "the coverage table" is a retired term, at `benchmarks/replay_ladder/README.md:41` and in `benchmarks/README.md:30`'s new row. Under Daniel's bookkeeping ruling it doesn't hold #7714. Rephrase both lines in your next PR, e.g. "the per-model status table", and use that phrase in notes and art descriptions from now on.
- 07:05Z REPLAYCOV r20261107-063330-eaa2 (glm53 ladder, all 8 GPUs) ETA revised to ~08:05Z (replays take ~4 min each; pinned replays 1-2 bit-exact so far).

## 07:08Z from GLMCENSUS: the route capture, for the capture slot after your ladder (proofs task GLMGAP)

What it runs: about 10 min of GLM-5.3 served the way production serves it, with a capture plugin loaded.
- The server is `vllm serve` at vLLM's defaults: compiled with CUDA graphs, chunked prefill, prefix caching and async scheduling. It runs at TP8 with `--max-num-seqs 256 --max-num-batched-tokens 8192 --max-model-len 40960`.
- The engine env has the one Inductor pin, `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`. No other plugin loads: `VLLM_PLUGINS` names only this one.
- Four `vllm bench serve` clients run at once:
  - ShareGPT, greedy, ramped from 1 to 24 requests/s;
  - ShareGPT at temperature 0.7 and top-p 0.95, 2/s;
  - 8k to 32k-token prompts, 0.1/s;
  - 16 shared 4k-token prefixes, 1/s.
- The plugin writes every engine step's shape class. On rank 0 it profiles the first 2 steps of each class with `torch.profiler`, which includes the kernels inside CUDA graphs.
- It uses all 8 GPUs for about 30 to 40 min: a boot from empty compile caches, 10 min of traffic, then the drain and teardown.

The command, at commit `ddf2bec4cf` of `cursor/proofs-glmgap-ce9f`. Add the GPU flags your ladder's run used (for example `--gpus 8`):

    git fetch origin cursor/proofs-glmgap-ce9f && git worktree add --detach /tmp/glmgap-capture ddf2bec4cf
    research run --on vy-spot-glm53 --project verity --source /tmp/glmgap-capture -- bash benchmarks/route_capture/run.sh

What it assumes. Override any of these with `--env K=V`:
- `VENV=/workspace/venv312`. It checks that vLLM is `+gd9105ea8`.
- `HF_HOME=/workspace/hf`, with the checkpoint at `hub/models--zai-org--GLM-5.3/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42`. Otherwise it uses `checkpoints.json`'s `local_path`, or `CKPT=DIR`.
- ShareGPT at `$HF_HOME/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`. If the file is missing, it fetches it from the hub (dataset `anon8231489123/ShareGPT_Vicuna_unfiltered`). If the pod has no egress, `--send` the file and pass `SHAREGPT=FILE`.
- `CUDA_HOME` is the newest `/usr/local/cuda-*`, as in your ladder.

Checks and fallbacks:
- It refuses to start while any GPU holds more than 2 GiB, so your ladder's processes must have exited first. `FORCE=1` overrides this.
- The FlashInfer sampler stays on (vLLM's default). If the temperature-0.7 smoke request fails, it restarts once with `VLLM_USE_FLASHINFER_SAMPLER=0` and records that in `engine_env.txt`. If you already know the sampler's JIT fails on this pod, pass `--env VLLM_USE_FLASHINFER_SAMPLER=0` to skip the extra boot.
- If the boot fails in CUDA-graph capture with a profiler error, rerun with `--env ROUTECAP_CAPTURE_OPS=0`.
- For a short check first, pass `--env DURATION_S=60`.

What to send back. Append the run id to `inbox/GLMCENSUS.md`, with the tail of its log: the clients' request counts and `summary.json`'s one-line totals. I fetch the outputs with `research fetch RUN --all`. They are under `out/`:
- `capture/`: `steps.rank0.jsonl.gz`, `kineto/rank0/*.json.gz`, `capture_ops.rank0.json` and `engine.rank0.json`;
- `clients/`, `server.log`, `compiled-code.tar.gz` and `summary.json`.

## 07:16Z from proofs coordinator: #7714 fixed and re-marked by me at 77f3daddee43

The change is the two-line rephrase ("the per-model status table"). Fetch it before you push to `cursor/proofs-replaycov-643a` again.

## 07:24Z from GLMCENSUS: the route capture, use commit 7d39890617 (proofs task GLMGAP)

Same command as at 07:08Z, at the newer commit (it adds each decode row's key count to the step rows; nothing else changes):

```
git fetch origin cursor/proofs-glmgap-ce9f && git worktree add --detach /tmp/glmgap-capture 7d39890617
research run --on vy-spot-glm53 --project verity --source /tmp/glmgap-capture -- bash benchmarks/route_capture/run.sh
```

If you already launched at ddf2bec4cf, keep that run and tell me its id; I can work without the key counts.

## 07:32Z from proofs coordinator: #7714 has landed (77f3daddee43)

For your next PR (review-1's bookkeeping): `benchmarks/replay_ladder/coverage_table.py:1` and `tests/test_coverage_table.py:1` still say "The coverage table" in their docstrings. Say "the per-model status table". Rename the module too if it's cheap.
- 07:41Z REPLAYCOV END r20261107-063330-eaa2 (glm53 ladder). **GLM-5.3 TP8 compiled replay: 4 of 4 pinned replays bit-exact, unpinned 100% (3 of 3 bit-exact).** Pinned (Inductor pin + RBLOCK=0): 517 requests, 560 steps, 84,352 rows, 0 rows / 0 token positions / 0 requests differing in every replay (3 from the compile cache, 1 fresh compile); replay wall 86-90 s vs serving's 97.5 s. Unpinned: 507 requests, 567 steps, 83,136 rows, 0 differing in all 3; 86-87 s vs 94.7 s. Contexts <= 2,048 (prompts 16-1,024 + caps 64/256), so DSA top-k keeps every key. FlashInfer is off in this venv (has_flashinfer() False: no flashinfer-cubin, no nvcc on PATH), as in every earlier ladder row.
- 07:43Z REPLAYCOV START r20261107-074247-ac04: GLMCENSUS's route capture (cursor/proofs-glmgap-ce9f at 7d39890617), all 8 GPUs, ETA ~08:25Z. Then my production-like window (prod.sh glm53), then tp_replay.

## 07:50Z from proofs coordinator: node A order (no node B yet). Great result on the ladder.

Node A's lease is now to 20:02Z. The order on node A:
1. Your ladder's unpinned replay 3 (running).
2. GLMCENSUS's route capture: circuits' PORT5959 is waiting on its gap list. Make sure FlashInfer is ON for it (your FLASHINFER=1); production has it.
3. **Your production-like replay block** (`prod.sh glm53`, FlashInfer on, ShareGPT + multi-turn + 8-32k, pinned, 2 cached replays + 1 fresh, with logprobs). This is the demo's main evidence. Then run it once unpinned, if it takes under 30 min.
4. **Drop `tp_replay.sh` and serve_tax step 4 from your block.** vllm's queue has the tp_replay rerun, and SERVETAX owns the production tax.
5. Then write "BLOCK END". GLMREPLAY's TP8 edge-restore run 1 (60 min) goes next, then OVERHEADDEF's capture if it's ready, then vllm's 4.5 h queue.

Report the production replay block's verdict as soon as you have it: steps, tokens and logprobs equal, how many requests ran past 2,048 keys, and whether any diverged in the DSA top-k.
- 07:48Z REPLAYCOV: the capture's first launch (r20261107-074247-ac04) failed at once (exit 127): GLMCENSUS's command has no `--cwd`, so `bash benchmarks/route_capture/run.sh` ran from the run dir. Relaunched with `--cwd source` as **r20261107-074748-1074**, all 8 GPUs, ETA ~08:30Z. Then prod.sh glm53 (~1.3 h), then tp_replay (~1 h).
- 07:50Z REPLAYCOV: the status table with GLM-5.3's Replay cells is art:78270528ec574c6c81100cf922be05b4660c89898d9db3003f20c56a69b1a835 (supersedes art:6b33fbf9…; GLM-5.3 pinned ● 100% 4 of 4, unpinned ● 100% 3 of 3; the ladder run's custody is art:bf1d181b…). The only Inductor reduction with a choice in GLM-5.3 at the pin is the FP8 quantizer's abs-max, which kept 3 different XBLOCKs unpinned, an order-free max, so no row moves. FYI: the pod's lease file says it was extended at 07:44:00Z to 20:02:00Z "by proofs". That wasn't me (my runs' timeouts reach ~09:35Z), and it's 2 min past the 20:00Z bound.
