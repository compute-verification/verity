# verity-bench: the shared benchmark harness

`verity-bench` measures what an arm (a protocol option or an experimental install) adds to served inference, against
stock vLLM, through the vLLM integration (`integrations/vllm`). It owns specs, runs, sweeps, results and tables. The
integration owns the engine pin, the hooks and the add-on interface (its README's "Hook interface"), and the harness uses
those like any other caller. This file is the harness's contract: how a point is timed and what a result says.
`benchmarks/harness/tests/test_bench.py::test_the_readme_states_the_contract_the_code_keeps` fails when the code drifts from it.

## Commands

~~~text
verity-bench plan SPEC       # expand a spec; each point's identity and its passing run, if any
verity-bench run SPEC        # one recorded `research run --queue` per point without a passing run
verity-bench check SPEC      # exit 0 only when every point has a passing run
verity-bench compare A B     # accept two results only when their identities agree (--allow PATH to permit one)
verity-bench table R1 R2...  # one table from one spec's per-point results, only when their identities agree
verity-bench vllm --preset SPEC --arms A,B --out DIR   # the same measurement in this process, unrecorded
~~~

A spec is `benchmarks/specs/<name>.toml` (`verity-bench/vllm-spec/v0`): the preset (model and revision, precision, GPU,
vLLM pin, engine settings), its arms and the modules of its experimental arms, its sweep (another file in
`benchmarks/specs/`, `verity-bench/vllm-sweep/v0`), reps, eager or CUDA graphs, and admission (`max_minutes` and friends,
which never enter identity). Loading is strict: an unknown key, a wrong type or an unknown GPU id is an error before
anything runs. A preset's model is a Hugging Face id at a pinned `revision`, or a node-local directory with its
`model_digest` (`presets.local_checkpoint`'s hash of its files), which a dispatched point recomputes and refuses to run on
when it differs.

A sweep is the product of its batches, prompt lengths, traffic shapes and samplings: `greedy`, or `stochastic`
(temperature 0.8, top-p 0.95). Every request carries its own seed, so a stochastic point's tokens are a function of the
engine's numerics like a greedy one's.

The engine settings `serving_defaults = true` build every engine as vLLM serves (torch.compile, its CUDA-graph mode and
capture sizes, no batch invariance: the execution label `compiled` of `verity_vllm.engine.served`), with one Inductor
cache directory for all arms' processes, so they run the kernel configurations the first one autotuned. Without it a
graph engine replays decode-only CUDA graphs at `capture_sizes`.

An experimental arm (`arm_modules`, `pkg.mod:NAME`) may need its own environment. The spec's `[env]` table (upper-case
names, string values, none of the dispatcher's, queue's or launcher's own) and its `setup`, a script in the source tree
that `run_point.sh` sources before the point's Python starts (to build the arm's libraries from the dispatched commit and
export their paths, say), are part of the resolved spec and so of every point's identity. The dispatcher forwards both
(`--env`, the setup as `VERITY_BENCH_SETUP`), and a point refuses to run under any other.

## How a point is timed

1. **Gates before timing.** Every arm runs every point, with the point's sampling, before anything is timed:
   - `tokens`: an arm with a control must produce the control's tokens exactly. A failure records the first divergent
     request, position and tokens.
   - `kv-pool`: an arm and its control must get the same KV-cache pool, in tokens. vLLM schedules by free KV blocks, and
     a step's batch composition reaches the numerics, so unequal pools can flip tokens. The preset pins it with
     `resident_kv_cache_memory_bytes` and `block_kv_cache_memory_bytes`. An engine that doesn't report its pool fails it.
   - `arm`: the arm's own gate, when it provides one.

   An arm can't be its own control. An arm without a control (the stock references, or an experimental arm such as
   Pearl-C's that computes different tokens) has only its own gate, and the table names it. Tokens are compared, not
   logits. One failed gate times nothing: the result keeps its gates and has no points. Three gates run after timing,
   and a failure there keeps the gates and drops the points:
   - `timed-tokens`: each timed rep of an arm with a control produces the control's tokens in that rep, so an arm can't
     compute something else only when timed.
   - `engine-identity`: in `blocks`, each timed engine reports the gated engine's KV pool, kernel digest, versions,
     GPU and overrides.
   - `arm-closing`: after an engine's warm-up and timed runs, before it closes, the runner reads the arm's own gate again
     (in `blocks`, one per engine lifetime, with its `cycle`), so an arm that judges every run it saw judges those too.
2. **Resident interleaving, else alternating blocks.** When every arm's engine fits on the GPU at once (`resident`), each
   point is warmed once per arm (discarded), then each rep runs every arm once. The baseline alternates between the first
   and last slot (`interleave_order`). An arm that needs its own engine lifetime makes the whole run `blocks`: each arm
   gets its own engine for two cycles, the second in reverse arm order, with a discarded warm-up per point in each.
   Either way each arm's mean timed position is the same over an even number of reps, so drift over the run (clocks,
   heat) cancels in the means; with odd reps it doesn't quite. `execution` records the mode, why resident failed, and the
   full warm-up and timed schedule. Prefix caching is off, so no run reuses an earlier run's KV.
3. **Measured, not estimated.** A timed rep synchronizes the device around the run and reads the engine's own request
   and step timings (`engine.arrivals.run_requests_arrivals` with `request_timing` and `step_timing`).
4. **An arm's run is its own.** Around every run (gate, warm-up or timed), the runner calls an experimental arm's
   `begin_run(target)` before the clock starts and `end_run(target)` inside the timed window, so the work an arm finishes
   at the end of a run (the Commit's hashing backlog, a commitment's last roots) is in the run's time. `target.phase`
   (`gate`, `warmup` or `measure`), `target.point` (the sweep point's id) and `target.tokens` (the prompt and output tokens
   it serves) describe the run `begin_run` opens. `extras(target)`, read after a timed run, reports that run.

## Metrics

Each timed rep records:

- `time_per_output_token_s`: wall time over the tokens generated;
- `time_to_first_token_s`: the mean over requests, from arrival to first token;
- `tokens_per_s`: tokens generated over wall time;
- `peak_gpu_memory_bytes`: the device's peak allocation;
- `decode_time_per_token_s` (optional): host time of the steps that returned a decode-only forward's tokens, over the
  tokens it decoded, i.e. decode cost without prefill. Under async scheduling a step waits on the forward
  `output_lag_steps` iterations back, so that step's time is charged to that forward. It's dropped when the work left
  after the last step, an arm's `end_run` included, exceeds 1% of it (work the host step times didn't see; each rep's
  `host_timing` records the lag and that tail), and when the steps' decode tokens add up to more than the tokens
  generated (prompt tokens miscounted as decode);
- `inter_token_latency_p50_s` and `inter_token_latency_p99_s` (optional): the median and 99th percentile, over every
  generated token after its request's first, of the time from the step that returned the request's previous token to the
  step that returned this one: the per-token latency a streaming client sees.

An optional metric is recorded for an arm's point only when every rep and every baseline rep has it (the engine's
scheduler is reachable; the requests generate more than one token), and the table shows the row only where every arm has
it.

Every metric is reported as a mean, and as a ratio to the baseline (`stock-fp8`, else `stock`) paired by rep, each with
`n` and, from two reps, a 95% Student-t interval (`ci95`, the quantile of the largest tabulated degrees of freedom at
or below n − 1). A zero baseline rep leaves the ratio empty rather than averaging the others; the table shows `n` where
there's no interval. Absolute decode and whole-run per-token times measure different things; compare ratios.

## Results

A result is `verity-bench/vllm-overhead/v1`: `identity`, `baseline`, `sweep`, `gates`, `execution` and `points`.
`identity` holds the preset, the vLLM version (which must match the pin), torch and CUDA versions and the GPU the
engines saw, each arm with its kernel digest, engine overrides and KV pool in tokens, the git commit, reps, the mode
and, for a dispatched point, the resolved spec's digest, the point's identity and the `uv.lock` digest. `validate`
requires every gate an arm should have and recomputes the points from their reps, so a dropped, reordered or edited rep
is refused. `judge` (`verity-bench compare`) refuses two results whose identities differ anywhere not under a path named
with `--allow` (`git_commit`, `config`, `arms[*].kernel_digest`), so results from different pools, pins or kernels never
sit in one ratio, and lists the differences it allowed. `verity-bench table` merges one spec's per-point results the
same way, allowing only `config.point_identity`.

## Dispatch and resume

`verity-bench run` launches each missing point as one queued, preemptible job of the kind registered for the spec's GPU
count in `infra/cluster/kinds/benchmarks.toml` (`vllm-bench-1gpu`, `vllm-bench-8gpu`). The job runs
`benchmarks/harness/run_point.sh`, which uses the node's pinned-vLLM environment (`benchmarks/harness/env.sh`, which a
benchmark's own launcher sources too): `$WORK/venv312` from the integration's `pod_bootstrap.sh`, or `VERITY_BENCH_VENV`,
which the dispatcher forwards. `benchmarks` is on its path, so a spec names an experimental arm's module from there
(`served_commit.arm:SERVED_COMMIT`). A point's identity hashes the resolved spec,
the commit, `uv.lock`, the vLLM pin and the point. The tool declaration (`verity_bench.tool:VLLM_BENCH_POINT`) records it
as the run's key param, and `plan`, `check` and `run` skip a point whose identity already has a passing run: validation
passed, exit code 0 and a clean source tree. A requeued point first deletes its run's old `result.json`. The lookup
reads the dispatching machine's catalog, so a machine that never pulled other machines' runs sees them as missing
(BENCH-10).

## Out of scope here

Kernel-level timing, meaning whether a timed CUDA loop measures what it claims (opcodes in the loop, bank conflicts, live
sinks), is compute accounting's GPU timing kit in `benchmarks/pouw/`, not this harness.
