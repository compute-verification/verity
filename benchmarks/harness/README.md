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
verity-bench compare A B     # accept two results only when their identities agree (--allow FIELD to permit one)
verity-bench vllm --preset SPEC --arms A,B --out DIR   # the same measurement in this process, unrecorded
~~~

A spec is `benchmarks/specs/<name>.toml` (`verity-bench/vllm-spec/v0`): the preset (model and revision, precision, GPU,
vLLM pin, engine settings), its arms and the modules of its experimental arms, its sweep (another file in
`benchmarks/specs/`, `verity-bench/vllm-sweep/v0`), reps, eager or CUDA graphs, and admission (`max_minutes` and friends,
which never enter identity). Loading is strict: an unknown key, a wrong type or an unknown GPU id is an error before
anything runs.

A sweep is the product of its batches, prompt lengths, traffic shapes and samplings: `greedy`, or `stochastic`
(temperature 0.8, top-p 0.95). Every request carries its own seed, so a stochastic point's tokens are a function of the
engine's numerics like a greedy one's.

The engine settings `serving_defaults = true` build every engine as vLLM serves (torch.compile, its CUDA-graph mode and
capture sizes, no batch invariance: the execution label `compiled` of `verity_vllm.engine.served`), with one Inductor
cache directory for all arms' processes, so they run the kernel configurations the first one autotuned. Without it a
graph engine replays decode-only CUDA graphs at `capture_sizes`.

## How a point is timed

1. **Gates before timing.** Every arm runs every point, with the point's sampling, before anything is timed:
   - `tokens`: an arm with a control must produce the control's tokens exactly. A failure records the first divergent
     request, position and tokens.
   - `kv-pool`: an arm and its control must get the same KV-cache pool, in tokens. vLLM schedules by free KV blocks, and
     a step's batch composition reaches the numerics, so unequal pools can flip tokens. The preset pins it with
     `resident_kv_cache_memory_bytes` and `block_kv_cache_memory_bytes`.
   - `arm`: the arm's own gate, when it provides one.

   One failed gate times nothing: the result keeps its gates and has no points.
2. **Resident interleaving, else alternating blocks.** When every arm's engine fits on the GPU at once (`resident`), each
   point is warmed once per arm (discarded), then each rep runs every arm once. The baseline alternates between the first
   and last slot (`interleave_order`). An arm that needs its own engine lifetime makes the whole run `blocks`: each arm
   gets its own engine for two alternating cycles, with a discarded warm-up per point in each. `execution` records the mode,
   why resident failed, and the full warm-up and timed schedule.
3. **Measured, not estimated.** A timed rep synchronizes the device around the run and reads the engine's own request
   and step timings (`engine.arrivals.run_requests_arrivals` with `request_timing` and `step_timing`).
4. **An arm's run is its own.** Around every run (gate, warm-up or timed), the runner calls an experimental arm's
   `begin_run(target)` before the clock starts and `end_run(target)` before it stops, so the work an arm finishes at the end
   of a run (a commitment's last roots) is in the run's time. `target.phase` (`gate`, `warmup` or `measure`) and
   `target.point` (the sweep point's id) name the run `begin_run` opens. `extras(target)`, read after a timed run, reports
   that run.

## Metrics

Each timed rep records:

- `time_per_output_token_s`: wall time over the tokens generated;
- `time_to_first_token_s`: the mean over requests, from arrival to first token;
- `tokens_per_s`: tokens generated over wall time;
- `peak_gpu_memory_bytes`: the device's peak allocation;
- `decode_time_per_token_s` (optional): host time of the steps that only decoded, over the tokens they decoded, i.e.
  decode cost without prefill;
- `inter_token_latency_p50_s` and `inter_token_latency_p99_s` (optional): the median and 99th percentile, over every
  generated token after its request's first, of the time from the step that returned the request's previous token to the
  step that returned this one: the per-token latency a streaming client sees.

An optional metric is recorded for an arm's point only when every rep and every baseline rep has it (the engine's
scheduler is reachable; the requests generate more than one token), and the table shows the row only where every arm has
it.

Every metric is reported as a mean, and as a ratio to the baseline (`stock-fp8`, else `stock`) paired by rep, each with
`n` and, from two reps, a 95% Student-t interval (`ci95`). Absolute decode and whole-run per-token times measure
different things; compare ratios.

## Results

A result is `verity-bench/vllm-overhead/v1`: `identity`, `baseline`, `sweep`, `gates`, `execution` and `points`.
`identity` holds the preset, the vLLM version (which must match the pin), each arm with its kernel digest, engine
overrides and KV pool in tokens, the git commit and, for a dispatched point, the resolved spec's digest, the point's
identity and the `uv.lock` digest. `judge` (`verity-bench compare`) refuses two results whose identities differ anywhere
not named with `--allow`, so results from different pools, pins or kernels never sit in one ratio.

## Dispatch and resume

`verity-bench run` launches each missing point as one queued, preemptible job of the kind registered for the spec's GPU
count in `infra/cluster/kinds/benchmarks.toml` (`vllm-bench-1gpu`, `vllm-bench-8gpu`). The job runs
`benchmarks/harness/run_point.sh`, which uses the node's pinned-vLLM environment (`benchmarks/harness/env.sh`, which a
benchmark's own launcher sources too): `$WORK/venv312` from the integration's `pod_bootstrap.sh`, or `VERITY_BENCH_VENV`,
which the dispatcher forwards. `benchmarks` is on its path, so a spec names an experimental arm's module from there
(`served_commit.arm:SERVED_COMMIT`). A point's identity hashes the resolved spec,
the commit, `uv.lock`, the vLLM pin and the point. The tool declaration (`verity_bench.tool:VLLM_BENCH_POINT`) records it
as the run's key param, and `plan`, `check` and `run` skip a point whose identity already has a passing run. The lookup
reads the dispatching machine's catalog, so a machine that never pulled other machines' runs sees them as missing
(BENCH-10).

## Out of scope here

Kernel-level timing, meaning whether a timed CUDA loop measures what it claims (opcodes in the loop, bank conflicts, live
sinks), is compute accounting's GPU timing kit in `benchmarks/pouw/`, not this harness.
