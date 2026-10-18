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
vLLM pin, engine settings), its arms, its sweep (another file in `benchmarks/specs/`, `verity-bench/vllm-sweep/v0`), reps,
eager or CUDA graphs, and admission (`max_minutes` and friends, which never enter identity). Loading is strict: an unknown
key, a wrong type or an unknown GPU id is an error before anything runs.

An experimental arm (`arm_modules`, `pkg.mod:NAME`) may need its own environment. The spec's `[env]` table (upper-case
names, string values, none of the dispatcher's, queue's or launcher's own) and its `setup`, a script in the source tree
that `run_point.sh` sources before the point's Python starts (to build the arm's libraries from the dispatched commit and
export their paths, say), are part of the resolved spec and so of every point's identity. The dispatcher forwards both
(`--env`, the setup as `VERITY_BENCH_SETUP`), and a point refuses to run under any other.

## How a point is timed

1. **Gates before timing.** Every arm runs every point greedily before anything is timed:
   - `greedy-tokens`: an arm with a control must produce the control's tokens exactly. A failure records the first
     divergent request, position and tokens.
   - `kv-pool`: an arm and its control must get the same KV-cache pool, in tokens. vLLM schedules by free KV blocks, and
     a step's batch composition reaches the numerics, so unequal pools can flip greedy tokens. Block-mode engines pin it
     with the preset's `block_kv_cache_memory_bytes`.
   - `arm`: the arm's own gate, when it provides one.

   One failed gate times nothing: the result keeps its gates and has no points.
2. **Resident interleaving, else alternating blocks.** When every arm's engine fits on the GPU at once (`resident`), each
   point is warmed once per arm (discarded), then each rep runs every arm once. The baseline alternates between the first
   and last slot (`interleave_order`). An arm that needs its own engine lifetime makes the whole run `blocks`: each arm
   gets its own engine for two alternating cycles, with a discarded warm-up per point in each. `execution` records the mode,
   why resident failed, and the full warm-up and timed schedule.
3. **Measured, not estimated.** A timed rep synchronizes the device around the run and reads the engine's own request
   and step timings (`engine.arrivals.run_requests_arrivals` with `request_timing` and `step_timing`). An experimental
   arm's `begin_run` runs before the clock starts and its `end_run` inside the timed window, so work an arm finishes at
   the end of a run (the Commit's hashing backlog) is in its time; the arm's `Target` names the run's phase (`gate`,
   `warmup` or `measure`) and point.

## Metrics

Each timed rep records:

- `time_per_output_token_s`: wall time over the tokens generated;
- `time_to_first_token_s`: the mean over requests, from arrival to first token;
- `tokens_per_s`: tokens generated over wall time;
- `peak_gpu_memory_bytes`: the device's peak allocation;
- `decode_time_per_token_s` (optional): host time of the steps that only decoded, over the tokens they decoded, i.e.
  decode cost without prefill. It's recorded for an arm's point only when every rep and every baseline rep has it (the
  engine's scheduler is reachable), and the table shows the row only where every arm has it.

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
`benchmarks/harness/run_point.sh`, which uses the node's pinned-vLLM environment: `$WORK/venv312` from the integration's
`pod_bootstrap.sh`, or `VERITY_BENCH_VENV`, which the dispatcher forwards. A point's identity hashes the resolved spec,
the commit, `uv.lock`, the vLLM pin and the point. The tool declaration (`verity_bench.tool:VLLM_BENCH_POINT`) records it
as the run's key param, and `plan`, `check` and `run` skip a point whose identity already has a passing run. The lookup
reads the dispatching machine's catalog, so a machine that never pulled other machines' runs sees them as missing
(BENCH-10).

## Out of scope here

Kernel-level timing, meaning whether a timed CUDA loop measures what it claims (opcodes in the loop, bank conflicts, live
sinks), is compute accounting's GPU timing kit in `benchmarks/pouw/`, not this harness.
