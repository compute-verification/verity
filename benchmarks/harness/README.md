# verity-bench: the shared benchmark harness

`verity-bench` measures what an arm (a protocol option or an experimental install) adds to served inference, against
stock vLLM, through the vLLM integration (`integrations/vllm`). It owns specs, runs, sweeps, results and tables. The
integration owns the engine pin, the hooks and the add-on interface (its README's "Hook interface"), and the harness uses
those like any other caller. This file is the harness's contract: how a point is timed and what a result says. In order:
the commands, specs, arms, how a point is timed, metrics, results, dispatch and resume, settings, and the two tables
`served-table` and `breakdown` render.
`benchmarks/harness/tests/test_bench.py::test_the_readme_states_the_contract_the_code_keeps` fails when the code drifts from it.

## Commands

~~~text
verity-bench plan SPEC       # expand a spec; each point's identity and its passing run, if any
verity-bench run SPEC        # one recorded run per point without a passing run, queued for any free GPU
                             # (--direct NODE: on NODE itself, through its GPU slots, instead)
verity-bench check SPEC      # exit 0 only when every point has a passing run, here or on the store's remote
verity-bench compare A B     # accept two results only when their identities agree (--allow PATH to permit one)
verity-bench table R1 R2...  # one table from one spec's per-point results, only when their identities agree
verity-bench served-table    # every protocol's served overhead from the store: runs of record, else newest passing (--json)
verity-bench breakdown [RESULT...]  # where the bandwidth audit's time goes, by span: the store's newest passing runs, or these files
verity-bench vllm --preset SPEC --arms A,B --out DIR   # the same measurement in this process, unrecorded
~~~

## Specs

A spec is `benchmarks/specs/<name>.toml` (`verity-bench/vllm-spec/v0`): the preset (model and revision, precision, GPU,
vLLM pin, engine settings), its arms and the modules of its experimental arms, its sweep (another file in
`benchmarks/specs/`, `verity-bench/vllm-sweep/v0`), reps, eager or CUDA graphs, and admission (`max_minutes` and friends,
which never enter identity). Loading is strict: an unknown key (an engine setting outside `presets.ENGINE_SETTINGS`
included), a wrong type or an unknown GPU id is an error before anything runs. A preset's model is a Hugging Face id at a pinned `revision`, or a node-local directory with its
`model_digest` (`presets.local_checkpoint`'s hash of its files but a Hub snapshot's dot-files and Markdown, which no loader
reads), which a dispatched point recomputes and refuses to run on when it differs.

A sweep is the product of its batches, prompt lengths, traffic shapes and samplings: `greedy`, or `stochastic`
(temperature 0.8, top-p 0.95). Every request carries its own seed, so a stochastic point's tokens are a function of the
engine's numerics like a greedy one's.

The engine settings `serving_defaults = true` build every engine as vLLM serves (torch.compile, its CUDA-graph mode and
capture sizes, no batch invariance: the execution label `compiled` of `verity_vllm.engine.served`), with one Inductor
cache directory for all arms' processes, so they run the kernel configurations the first one autotuned. Without it a
graph engine replays decode-only CUDA graphs at `capture_sizes`, and `verity_vllm.LLM` builds every engine batch
invariant (`bi-eager`, `bi-compiled`).

## Arms

A protocol arm (`pouw:SCHEME`, `pous:SCHEME`, `sampled-proofs:SCHEME`) runs the preset's precision with stock FP8 as its
control, unless the spec's `[arm_settings."ARM"]` table says otherwise: its `precision` (PoUW's Pearl-C loads the BF16
checkpoint and quantizes it itself), its `control` (another arm of the spec, or `"none"`), and its `protocols_config`,
which `verity_vllm.LLM` takes as it is, `{protocol: {...}}`, except that a string `"$NAME"` is the point's environment
variable NAME (the cubin the spec's setup builds, say). The harness gives an arm with PoUW an in-process challenger as
its `coins` (`pouw.challenged`). The adapter ends one Pearl-C pass per model forward itself; the passes'
counts (`passes`, `units`, `tiles`, screened, failing and unsure rows) are each timed rep's `pearl_c` spans, and the
experimental arm's call check on the installed run is the arm's own gate. Pearl-C's y carries the epoch's live salt, so
no stock arm is its token control (`llama31-8b-1gpu-pouw-protocol-served` has `"none"`). The table is part of the resolved spec
and so of every point's identity, and the arm's cubin digest is its `kernel_digest`.

The precision `int8_exact` is the eager exact-int8 engine of an int8 pin (every linear and attention layer on the plain
exact-int8 arm, read back from the layers), and `stock-int8` is that engine without a protocol. PoUW's H1 + fx1
(`pouw:eps1-sm120-h1fx1`) runs on it with `stock-int8` as its token control, since its arm serves its own y. The harness
builds it through `verity_vllm.LLM`'s `protocols` with only H1's public parameters, which fills the engine's model runner
and in-process coins after the build, and gives each run its own window with `reinstall`: its `steps` (S) the run's step
bound (`arms.window_steps`), its `rows` (T) the spec's, which must be the engine's `max_num_batched_tokens`, and its
coins a fresh in-process salt coin and S round coins, so it credits nothing and measures cost only. The window opens
before the clock starts and closes after it stops; each timed rep's `extras.h1_window` says what it served, and the
arm's own gate is every window static, every served call passing the credit gate and every request entered
(`llama31-8b-1gpu-pouw-h1`).

A served variant `BUILTIN@eager` or `BUILTIN@compiled` (`stock-fp8@compiled`) is a built-in arm's engine at vLLM's
serving defaults in that execution, whatever the spec's `eager` and the preset's `serving_defaults` say for the other
arms, so one spec times one engine in two executions (`llama31-8b-fp8-1gpu-compiled-full`). Its control is the built-in
unless its `[arm_settings."ARM"]` names another arm or `"none"`, and its `execution` is in the arm's identity. An
experimental arm may name its own `execution` the same way: `commit@compiled` (`verity_bench.vllm.commit_arm`) is the
Commit on vLLM's compiled serving, token-controlled by `stock-fp8@compiled` (`llama31-8b-fp8-1gpu-commit-served`).
`llama31-8b-fp8-1gpu-bandwidth-ablation` times where PoUS's bandwidth audit's time goes: the two credited arms beside
the bandwidth audit's spans arms (`benchmarks/vllm_encoded/README.md`) and the spans arms with the top-up off
(`verity_bench.vllm.bandwidth_ablation`, `BandwidthArm(..., spans=True, topup=False)`), each token-controlled by
`bandwidth-plain`; `verity-bench breakdown` renders it.

A served variant's `[arm_settings."ARM"]` may also name its own checkpoint, `model` at `revision`, when its precision
can't load the preset's: in `llama31-8b-1gpu-pouw-h1-served`, H1's served ratio, `stock-fp8@compiled` loads the int8
pin's BF16 base.

An experimental arm (`arm_modules`, `pkg.mod:NAME`) may need its own environment. The spec's `[env]` table (upper-case
names, string values, none of the dispatcher's, queue's or launcher's own) and its `setup`, a script in the source tree
that `run_point.sh` sources before the point's Python starts (to build the arm's libraries from the dispatched commit and
export their paths, say), are part of the resolved spec and so of every point's identity. The dispatcher forwards both
(`--env`, the setup as `VERITY_BENCH_SETUP`), and a point refuses to run under any other.

## How a point is timed

1. **Gates before timing.** Every arm runs every point, with the point's sampling, before anything is timed:
   - `tokens`: an arm with a control must produce the control's tokens exactly. A failure records the first divergent
     request, position and tokens.
   - `kv-pool`: an arm and its control must get the same KV-cache pool, in tokens. vLLM schedules by free KV pages, and
     a step's batch composition reaches the numerics, so unequal pools can flip tokens. The preset pins it with
     `resident_kv_cache_memory_bytes` and `block_kv_cache_memory_bytes`. An engine that doesn't report its pool fails it.
   - `arm`: the arm's own gate, when it provides one.

   An arm can't be its own control. An arm without a control (the stock references, an experimental arm such as
   Pearl-C's that computes different tokens, or a protocol arm whose `control` is `"none"`) has only its own gate, and
   the table names it. Tokens are compared, not
   logits. One failed gate times nothing: the result keeps its gates and has no points. Three gates run after timing,
   and a failure there keeps the gates and drops the points:
   - `timed-tokens`: each timed rep of an arm with a control produces the control's tokens in that rep, so an arm can't
     compute something else only when timed. In `resident`, each timed rep of an arm without a control produces the
     tokens its own gate saw from the same engine (`"against": "gate"`).
   - `engine-identity`: in `blocks`, each timed engine reports the gated engine's KV pool, kernel digest, versions,
     GPU and overrides.
   - `arm-closing`: after an engine's warm-up and timed runs, before it closes, the runner reads the arm's own gate again
     (in `blocks`, one per engine lifetime, with its `cycle`), so an arm that judges every run it saw judges those too.
2. **Resident interleaving, else `blocks`.** When every arm's engine fits on the GPU at once (`resident`), each
   point is warmed per arm (discarded), then each rep runs every arm once. The baseline alternates between the first
   and last slot (`interleave_order`). An arm that needs its own engine lifetime makes the whole run `blocks`: each arm
   gets its own engine for two cycles, the second in reverse arm order, with discarded warm-ups per point in each. The
   spec's `warmups` (default 1) is how many discarded runs each engine makes of a point before timing it; with
   `max_warmups` above it, the engine keeps warming, up to that many runs, until two consecutive runs' wall times agree
   within 5% (`WARMUP_AGREEMENT`), and an engine that never settles is marked `settled: false` in the schedule. Both
   are part of the identity. The schedule keeps each warm-up's wall time, time to first token and host counters.
   Either way each arm's mean timed position is the same over an even number of reps, so drift over the run (clocks,
   heat) cancels in the means; with odd reps it doesn't quite. `execution` records the mode, why resident failed, and the
   full warm-up and timed schedule. Prefix caching is off, so no run reuses an earlier run's KV.
3. **Measured, not estimated.** A timed rep synchronizes the device around the run and reads the engine's own request
   and step timings (`engine.arrivals.run_requests_arrivals` with `request_timing` and `step_timing`).
4. **An arm's run is its own.** Around every run (gate, warm-up or timed), the runner calls an experimental arm's
   `begin_run(target)` before the clock starts and `end_run(target)` inside the timed run, so the work an arm finishes
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

Each timed rep also keeps where its time went: `trace` (every `step()` iteration's host time and its prefill and decode
tokens, every request's arrival and first output from the run's start, and the seconds since its engine became ready) and
`host` (the process's CPU time, the engine thread's, context switches, page faults and garbage-collection time over the
run, the 1-minute load average when it began, and under `device` the caching allocator's device allocations, frees,
out-of-memory retries and all-stream synchronizations).

Every metric is reported as a mean, and as a ratio to the baseline (`stock-fp8`, else `stock`) paired by rep, each with
`n` and, from two reps, a 95% Student-t interval (`ci95`, the quantile of the largest tabulated degrees of freedom at
or below n − 1). A zero baseline rep leaves the ratio empty rather than averaging the others; the table shows `n` where
there's no interval. Absolute decode and whole-run per-token times measure different things; compare ratios.

An experimental arm's `spans(target)`, read before and after a timed run, is that rep's `spans` (each numeric leaf's
change), and its `extras` is the rep's `extras`; a point keeps their summaries per arm (`span_totals`, `extras`). They are
reported only: no gate reads them, and no ratio is taken of them.

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

`verity-bench run` queues each missing point, as below. `verity-bench run SPEC --direct NODE` launches each missing point
on NODE itself instead, for a node whose launcher takes a run only through a slot. Each point is a recorded run on NODE with the queued point's tool, identity and environment, so `plan` and `check`
find it the same way, and its command takes one of the node's GPU slots (`/workspace/research/locks/gpu/slots`, one per GPU)
through `tools/verity/check/slot.py`, then its GPUs from the node's `gpu-lease`, preemptible. The spec's admission
`max_minutes` bounds the run, its slot and its lease, so `--direct` refuses a spec without it, and it refuses any node but the
GPU nodes in `verity_bench.jobs.DIRECT_NODES` (multi-GPU nodes, and single-GPU VMs whose one GPU kernels' leases
share). A preempted direct point is a failed run, which the
next `run --direct` launches again. Use the queue otherwise: it places each point on whichever node has a free GPU.

`verity-bench run` launches each missing point as one queued, preemptible job of the kind registered for the spec's GPU
count in `infra/cluster/kinds/benchmarks.toml` (`vllm-bench-1gpu`, the only one). The job runs
`benchmarks/harness/run_point.sh`, which uses the node's pinned-vLLM environment (`benchmarks/harness/env.sh`, which a
benchmark's own launcher sources too): `VERITY_BENCH_VENV`, which the dispatcher forwards, else the first that exists of
`$WORK/venv312` (the integration's `pod_bootstrap.sh`), `$WORK/jobs/venv312` (a jobs node's),
`$WORK/pouw/mvp-e2e/venv312` and `$WORK/cp/venv312` (the DC VMs'). `benchmarks` is on its path, so a spec names an experimental arm's module from there
(`served_commit.prof_arm:SERVED_COMMIT_ROWS`). A point's identity hashes the resolved spec, the commit, `uv.lock`, the
vLLM pin and the point. The tool declaration (`verity_bench.tool:VLLM_BENCH_POINT`) records it as the run's key param,
and `plan`, `check` and `run` skip a point whose identity already has a passing run: validation passed, exit code 0 and a
clean source tree. A requeued point first deletes its run's old `result.json`. The lookup asks the machine's catalog of
the evidence store for an attempt of the tool `verity_bench.tool:VLLM_BENCH_POINT` with the point's `identity` param, at
the commit's source tree (git must know the commit), and when a point isn't there, the store's remote: once per commit it fetches and indexes the
attempts this store lacks among those started since a day before the commit's committer date (a point's runs can't start
before its commit exists, and a run id begins with its UTC start), so a fresh machine finds the points other machines
ran. Without a usable remote it refuses rather than report them missing; `--local-only` reads the catalog alone.

## Settings

Every environment variable the harness reads is declared once in `verity_bench.settings`, with its type, default, doc and
role: `value` (it can change what a point measures), `where` (only where and how long things run) or `debug` (refused on a
point, which is always a run of record). A point writes the value settings in force, with their SHA-512 digest, to
`out/settings.json` beside its result. `verity-bench` refuses a `VERITY_BENCH_*` variable that nothing in the tree reads,
naming it. A variable the vLLM integration declares is never declared here: the harness writes one through
`verity_vllm.engine.env`. `benchmarks/harness/tests/test_settings.py` holds the harness's other direct uses of `os.environ` to a shrink-only
baseline.

## The served table

`verity-bench served-table` (`verity_bench.served_table`) renders each protocol's served overhead on Llama-3.1-8B from
the store, never from numbers or ids in code. A record is named by a store label: a run labelled `run_of_record=NAME`
(`research data label RUN run_of_record NAME --by benchmarks`) is one of record NAME's runs, and an art so labelled names
its runs in its `runs` meta (a list, or a map from point to run). A record's runs are in the order they were labelled,
and a target's latest label names its record. The command first pulls the remote's labels. It has four sections.

- **PoUW's run of record** is its live rows: the entries (device, run, PoUW arm, control, and `out` where the run's
  records aren't under `out/`) of the `runs` meta of an art of record `LIVE`. Each row is read from the
  `summary.json` and `stop.json` in the run's record: the arm's median decode step at batch 32 over its control's
  (the int8-exact engine without PoUW) in the same run. That ratio is per step, so it sits in its own section, apart
  from the per-token ratios over stock FP8. A row stands only when every gate the run records passed: the device gate,
  the tokens at the gate and at every timed rep, and a live window at the run's end on off-node coins. A run that fails
  one is passed over and listed with why.
- **The served rows** (`PROTOCOLS`), each on its device (the GPU its run's engines saw, the result's `identity.gpu`: the
  RTX PRO 6000's, and the Commit's on an H200 and a B200 too), each come from a protocol's run of record
  (`Protocol.record`) or, without one, from the newest passing run. A run of record is read by id, from the remote where
  this machine lacks it, and every point of each run is a row. Without a record, each point of each spec takes the newest passing
  `vllm_bench_point` attempt whose `spec` and `point` params name it (the resume lookup's rule, without its one-commit
  identity), so a newer passing run supersedes an older one; the command first pulls the remote's attempts started
  since the oldest such spec was added, the same pull as `check`'s. A result labelled `superseded_by`, an invalid
  result, a result without the row's arms, or a record run that isn't a passing point run is passed over and listed.
  Ratios are paired by rep from the result's own reps, each metric over `stock-fp8@compiled` and, where the run times
  an eager stock, over that.
- **The conversion factors** (`stock-fp8@compiled` over batch-invariant eager `stock-fp8`, `FACTORS`) are rendered the
  same way.
- **Where the bandwidth audit's time goes** (`BREAKDOWNS`: a protocol and its record) is the bandwidth
  breakdown's rows (below) at each point of the record's runs, in ms an output token and not a ratio, beside the
  protocol's run of record.

The caveat column is the newest `label` and `note` on the result's artifact, its run and the record art that named it
(and a live run's `row` label).

## The bandwidth breakdown

`verity-bench breakdown` (`verity_bench.breakdown`) reads, for each point of `llama31-8b-fp8-1gpu-bandwidth-ablation`,
the newest passing run that times all six arms (the served rows' rule and pull), or the result files it's given. Each
row is ms an output token, paired by rep: a part's ms a step (`extras.span_per_step.<kind>.<part>_ms`) times its kind's
steps over the run's generated tokens. `bandwidth-audit-spans` minus `bandwidth-plain-spans` is split into rows that sum
to it: the top-up's contention, `(audit-spans − audit-spans-no-topup) − (plain-spans − plain-spans-no-topup)` in time
per output token, since the top-up runs beside the step and is in no span; each decode part and the prefill steps from
the top-up-off spans arms; and the time outside every span. The credited lines' difference, spans off, is beside it as
the audit's cost (the marks are in the spans arms' time). Then each spans arm's step by part, decode and prefill apart.

## Out of scope here

Kernel-level timing, meaning whether a timed CUDA loop measures what it claims (opcodes in the loop, bank conflicts, live
sinks), is the GPU timing kit in `benchmarks/pouw/`, not this harness.
