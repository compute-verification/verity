# verity-bench

This is the harness benchmarks share. Its main job is one measurement: what an arm, a protocol option or an
experimental install, adds to served inference over stock vLLM, on the same GPU, through the vLLM adapter's public
interfaces. A spec names a model, its arms and a sweep. The harness expands it into points, times each point with the
arms interleaved and gated to give the same tokens, and records one run per point. A table only ever puts results
side by side whose identities agree.

It also holds the kit: the few calls every benchmark's runner shares to lease a GPU, stamp the host, pair a ratio,
write a result and print a core benchmark's number.

## Try it

~~~sh
uv run verity-bench plan llama31-8b-fp8-1gpu-commit-served --local-only
~~~

This expands a spec from [`specs/`](specs/README.md) and prints, as JSON, each point with its identity and the run that
passed it, if this machine's store has one. It needs no GPU. `verity-bench run SPEC` queues a run for each point
without one, for any free GPU.

## Commands

~~~text
verity-bench plan SPEC       # expand a spec; each point's identity and its passing run, if any
verity-bench run SPEC        # one recorded run per point without a passing run, queued for any free GPU
                             # (--direct NODE: on NODE itself, through its GPU slots, instead; --again POINT: that
                             # point too, though a passing run stands for it; plan takes --again to show it first)
verity-bench check SPEC      # exit 0 only when every point has a passing run, here or on the store's remote
verity-bench compare A B     # accept two results only when their identities agree (--allow PATH to permit one)
verity-bench table R1 R2...  # one table from one spec's per-point results, only when their identities agree
verity-bench served-table    # every protocol's served overhead from the store: runs of record, else newest passing (--json)
verity-bench breakdown [RESULT...]  # where the bandwidth audit's time goes, by span: the store's newest passing runs, or these files
verity-bench vllm --preset SPEC --arms A,B --out DIR   # the same measurement in this process, no point's run: under a
                                                        # research run its numbers go into the run's result (the kit's writer)
verity-bench overhead points|run|table  # the front door for γ at a card's operating points (`verity bench overhead`,
                                        # benchmarks/overhead/README.md): the PoUW harness's kernels, not served vLLM
verity-bench prove --system S --workload W  # a proof system's prover on a workload, a block type a row, from the store's
                                            # labelled records (`verity bench prove`, benchmarks/prove/README.md)
verity-bench lean-verify --lean F --proof TREE  # Flock's Lean verifier timed on one kept proof, as the record `prove`
                                                # reads (`verity bench prove lean-verify`, the same README)
verity-bench prove-layers --flock R --zk R --sampling R --recursion R  # the proof benchmark by layer, the benchmark result
                                                # for each layer read from its run's out/ or run id (`verity bench prove layers`, the same README)
~~~

## What a point and a result hold

`tests/benchmarks/harness/test_bench.py::test_the_readme_states_the_contract_the_code_keeps` holds each name below to the
code, and fails when the code has one this list lacks.

| | |
| --- | --- |
| Arms | a protocol arm (`pouw:SCHEME`, `pous:SCHEME`, or SPOT's under the vLLM adapter's key, `sampled-proofs:SCHEME`), a built-in such as `stock-fp8`, or an experimental arm a spec names by its module |
| Gates before timing | `tokens` (an arm gives its control's tokens exactly), `kv-pool` (an arm and its control get the same KV-cache pool), `arm` (the arm's own gate) |
| Gates after timing | `timed-tokens`, `engine-identity`, `arm-closing` |
| Modes | `resident`, every arm's engine on the GPU at once and the arms interleaved rep by rep; else `blocks`, an engine for each arm in turn |
| Metrics of a timed rep | `time_per_output_token_s`, `time_to_first_token_s`, `tokens_per_s`, `peak_gpu_memory_bytes`; and where every rep has them, `decode_time_per_token_s`, `inter_token_latency_p50_s`, `inter_token_latency_p99_s` |
| A result | `verity-bench/vllm-overhead/v1`: its identity, its baseline, its sweep, its gates, how it ran, and its points |
| A queued point | one preemptible job of the kind `vllm-bench-1gpu` |

One failed gate times nothing. Every metric is reported as a mean and as a ratio to the baseline paired by rep.

## The kit

`verity_bench.kit` is what benchmarks' runners share, on the standard library and the harness's settings alone, so a
runner imports it without vLLM or torch: `kit.lease` (the wait for a job's GPUs), `kit.host_stamp` (where a number was measured), `kit.paired`
(an arm's cost over stock's from matched reps), `kit.writer` (a measurement and a validation into the run's result),
`kit.report` (how `verity bench compute`, `verity bench memory` and `verity bench inference_only` print and write their
number), `kit.record_dir` and `kit.record` (a run's record by directory or run id), and `kit.ran_with` (the core
benchmarks' `ran with` line). Nothing it writes is a verdict of record.

## A steady-state spec

`plan`, `run` and `check` take a second schema, told from the first by the file's `schema`: a steady-state spec
(`verity-bench/steady-spec/v0`, `verity_bench.steady_spec`), the proof server's rate at steady state. A point is one
staged block type at K provers with C cores on one card, timed by `benchmarks/one_stage/server_rate.py`. The spec says
the `workload` whose steady-state record its rows belong to, the `card`, the `cores` (C), the timed `sessions` and the
sessions a prover runs `ahead`, the GPU `driver`'s version and the CUDA toolkit's (`cuda_toolkit`), a `setup` script,
`[env]` and `[admission]` (`max_minutes`, `mem_gb`); each `[[row]]` is a block type at one statement size: the
stager's `type` (`benchmarks/served_units/stage.py`), its statement's units `n` (its proof blocks), the `pin` (the
SHA-512 of the block's circuit), the `provers` counts (a point each), the `warm` sessions a prover and, optionally,
`period` and `staged`, a statement staged already: a directory on the node, or one of an artifact in the store
(`art:ID/DIR`). Loading is as strict as a vLLM spec's. Two rows may name one type at two `n` (a full statement and a
quarter of one, which share the `pin`); each point of such a type is `NAME-nN-kK`, and every other point `NAME-kK`.

**Timed sessions.** A point times the spec's `sessions`, and at least eight a prover, the fewest the timing script
times. Where a row's sessions file in bursts, its `period` says the
sessions a prover's filings repeat over (a burst is `period` times the provers), and the point times the next count
whose rated window is a whole number of bursts and a multiple of three of them: a window, or a third of one, that cuts
a burst reads the burst's phase and not the rate. The period is declared, not read off the run's first sessions, so
the count is fixed before the run and is part of the point; `plan` shows it (`sessions`), and the node side writes the
`period` in the row, where the `level` check reads it and cuts the window at whole bursts (`steady.burst_parts`).

**Identity.** A point's identity is a digest of what its time depends on, and nothing else: its row's `type`, `n`,
`pin`, `warm` and `period`, its prover count, the spec's `card`, `cores`, `sessions`, `ahead`, `driver`, `cuda_toolkit`,
`env` and `setup`, and the SHA-256 of every source file of the prover and of the scripts that build it (their flags)
(`verity/protocols/proofs/flock/prover`, `verity/protocols/proofs/flock/prover/pod`; no tests, no Markdown), of the timing script and of the
setup script. It names no commit, so a commit that changes none of those times no point again, and neither does an edit
to another row, to `admission`, to `staged` or to the spec's name. The driver and the toolkit are the spec's because
the node can't be asked before the run: the node side reads both on the node, refuses a node whose versions aren't the
spec's, and writes what it read in the row (`versions`). The toolkit's version is `cuda_toolkit`, in the spec and in the
row: `cuda` is the kit's name, in its host stamp, for the newest CUDA the driver serves, which is another number.

**A passed point** is a row whose `pin`, statement blocks (its `n` and `stage.blocks`), `provers` and `cores.count`
are the point's and its spec's, and that passes the
ten checks of `verity_bench.steady` against the files of the run that measured it (`steady_spec.point_why`); a row
of a run of the point itself also holds the point's timed and warm sessions, its period and the versions. `plan` and
`check` recompute that from each run's files in the store, by the rules of the checkout they run from; a verdict a
run wrote of itself is never read. A point's passing run is the newest of: a run of the point (the tool
`verity_bench.tool:STEADY_POINT` with the point's `identity`, at any commit) that ran to its end from a clean tree; or
a row of the workload's steady-state record (`prove_records`' `flock-steady`: a labelled run, or a run a labelled art
names), timed outside the harness. A labelled run is judged on its own row, or, where one labelled art holds its rows as
completed after the run, on the art's (`steady_plan.completed_rows`, `completed`): the art names that one run in its
`runs`, isn't superseded, and carries how each field it added was derived from the run's own records
(`derivation.json`); each of its rows is one row of the run, keeps every field of it and changes only the fields the
derivation names for that row. An art that changes anything else isn't taken, two standing arts that complete one run
complete it for neither, and the run then counts on its own row, with the reason. So a run from before the timing
script wrote a field counts on its completed row, and one whose own row lacks the field and that has no such art is
"cannot tell". A point without a passing run is timed again, and `plan` and `check` say why each run that could have
been it isn't (`why`), a run of the record whose rows can't be read among them. The lookup pulls the remote's labels of
runs and arts, the targets a record's labels are on, and its attempts started since the spec's file was added, and
says on stderr when it starts and what it pulled.

**Run.** `verity-bench run SPEC --direct NODE` launches each point without a passing run as one recorded run on NODE
through one of its GPU slots, which gives the run its cores, by the harness's one dispatch and its one slot run
(`jobs.dispatch`, `jobs.slot_run`); a steady-state spec has no queued form. `--again POINT` (repeatable, each point by
its id, never all of them) launches that point too though a passing run stands for it, as a labelled row timed before
the harness does whatever its closure: `plan SPEC --again POINT` shows it as `again: passed by RUN` before anything is
launched, and the new run labels and supersedes nothing, so the passing run stays the record until a run is labelled
on purpose. `--cores CPUS` (a booked slot's, `34-43`) runs the slot taker under `taskset -c CPUS`, so it takes that
slot or waits for it, and `--card INDEX` (a booked card's, with `--cores`) asks the lease for that card alone, which
the node side refuses before the statement unless it is on the NUMA node of the run's CPUs. The run waits for its slot,
`jobs.DIRECT_WAIT_S` at most or until the spec's `max_minutes` and `jobs.SLOT_GRACE_S` (for `slot.py`'s 20 s poll)
before `--end UTC` (a booked period's end), and then has `max_minutes`; its timeout is the two, so it ends before the
end whichever `slot.py` the node runs, and `run` refuses a point whose `max_minutes` don't fit before `--end` at
launch. On a node whose launcher
takes a run only through `slot.py` (`/etc/vy/runs-via-slot`), the launcher refuses a `--cores` launch. The job is
`benchmarks/harness/steady_point.sh`, which sources the spec's `setup` in the source tree and runs
`python -m verity_bench.steady_node` under the Python the setup names. The setup builds the prover and exports
`VERITY_BENCH_STEADY_BIN` (the prover) and `VERITY_BENCH_STEADY_PY` (a Python the stager, the timing script and the
harness run under), with the CUDA toolkit the prover was built with first on `PATH`.
`benchmarks/harness/setup/flock_prover.sh` is that setup for this tree's prover: it builds it with the prover's own
build script, which keeps the binary on the node by a hash of its sources, makes the Python, and exports a scratch on
the node's disk. A point whose row names its statement in the store is launched with a read-only key to it
(`research run --store-read`). The node side:

1. before any card is asked for, refuses a tree that gives the point another identity than the one it was launched with,
   an environment that isn't the spec's, a node that can't run `gpu-lease` (the kit's lease calls it by its bare name:
   the refusal names the tree's script to put on the node's path), a node whose nvcc's version isn't the spec's
   `cuda_toolkit`, a run whose CPUs aren't exactly the spec's `cores`, a booked card (`--card`) not on their NUMA node,
   and a statement (staged there, copied from `staged` on the node, or fetched from `staged` in the store) whose
   circuit isn't the row's `pin` or whose units aren't its `n`. A statement from the store is fetched into the scratch, on the node's disk: a scratch
   in its memory (`/dev/shm`, the default of `VERITY_BENCH_STEADY_SCRATCH`, which the setup then exports) is refused,
   as is a disk with less free than the statement twice, and the node's own store keeps no copy;
2. takes one lease of one card for the spec's `max_minutes`, the kit's (`kit.lease`), preemptible and with the spec's
   `mem_gb`. A slot gives a run its CPUs and no card, so the lease is asked of the cards on the NUMA node of the run's
   CPUs (gpu-lease's `--on`) where the node says which those are, of the booked card alone where the run names one,
   and of any card where it doesn't; `out/point.json`
   says which (`lease_on`). The lease declares the run's CPUs (gpu-lease's `--cores`), so gpu-lease refuses, or holds
   until they fit, a lease whose CPUs and the live leases' pass the machine's CPU quota, and the check `no_throttling`
   says whether the machine throttled them while the row ran. Inside the lease it reads the kit's host stamp (`kit.host_stamp`, the leased card's) and
   the toolkit's version from the node's nvcc, refuses a card that isn't the spec's or a driver or toolkit version
   that isn't, reads the NUMA node the kernel says the leased card is on (each row's `card_numa_node`, null where it
   says none, which `one_numa_node` holds to the row's CPUs' node), records what else is on the node before and after
   the provers run (`steady.card_file`), and times the row. It keeps one accepted timed session's proof as its verifier filed it, for the Lean verifier
   (`out/rate/TYPE-kK/kept-session/`, `steady_node.keep_session`): the newest session whose proofs are the ones it
   recorded and that exactly one verifier's log line and one row of `sessions.jsonl` say was timed, accepted and of the
   row's `pin`. `out/timed.json` names it with its label and its proofs' digests, or says why none was kept, and is
   written however the step ends: a step that raised says what (`error`), and the point is refused with it;
3. after the lease has ended, so that its usage line is among the run's files, computes the verdict from the run's own
   files and writes `out/verdict.json`, `out/point.json` and `result.json`, whose validation passes with it. A passed
   point's rate is the result's one measurement, written with it once, as the kit's writer makes it (`kit.Writer`'s `entry`)
   under the host stamp the timed step ran under, with the point's id as its `member` and the spec and the point's
   identity as its inputs.

It exits 0 for a passed point, 3 for a point that was timed and didn't pass, 4 where nothing was timed, and 143 where
TERM or INT ended it (a cancel, a timeout): it then ends the command it was waiting for first and removes its scratch,
the staged statement with it. Each is a run without a passed point, as is one that was preempted or that waited for a
slot past its bound, so the next `run` launches the point again.

## Settings

Every environment variable the harness reads is declared once in `verity_bench.settings`, a `verity_config.settings`
`Registry`, with its type, default, doc and role: `value` (it can change what a point measures), `where` (only where and
how long things run) or `debug` (refused on a point, which is always a run of record). A point writes the value settings
in force, with their SHA-512 digest, to `out/settings.json` beside its result. `verity-bench` refuses a `VERITY_BENCH_*`
variable that nothing in the tree reads, naming it. A variable the vLLM integration declares is never declared here: the
harness writes one through `verity_vllm.engine.env`, part of the vLLM adapter's public interface.
`tests/benchmarks/harness/test_settings.py` holds the harness's
other direct uses of `os.environ` to a shrink-only baseline.
