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

[`DESIGN.md`](DESIGN.md) holds the rest: how a point is timed, what a result holds and what refuses it, dispatch and
resume, the steady-state spec, settings, the served table and the bandwidth breakdown, and each kit call's arguments.
