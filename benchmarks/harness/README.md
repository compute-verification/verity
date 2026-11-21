# verity-bench

The harness establishes one measurement per recorded run: a value with its unit and basis (`measured`, `modeled` or
`partial`), taken once on a stated card. It builds that from a spec or a runner a benchmark brings, the machine's record
and the GPU lease, and reads the vLLM adapter only through its public interfaces.

Not yet: a served overhead is not `verity run` of an application against stock vLLM, and the launching layer builds the
lease's command line itself.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the kit | `measurement()` and its writer, the host stamp, card and machine records, the lease's line, `report` | `verity_bench/kit.py`, `kit_report.py` |
| launching | a point as one `research run`, with its identity and passing-run lookup | `verity_bench/jobs.py`, `tool.py`, `run_point.sh` |
| the steady path | the proof server's rate at steady state, on one card | `verity_bench/steady*.py`, `steady_point.sh` |
| the overhead door | γ at a card's operating points | `verity_bench/overhead*.py`, `overhead_node.sh` |
| the served arms | engines timed against stock, interleaved and gated | `verity_bench/vllm/arms.py`, `commit_arm.py`, `runner.py` |
| the served table | each protocol's served overhead, from the store | `verity_bench/served_*.py`, `served_overhead.py` |
| readers of stored runs | proving, exfiltration and the Lean verifier's timing | `verity_bench/prove*.py`, `exfiltration*.py`, `lean_verify.py` |
| specs | one measurement written down: model, cards, arms, sweep | `specs/` |
| the registry | each metric's row, its command, unit and basis | `metrics.toml` |
| profiles | the serving profiles a benchmark serves | `verity_bench/profiles.py`, `../production.toml` |

## What varies

| path | h100 | h200 | rtx-pro-6000 | b200 | arms it times |
| --- | --- | --- | --- | --- | --- |
| served arms (`specs/`) | served | served | served | never written | stock, commit, pearl-plus, pouw-pearl-c-whole, bandwidth audit |
| steady proof-server path | never written | served | never written | never written | 1, 2 or 4 provers |
| overhead door | exists | exists | exists | never written | PoUW schemes against stock |
| served table (reads the store) | — | — | — | — | reads whatever card a stored result names |
| `prove` (reads the store) | — | exists | — | — | a proof system on a workload, priced at h200 by default |

A new benchmark supplies: a metric row with its command, unit and basis, and a spec or a runner that calls `measurement()`.
Fixed for every benchmark: one record per run; the card is read from the machine's record, never typed; a trial
(`VERITY_BENCH_TRIAL`) is never of record; a timed point passes its gates (`tokens`, `kv-pool`, `arm`, then `timed-tokens`,
`engine-identity`, `arm-closing`, `host-quiet`), and one failed gate times nothing.

## Interfaces

- From a benchmark: a spec (`verity-bench/vllm-spec/v0`, `vllm-sweep/v0` or `steady-spec/v0`) or a runner, and its `metrics.toml` row.
- From the machine: its record's `[properties]`, the card files in `calibration/hardware/cards`, `gpu-lease`.
- From the store: the runs and labels `plan`, `check`, `served-table`, `prove` and `exfiltration` read.
- It gives: a measurement in the run's `result.json`, and a point on the dashboard through its registry row.

## Try it

~~~sh
uv run verity-bench plan llama31-8b-fp8-1gpu-commit-served --local-only
~~~

Prints JSON and exits 0, with no GPU: the spec's `points` (`b8-p512-mixed` and `b32-p512-mixed`), each with its identity and
`passing_run` (`null` when this machine's store has none), and the `resolved_spec`. `verity-bench --help` lists the rest.
