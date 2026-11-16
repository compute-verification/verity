# Pearl-C4's hidden checks under zero knowledge

This directory proves Pearl-C4's hidden-check proof blocks under Flock `--zk` on CPUs: each proof block is staged for
Flock, proved by Rust `prove --zk` and checked by the Lean verifier, beside controls that must be refused or rejected.
It also measures what staging and proving a proof block costs, and what the ZK server's send check costs per proved
witness bit.

## Try it

Check that the summaries hold every Lean check to the verdict it must give:

~~~sh
uv run pytest tests/benchmarks/pouw/test_hidden_zk_summarize.py -q
~~~

It prints `4 passed`.

## Runner

**Hardware:** a node's CPUs, no GPU. `scope` exists for a node whose sessions carry a CPU mask narrower than the CPUs a
job needs (the RTX PRO 6000 node's).

## Runner

`run.sh JOB [ARGS...]` starts one job's script with the arguments and environment as given; run it from the source tree,
as a `research run`'s command. Each script's header has its knobs (`HZ_*`, `SC_*`) and what each step writes.

~~~text
research run --on MACHINE --project verity --source . --cwd source --declared-output 'out/*' \
  --env HZ_CONFIGS="row_a:256 tile:256" -- bash benchmarks/pouw/hidden_zk/run.sh job
~~~

| job | what it runs | writes under `$OUT` |
|---|---|---|
| `job` | the builds (Flock's proving and selftest builds, the Lean verifier, a venv), then per config `ROLE:K` in `HZ_CONFIGS`: `stage.py`'s N instances, one loopback `--zk` session sampling them all, the selftest build's controls and the Lean verifier on each; then `summarize.py` | `<role>-k<K>/summary.json` and `summary.json`, each check beside the verdict it must give |
| `registered` | `job`'s proving build on N instances of the registered sm80 bf16 GEMM-coordinate template per K in `HZ_K`, staged and proved in one loopback `--zk` session | `k<K>/` (the generate, stage, serve and prove times, `prove.out`, `stage.out`, `registered.json`), `registered.json` |
| `send_check` | per `SC_PLAN` entry `ROLE:K:N`, `SC_REPS` rounds of a loopback `--zk` session with the ZK server's send check off and on, and on through a second build when the header's variable names one, the arms' order rotated each round | `send_check.tsv` (a row a session), `send_check.json` (each arm's medians, and the seconds a proved bit) |
| `scope` | `scope.sh CPUS MEM`: `job`, or the job script `HZ_JOB` names (`verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk/job.sh`), in a systemd scope of its own on CPUS under a memory limit of MEM | what the job it runs writes |

`job` keeps its builds and staged configs under `HZ_CACHE`, keyed by their inputs, so a later run with the same key
proves the kept files. Jobs in `../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk/`, `h1_prove/` and `h1_lanes/` source `job.sh` for those builds.

[`DESIGN.md`](DESIGN.md) says what each Python module is and who reads it.
