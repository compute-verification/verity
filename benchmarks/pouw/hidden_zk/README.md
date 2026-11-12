# hidden_zk

Pearl-C4's hidden-check units proved under Flock `--zk` on CPUs: each unit staged for Flock, proved by Rust `prove --zk`
and checked by the Lean verifier, beside controls that must be refused or rejected; what staging and proving a unit
costs; and the ZK server's send check, its cost per proved witness bit.

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
| `scope` | `scope.sh CPUS MEM`: `job`, or the job script `HZ_JOB` names (`served_zk/job.sh`), in a systemd scope of its own on CPUS under a memory limit of MEM | what the job it runs writes |

`job` keeps its builds and staged configs under `HZ_CACHE`, keyed by their inputs, so a later run with the same key
proves the kept files. Jobs in `served_zk/`, `h1_prove/` and `h1_lanes/` source `job.sh` for those builds.

## The Python modules

`stage.py ROLE K N OUT` stages one size class (`row_a`, `row_b`, `tile`) through `verity_flock.class_statement`;
`fast_stage.py` and `compact_stage.py` make that staging take less time and memory while writing the same bytes;
`summarize.py OUT` folds what `job` leaves into its summaries; `timed.py OUT_FILE CMD...` records a command's wall
seconds and peak memory. `core3_scratch.py` and `hash_scratch.py` are development bindings, Flock pieces for primitives
Flock has none for yet, and every result staged through them is labelled so. `h1_prove/stage.py` and
`served_zk/stage_tile.py` import this directory's modules, and `compact_stage.py`, `fast_stage.py`, `core3_scratch.py`
and `hash_scratch.py` list in `__all__` the names other files read, the tests' included. Their tests are
`test_hidden_zk_stage.py` and `test_hidden_zk_summarize.py` under `benchmarks/pouw/tests/`.
