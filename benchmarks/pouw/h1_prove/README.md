# h1_prove

H1 tile units through SPOT: a recorded or served int8 Llama-3.1-8B window's units sampled by the live coin,
proved by Flock `--zk` under the ZK server and checked by the Lean verifier against the verifier's own public files,
beside controls each refused or rejected by the check it breaks; and the units a served window's own sampling coin
selects, read from that window's run.

**Hardware:** a node's CPUs, no GPU, with the memory the job's header sizes: `job` at K = 4096 wants hundreds of GB
(`H1_MEM_GB`), `live` 128 GB, `served` 16 GB.

## Runner

`run.sh JOB [ARGS...]` starts one job's script with the arguments and environment as given; run it from the source tree,
as a `research run`'s command (or, for PoUW's piece of a served window, `benchmarks/one_stage/served_verdicts.sh`'s).
Each script's header has its full command, its knobs (`H1_*`) and what it writes.

~~~text
research run --on MACHINE --project verity --source . --cwd source --mem-gb G --declared-output 'out/*' \
  --env H1_MEM_GB=G -- bash benchmarks/pouw/h1_prove/run.sh job
~~~

| job | what it runs | writes |
|---|---|---|
| `job` | one recorded window: `layer.py` records two windows' int8 codes from the checkpoint, `run.py prepare` commits, serves, samples, opens and stages, one loopback `--zk` session a statement, the Lean checks of the plan and the selftest build's controls, then `run.py summarize` | `$OUT/summary.json`, and the sessions', Lean checks' and controls' logs under `$OUT` |
| `live` | a served window's live units, from its dump (`H1_LIVE`): `live.py prepare`, a session a round, the Lean check of each, `live.py summarize` and `evidence` | `out/<window>/live-summary.json`, `out/<window>/live-evidence.json` |
| `served` | `served.py` over a served window's run (`H1_SERVED`), against the challenger's pinned key, the pins log and `main`'s commits, then three negatives each refused | the selected units, `out/program/window-program.json.gz`, a log a negative |

One more job has no row, because its name is a term the Glossary retires; `run.sh` with no argument lists every job. It
is PoUW's piece of a served window with its sampled proof blocks proved: each sampled call's units staged on the
window's registered values, a session a statement, the statements file checked as credit does, then
`served_verdicts.py pouw`. It writes `PIECE_DIR/` (the statements file, a directory of logs a session).

`run.py` is a step of `job` (its `prepare`, `summarize` and the Lean plan), and `run.sh` never starts it.

## The Python modules

`h1unit.py` is the tile unit as a Flock statement and the commitments it is opened against, which the committer, prover
and verifier share; `window.py` is the window's committer and prover, which `run.py` drives and the verifier never
imports; `stage.py K WORK` builds the unit's circuit once and stages each round on it; `layer.py` records a decoder
layer's int8 codes and scales from a checkpoint on the CPU. `live.py` is also read by the vLLM adapter's tests and
`benchmarks/one_stage/`, and `h1_lanes/served.py` takes the units `served.py` samples. `h1unit.py`, `window.py`,
`run.py`, `stage.py`, `layer.py` and `live.py` list in `__all__` the names other files read through them, the tests'
included, except underscore names. Their tests are the `test_h1_prove*.py` files under `benchmarks/pouw/tests/`.
