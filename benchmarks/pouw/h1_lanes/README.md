# h1_lanes

One H1 leaf proved as its 32 lanes and its combine, one Flock `--zk` statement each, and checked by the Lean verifier
against the verifier's own files: a leaf at a small K made here, J10-R's leaf, or the unit a served window's own sample
selected, beside negatives each refused.

**Hardware:** a node's CPUs, no GPU, with the memory the job's header sizes (`H1L_MEM_GB`, the job's `--mem-gb`).

## Runner

`run.sh JOB [ARGS...]` starts one job's script with the arguments and environment as given; run it from the source tree,
as a `research run`'s command. Each script's header has its full command, its knobs (`H1L_*`, `H1S_*`) and what each
step writes.

~~~text
research run --on MACHINE --pool proofs --project verity --source . --cwd source --mem-gb G \
  --declared-output 'out/*' --env H1L_MEM_GB=G -- bash benchmarks/pouw/h1_lanes/run.sh job
~~~

| job | what it runs | writes |
|---|---|---|
| `job` | `run.py prepare` (each statement's Program, the leaf's registered values, every statement staged), one loopback `--zk` session a statement, the plan's Lean checks and PoUW's own-window check of the registered reads, then `run.py summarize`; `H1L_J10R=1` runs `j10r.py`'s J10-R leaf in its place | `out/summary.json`, and the sessions' and Lean checks' logs under `$OUT` |
| `served` | the unit a served window's sample selected (`h1_prove/served.py`), its openings extracted into the job's private scratch, `served.py prepare`, a session a lane under the ZK server, the Lean checks, `served.py summarize` | `out/summary.json`, and the sessions' and Lean checks' logs under `$OUT` |

`run.py` is a step of `job` (its `prepare`, `summarize` and `cost`), and `run.sh` never starts it.

## The Python modules

`run.py` holds the leaf's protocol and its negatives, and the verifier's `verity/protocols/pouw/verifier/leaf_lanes.py`
names it by its path, while the Lean window's tests and `benchmarks/unit_lowering/` load it, reading the names its
`__all__` lists; `j10r.py` is the same protocol for a J10-R leaf; `served.py` is the served unit's extract, prepare and
summary. Their test is `benchmarks/pouw/tests/test_h1_lanes.py`.
