# H1 leaves proved lane by lane

This directory proves one H1 leaf as its 32 lanes and its combine, one Flock `--zk` statement each, and has the Lean
verifier check them against the verifier's own files, beside negatives each refused. The leaf is one at a small K made
here, J10-R's leaf, or the proof block a served execution phase's own sample selected.

## Try it

Check, on a few leaves, that every statement's known rows are the ones the harness builds:

~~~sh
uv run pytest tests/benchmarks/pouw/test_h1_lanes.py -q
~~~

It prints `4 passed`.

## Runner

**Hardware:** a node's CPUs, no GPU, with the memory the job's header sizes (`H1L_MEM_GB`, the job's `--mem-gb`).

## Runner

`run.sh JOB [ARGS...]` starts one job's script with the arguments and environment as given; run it from the source tree,
as a `research run`'s command. Each script's header has its full command, its knobs (`H1L_*`, `H1S_*`, `ZC_*`) and what
each step writes.

~~~text
research run --on MACHINE --pool proofs --project verity --source . --cwd source --mem-gb G \
  --declared-output 'out/*' --env H1L_MEM_GB=G -- bash benchmarks/pouw/h1_lanes/run.sh job
~~~

| job | what it runs | writes |
|---|---|---|
| `job` | `leaf.py prepare` (each statement's Program, the leaf's registered values, every statement staged), one loopback `--zk` session a statement, the plan's Lean checks and PoUW's own-window check of the registered reads, then `leaf.py summarize`; `H1L_J10R=1` runs `j10r.py`'s J10-R leaf in its place | `out/summary.json`, and the sessions' and Lean checks' logs under `$OUT` |
| `served` | the proof block a served window's sample selected (`h1_prove/served.py`), its openings extracted into the job's private scratch, `served.py prepare`, a session a lane under the ZK server, the Lean checks, `served.py summarize` | `out/summary.json`, and the sessions' and Lean checks' logs under `$OUT` |
| `zk_cost` | what the ZK server adds to one lane statement: `served` on a stand-in window (`ZC_K`, 2048), lane 0 alone and no negatives, once under the ZK server and once without it (`ZC_ARMS`), each lane checked by the Lean verifier | `out/arms.txt` (each arm's exit code and the job cgroup's CPU seconds), a text file beside it with a line an arm (the ZK server's processes' CPU seconds, how many they were and their summed peak RSS), and each arm's log and `served` outputs under `out/` |

`leaf.py` is a step of `job` (its `prepare`, `summarize` and `cost`), and `run.sh` never starts it.

[`DESIGN.md`](DESIGN.md) says what each Python module is and who reads it.
