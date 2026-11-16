# scan_cost

What the network attester's scan costs to prove and to verify, for each second of execution phase it covers. The
scan is `benchmarks/network_traces/scan_zk.sh`'s: the step-indexed scan of one link, composed in zero knowledge, on a
CPU machine. This benchmark runs it once at a fixed identity, puts two numbers on its honest round with the stages
they are made of, and refuses to write them unless the round's answer is the pinned one.

The two numbers are the proving seconds, and the Lean verifier's CPU seconds on one core, each for a second of
execution phase. A candidate that makes the scan faster is compared with the baseline at that identity, stage by stage,
and has a number only if the guard holds on its run.

It states no measured figure. `baseline.json` names the baseline's run, the numbers are in that run's `result.json`
and `out/scan_cost.json` in the evidence store, and `result.py table RUN` prints them.

## Try it

~~~sh
uv run python benchmarks/scan_cost/result.py baseline
uv run python benchmarks/scan_cost/result.py table RUN
~~~

The first prints the baseline: its run, its commit and its identity. The second, with that run's id, fetches its
document from the evidence store and prints the identity, the group, whether the guard holds, the two numbers, and the
round a session a line and a stage a column. It needs the store's credentials and no GPU.

## Running it

`run.sh` is the command of a `research run` (the team's job runner), from a clone, in one slot of the network pool, as a job of at most 180
minutes. The slot must give at least 21 cores and 128 GB:

~~~sh
research run --on NODE --project verity --source . --cwd clone --pool network --as LANE --mem-gb 128 \
  --timeout 10800 -- bash benchmarks/scan_cost/run.sh
~~~

`run.sh`'s header has its settings (`SC_GROUP`, `SC_CORES`, `SC_CLASS`) and its exit codes. How the numbers are
taken, the identity, the guard and the targets are in [`DESIGN.md`](DESIGN.md).

~~~text
result.py table DOC                      a result's round, a session a line and a stage a column
result.py compare CANDIDATE [--to DOC]   a candidate beside the baseline, a second of execution phase by stage
result.py baseline [--lock DOC]          the baseline's run, or pin DOC's run as the baseline
~~~

DOC is a `scan_cost.json`, a directory holding one, or a run's id; run them with the workspace's Python (`uv run
python benchmarks/scan_cost/result.py …`). `baseline --lock` refuses a run whose guard does not hold or that lacks
either number.
