# compute: the share of a served model's MACs that Pearl+ credits

What you may change is how the work is done: the kernels that compute the served model's matmuls, how the model is
served under Pearl+ (how the load's requests become engine steps and passes, and the streams, graphs and overlap on the
card), and the give-up rule (which rows each matmul gives up before the coin, within the verifier's cap). A better
kernel, schedule or give-up raises the number.

What is fixed is what the number is of: the model and its revision, and the load (`load.toml`: Llama 3.2 1B at its
pinned revision, 4 requests of 32 prompt tokens and 4 output tokens each), the verifier and its Lean (`pouw-pearlplus`,
whose verdict the number needs), and the formula (`run.py`'s docstring). A record whose policy registers another model,
revision or load is refused. Change any of these and the number is another benchmark's.

`verity bench compute` prints the compute audit's number for one served run. The number is a share: the
multiply-accumulates (MACs) that the Pearl+ verifier credits, over the MACs the served model computes on the tokens of
the run's registered class. The command measures nothing itself. It reads the verifier's side of the run's record, and
nothing the developer's node reports of what it served. The number is the Pearl+ piece's own answer, not PoUW's row in
the run's record of verdicts, and the line under it says which coins and parameters the record states the run used.

You give it a run's record in one of two ways:

- `--record DIR`: a directory on disk holding the record. This needs nothing of ours: no evidence store, no host and
  no key. The command imports no store client, looks up no credential and opens no network connection.
- `--from RUN`: a run id of our evidence store. This is ours: it needs research's store client and a key that reads
  the store.

~~~text
verity bench compute --record DIR
verity bench compute --from RUN
verity bench compute --record DIR --per-call [--uncredited F] [--top N]
uv run python benchmarks/compute/h200_share.py --from RUN --duty RUN
~~~

The two options are separate, so the command never guesses: a path that names no directory is refused as a path, and
a run id never reads a local file. `tests/fixtures/record/` is a record directory, so
`verity bench compute --record benchmarks/compute/tests/fixtures/record` runs from a clone. Its record is made up: it
shows the format, not a served run's number. Run the command from the clone's root as `verity bench compute`, which
starts it there, so a relative DIR is taken from the root.

The tests run from this directory, which is not a workspace member, in the workspace's environment:
`uv run --project ../.. pytest tests`.

Outside `research run` it only prints. To keep what it prints as files too (`compute.json` and `result.json`, below),
set `OUT` to a directory: `OUT=/tmp/compute verity bench compute --record DIR`.

Producing a record (serving a model and auditing the run) is not yet one command from a clone. Today a record comes
from a run on our machines.

## A record directory

A record directory is a run's directory as `research run` leaves it and our store keeps it. The paths below are
relative to it.

- `out/pieces/pouw/pearlplus-answer.json` (required): the Pearl+ verifier's answer. It holds the verdict, the credit
  in MACs, and the sample's ε_s at δ_s.
- `out/inference-only-policy.json` (required): the registered policy. It names the served model and the class: its
  slots, prompt tokens and output tokens. The model's MACs are counted from it.
- `out/pieces/pouw/pearlplus-lean.json` and `out/pieces/pouw/pearlplus-lean-check.json`: the Lean verifier's verdict,
  and the record's comparison of it with the answer. Without either there is no number (exit 1, naming the file).
- `out/record/verdicts.json` (optional): the run's composed record. PoUW's row is printed beside the number.
- `out/pieces/pouw/verify.json` (optional): the Pearl+ verifier's document. Its per-call split says whether the
  passes hold the LM head's calls, and `--per-call` prints it.
- `out/pieces/pouw/development.json` (optional): the Pearl+ piece's own document. Only its `coin_source`, where the
  passes' salt coins came from, is read, for the coins line below.
- `job.json` (optional): the run's own record of itself. The command prints its `run.id` as the run, and the commit
  its `store.conditions` name (`source.commit`, with `-dirty` when `source.dirty`). Without it the run is the
  directory's name and the commit is unknown.

A directory missing a required file is refused with exit 2, and the message names the file.

## What it refuses

It prints no number, and exits 1, unless the Pearl+ answer is an accept with a credit, and not a test instance's (the
answer's `profile.test_instance`, whose passes are not the served instance's). It also prints none when the
record holds no Lean verdict, or when the Lean verifier's verdict differs from the answer in verdict, run commitment or
credit. It
exits 2 when a file can't be read, when the policy's model or class can't give the model's MACs, or when the credit is
more than one load of the class can hold. It reads only the verifier's side of the record, so the tokens and seconds
the developer's node reports are never part of the number.

## Files

- `run.py`: the command. Its docstring states the number's formula, each file of a record it reads, what it prints
  beside the number, the result it writes and when it refuses.
- `h200_share.py`: the number of the goal `accounting/h200-97-percent`, modeled: an honest H200's e4m3 rate (a
  `benchmarks/tensor_duty --arms fp32-beside` run's arm A on uniform operands) over the card's ceiling at each reading
  of its tensor clock, times `run.py`'s share of a served run of record. Its docstring states the model and what it
  leaves out, which makes the number an upper bound on the served share.
- `tests/`: the printer, the model's MAC count and the written result, held to the small record under
  `tests/fixtures/`, read as a directory and through a stand-in for the evidence store, so the suite fetches nothing.
  Capacity is held to one card file's two clocks.

**What a caller may import:** nothing outside this directory imports `run.py`. It is a command, which
`verity bench compute` starts as a process of its own, and its `__all__` lists the names its own tests read. **What it
never does:** it starts no job and measures nothing, it reads nothing the developer's node reports of what it served,
and it prints no number for an answer that is not an accept. With `--from`, fetching a record is the evidence store's
client's work; the command itself writes only the two files named below.

- **Hardware:** none. It runs on any machine, in seconds: with `--record` on any clone, with `--from` on one that can
  read the evidence store. The GPU work is the recorded run's.
- **What it prints:** the number as text and as one line of JSON (`value`, `unit`, `commit`, `run`), beside its
  target. The first line says the number is what the Pearl+ piece's own answer credits, not PoUW's row in the run's
  record of verdicts. The line under it (`coins` in the JSON) says which coins and parameters the record states the
  run used: the answer's sample source (`profile.source_from`: `challenger` or `node`), the passes' salt coins
  (`development.json`'s `coin_source`: `challenger` or `in-process`), whether the Lean verifier ran as of record
  (`--of-record` in `pearlplus-lean-check.json`'s `argv`), and the answer's draws and δ. What the record doesn't state
  it prints as not stated. It ends the line with `development` where the record holds PoUW's development piece
  (`development.json`) or any of the three is not live, which today is every record, since only that piece states the
  salt coins. `challenger` names a challenger session, the auditor's or a local one: the record doesn't say whose. It
  never refuses on them. With `--per-call`, also where the MACs the credit leaves out went, by cause and by matmul call (the top N
  calls, 10 by default), from the Pearl+ verifier's split of its credit (the record's `verify.json`, or the file
  `--uncredited` names).
- **What it writes:** two files where `OUT` names a directory, as it does under `research run`, and otherwise
  nothing. With no number printed it writes neither.
  - `$OUT/compute.json`: the printed document.
  - `result.json`: the same result in the fields every resource's benchmark reports, through the harness kit's writer
    (`verity_bench.kit`). Under `research run` it is the run directory's (`$RESEARCH_RUN_DIR/result.json`), where
    `research` reads a run's result; outside one it is `$OUT/result.json`. Its measurements are
    `compute-accounting/verified-compute-share` (the printed number, its unit and its target, under its id in
    `metrics.toml`, where the console's Metrics tab reads it), `verified` (the credited MACs), `capacity`, `free`
    (capacity less verified) and `error` (the answer's δ_s, sampling's alone). A recorded run's record holds no
    registration of its hardware and no span the verifier holds its passes to, so `capacity` and `free` are one entry
    each, with no value and why.
    `capacity()` in `run.py` gives them for a registered site and a span, at each data-sheet reading: a clock of a
    registered device's card file (`clock.tensor`, `clock.sm`).
- **Exit code:** 0 with the number printed; 1 with none (the answer is not an accept, the record holds no Lean
  verdict or one that is not the answer's, or the run left no answer); 2 when the command line, the run, its record
  directory or its policy can't be read, when the policy's load is not the benchmark's (`load.toml`), or when the
  split `--per-call` reads is not the answer's.
