# compute: the share of a served model's MACs that Pearl+ credits

`verity bench compute` prints the compute audit's number for one recorded run: the multiply-accumulates (MACs) the
Pearl+ verifier credits, over the MACs the served model computes on the tokens the run's registered class holds. It
measures nothing itself. It reads the verifier's side of the run's record from the evidence store and nothing the
developer's node reports of what it served.

- `run.py`: the command. Its docstring states the number's formula, each file of a record it reads, what it prints
  beside the number, the result it writes and when it refuses.
- `tests/`: the printer, the model's MAC count and the written result, held to the small record under
  `tests/fixtures/` with a stand-in for the evidence store, so the suite fetches nothing. Capacity is held to one
  card file's two clocks.

~~~text
verity bench compute --from RUN
verity bench compute --from RUN --per-call [--uncredited F] [--top N]
~~~

- **Hardware:** none. It runs on any machine that can read the evidence store, in seconds; the GPU work is the
  recorded run's.
- **What it prints:** the number as text and as one line of JSON (`value`, `unit`, `commit`, `run`), beside its
  target. With `--per-call`, also where the MACs the credit leaves out went, by cause and by matmul call (the top N
  calls, 10 by default), from the Pearl+ verifier's split of its credit (the record's `verify.json`, or the file
  `--uncredited` names).
- **What it writes:** two files where `OUT` names a directory, as it does under `research run`, and otherwise
  nothing. With no number printed it writes neither.
  - `$OUT/compute.json`: the printed document.
  - `result.json`: the same result in the fields every resource's benchmark reports, through the harness kit's writer
    (`verity_bench.kit`). Under `research run` it is the run directory's (`$RESEARCH_RUN_DIR/result.json`), where
    `research` reads a run's result; outside one it is `$OUT/result.json`. Its measurements are `metric` (the printed
    number, its unit and its target), `verified` (the credited MACs), `capacity`, `free` (capacity less verified) and
    `error` (the answer's δ_s, sampling's alone). A recorded run's record holds no registration of its hardware and no
    span the verifier holds its passes to, so `capacity` and `free` are one entry each, with no value and why.
    `capacity()` in `run.py` gives them for a registered site and a span, at each data-sheet reading: a clock of a
    registered device's card file (`clock.tensor`, `clock.sm`).
- **Exit code:** 0 with the number printed; 1 with none (the answer is not an accept, the Lean verifier's verdict is
  not the answer's, or the run left no answer); 2 when the command line, the run or its policy can't be read, or the
  split `--per-call` reads is not the answer's.
