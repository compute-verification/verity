# compute: the share of a served model's MACs that Pearl+ credits

`verity bench compute` prints the compute audit's number for one recorded run: the multiply-accumulates (MACs) the
Pearl+ verifier credits, over the MACs the served model computes on the tokens the run's registered class holds. It
measures nothing itself. It reads the verifier's side of the run's record from the evidence store and nothing the
developer's node reports of what it served.

- `run.py`: the command. Its docstring states the number's formula, each file of a record it reads, what it prints
  beside the number and when it refuses.
- `tests/`: the printer and the model's MAC count, held to the small record under `tests/fixtures/` with a stand-in
  for the evidence store, so the suite fetches nothing.

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
- **What it writes:** under `research run`, the printed document as `$OUT/compute.json`; otherwise nothing.
- **Exit code:** 0 with the number printed; 1 with none (the answer is not an accept, the Lean verifier's verdict is
  not the answer's, or the run left no answer); 2 when the command line, the run or its policy can't be read, or the
  split `--per-call` reads is not the answer's.
