# Multi-unit statement forms

One Definition, `ScaledMmFp8Block_v2{K,N,128}`, can be handed to Flock as one statement, as two statements joined by a
row, or as two tables with glue. This benchmark states it each way (`verity_flock.multi_unit`) at a few shapes and proves
each on CPUs, so the forms can be compared on what the prover and verifier pay: nonzeros, proving and verifying seconds,
peak memory and proof bytes.

## Try it

Stage the one-statement form at a small shape, K = 1024, N = 4, four instances:

~~~sh
uv run python benchmarks/multi_unit/stage.py c 1024 4 4 /tmp/multi_unit
~~~

It checks every instance against the served kernel's numpy twin and the Definition's evaluation, prints one JSON line
(each slot type's ANDs and nonzeros, k_log, seconds) and writes it to `/tmp/multi_unit/c/stage.json`; it takes about a
minute.

The whole comparison, every form at every shape with its proofs, runs in a proofs slot (its settings are the
environment variables `job.sh` lists at its top):

~~~text
research run --pool proofs --cwd source -- bash benchmarks/multi_unit/job.sh
~~~

Status: a comparison of statement forms, not a served path. The verifier refuses a form its soundness proof doesn't cover, and
the Lean verifier refuses each `--zk` session for want of a Program once every check has passed; the table reports each verdict
without that scope refusal.
