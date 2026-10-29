# Multi-unit statement forms

Three ways to state `ScaledMmFp8Block_v2{K,N,128}` for C-Flock (`verity_flock.multi_unit`), measured side by side on CPUs at
each shape `K:N:n` (n instances, one x row each):

- `c`: one statement, N * K/128 k-step units and a scale stage of N slots per VU, the stage returning the outputs;
- `b`: two statements joined by a row, the writer's units returning temp and s and the reader's units the scale stage;
- `a`: two tables with glue, `c`'s unit and stage circuits as tables and every stage port a glue relation (no rows, no `--zk`).

For each shape, `stage.py` stages `c` and `b` from synthetic values (checked against the served kernel's numpy twin and, when
small enough, the Definition's IR reference); `prove.py` runs one loopback session of each without `--zk` and one with it,
drawing every instance, with the prover's and verifier's peak memory; the Lean verifier of record checks each `--zk` session;
flock-live's test binary runs `a`; and selected shapes run `c`'s selftest negatives. After each shape the job runs `summarize.py`, which gathers every shape
into one table (`summary.json`): each slot type's nonzeros, k_log, the committed m, prover and verifier seconds and peak memory,
proof bytes and the verdict.

Run it from the source tree's root in a proofs slot; there is no `run.py` or `run.sh`, so not through `verity bench`:

~~~text
research run --pool proofs --cwd source -- bash benchmarks/multi_unit/job.sh
~~~

Its settings are environment variables: `MU_SHAPES` (default `1024:4:4 6144:16:2`), `MU_SELFTEST`, `MU_SELFTEST_ALL`,
`MU_CASES` and `MU_CACHE` (where the builds are cached). It builds with `benchmarks/pouw/hidden_zk/job.sh` (sourced: flock-circuit,
the Lean `flock-verify`, a venv with numpy) and builds flock-live's tests from an upstream Flock checkout with C-Flock's patches.
Results go to `$RESEARCH_RUN_DIR/out`, one directory per shape (`k<K>-n<N>-i<n>/`), and scratch to a temporary directory the job
removes.

Status: a comparison of statement forms, not a served path. The verifier refuses a form its soundness proof doesn't cover, and
the Lean verifier refuses each `--zk` session for want of a Program once every check has passed; the table reports each verdict
without that scope refusal.
