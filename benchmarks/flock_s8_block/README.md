# Flock --zk on an int8 row block

What does Flock charge to prove a few cells of an int8 matrix product? Each proof block here is `ROWS` adjacent cells
of one output row, `S8Matmul_v1{N=ROWS, K}` in its plain arm (`verity.circuits.ops.s8`), with its operand rows and
output row hashed inside the circuit. The benchmark stages N such proof blocks as one Flock statement and proves it in
loopback `--zk` sessions on the GPU and on the CPU, reporting prove time, the GPU's use and peak memory. `run.sh`'s
header lists its steps and settings.

## Try it

Stage the statement `run.sh` checks first, K = 64 and two proof blocks, without proving (about twenty seconds):

~~~sh
K=64 N=2 uv run python -m verity_flock.class_statement --program-module benchmarks/flock_s8_block/program.py:program \
  --partition q-template-instance --out /tmp/flock_s8_block --stage-only --n 2
~~~

It prints the statement's Definition and its proof blocks' count as one JSON line.

On a GPU node (the build and staging run outside the lease; staging is about the template, not N):

~~~text
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/flock_s8_block/run.sh \
  'STATEMENTS=4096:8 4096:32 14336:1' 'CPU_STATEMENTS=4096:8' NSYS=1
~~~

A session's cost is mostly fixed, so its blocks' marginal cost comes from two statements at the same K and different N.

On a served window instead (the sampled audit's one-stage proof blocks over the values serving committed): `WINDOW` names
`prove_path.py commit`'s window dir, prover's files included, and `SUBCIRCUIT` its member's census-style id.
`rec_inner.py stage-window` stages every proof block of the member as the statement `served` (hm96 leaves, today's `--zk`
statement) and `served-plain` (plain SHA-512 leaves). With `SERVE_RUN`, staging first checks that the window's run root is the
one the served run published. `DRAW=all` has each verifier sample every block, as a deployed one must: a `--zk` one by
`--draw subset:N`, a plain one by the staged sample of every block (`draw.json`). The step `lean` (add it to `STEPS`)
verifies every kept session of the two statements with Lean's `verity-verify-proof`, the verifier: `--zk` sets as
`verity/flock-circuit`, and `served-plain`'s plain sets as `verity/flock-circuit+plain-leaves` (hm96 leaves without `--zk`
are outside its plain scope).
The job's `FLOCK_WORK` then holds the prover's values, so it is a directory the job makes and deletes:

~~~text
research run --on NODE --project verity --source . --cwd source -- bash -c 'w=$(mktemp -d -p /workspace/tmp); \
  trap "rm -rf $w" EXIT; bash benchmarks/flock_s8_block/run.sh FLOCK_WORK=$w WINDOW=DIR SUBCIRCUIT=ID SERVE_RUN=DIR \
  STATEMENTS=served "PLAIN_STATEMENTS=served served-plain" CPU_STATEMENTS=served CPU_PLAIN_STATEMENTS=served DRAW=all \
  SANITY=0 NSYS=1 NSYS_STATEMENT=served'
~~~
