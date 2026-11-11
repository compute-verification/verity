# Flock --zk on an int8 row block

The prover's price for proof blocks of `ROWS` adjacent cells of one int8 GEMM output row: `S8Matmul_v1{N=ROWS, K}` in its
plain arm (`verity.definitions.ops.s8`), each instance's operand rows and output row hashed in the circuit, staged as a
`verity/flock-circuit` statement of N instances (`class_statement --partition q-template-instance`) and proved in loopback
`flock-circuit serve --zk` / `prove --zk` sessions, on the GPU under `gpu-lease` and on the CPU.

Files:

- `program.py`: the statement's Program, N instances of the template (`K`, `ROWS`, `N` from the environment).
- `run.sh`: builds the CUDA prover (CUDA 13.3, `verity/protocols/flock/pod/cuda-13.3.sh`), stages each `K:N[:R]` of `STATEMENTS`, and
  runs a session set per statement on the GPU (one lease, with GPU, DCGM and prover-core sampling, and an optional Nsight
  Systems session) and per statement of `CPU_STATEMENTS` on the CPU; `PLAIN_STATEMENTS` and `CPU_PLAIN_STATEMENTS` run the
  same sets without `--zk`. Each set passes only if the verifier recorded and accepted every session. Its header lists the
  steps and the environment.
- `summary.py`: `out/summary.json`, per statement and mode: the timed sessions' prove time and phases, the verifier's
  verdicts, the GPU's busy share, tensor-pipe and SM activity and peak memory, the prover cores' CPU seconds a session, the
  statement's size, and with the step `lean`, Lean's verdict, wall and peak memory per kept session.

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
verifies every kept session of the two statements with Lean's `flock-verify`, the verifier: `--zk` sets as
`verity/flock-circuit`, and `served-plain`'s plain sets as `verity/flock-circuit+plain-leaves` (hm96 leaves without `--zk`
are outside its plain scope).
The job's `FLOCK_WORK` then holds the prover's values, so it is a directory the job makes and deletes:

~~~text
research run --on NODE --project verity --source . --cwd source -- bash -c 'w=$(mktemp -d -p /workspace/tmp); \
  trap "rm -rf $w" EXIT; bash benchmarks/flock_s8_block/run.sh FLOCK_WORK=$w WINDOW=DIR SUBCIRCUIT=ID SERVE_RUN=DIR \
  STATEMENTS=served "PLAIN_STATEMENTS=served served-plain" CPU_STATEMENTS=served CPU_PLAIN_STATEMENTS=served DRAW=all \
  SANITY=0 NSYS=1 NSYS_STATEMENT=served'
~~~
