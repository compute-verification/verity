# C-Flock --zk on an int8 row block

The prover's price for proof blocks of `ROWS` adjacent cells of one int8 GEMM output row: `S8Matmul_v1{N=ROWS, K}` in its
plain arm (`verity.ml.circuits.ops.s8`), each instance's operand rows and output row hashed in the circuit, staged as a
`verity/flock-circuit` statement of N instances (`class_statement --partition q-template-instance`) and proved in loopback
`flock-circuit serve --zk` / `prove --zk` sessions, on the GPU under `gpu-lease` and on the CPU.

Files:

- `program.py`: the statement's Program, N instances of the template (`K`, `ROWS`, `N` from the environment).
- `run.sh`: builds the CUDA prover (CUDA 13.3, `verity/ml/flock/pod/cuda-13.3.sh`), stages each `K:N[:R]` of `STATEMENTS`, and
  runs a session set per statement on the GPU (one lease, with GPU, DCGM and prover-core sampling, and an optional Nsight
  Systems session) and per statement of `CPU_STATEMENTS` on the CPU. Its header lists the steps and the environment.
- `summary.py`: `out/summary.json`, per statement and mode: the timed sessions' prove time and phases, the GPU's busy share,
  tensor-pipe and SM activity and peak memory, the prover cores' CPU seconds a session, and the statement's size.

On a GPU node (the build and staging run outside the lease; staging is about the template, not N):

~~~text
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/flock_s8_block/run.sh \
  'STATEMENTS=4096:8 4096:32 14336:1' 'CPU_STATEMENTS=4096:8' NSYS=1
~~~

A session's cost is mostly fixed, so its blocks' marginal cost comes from two statements at the same K and different N.
