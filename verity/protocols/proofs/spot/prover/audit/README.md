# prover/audit

The prover's side of SPOT's compute audit that loads neither the adapter, nor Flock's prover, nor `benchmarks/`: a
development challenger that runs in this process, for the audit's commands and their tests, and a class circuit's
padded size, for a class too large to lower. Each file's docstring says what it does.

The commands that run an audit on a served run do load those three, so they are under `benchmarks/one_stage/` and
`benchmarks/prove_path/`.
