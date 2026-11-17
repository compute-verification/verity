# prover/audit

What SPOT's compute audit runs on the prover's side and loads neither the adapter nor Flock's prover nor `benchmarks/`:

- `local_challenger.py`: a development challenger on this machine, its session token and gate secret, for the audit's
  commands and their tests.
- `class_size.py`: a class circuit's padded size (`../cost/README.md`).

The commands that run an audit on a workload (the template and window audits, the stop audit, the two-stage driver,
the proved bundle, the prove path and the audit from the records alone) load the adapter, Flock's prover and the
benchmarks' scripts, so they are under `benchmarks/one_stage/` and `benchmarks/prove_path/` (its README).
