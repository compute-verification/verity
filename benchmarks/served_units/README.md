# Served units under C-Flock

A served model's units proved one type at a time, from synthetic values at the model's served statics, through the path the
one-stage audit takes: the unit's Boolean Definition, its `verity/flock-circuit` statement with hidden outputs
(`class_statement.stage_unit`), a loopback `serve --zk --draw subset:N` / `prove --zk` session of `flock-circuit` (M0, Flock
b684b12 with `verity/ml/flock`'s patches), and the Lean verifier of record (`flock-verify verify --zk`, built from
`verity/core/service/flock`), which must accept the honest session and reject the selftest build's forgeries (a committed
false output word, a false output claim, a flipped internal bit).

- `stage.py UNIT N OUT`: the unit types (`UNITS`, GLM-5.3's statics by default) and their synthetic values; each instance's
  output is held to the IR reference and to the served kernel's numpy twin before anything is written.
- `job.sh`: builds, stages, proves and verifies every unit of `SU_UNITS` (its header lists the checks and the environment);
  the builds, the session and the controls are `benchmarks/pouw/hidden_zk/job.sh`'s.
- `summarize.py`: `out/<tag>/summary.json` and one table row per unit.

~~~text
research run --on vy-cpu-1 --project verity --campaign glm53 --source . --cwd source --declared-output 'out/**' \
  --env SU_UNITS="fp8-coord:K=2048 fp8-coord:K=6144" --env SU_N=4 --env SU_MEM_GB=200 -- bash benchmarks/served_units/job.sh
~~~
