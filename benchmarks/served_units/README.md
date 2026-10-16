# Served units under C-Flock

A served model's units proved one type at a time, from synthetic values at the model's served statics, through the path the
one-stage audit takes: the unit's Boolean lowering, its `verity/flock-circuit` statement with hidden outputs, a loopback
`serve --zk --draw subset:N` / `prove --zk` session of `flock-circuit` (M0, Flock b684b12 with `verity/ml/flock`'s patches),
and the Lean verifier of record (`flock-verify verify --zk`, built from `verity/core/service/flock`), which must accept the
honest session and reject the selftest build's forgeries (a committed false output word, a false output claim, a flipped
internal bit).

Two kinds of type (`stage.py`'s docstring has both):

- a unit Definition (`UNITS`: the FP8 block GEMM coordinate), staged by `class_statement.stage_unit` on PoUW's row
  convention, each instance's output held to the IR reference and to the served kernel's numpy twin;
- a Q_word class of a word Definition (`PARTS`: router, MoE tiles, SiLU-mul block quant, RMSNorm, RoPE, DSA indexer, sparse
  MLA head, ...), the class of rank `C` by word gates (`TYPE:C=1`), staged by `class_statement.stage` with every instance's
  own `sha512/row/v1` rows; the circuit binds the Q_word partition, and Lean gets its own copy of the partition object
  (`--partition`) and takes the units as stated.

Files:

- `stage.py UNIT N OUT`: the unit types at GLM-5.3's statics (per TP8 rank) and their synthetic values.
- `job.sh`: builds, stages, proves and verifies every unit of `SU_UNITS` (its header lists the checks and the environment);
  the builds, the session and the controls are `benchmarks/pouw/hidden_zk/job.sh`'s.
- `summarize.py`: `out/<tag>/summary.json` and one table row per unit.

On vy-nebius-cpu-1, through its check slots:

~~~text
research run --on vy-nebius-cpu-1 --project verity --campaign glm53 --source . --cwd source --mem-gb 160 \
  --declared-output 'out/**' --env SU_UNITS="fp8-coord:K=6144 router moe-w13" --env SU_N=4 --env SU_MEM_GB=160 \
  --env SU_CACHE=/workspace/research/cache/served-units-glm53 \
  -- python3 tools/verity/check/slot.py --by LANE -- bash benchmarks/served_units/job.sh
~~~
