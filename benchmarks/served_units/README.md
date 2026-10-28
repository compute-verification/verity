# Served proof blocks under Flock

A served model's proof blocks proved one type at a time, from synthetic values at the model's served statics, through the path the
one-stage audit takes: the proof block's Boolean lowering, its `verity/flock-circuit` statement with hidden outputs, a loopback
`serve --zk --draw subset:N` / `prove --zk` run of `flock-circuit` (Flock's circuit prover, upstream Flock b684b12 with
`verity/ml/flock`'s patches), and the Lean verifier of record (`flock-verify verify --zk`, built from `verity/core/service/flock`), which must accept the
honest run and reject the selftest build's forgeries (a committed false output word, a false output claim, a flipped
internal bit).

Two kinds of type (`stage.py`'s docstring has both):

- a Definition proved as one proof block (`UNITS`: the FP8 GEMM coordinate with scale groups of 128, the W8A8 GEMM
  coordinate), staged by `class_statement.stage_unit` on PoUW's row convention, each input's output held to the Program's
  reference and to the served kernel's numpy twin;
- a Q_word class of a word Definition (`PARTS`: router, MoE tiles, SiLU-mul with its scale-group quantization, RMSNorm, RoPE,
  DSA indexer, sparse MLA head, ...), the class of rank `C` by word gates (`TYPE:C=1`), staged by `class_statement.stage` with every input's
  own `sha512/row/v1` rows; the circuit binds the Q_word partition, and Lean gets its own copy of the partition object
  (`--partition`) and takes the proof blocks as stated.

Files:

- `stage.py UNIT N OUT`: the proof-block types at GLM-5.3's statics (per TP8 rank), the W8A8 coordinate at Llama-3.2-1B's, and
  their synthetic values.
- `job.sh`: builds, stages, proves and verifies every proof block of `SU_UNITS` (its header lists the checks and the environment);
  the builds, the run and the controls are `benchmarks/pouw/hidden_zk/job.sh`'s.
- `summarize.py`: `out/<tag>/summary.json` and one table row per proof block.

From the source tree's root on a CPU node with 160 GB for the job (results go to `$RESEARCH_RUN_DIR/out`; `SU_CACHE` keeps
builds and stagings between runs):

~~~text
RESEARCH_RUN_DIR=DIR SU_UNITS="fp8-coord:K=6144 router moe-w13" SU_N=4 SU_MEM_GB=160 SU_CACHE=CACHE \
  bash benchmarks/served_units/job.sh
~~~
