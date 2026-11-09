# Served proof blocks under Flock

A served model's proof blocks proved one type at a time, from synthetic values at the model's served statics, through the path the
one-stage audit takes: the proof block's Boolean lowering, its `verity/flock-circuit` statement with hidden outputs, a loopback
`serve --zk --draw subset:N` / `prove --zk` run of `flock-circuit` (Flock's circuit prover, upstream Flock b684b12 with
`verity/ml/flock`'s patches), and the Lean verifier (`flock-verify verify --zk`, built from `verity/core/service/flock`), which must
pass the honest run through every check and reject the selftest build's forgeries (a committed false output word, a false
output claim, a flipped internal bit). The verifier holds no program for a proof block (`stage.py` says why), so it then
refuses the honest run for that alone (`ProvedScope.NO_PROGRAM`), which is what `summarize.py` passes it on.

Three kinds of type (`stage.py`'s docstring has each):

- a Definition proved as one proof block (`UNITS`: the FP8 GEMM coordinate with scale groups of 128, the W8A8 GEMM
  coordinate), staged by `class_statement.stage_unit` on PoUW's row convention, each input's output held to the Program's
  reference and to the served kernel's numpy twin;
- a Q_word class of a word Definition (`PARTS`: router, MoE tiles, SiLU-mul with its scale-group quantization, RMSNorm, RoPE,
  DSA indexer, sparse MLA head, ...), the class of rank `C` by word gates (`TYPE:C=1`), staged by `class_statement.stage` with every input's
  own `sha512/row/v1` rows; the circuit binds the Q_word partition, and Lean gets its own copy of the partition object
  (`--partition`) and takes the proof blocks as stated;
- a block type of the serving rows by its census name (`CENSUS`: the FP8 group quantization, `gq:K=384` a column block of
  K = 6144), one whole Definition a proof block, staged like a `UNITS` type on bf16 rows, held to the Program's reference.

Files:

- `stage.py UNIT N OUT [--public-weights [--packed]]`: the proof-block types at GLM-5.3's statics (per TP8 rank), the W8A8
  coordinate at Llama-3.2-1B's, and their synthetic values; `--public-weights` stages a `UNITS` type's weight row as public
  input ports on the same values, the block without its weight row's in-circuit opening (a statement the verifier refuses as
  outside the proved scope, after every check), each leaf in its own u16 words, or with `--packed` as the row lays them.
- `job.sh`: builds, stages, proves and verifies every proof block of `SU_UNITS` (its header lists the checks and the environment);
  the builds, the run and the controls are `benchmarks/pouw/hidden_zk/job.sh`'s.
- `summarize.py`: `out/<tag>/summary.json` and one table row per proof block.
- `prove_time.sh`: the prover's time on statements of each `PT_UNITS` type (`TYPE[:S=V,...]`, `+public-weights` or
  `+packed-weights` for those arms)
  at each size of `PT_NS`, on one leased GPU (`PT_DEVICE=gpu`) or the CPU prover (`cpu`); `prove_time.py` fits each type's
  fixed seconds a statement and seconds a block, and on a GPU gives its proof overhead (seconds a block × the card's peak MACs
  a second ÷ the block's MACs), marginal and amortized at each size. `PT_STATEMENTS` proves statements staged elsewhere as
  they are, and `PT_PROVERS="1 4"` proves each one by K prover processes at once on the one device as well, priced from
  their throughput.
- `stage_window.py WINDOW_DIR OUT (--member K | --pick P N)`: a served window's member (a TP Commit's `--serving-rows`
  window, or `verity-vllm tp-serving-rows`') as its served statement for `PT_STATEMENTS`: the verifier's circuit for its
  template bound to the window's digests, and the window's own bodies (its `inst.bin` is the prover's: keep OUT private).

From the source tree's root on a CPU node with 160 GB for the job (results go to `$RESEARCH_RUN_DIR/out`; `SU_CACHE` keeps
builds and stagings between runs):

~~~text
RESEARCH_RUN_DIR=DIR SU_UNITS="fp8-coord:K=6144 router moe-w13" SU_N=4 SU_MEM_GB=160 SU_CACHE=CACHE \
  bash benchmarks/served_units/job.sh
~~~

Prove times (results go to `$RESEARCH_RUN_DIR/out`, the build and stagings to `FLOCK_WORK`, kept between runs):

~~~text
research run --on NODE --cwd source -- bash benchmarks/served_units/prove_time.sh \
  PT_UNITS="fp8-coord:K=6144 fp8-coord:K=6144+public-weights" PT_NS="16 64" PT_DEVICE=gpu SM=120
~~~
