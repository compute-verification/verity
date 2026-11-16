# The shared numerical Python: design

The package's parts, the subcircuit catalog, the reference and the checker, beside [the README](README.md).

## The parts

`verity_numerical` has three parts:

- `bench`: the result contract (`contract`) and the comparison of two results under it (`judge`), a result's interaction
  time (`interaction`, `latency`), the census as Python (`census`, which reads `benchmarks/census/`), input sets (`generate`,
  `fixture_sets`, `instances`), one benchmark result measured on its prover and verifier pods (`cell`, `placement`, `drivers`), the
  subcircuit catalog and the per-unit-type cost inventory (`unit_costs` and its padding, prover and menu tables).
- `reference`: the pinned GPU simulator's tensor-core step as integer operations (`hawkeye_close`), an independent reference
  for `verity.circuits.targets.target.FIRST`.
- `checker`: a reference checker of that step with auxiliary witnesses, and the search for an output it accepts and should not.

`verity bench numerical JOB [ARGS]` (`run.py`) starts one of the package's command-line modules with ARGS, as
`python -m verity_numerical.<module>` does; with no JOB it prints the list of jobs. Each module's docstring is its manual. None needs
a GPU: every job runs on a CPU machine, and `cell` and `placement` reach a benchmark result's pods through `research`.

The tests are the suite `tests/benchmarks/numerical/` (`uv run tools/check/suites.py`).

## The subcircuit catalog

`bench.subcircuit_catalog` builds `verity/subcircuit-catalog/v1`, one artifact per build: every census subcircuit's Boolean
export (one payload file each), descriptor id, program digests per variant, and gate, AND and unit-row counts cited from a
`circuit-check-report/v1` artifact, never recomputed. An entry with no export, digest or record stays in, with a `missing`
reason. The module docstring holds the meta schema. A build checks the census first, then builds:

~~~text
circuit-check $(python -c 'from verity_numerical.bench import subcircuit_catalog as S; print(*S.census_ids())') --out cc.json
python -m verity_numerical.bench.subcircuit_catalog --records cc.json --out DIR --put [--preserve]
~~~

A `research run` workload holds no store credentials. There, `--put` goes into a scratch store (`RESEARCH_STORE` and
`RESEARCH_STORE_CONFIG` under its scratch), and `DIR/catalog.json` records the ids. Afterwards, `research data put` of the
run's `cc.json` (same name) and `DIR/tree` with `--meta @DIR/meta.json --ref circuit_check=ART --preserve` gives the same ids.

## The reference: its files and commands

~~~text
python/verity_numerical/reference/hawkeye_close.py             the source-close evaluator: the pinned routines as defined integer operations
                                                               (multiply_bfloat16_to_gfloat, group_sum, gfloat_to_float_bits, the constants),
                                                               wrapper_element (the wrapper's loop incl. its finalisation), fp32_to_gfloat +
                                                               tc_dot16 (the arbitrary-accumulator adapter, D7), normalise_closed_form (D3)
tests/benchmarks/numerical/reference/test_hawkeye_close.py     plan items a-d against silicon and every all-finite hardware record (seconds)
tests/benchmarks/numerical/reference/test_normalisation_d3.py  D3: SMT proof + exhaustive structure; the proof sketch is its module docstring
tests/benchmarks/numerical/reference/wp1_cases.py              fixture loaders, structured/random cases, boundary groups (shared with the driver)
tests/benchmarks/numerical/reference/exhaustive_driver.py      the recorded run, stage wp1.exhaustive: counts + wp1_exhaustive.json + result.json
tests/benchmarks/numerical/reference/hawkeye_harness.cpp       drives the compiled pinned simulator (unmodified sources) on the same cases
tests/benchmarks/numerical/reference/exhaustive_pod.sh         clone at the pin, hash-object check, CPU libtorch venv, build, run
../tc_probe/                                                   the hardware probe (mma_tiles.cu, tc_probe.py): golden replay + finite sweep
~~~

~~~sh
uv run pytest tests/benchmarks/numerical/reference                    # fast; the SMT proof needs z3 (a workspace dependency)
uv run research run --on vy-sp1 --project verity --source . --exclusive --stage wp1.exhaustive --require-result \
  -- bash tests/benchmarks/numerical/reference/exhaustive_pod.sh
uv run research run --on vy-sp1 --project verity --source . --exclusive --stage wp1.probe.4090 --require-result \
  -- bash -c 'export PYTHONPATH=.:benchmarks/numerical/python PATH=/usr/local/cuda/bin:$PATH; \
     /root/.local/bin/uv run --no-project --with numpy --python 3.12 python benchmarks/tc_probe/tc_probe.py --arch sm_89 \
       --out "$RESEARCH_RUN_DIR" --procs 48 --replay-golden tests/verity/circuits/ops/fixtures/golden/ada_bf16_m16n8k16.json --sweep'
# A100 (sm_80): same, with --on vy-a100 --stage wp1.probe.a100, --arch sm_80, and instead of the (Ada) golden replay
#   --replay-special PATH, PATH the A100 special-value fixture
#   (its a100/mma_special.reduced.json.gz path is in artifacts.json)
~~~

The evidence store holds the recorded run of each of the three stages: the files its command writes and its result, whose
validation passes only when every mismatch count is zero. wp1.exhaustive is the driver's steps a to g: the source-close
evaluator, `silicon` and the compiled pinned simulator held to one another, and to every all-finite hardware record.
wp1.probe.4090 and wp1.probe.a100 hold the hardware's words to `silicon.tc_dot`: the golden records replayed on the RTX 4090,
the A100's all-finite special records replayed on the A100, and one finite sweep, same seed and families, on both. The sweep's
family `zero_acc_tiny_cancellation` is the measurement behind log entry D9.

## The reference: the adapter and the wrapper's finalisation (D7)

The pinned wrapper (`Ampere_simulator::matrix_multiply_template`) starts every coordinate from `Gfloat(false, -132, 0)`,
appends products, collapses each full vector of 9 by `group_sum`, and after the last product applies `group_sum` once more to
whatever remains -- for `k = 16` the single Gfloat the second group produced -- and casts it with `operator float`. The
finalisation is therefore `group_sum([g])` followed by the cast. `group_sum([g]) = g` for every reachable `g`: a 24-bit
significand at exponent `>= -126` rescales to 25 bits, has width 25, keeps its exponent and shifts back; a subnormal at `-126`
with `l < 24` bits is shifted left by `25 - l`, gets exponent `l - 150 < -126`, is shifted right by `24 - l` and then by 1 --
net zero (checked on every subnormal significand and on 3,072 patterns per exponent in the recorded run). The adapter
`fp32_to_gfloat` maps a word to the Gfloat `group_sum` would have produced for it -- zero to `(sign, -132, 0)`, a subnormal to
`(sign, -126, m)`, a normal to `(sign, e - 127, 2^23 | m)` -- and is the left inverse of `operator float` on those outputs
(same check). So chaining through FP32 words is chaining through Gfloats, the adapter preserves the cast and provably omits the
identity `group_sum`; `tests/benchmarks/numerical/reference` compares `tc_dot16` with the full wrapper on every zero-accumulator case. A zero must
carry `-132`: the C++ maximum ranges over every term, and a zero at `-126`/`-127` changes the truncation of sums of products
below `2^-145` (`tc_probe` family `zero_acc_tiny_cancellation` measures which the hardware does; no earlier capture
distinguishes them).

## The reference: where the simulator is not one

On finite inputs whose group result leaves the FP32 range (`exponent > 127`) the simulator masks the exponent field
(`(e + 127) & 0xFF`: an exponent of 128 gives an infinity or NaN word by the mantissa, 129 and above wrap into a small finite
field) and keeps computing into group 2, where the value can come back into range; `silicon` saturates to an infinity of the
sign and keeps it through the remaining group, and every measured case agrees with `silicon` (2,197 A100 + 2,197 RTX 4090 special
cases and 114 golden records saturate; 18 of them cancel to zero in group 2 under the simulator's arithmetic and one golden
record returns to a finite value). Log entry D8.

## The checker: FP8

`verity_numerical.checker` checks the FP8 step beside BF16's. On BF16 the adder is at least as wide as the accumulator's
significand (`width >= acc_sig`, 25 >= 24), and one exponent width serves operands and accumulator alike. Hopper/Ada E4M3
have neither -- the FP8 adders are 14 bits wide and truncate the incoming FP32 accumulator's 24-bit significand to 14 before
the grouped add (`verity.circuits.silicon.models.GroupSum.step`; a group of zero products is therefore not a no-op, log entry D10), and
E4M3 has a 4-bit exponent with its own NaN rule (`0x7F`/`0xFF` are the NaNs, `0x7E` = 448 is finite, there are no infinities).

Where (`python/verity_numerical/checker/`):

- `params.py`: `Params` carries the operand format (`sig`, `exp`, `op_nan_rule` in `{"ieee", "e4m3"}`) and the accumulator's
  (`acc_sig`, `acc_exp`, defaulting to `exp`) and admits `width < acc_sig` -- the *lossy* case, with `acc_drop = acc_sig - width`.
  `Params.from_model(model)` derives them from a `GroupSum`; at `AMPERE_BF16_M16N8K16` it is `REAL`. `TOY_LOSSY` is the small
  lossy instance; `REAL`, `TOY` and `TINY` are not lossy.
- `gadgets.py`: `RescaleState` (`M = 2^drop * P + r`, `r < 2^drop`, `P < 2^W`): the truncation of the incoming
  significand to the adder width, placed before `AlignState` on lossy paths only. `Normalise` produces a `width`-bit magnitude
  and scales it back up exactly (`norm_up`) where the truncation no longer folds into one floor division. `Bf16Decode` (alias
  `OperandDecode`) admits the E4M3 NaN rule through one extra range check; the accumulator gadgets read the `acc_*` widths.
  The rescale is no family of its own: the cast-free `veritor.tensor-core.<pipeline>@2` sets (`FP8_HOPPER_FAMILY` and
  `FP8_ADA_FAMILY` in `verity.circuits.targets.target`) describe the FP8 pipelines' `tc_dot32` / `tc_dot32_0`, and
  it is a gadget of the checker's decomposition of that step.
- `reference.py`, `step.py`, `search.py`: use the split formats; `gen_step` / `check_step_expr` route lossy paths through
  `RescaleState`. `z3_group_step` and the `v2` arithmetisation remain BF16-only and raise `Reject` / `ValueError` when asked
  for a lossy or two-exponent-width configuration rather than answering wrongly.
- `tests/benchmarks/numerical/checker/test_fp8.py`: the reference evaluator and the gadget relation equal `model.step` on 2,000 random finite FP8
  cases each for `HOPPER_E4M3_K32` and `ADA_E4M3_M16N8K32` (zero accumulators, cancellation, subnormal operands, 448 included),
  the zero-product truncation of D10, the E4M3 NaN words, rescale-hint mutations rejected, the E5M2 hypotheses through the same
  `Params`, and an exhaustive slice of `TOY_LOSSY`.

The FP8 drafts these serve are `verity.circuits.targets.target.FP8_HOPPER_WGMMA` / `FP8_ADA_MMA` (`status="draft"`, no guest bound);
their instructions and evidence are in `verity.circuits.silicon.instructions`, the replicas in `verity/circuits/silicon/captures/`, the open hardware questions
in `verity/circuits/silicon/discrepancy_log.json` D13-D15.
