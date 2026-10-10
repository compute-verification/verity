# benchmarks/numerical

Frozen: kept, still in the tables, not developed. The exception is the bench spine `verity_numerical.bench` (the result
contract, workloads, lowerings, the tables and their views), which C-Flock, `circuit-check` and the table renderers read,
and `bench.unit_costs`, the per-unit-type cost inventory (gates, nonzeros, sampled units, work weight) over the census
workloads.
What the merge gate runs here: `../../archive/README.md`.

Campaign-side numerical work for `verity.ml.circuits.targets.target.FIRST` (rules: `archive/AGENTS.md`). `verity_numerical.reference`
is lane WP1 (independent numerical references); `verity_numerical.checker` is lane WP2. `security` is the soundness
calculator for A-GKR and B-Ligero (`accounting`, `census`), `gkr_export` A-GKR's circuit exporter, `explore` cost-model
explorations and `redteam` the red team's forgeries against the checkers.

## WP1: independence audit of the existing columns

The target names three executable columns of the finite-domain step: Python `verity.ml.circuits.silicon.silicon`, Rust
`archive/sp1/common/src/tc.rs`, Lean `veritor/lean/VeritorSpec/TC.lean` in the `veritor` repository (reused unchanged by `TCB.lean`, which only adds the
casts). Reading them against Hawkeye's pinned simulator (`verity/ml/circuits/silicon/hawkeye/PIN.json`, `src/Ampere_simulator.cpp::group_sum`,
`utils/utils.cpp::multiply_bfloat16_to_gfloat`, `src/gfloat.cpp::Gfloat::operator float`):

- `silicon.py` states it is "a bit-exact pure-Python port of the reference simulator" at that commit. `silicon.group_sum` is the
  C++ `group_sum` statement for statement -- scan for the maximum exponent from the floor, align each term by
  `(magnitude << rescale) >> (max - exponent)`, sum with signs, `bit_length`, `exponent = max + width - W`, shift to `W` bits,
  denormalise below `-126`, shift back by `rescale`, the two zero rules -- generalised by the `width`, `zero_exponent` and
  `groups` parameters. `bf16_product` is `multiply_bfloat16_to_gfloat` (same `<< 9`, same `ea or 1`), `pack_fp32` is
  `operator float` with saturation to an infinity added where the C++ masks the exponent field, and `tc_dot` adds the sticky
  saturation of an overflowing group (measured on the RTX 4090, not in the simulator). Two representational differences: zero
  terms are skipped in the maximum (the C++ gives every zero the exponent `E_ZERO`, so the maximum is the same), and Python
  integers replace the `int32_t` with its `shift >= 32` guard (equal for every input the step forms; see
  `hawkeye_close.group_sum`).
- `tc.rs` says it is "bit for bit as `veritor/core/silicon.py` models it and `lean/VeritorSpec/TC.lean` specifies it"; every
  function's doc comment names the `silicon` function it ports, the bodies keep the Python's names (`max_exponent`, `total`,
  `bit_length`, `rescale`) and order, and `shift_int`/`unshift_int` are documented as `TC.shiftInt`/`TC.unshiftInt`.
- `TC.lean` says it is "bit for bit as `veritor/core/silicon.py` models it"; `groupSum` is the same statement sequence as
  `foldl`s, `packFp32`'s docstring says it is written "with `+` and `%` where the Python uses `|` and `&`". `TCTotal.lean`
  proves the total transcription agrees with it on finite inputs -- a theorem about two transcriptions.

Finding: the three columns are transliterations of one derivation -- the simulator's `group_sum`, ported once into Python and
from there into Rust and Lean. Their agreement establishes transcription fidelity, not independent confirmation of the hardware
semantics. The independent evidence for the target is (i) the hardware captures (`verity/ml/circuits/ops/tests/fixtures`, and the
recorded probe runs below), (ii) the compiled pinned simulator itself, driven on the same cases by
`tests/reference/hawkeye_harness.cpp`, and (iii) the closed-form normalisation of log entry D3, a different formula shown equal
to the successive shifts by an SMT proof over the whole reachable domain (`tests/reference/test_normalisation_d3.py`).

Two facts about the simulator noted on the way, neither used by the mma path: `utils.cpp::bfloat16_to_gfloat_tensor` decodes a
BF16 subnormal with exponent `-127` and no forced `exp = 1` (unlike `multiply_bfloat16_to_gfloat`), and `Gfloat`'s default
constructor carries Hopper's `-133`.

## WP1: what is here

~~~text
python/verity_numerical/reference/hawkeye_close.py   the source-close evaluator: the pinned routines as defined integer operations
                                                     (multiply_bfloat16_to_gfloat, group_sum, gfloat_to_float_bits, the constants),
                                                     wrapper_element (the wrapper's loop incl. its finalisation), fp32_to_gfloat +
                                                     tc_dot16 (the arbitrary-accumulator adapter, D7), normalise_closed_form (D3)
tests/reference/test_hawkeye_close.py                plan items a-d against silicon and every all-finite hardware record (seconds)
tests/reference/test_normalisation_d3.py             D3: SMT proof + exhaustive structure; the proof sketch is its module docstring
tests/reference/wp1_cases.py                         fixture loaders, structured/random cases, boundary groups (shared with the driver)
tests/reference/exhaustive_driver.py                 the recorded run, stage wp1.exhaustive: counts + wp1_exhaustive.json + result.json
tests/reference/hawkeye_harness.cpp                  drives the compiled pinned simulator (unmodified sources) on the same cases
tests/reference/exhaustive_pod.sh                    clone at the pin, hash-object check, CPU libtorch venv, build, run
../tc_probe/                                the hardware probe (mma_tiles.cu, tc_probe.py): golden replay + finite sweep
~~~

~~~sh
uv run pytest benchmarks/numerical/tests/reference                    # fast; the SMT proof needs z3 (a workspace dependency)
uv run research run --on vy-sp1 --project verity --source . --exclusive --stage wp1.exhaustive --require-result \
  -- bash benchmarks/numerical/tests/reference/exhaustive_pod.sh
uv run research run --on vy-sp1 --project verity --source . --exclusive --stage wp1.probe.4090 --require-result \
  -- bash -c 'export PYTHONPATH=.:benchmarks/numerical/python PATH=/usr/local/cuda/bin:$PATH; \
     /root/.local/bin/uv run --no-project --with numpy --python 3.12 python benchmarks/tc_probe/tc_probe.py --arch sm_89 \
       --out "$RESEARCH_RUN_DIR" --procs 48 --replay-golden verity/ml/circuits/ops/tests/fixtures/golden/ada_bf16_m16n8k16.json --sweep'
# A100 (sm_80): same, with --on vy-a100 --stage wp1.probe.a100, --arch sm_80, and instead of the (Ada) golden replay
#   --replay-special verity/ml/circuits/ops/tests/fixtures/tc-total-2026-09-07/a100/mma_special.reduced.json.gz
~~~

Recorded runs (`research inspect <id>`): `r20260921-230852-4f1e` (wp1.exhaustive, SUCCESS, result valid: every counter zero over
1,076,644 transition-unit cases, 65,536 encodings, 131,072 products, 28 boundary groups, 3,170,304 + 88,602 D3 triples,
18,337,790 adapter pairs, the compiled simulator on all 1,076,644 lines); `r20260921-231618-ea24` (wp1.probe.4090, SUCCESS,
result valid: 360/360 golden records, 25,698,304 sweep elements of which 12,065,535 with a nonzero accumulator, 0 mismatches
against `silicon.tc_dot`); `r20260921-232750-56c5` (wp1.probe.a100, NVIDIA A100 80GB PCIe sm_80, SUCCESS, result valid: the
7,713 all-finite A100 special records (4,565 with a nonzero accumulator) replayed with 0 mismatches, the same sweep -- same seed,
same 10 families -- 25,698,304 elements, 12,065,535 with a nonzero accumulator, 0 mismatches against `silicon.tc_dot`; D9's family
`zero_acc_tiny_cancellation`: 163,533 elements would differ under a zero at -126 and 87,058 under -127, the hardware matched
neither alternative on either device).

## WP1: the adapter and the wrapper's finalisation (D7)

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
identity `group_sum`; `tests/reference` compares `tc_dot16` with the full wrapper on every zero-accumulator case. A zero must
carry `-132`: the C++ maximum ranges over every term, and a zero at `-126`/`-127` changes the truncation of sums of products
below `2^-145` (`tc_probe` family `zero_acc_tiny_cancellation` measures which the hardware does; no earlier capture
distinguishes them).

## WP1: where the simulator is not a reference

On finite inputs whose group result leaves the FP32 range (`exponent > 127`) the simulator masks the exponent field
(`(e + 127) & 0xFF`: an exponent of 128 gives an infinity or NaN word by the mantissa, 129 and above wrap into a small finite
field) and keeps computing into group 2, where the value can come back into range; `silicon` saturates to an infinity of the
sign and keeps it through the remaining group, and every measured case agrees with `silicon` (2,197 A100 + 2,197 RTX 4090 special
cases and 114 golden records saturate; 18 of them cancel to zero in group 2 under the simulator's arithmetic and one golden
record returns to a finite value). Log entry D8.

## WP2 checker: the FP8 generalisation (silicon Phase 0, lane `silicon-p0`, 2026-09-22)

`verity_numerical.checker` was BF16-only by construction: `Params` asserted an adder at least as wide as the accumulator's
significand (`width >= acc_sig`, 25 >= 24), and one exponent width served operands and accumulator alike. Hopper/Ada E4M3
break both -- the FP8 adders are 14 bits wide and truncate the incoming FP32 accumulator's 24-bit significand to 14 before
the grouped add (`verity.ml.circuits.silicon.models.GroupSum.step`; a group of zero products is therefore not a no-op, log entry D10), and
E4M3 has a 4-bit exponent with its own NaN rule (`0x7F`/`0xFF` are the NaNs, `0x7E` = 448 is finite, there are no infinities).

What changed (`python/verity_numerical/checker/`):

- `params.py`: `Params` carries the operand format (`sig`, `exp`, `op_nan_rule` in `{"ieee", "e4m3"}`) and the accumulator's
  (`acc_sig`, `acc_exp`, defaulting to `exp`) and admits `width < acc_sig` -- the *lossy* case, with `acc_drop = acc_sig - width`.
  `Params.from_model(model)` derives them from a `GroupSum`; at `AMPERE_BF16_M16N8K16` it is `REAL`. `TOY_LOSSY` is the small
  lossy instance. Contracts, hint layout and outputs at `REAL` / `TOY` / `TINY` are byte-identical to before (every BF16 test
  passes unchanged); the census over the WP2 contracts is unaffected at `REAL`.
- `gadgets.py`: one new gadget, `RescaleState` (`M = 2^drop * P + r`, `r < 2^drop`, `P < 2^W`): the truncation of the incoming
  significand to the adder width, placed before `AlignState` on lossy paths only. `Normalise` produces a `width`-bit magnitude
  and scales it back up exactly (`norm_up`) where the truncation no longer folds into one floor division. `Bf16Decode` (alias
  `OperandDecode`) admits the E4M3 NaN rule through one extra range check; the accumulator gadgets read the `acc_*` widths.
  No new *family* in `verity_sp1.proofs.families` was needed: the cast-free `veritor.tensor-core.<pipeline>@2` sets already
  describe the FP8 pipelines' `tc_dot32` / `tc_dot32_0`; the rescale is a gadget of the checker's decomposition of that step.
- `reference.py`, `step.py`, `search.py`: use the split formats; `gen_step` / `check_step_expr` route lossy paths through
  `RescaleState`. `z3_group_step` and the `v2` arithmetisation remain BF16-only and raise `Reject` / `ValueError` when asked
  for a lossy or two-exponent-width configuration rather than answering wrongly.
- `tests/checker/test_fp8.py`: the reference evaluator and the gadget relation equal `model.step` on 2,000 random finite FP8
  cases each for `HOPPER_E4M3_K32` and `ADA_E4M3_M16N8K32` (zero accumulators, cancellation, subnormal operands, 448 included),
  the zero-product truncation of D10, the E4M3 NaN words, rescale-hint mutations rejected, the E5M2 hypotheses through the same
  `Params`, and an exhaustive slice of `TOY_LOSSY`.

The FP8 drafts these serve are `verity.ml.circuits.targets.target.FP8_HOPPER_WGMMA` / `FP8_ADA_MMA` (`status="draft"`, no guest bound);
their instructions and evidence are in `verity.ml.circuits.silicon.instructions`, the replicas in `verity/ml/circuits/silicon/captures/`, the open hardware questions
in `verity/ml/circuits/silicon/discrepancy_log.json` D13-D15.
