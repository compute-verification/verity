# backends/numerical

Campaign-side numerical work for `verity.verification.target.FIRST` (rules: `backends/AGENTS.md`). `verity_numerical.reference`
is lane WP1 (independent numerical references); `verity_numerical.checker` is lane WP2.

## WP1: independence audit of the existing columns

The target names three executable columns of the finite-domain step: Python `verity.ml.tc.silicon`, Rust
`backends/sp1/common/src/tc.rs`, Lean `veritor/lean/VeritorSpec/TC.lean` (reused unchanged by `TCB.lean`, which only adds the
casts). Reading them against Hawkeye's pinned simulator (`fixtures/hawkeye/PIN.json`, `src/Ampere_simulator.cpp::group_sum`,
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
semantics. The independent evidence for the target is (i) the hardware captures (`packages/verity/tests/ml/fixtures`, and the
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
../../tools/tc_probe/                                the hardware probe (mma_tiles.cu, tc_probe.py): golden replay + finite sweep
~~~

~~~sh
uv run pytest backends/numerical/tests/reference                    # fast; the SMT proof needs z3 (a workspace dependency)
uv run research run --on vy-sp1 --project verity --source . --exclusive --stage wp1.exhaustive --require-result \
  -- bash backends/numerical/tests/reference/exhaustive_pod.sh
uv run research run --on vy-sp1 --project verity --source . --exclusive --stage wp1.probe.4090 --require-result \
  -- bash -c 'export PYTHONPATH=packages/verity/src:backends/numerical/python PATH=/usr/local/cuda/bin:$PATH; \
     /root/.local/bin/uv run --no-project --with numpy --python 3.12 python tools/tc_probe/tc_probe.py --arch sm_89 \
       --out "$RESEARCH_RUN_DIR" --procs 48 --replay-golden packages/verity/tests/ml/fixtures/golden/ada_bf16_m16n8k16.json --sweep'
~~~

Recorded runs (`research inspect <id>`): `r20260921-230852-4f1e` (wp1.exhaustive, SUCCESS, result valid: every counter zero over
1,076,644 transition-unit cases, 65,536 encodings, 131,072 products, 28 boundary groups, 3,170,304 + 88,602 D3 triples,
18,337,790 adapter pairs, the compiled simulator on all 1,076,644 lines); `r20260921-231618-ea24` (wp1.probe.4090, SUCCESS,
result valid: 360/360 golden records, 25,698,304 sweep elements of which 12,065,535 with a nonzero accumulator, 0 mismatches
against `silicon.tc_dot`).

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
