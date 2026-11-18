# The shared numerical Python

`verity_numerical` (`python/verity_numerical/`) is the Python that the proof system's benchmarks share: what a measured
result must say about itself, the input sets results are measured on, and what each census workload costs as a Boolean
circuit. Beside them it holds an independent reference for the tensor-core step every expected word comes from, and a
reference checker of that step.

## Try it

List the package's command-line jobs; none needs a GPU:

~~~sh
uv run verity bench numerical
~~~

It prints each job with one line on what it does, and exits 2. `verity bench numerical JOB [ARGS]` runs one, as
`python -m verity_numerical.<module>` does, and each module's docstring is its manual.

## What a caller may rely on

**What a caller may import** is each module's `__all__`: every public name a file outside this directory reads is in one,
the Python inside a shell script counted. A name that starts with an underscore is in no list and is no part of what a
caller may rely on; one test of Flock's prover calls `bench.contract._check_rates` all the same. `settings` also answers
each setting it declares as an attribute. The package's readers outside this directory are Flock's prover and pod scripts
(`bench.contract`, `bench.interaction`, `bench.generate`, `bench.cell`, `bench.instances`) and other benchmarks: `tc_probe/`
(`bench.contract`, `bench.instances`, `reference.hawkeye_close`) and `proof_throughput/` (`bench.contract`). Several
benchmarks' scripts start `bench.generate` as a command, so what it prints is read too.

**What it never does.** It states no relation of its own: every expected word comes from `verity.circuits.silicon`, and the
reference and the checker are held to it by tests. It reads the census and the committed input sets and never writes them
outside a job that says so. It is no verifier, and no guarantee reads a number it computes.

## The reference: how independent the existing columns are

Two columns of the finite-domain step are read here: Python `verity.circuits.silicon` (`term.py`, `models.py`) and Lean
`security/Definitions/Core/TC/Spec.lean`. The Python is read against Hawkeye's pinned simulator
(`verity/circuits/silicon/hawkeye/PIN.json`, `src/Ampere_simulator.cpp::group_sum`,
`utils/utils.cpp::multiply_bfloat16_to_gfloat`, `src/gfloat.cpp::Gfloat::operator float`), the Lean by what its header says:

- `models.py` states its body is "a bit-exact pure-Python port of" the reference simulator at that commit. `models.group_sum` is the
  C++ `group_sum` statement for statement -- scan for the maximum exponent from the floor, align each term by
  `(magnitude << rescale) >> (max - exponent)`, sum with signs, `bit_length`, `exponent = max + width - W`, shift to `W` bits,
  denormalise below `-126`, shift back by `rescale`, the two zero rules -- generalised by the `width`, `zero_exponent` and
  `groups` parameters. `term.bf16_product` is `multiply_bfloat16_to_gfloat` (same `<< 9`, same `ea or 1`), `term.pack_fp32` is
  `operator float` with saturation to an infinity added where the C++ masks the exponent field, and `tc_dot` adds the sticky
  saturation of an overflowing group (measured on the RTX 4090, not in the simulator). Two representational differences: zero
  terms are skipped in the maximum (the C++ gives every zero the exponent `E_ZERO`, so the maximum is the same), and Python
  integers replace the `int32_t` with its `shift >= 32` guard (equal for every input the step forms; see
  `hawkeye_close.group_sum`).
- `Spec.lean`'s header calls it "A transcription of `term.py`", of `models.py`'s `group_sum` and of `total.py`, "function for
  function and branch for branch, on the Python functions' domain". `verity/circuits/evaluate/Evaluate/TC/Spec.lean` is, by
  its own header, a copy of that file and of the `Pipeline.lean` beside it, "definition for definition". Neither body is read
  against the Python here.

Finding: the two columns are one derivation -- the simulator's `group_sum`, ported once into Python and, by the Lean's header,
transcribed from there. Agreement between them shows transcription fidelity, not an independent confirmation of the hardware
semantics. The independent evidence for the target is (i) the hardware captures (`tests/verity/circuits/ops/fixtures`, and the
probe's recorded runs), (ii) the compiled pinned simulator itself, driven on the same cases by
`tests/benchmarks/numerical/reference/hawkeye_harness.cpp`, and (iii) the closed-form normalisation of log entry D3, a different formula shown equal
to the successive shifts by an SMT proof over the whole reachable domain (`tests/benchmarks/numerical/reference/test_normalisation_d3.py`).

Two facts about the simulator noted on the way, neither used by the mma path: `utils.cpp::bfloat16_to_gfloat_tensor` decodes a
BF16 subnormal with exponent `-127` and no forced `exp = 1` (unlike `multiply_bfloat16_to_gfloat`), and `Gfloat`'s default
constructor carries Hopper's `-133`.
