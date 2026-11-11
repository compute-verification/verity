# verity.definitions.silicon

The GPUs' exact semantics as integer arithmetic on encodings: one tensor-core accumulator step per instruction class, the
scalar FP32 pipe and the casts, and the same steps as exact GF(2) circuits, which the Boolean Definitions are built from.
The semantics import nothing but the standard library. The package's docstring lists every module; the core API:

- `models.MODELS`: the step semantics, `GroupSum` (grouped truncating accumulation) and `BlockScaledAlignAdd` (the
  FP4 path with scale groups), each `model.step(acc_word, a_words, b_words) -> acc_word'`. `total` and `total_fp8` give the
  total step on every encoding (NaN, infinities, saturation), and `integer.IntegerMma` the integer mma step.
- `instructions.INSTRUCTIONS`: for each PTX instruction class, the model it uses and its status: pinned (the model matched
  every hardware probe), hypothesis or unprobed, with the evidence.
- `term`, `fp32` and `cast`: the operand formats' exact decoders and products, IEEE binary32 on words, and the word
  casts to and from FP32.
- `tc_step`, `fp4` and `fp`: a `GroupSum` step, an NVFP4 `BlockScaledAlignAdd` step and f32 / bf16 arithmetic as exact
  GF(2) circuits over `verity.circuits.boolean.forms`.

The models are held to the hardware: [`captures/`](captures/README.md) holds replicas of evidence measured on silicon
elsewhere, [`../ops/tests/fixtures/`](../ops/tests/fixtures/README.md) the captures the tests replay, `hawkeye/PIN.json`
the pinned source-close reference, and `discrepancy_log.json` each place where the paper, the pinned simulator, the
reference and the hardware disagree, open until its resolution names its evidence. `benchmarks/tc_probe/` and
`benchmarks/tc_probe_fp4/` capture the hardware.

## Tests

`verity/definitions/ops`'s suite tests this package (`uv run tools/verity/check/suites.py verity/definitions/ops`).
