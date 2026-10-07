# The descriptor v1 evaluator in Lean

`verity.ml.circuits.evaluation.reference.evaluate` on a Program, as executable Lean: the Lake package `verity_evaluate`
(lib `Evaluate`, exe `verity-evaluate`). It decodes a descriptor v1 Program ([format](../../../../ml/circuits/PROTOCOL.md)),
binds the run inputs, evaluates every gate in canonical order and returns the root's outputs, or refuses. The Program is read
through C-Flock's verifier (`Flock.Program`, required by path from `verity/core/service/flock`), which evaluates no primitive;
this package adds the primitives. It is code, so it requires nothing of `Security/`.

- `Evaluate/Prims.lean`: the primitives implemented, each with the signature the registry holds for it. E0 is the Boolean
  profile (§9: `And<n>_v1`, `Xor<n>_v1`, `Not_v1`) and `Input<w>_v1`; E1 the constant family `Const<w>[0x<hex>]_v1` and the
  scalar primitives. Every other primitive, and `RegisteredInput<w>_v1`, is refused by name.
- `Evaluate/Scalar.lean`: the scalar primitives (`verity.ml.circuits.ops.scalar`, and the vLLM integration's `I32Sub`,
  `I64ToI32`, `U32And`, `U32Xor`, `BitXor`, `SelectBit`), transcribed branch for branch from the Python. Each names the
  definition of `Security/Definitions/Core/Scalar.lean` it must equal, or that Security has none yet; the equalities are
  proved in `Security/Proofs`, as the certifier's are, since code never requires `Security/`.
- `Evaluate/Walk.lean`: the gate walk (`Evaluate.evaluate`).
- `Evaluate/Serve.lean`: `Evaluate.evaluateFn`, which `verity-evaluate` serves through the difftest harness's line protocol
  (`Difftest.serve`, [harness](../../difftest/README.md)): `{"fn": "evaluate", "input": {"program": <descriptor>, "inputs":
  [[word, ...], ...]}}` gives `{"output": [word, ...]}` or `{"reject": "<reason>"}`, with one list of words per parameter of the
  program function, each in leaf order.

~~~sh
cd verity/experimental/core/primitives/evaluate && lake build      # .lake/build/bin/verity-evaluate
python tools/verity/lean/check.py verity/experimental/core/primitives/evaluate   # the Lean check, from the repository root
uv run tools/verity/check/suites.py verity-core-evaluate           # the difftest against the Python
~~~

`tests/test_difftest.py` holds it to the Python reference through `verity_difftest`, output for output and refusal for
refusal: seeded random Boolean composites over every node form, drawn by a Hypothesis strategy with their inputs and with
mutations both sides refuse; each scalar primitive on the whole grid of its operands' edge words, on drawn words, and on
words out of range; every constant of the registry and the constant ids the Python reads; and explicit examples: the pinned
2-bit adder of `boolean_vectors.json`, Boolean Definitions traced from the builders (up to the Ampere BF16 tensor-core
step), and malformed programs and inputs.
