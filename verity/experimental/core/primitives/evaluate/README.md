# The descriptor v1 evaluator in Lean

`verity.ml.circuits.evaluation.reference.evaluate` on a Program, as executable Lean: the Lake package `verity_evaluate`
(lib `Evaluate`, exe `verity-evaluate`). It decodes a descriptor v1 Program ([format](../../../../ml/circuits/PROTOCOL.md)),
binds the run inputs, evaluates every gate in canonical order and returns the root's outputs, or refuses. The Program is read
through C-Flock's verifier (`Flock.Program`, required by path from `verity/core/service/flock`), which evaluates no primitive;
this package adds the primitives. It is code, so it requires nothing of `Security/`.

- `Evaluate/Prims.lean`: the primitives implemented, each with the signature the registry holds for it. E0 is the Boolean
  profile (§9: `And<n>_v1`, `Xor<n>_v1`, `Not_v1`, `Const1[0x0]_v1`, `Const1[0x1]_v1`) and `Input<w>_v1`; every other
  primitive, and `RegisteredInput<w>_v1`, is refused by name.
- `Evaluate/Walk.lean`: the gate walk (`Evaluate.evaluate`).
- `Evaluate/Serve.lean`: the line protocol, one JSON request per line on stdin, one answer per line on stdout:
  `{"fn": "evaluate", "input": {"program": <descriptor>, "inputs": [[word, ...], ...]}}` gives `{"output": [word, ...]}` or
  `{"reject": "<reason>"}`, with one list of words per parameter of the program function, each in leaf order.

~~~sh
cd verity/experimental/core/primitives/evaluate && lake build      # .lake/build/bin/verity-evaluate
python tools/verity/lean/check.py verity/experimental/core/primitives/evaluate   # the Lean check, from the repository root
uv run tools/verity/check/suites.py verity/experimental/core/primitives/evaluate # the difftest against the Python
~~~

`tests/test_difftest.py` holds it to the Python reference, output for output and refusal for refusal: the pinned 2-bit adder
of `boolean_vectors.json`, Boolean Definitions traced from the builders (up to the Ampere BF16 tensor-core step), seeded
random Boolean composites over every node form, and malformed programs and inputs.
