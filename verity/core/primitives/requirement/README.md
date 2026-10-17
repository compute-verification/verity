# A declared circuit's requirement in Lean

`Accounting.Requirement` (`Security/Definitions/Accounting/Requirement.lean`) as executable Lean: the Lake package
`core_requirement` (lib `CircuitRequirement`, exe `requirement-difftest`). On one instance (a declared circuit's wiring, one
scheduled execution of it and a layout of its values) and the audit's store, it computes each step's live and read values,
their bits by value and by block, the resident blocks, `held` and `read` over a window, and whether the store and a step's
plan are W's (`StoreMatches`, `PlanMatches`); and, the execution placed on devices (each value's device and the tick at
which each device begins each engine step), PoUW's regions, ingress and credits over it (`RequirementCredits`,
`RequirementHelp`). A step is always an engine step. It is code, so it requires nothing of `Security/`: the Lean check's
`security_properties` package states that each function is the definition it computes (`Properties.Accounting.Requirement`)
and proves it (`Properties.Proofs.Accounting.Requirement`).

- `CircuitRequirement.lean`: `Instance`, `Store`, `Placement` and the functions, on lists with membership through hash sets, so a
  registry row of tens of thousands of gates runs in seconds.
- `CircuitRequirement/Difftest.lean`: `requirement`, which `requirement-difftest` serves through the difftest harness's line
  protocol (`Difftest.serve`, [harness](../../difftest/README.md)): `{"fn": "requirement", "input": {"width", "registered",
  "ops", "uses", "step", "value", "block", "offset", "store", "plans", "block_bits", "steps", "leaf", "ticks", "price",
  "windows"}}` gives each step's values, bits, blocks and verdicts and each window's region (at its ticks and at its
  engine steps), ingress and credits, or `{"reject": "<reason>"}` for a malformed instance or an execution that doesn't
  follow its wiring. Payloads travel as lowercase hex.

~~~sh
cd verity/core/primitives/requirement && lake build                 # .lake/build/bin/requirement-difftest
python tools/verity/lean/check.py verity/core/primitives/requirement   # the Lean check, from the repository root
uv run tools/verity/check/suites.py verity/core/primitives/requirement # the difftest against the Python
~~~

`tests/test_requirement_difftest.py` holds it to `verity.ml.circuits.requirement` through `verity_difftest`, output for
output and refusal for refusal: drawn wirings with executions that follow them and some that don't, under `pack`'s layout
and store, a mutation of them or a random one, placed on a few devices with PoUW's windows over them, and malformed
instances; then Programs end to end through `requirement.execution`, `pack` and a placement on three devices: the toy
decoder, `b1.Serve` at the vLLM tests' tiny configuration and
`b1.RMSNormFusedCudaV1` at N = 64.
