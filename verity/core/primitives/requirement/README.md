# A declared circuit's requirement in Lean

`Accounting.Requirement` (`Security/Definitions/Accounting/Requirement.lean`) as executable Lean: the Lake package
`core_requirement` (lib `CircuitRequirement`, exe `requirement-difftest`). On one instance (a declared circuit's wiring, one
scheduled execution of it and a layout of its values) and the audit's store, it computes each step's live and read values,
their bits by value and by block, those in complete blocks (every value produced, and the step at or after the block's
first encoded read, `Layout.encoded`: memory-map/v1's deferred residency), which the audit credits (`credited`,
`creditedReads`), the resident blocks, `held` and `read` over a window, and whether the store and a step's plan are W's (`StoreMatches`,
`PlanMatches`: the credited values' blocks are resident, and planned for the steps that read them, and other blocks, filler
among them, may be too, earning W nothing); and, the execution placed on devices (each value's device and the tick at
which each device begins each engine step), PoUW's regions, ingress and credits over it when the node runs it as placed
(`RequirementCredits`, `RequirementHelpBound`, and `RequirementHelp` when the wiring is `topological`), a run input an
input gate wherever it is placed. A step is always an engine step. It is code, so it requires nothing of `Security/`: the Lean check's
`security_properties` package states that each function is the definition it computes (`Properties.Accounting.Requirement`)
and proves it (`Proofs.Properties.Accounting.Requirement`).

- `CircuitRequirement.lean`: `Instance`, `Store`, `Placement` and the functions, on lists with membership through hash sets, so a
  registry row of tens of thousands of gates runs in seconds.
- `CircuitRequirement/Difftest.lean`: `requirement`, which `requirement-difftest` serves through the difftest harness's line
  protocol (`Difftest.serve`, [harness](../../difftest/README.md)): `{"fn": "requirement", "input": {"width", "registered",
  "inputs", "ops", "uses", "step", "value", "block", "offset", "encoded", "store", "plans", "block_bits", "steps", "leaf",
  "ticks", "price", "windows"}}` gives whether the wiring is topological, each step's values, bits, blocks and verdicts and each window's
  region (at its ticks and at its engine steps), ingress and credits, or `{"reject": "<reason>"}` for a malformed
  instance, an execution that doesn't follow its wiring, or, asked for a window, a placement that leaves a value without
  a device or a gate without a tick at its step. Payloads travel as lowercase hex.

~~~sh
cd verity/core/primitives/requirement && lake build                 # .lake/build/bin/requirement-difftest
python tools/verity/lean/check.py verity/core/primitives/requirement   # the Lean check, from the repository root
uv run tools/verity/check/suites.py verity/core/primitives/requirement # the difftest against the Python
~~~

`tests/test_requirement_difftest.py` holds it to `verity.ml.circuits.requirement` through `verity_difftest`, output for
output and refusal for refusal: drawn wirings with executions that follow them and some that don't, under `pack`'s layout
and store, a KV pool's (`paged`, whose blocks fill over several steps and credit nothing until complete), either with its
coins back late (`deferred`, each block credited from its first encoded read, withdrawn when its values are dead by
then), a mutation of `pack`'s or a random one, with run inputs, placed on a few devices with PoUW's windows over them (some bounding their
steps from inside or outside, `RequirementStepsInner` and `RequirementStepsOuter`, some holding run inputs), and
malformed instances; then Programs
end to end through `requirement.execution`, `pack` and a placement on three devices: the toy decoder, `b1.Serve` at the
vLLM tests' tiny configuration and `b1.RMSNormFusedCudaV1` at N = 64.

`tests/test_census_and_memory_map.py` holds two more producers to the same Lean through the same instance format:
`requirement.from_census` on those Programs' censuses (a census value per value, a value of 0 bits per scheduled root
Call), and memory-map/v1 as the Commit's builder (`verity_vllm.commit.memory_map`) records it, a session's KV rows in the
blocks it wrote under the free convention of 21:52Z (step s's record takes effect after step s's reads). The audit
credits the live rows in complete blocks, and those it doesn't, a page's first rows until its last is born, lie in one
block per request, layer and plane (`OneBlockEach`). The builder frees each block in the record of its rows' last
structural reader, a gather that may select one included, and leaves out a block no step reads while it is resident
(one whose last structural reader is the step that completes it), and the link holds at every step. With r0's last step
selecting its first page's rows instead of reading them (a stand-in census), the builder's free is unchanged and the
link holds, and the same map freeing the page at the rows' last use fails `StoreMatches` at that step alone. The link
allows a dead block resident, so the tests check separately that the map holds none.
With the session's coins back two steps late (`deferred`), each block is credited from its first encoded read, two are
withdrawn, and the uncredited rows lie in one block per request, layer, plane and page parity.
