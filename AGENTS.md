# Verity

A common representation of computation, proofs about that computation,
mappings from real systems into the representation, and accounting of the
physical resources those systems use. Monorepo; structure is being decided.
Code migrates incrementally from `veritor`.

## Map and commands

- `tools/research/`: shared tooling for running, observing, and coordinating
  computational work (`research run | status | inspect`). Imports nothing
  from `packages/`. Developer notes in its `README.md`.
- `packages/ir/` (`verity_ir`): the IR core — types, definitions, canonical
  layout, evaluator, descriptor codec. Stdlib only; knows no semantics package.
- `packages/ml/` (`verity_ml`): exact hardware semantics of the tensor-core
  accumulator step (`verity_ml.tc`, stdlib only) and their IR registrations
  (`prims`, `gemm`). `verity_ml.tc.relation` restates the step as witness
  columns + mul/linear/lookup predicates and counts them (`census()`): the
  input to the proof-backend decision. Tests replay the A100 / RTX 4090 / H100
  captures under `packages/ml/tests/fixtures/` (see its README for provenance).
- `adapters/`, `benchmarks/`: reserved; nothing lives there yet.

~~~text
uv sync            # workspace root; installs every member editable
uv run pytest      # all tests
~~~

## Working principles

Solve the current task with the smallest clear implementation. Reuse
existing code and commands before adding abstractions or procedures. Add
shared infrastructure when a concrete use case, recurring failure, or
material risk justifies it; do not build for speculative requirements.

Enforce deterministic requirements in code where practical. Use skills for
recurring, non-obvious workflows, not as a substitute for tooling or as a
reason to add approval steps.

Make ordinary reversible engineering decisions independently. Surface changes
to agreed semantics, acceptance criteria, material spending, or destructive
behavior.

Architectural rationale and human-facing reports belong in Notion;
experimental evidence belongs in artifact storage. Temporary working notes
need not become permanent documentation. Remove obsolete rules and machinery
rather than accumulating them.

## Skills

When creating or modifying a skill in `.agents/skills/`, use
`.agents/skills/authoring-skills/SKILL.md`.
