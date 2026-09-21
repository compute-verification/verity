# Verity

A common representation of computation, proofs about that computation,
mappings from real systems into the representation, and accounting of the
physical resources those systems use. Monorepo; structure is being decided.
Code migrates incrementally from `veritor`.

## Map and commands

- `tools/research/`: shared tooling for running, observing, and coordinating
  computational work (`research run | status | inspect`). Imports nothing
  from `packages/`. Developer notes in its `README.md`.
- `packages/`, `adapters/`, `benchmarks/`: reserved; nothing lives there yet.

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
