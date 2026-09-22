# Verity

Representing computations precisely, verifying claims about them, and
deriving compute/memory/network resource guarantees. The architecture and
component boundaries are in `README.md`; that document is settled, read it
first. Code migrates incrementally from `veritor` as real consumers need it;
do not bulk-migrate or scaffold directories the README merely anticipates.

## Map and commands

- `packages/verity/`: the core, one ordinary distribution `verity`. Import
  from the owning module (`from verity.ir import Program`), never a
  convenience re-export at `verity`.
  - `verity.ir`: types, definitions, canonical layout, evaluator, descriptor
    codec. Stdlib only; imports no other core module.
  - `verity.ml`: exact hardware semantics of the tensor-core accumulator step
    (`verity.ml.tc`, stdlib only) and their IR registrations (`prims`,
    `gemm`). `verity.ml.tc.relation` restates the step as witness columns +
    mul/linear/lookup predicates and counts them (`census()`). Tests replay
    the A100 / RTX 4090 / H100 captures under `tests/ml/fixtures/` (see its
    README for provenance).
  - `verity.commitments`, `verity.verification`, `verity.resource_accounting`:
    created when code lands, not before.
  - Tests live in `packages/verity/tests/<module>/`.
- `tools/research/`: development infrastructure for running, observing, and
  coordinating recorded work (`research run | status | inspect | fetch |
  compare`). Separate distribution; must not import `verity`, and a test
  enforces it. Developer notes in its `README.md`.
- `backends/`, `frontends/`, `integrations/`, `benchmarks/`: created when a
  component has real code. The first is `backends/sp1`; its `README.md` is
  the how-to (build, identity, Python API, tests, recorded pod runs).
  **SP1 proving never runs on a laptop or the control pod**: use
  `research run --on vy-sp1 ...` (`~/.research/machines.toml`).

Dependency direction (README "Development"): `ir <- ml`, `ir <- commitments`,
`ir + commitments <- verification`, everything else depends inward on the
core. Backends must be testable without a frontend or runtime integration.

~~~text
uv sync            # workspace root; installs research and verity editable
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

## Where writing goes

- This repo: code, tests, small fixtures, and maintained current-truth docs only (`README.md`, `AGENTS.md`, `PROTOCOL.md`, `DESIGN.md`, `DISCREPANCIES.md`, `SKILL.md`, `CHANGELOG.md`, `*.template.md`).
- Lane reports, handoffs, findings, drafts: markdown under `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-<kind>-<slug>.md` with YAML front-matter (`id`, `campaign`, `lane`, `kind: report|handoff|finding|draft`, `status`, `repo`, `origin`); cite them from here by id as `note:<id>`.
- Evidence (explore JSON, plots, gate outputs, run results): `~/.research/notes/campaigns/<campaign>/assets/<lane>/...`, cited as `notes-asset:<path>`; the overhead ledger (`backends/numerical/reports/ledger/*.jsonl`) stays here for now.
- Architecture decisions: Notion (the coordinator publishes); no `docs/adr/` in the repo.
- `tests/test_repository.py` rejects report-genre files (`*HANDOFF*`, `*REPORT*`, `*URGENT*`, `*DRAFT*`, `*TRIAGE*`, `*STATUS*`, anything under `reports/` or `results/`), and `.gitignore` keeps them out of `git add`.

## Skills

When creating or modifying a skill in `.agents/skills/`, use
`.agents/skills/authoring-skills/SKILL.md`.
