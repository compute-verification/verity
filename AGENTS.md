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
    codec. Stdlib only; imports no other core module. The partition checker's
    references live here: `cut` (the query `Q_word` v1's canonical evaluator:
    each Call's computing graph, the cut, the width rule in bits; the algorithm
    is written in its docstring), `partition.validate_unit_cut` (the invariant)
    and `partition_object` (`verity/partition/v1` = program digest + query:
    `evaluate`, `locate`, `validate`, `verify`; pinned by
    `tests/ir/partition_vectors.json`).
  - `verity.evaluation`: the one evaluator. The plain-integer reference
    (`evaluate`, the IR oracle), `evaluate_batch` over numpy arrays,
    `register_kernel`, `self_check`. Every registered kernel must be listed
    in `tests/evaluation/test_evaluation.py`, which self-checks it.
  - `verity.ml`: exact hardware semantics of the tensor-core accumulator step
    (`verity.ml.tc`, stdlib only) and their IR registrations (`prims`,
    `gemm`), with numpy kernels beside them (`kernels`; numpy is a core
    dependency, kept out of `verity.ir` and `verity.ml.tc` by
    `tests/test_boundaries.py`). `verity.ml.tc.relation` restates the step as witness columns +
    mul/linear/lookup predicates and counts them (`census()`). Tests replay
    the A100 / RTX 4090 / H100 captures under `tests/ml/fixtures/` (see its
    README for provenance).
 - `verity.randomness`: `derive(source, domain, context) -> Key`, the one
 challenge derivation, with exact samplers (`uniform`, `bernoulli`, `subset`)
 over counter-mode streams. Stdlib only, and it imports no other core module.
 Its byte formats are pinned by `tests/randomness/vectors.json`.
 - `verity.proofs`: single statements and how a backend is judged on them,
 plus `proofs.profile.IntegrityProfile`, which records what verification
 established: the exhaustive facts, a verifier's law over named levels of a
 partition hierarchy (`verity.ir.partition.Level`) and δ. Consumers read
 it with their own utilities; their readers stay out of core. Its tests live
 in `tests/proofs/`.
 - `verity.commitments`: see its module docstring.
 - `verity.claims`: the ids of security claims (guarantees, assumptions as `property/instance`
 such as `cr/sha-512`, models, setup) that Table 1 cites; the docs site imports them from the
 entities render. Stdlib only; imports no other core module.
 Other modules are created when code lands, not before.
 - Tests live in `packages/verity/tests/<module>/`.
- `protocols/`: one distribution per protocol, and each imports only the
 stdlib, `verity` and itself (`protocols/tests/test_protocol_boundaries.py`).
 The first is `protocols/sampled_proofs` (`verity_sampled_proofs`): the
 two-stage law, its beacon-keyed draws, and verification plans. See its
 `PROTOCOL.md`. `protocols/one_stage` (`verity_one_stage`) is the one-stage
 audit: registration, the verifier's draw over a partition's units, and the
 audit record with its profile; `benchmarks/one_stage` runs it end to end with
 M0's prover and the Lean verifier. Applications (vLLM) compose protocols.
- `census/`: data only, meant to move to its own repository: `hardware.json`, `datatypes.json`, `networks.json`,
 `subcircuits.json`, `input_sets.json`, `workloads.json` (stable ids, sources), a JSON Schema for each under `schemas/`,
 and a README with the id scheme and sourcing rules. Nothing imports it; the renderer and benchmarks
 read it with `verity_numerical.bench.census` (a few lines), the docs site reads the same files, and
 core never reads it (`test_boundaries.py`): core takes such numbers as parameters.
- `tools/research/`: development infrastructure for running, observing, and
  coordinating recorded work (`research run | status | inspect | fetch |
  compare`). Separate distribution; must not import `verity`, and a test
  enforces it. Developer notes in its `README.md`.
- `tools/circuit_check/`: `circuit-check`, the property checks every circuit passes (see Circuits below).
- `tools/check/`: `check`, the one repository check a merge requires (see Checking and merging below).
- `backends/`, `frontends/`, `integrations/`, `benchmarks/`: created when a
  component has real code. The first is `backends/sp1`; its `README.md` is
  the how-to (build, identity, Python API, tests, recorded pod runs).
  **SP1 proving never runs on a laptop or the control pod**: use
  `research run --on vy-sp1 ...` (`~/.research/machines.toml`).

Dependency direction (README "Development"): `ir <- ml`, `ir <- commitments`,
`ir + commitments <- proofs`, everything else depends inward on the
core. Backends must be testable without a frontend or runtime integration.

~~~text
uv sync                                   # workspace root; installs research and verity editable
uv run pytest                             # all tests
uv run python tools/check/check.py        # the check a merge requires; --record publishes it as an Attempt
~~~

## Circuits

Any change that adds or modifies a circuit (a Definition, a subcircuit template, a Boolean lowering) must pass
`circuit-check` (`uv run circuit-check <definition|template|row>`); include its report in the PR. A new Definition needs a
small binding in `circuit_check.targets`, which the suite enforces.

## Checking and merging

There is no GitHub Actions CI. `check` (`tools/check/check.py`) is the one check: pytest, `circuit-check --all`, and the Lean
verifier's build with its upstream cross-check once PR #85 is on `main`. `--record` runs it through `research run` as an
Attempt of the checked-out clean commit. Merge into `main` only through `research merge <branch>`, which refuses without a
passing `check` of that exact commit in the evidence store, or when `main` has moved past it.

## Working principles

Solve the current task with the smallest clear implementation. Reuse
existing code and commands before adding abstractions or procedures. Add
shared infrastructure when a concrete use case, recurring failure, or
material risk justifies it; do not build for speculative requirements.

Use the Glossary's terms (`README.md`); when a term changes, update the Glossary
and roll it out in the same change.

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
- Evidence goes through `research run --tool … --campaign …`: every completed or failed run is auto-published as an Attempt in the evidence store (`~/.research/store`, `research data …`; contract in `tools/research/src/research/store/README.md`).
- Findings and claims about a run are labels, never edits to a shared table: `research data label <run id|art:…> <key> <value> --by <lane> --ref <run id>` (a red-team verdict is another label by the red-team; tables pick a policy, `store_tables.py --proof-class-by 'red-team-*'`).
- The ledger `backends/numerical/reports/ledger/*.jsonl` is frozen history (imported once by `ledger_to_store.py`); new rows are labels on the attempt they describe.
- Repo-replicated inputs are registered in `fixtures/artifacts.json` (`{art id: path}`, `research data put --kind fixture/v1 --file …`); `tools/research/tests/test_repo_replicas.py` checks every entry hashes to its id.
- `tests/test_repository.py` rejects report-genre files (`*HANDOFF*`, `*REPORT*`, `*URGENT*`, `*DRAFT*`, `*TRIAGE*`, `*STATUS*`, anything under `reports/` or `results/`), and `.gitignore` keeps them out of `git add`.

## Lanes (worker agents)

If your launch message names you a lane, `~/.research/notes/kb/LANE-CONTRACT.md` is your rulebook. Keep chatter low
(contract §3a): every message and every background shell you start reaches the coordinator and the human, so run long
jobs detached on the pod, write one-line checkpoints, and hand off only to unblock or to change someone's plan.

Push your branch after every commit: `research notes push <you>` (never `main`, never `--force`). Only the research
coordinator merges `main`, from origin and through `research merge`; `research notes branches --fetch` lists open lanes' work
origin lacks.

## Skills

When creating or modifying a skill in `.agents/skills/`, use
`.agents/skills/authoring-skills/SKILL.md`.
