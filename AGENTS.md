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
    codec. Stdlib only; imports no other core module. The program format
    (descriptor v1, canonical JSON = RFC 8785, layout, scope resolution) is
    specified in `verity/ir/PROTOCOL.md`, every rule pinned by
    `tests/ir/format_vectors.json`. The partition checker's
    references live here: `cut` (the query `Q_word` v1's canonical evaluator:
    each Call's computing graph, the cut, the width rule in bits; the algorithm
    is written in its docstring), `partition.validate_unit_cut` (the invariant)
    and `partition_object` (`verity/partition/v1` = program digest + query,
    `Q_word` v1 or the instance-granular `Q_template_instance(s)` v0:
    `evaluate`, `locate`, `validate`, `verify`; pinned by
    `tests/ir/partition_vectors.json` and `template_instance_vectors.json`).
    `units` cuts each proof unit of a partition out as a standalone circuit,
    with the shape that units of equal circuits share.
    `annotations` is `verity/annotations/v0`: names, descriptions, formats and
    groupings keyed by descriptor ids, validated against a descriptor's
    structure, never inside a digest.
    `boolean` is the Boolean profile of descriptor v1 (PROTOCOL §9): the gates
    `And<n>_v1`, `Xor<n>_v1` (any n >= 2), `Not_v1`, the constant bits,
    `is_boolean` and the word view (`word_type`, `bit_type`).
  - `verity.evaluation`: the one evaluator. The plain-integer reference
    (`evaluate`, the IR oracle, which also takes a table held in a registered
    input lazily, as a `RegisteredTable`), `evaluate_batch` over numpy arrays,
    `register_kernel`, `self_check`. Every registered kernel must be listed
    in `tests/evaluation/test_evaluation.py`, which self-checks it.
    `bits` evaluates Boolean Definitions bit-sliced (`evaluate_bits`, and
    `evaluate_words` through the word view); `evaluate_batch` uses it for one
    without a kernel.
  - `verity.ml`: exact hardware semantics of the tensor-core accumulator step
    (`verity.ml.tc`, stdlib only) and their IR registrations (`prims`,
    `gemm`), the other IR primitives (`scalar`: integer, bit, select and
    comparison operations; `fp32`: FP32 add, mul, FMA and divide, their
    flush-to-zero forms, and BF16 element-wise arithmetic), the library of
    hardware and protocol objects core owns (`library`), the pinned list of
    operations with the guard that a program holds no data (`operations`), and numpy kernels
    beside them (`kernels`; numpy is a core dependency, kept out of `verity.ir`
    and `verity.ml.tc` by `tests/test_boundaries.py`). `verity.ml.boolean` is
    the word library as Boolean Definitions: C-Flock's linear-form builder and
    pieces (`forms`, `fp`, `tc_step`, `softmax`, `fp4`), the multiplexer tree (`gather`),
    `trace.definition` (a builder traced into And/Xor/Not gates, with `word_of`
    its word Definition), the BF16 dot, coordinate and GEMM and the E4M3, E5M2,
    NVFP4 and MXFP4 steps and coordinates on bits (`gemm`: `DotBf16_v3`,
    `GemmCoordinate_v3`, `Gemm_v3`, `GemmCoordinateE4m3_v2`, `GemmCoordinateNvf4_v2`,
    `GemmCoordinateMxf4_v2`), attention's softmax operations and inverse sum on bits
    (`attention`), the element-wise FP32, BF16 and I32 operations on bits
    (`elementwise`), the bf16 widening on bits (`cast`), and the bf16 comparator,
    the selects, PTX max / min and the E4M3 cast on bits (`scalar`). `verity.ml.tc.relation`
    restates the step as witness columns +
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
 stdlib, `verity`, itself and the protocols Daniel has allowed it (PoUW imports sampled proofs;
 `EDGES` in `tests/test_protocol_boundaries.py`).
 The first is `protocols/sampled_proofs` (`verity_sampled_proofs`): the one-stage
 audit (`one_stage`: registration, the verifier's draw over a partition's proof units, and the
 audit record with its profile), the two-stage law, its draws from the verifier's own
 randomness, and verification plans. The Lean states it; its `README.md` signposts that
 Lean and lists what it does not state yet. `benchmarks/one_stage` runs the one-stage audit end to end with
 C-Flock's circuit prover (M0, `verity_flock.circuit`, run from the pinned tree
 that `--m0-pythonpath` names, not from the checkout) and the Lean
 verifier. `protocols/pous` (`verity_pous`) is proof of useful space: the reference
 codecs, setup and timed audit, its `PROTOCOL.md`, and its Lean package (`lean/`: the
 statements of its guarantees, their proofs and a submission grader). `protocols/pouw` (`verity_pouw`)
 is proof of useful work: the `PoUWScheme` interface (`protocol`), its lifecycle
 (`audit`) and the named schemes (`schemes.get`: NCP, Pearl's FP8 scheme, Pearl-C), each with
 pinned vectors, `circuit`, `ncp-v2` as a Program that sampled proofs verifies, and its Lean
 package (`lean/`, whose `Pouw.SecurityProofs.EndToEnd` is the guarantee); see its `PROTOCOL.md`.
 `protocols/network_warden` (`verity_network_warden`) is network transparency's timing
 channel: the warden, its audit and calibration, and its Lean package (`lean/`).
 `protocols/nci` is novel communication intensity, the policy that throttles frontier training
 and leaves inference alone; so far only its Lean package (`lean/`).
 Applications (vLLM) compose protocols.
- `census/`: data only, meant to move to its own repository: `hardware.json`, `datatypes.json`, `networks.json`,
 `subcircuits.json`, `input_sets.json`, `workloads.json` (stable ids, sources), a JSON Schema for each under `schemas/`,
 and a README with the id scheme and sourcing rules. Nothing imports it; the renderer and benchmarks
 read it with `verity_numerical.bench.census` (a few lines), the docs site reads the same files, and
 core never reads it (`test_boundaries.py`): core takes such numbers as parameters.
- `tools/research/`: development infrastructure for running, observing, and
  coordinating recorded work (`research run | status | inspect | fetch |
  compare`). Separate distribution; must not import `verity`, and a test
  enforces it. Developer notes in its `README.md`. Code reaches the Nebius
  nodes only through `research deploy install` from a pushed commit, listed in
  `pods/nebius/deploy.toml`; `research deploy drift` names what isn't `main`'s.
  Delete data you don't own on a node only with `research retention gc` or `research retention rm --approved-by @OWNER`, never a bare `rm`.
- `tools/circuit_check/`: `circuit-check`, the property checks every circuit passes (see Circuits below).
- `tools/check/`: `check`, the one repository check a merge requires (see Checking and merging below).
- `tools/lean/`: the Lean audit (`audit.py`), which `check` runs on every Lake package (see Lean below).
- `tools/cluster/`: `cluster`, the shared machines' description, a pure per-node planner and the allocation ledger. Stdlib
 only, imports neither `verity` nor `research`. On vy-nebius-2 its agent runs in shadow mode (read-only, beside
 `gpu-lease`); nothing it decides is carried out yet (its `README.md`).
- `backends/`, `frontends/`, `integrations/`, `benchmarks/`: created when a
  component has real code; there is no `frontends/` yet. `backends/flock`
  (`verity_flock`) is C-Flock, the backend in active development; its verifier
  of record is the Lean one under `backends/flock/verifier/` (see Lean below).
  A-GKR (`backends/gkr`), B-Ligero (`backends/direct`, `backends/ligero-verify`,
  `backends/ligerito-verify`) and D-SP1 (`backends/sp1`) are frozen: kept, and
  still in the benchmark tables, but not developed; so is the VOLE research
  backend (`backends/vole`). `backends/numerical` holds the numerical references
  and the benchmark tooling that renders the tables. `backends/sp1/README.md` is
  D-SP1's how-to (build, identity, Python API, tests, recorded pod runs).
  **SP1 proving never runs on a laptop or the control pod**: use
  `research run --on vy-sp1 ...` (`~/.research/machines.toml`).
  `integrations/vllm` (`verity_vllm`) is the vLLM integration.

Dependency direction (README "Development"): `ir <- evaluation <- ml`,
`ir <- commitments`, `ir + ml + commitments <- proofs`; `randomness` and
`claims` import no core module; everything else depends inward on the core
(`packages/verity/tests/test_boundaries.py`). Backends must be testable
without a frontend or runtime integration.

~~~text
uv sync --all-packages --extra torch-cpu  # the whole workspace, editable, with CPU-only torch (GPU work: a member's `torch` extra)
uv run tools/check/suites.py              # every test suite whose inputs changed since it last passed; --quick skips slow tests
uv run --extra torch-cpu python tools/check/check.py  # the check a merge requires; --record publishes it as an Attempt
~~~

## Tests

Every package's tests live in its own `tests/` and run as one suite, in its own process (a `parallel` suite in pytest-xdist
workers), cached by input; the repository's own suite (`tests/`) holds only invariants of the whole tree. How to run them,
what a suite may read (`[tool.verity.tests] inputs`), and the rules for slow tests, fixtures and dummy credentials:
`.agents/skills/testing/SKILL.md`.
No test depends on wall-clock time (no sleeps, clock readings or timeouts; wait on the event itself or inject the clock):
`tests/test_no_wall_clock.py` enforces it, and its `ALLOWED` list, one reason per entry, holds the exceptions.
A test that adds more than about a minute to `check` says in its docstring what it catches that a cheaper test can't.

## Circuits

Any change that adds or modifies a circuit (a Definition, a subcircuit template, a Boolean lowering) must pass
`circuit-check` (`uv run circuit-check <definition|template|row>`); include its report in the PR. A new Definition needs a
small binding in `circuit_check.targets`, which the suite enforces.

## Checking and merging

There is no GitHub Actions CI. `check` (`tools/check/check.py`) is the one check: every test suite (`tools/check/suites.py`),
`circuit-check --all`, C-Flock's Rust build and tests on its patched checkout, the Lean verifier's build and its partition check
against `verity.ir`, the Lean audit, and the verifier's cross-check against upstream's Rust verifier, at once on a multi-core
CPU pod, with caches that hit only on identical inputs. `--record` runs it through `research run` as an Attempt of the checked-out clean commit. Merge into `main`
only through `research merge <branch>`, which refuses without a passing `check` of that exact commit in the evidence store, or
when `main` has moved past it; `research merge --train A B C --on POD --project verity` lands several at once on one `check`
of the merged tree. The lander checks a train-prep commit with `tools/check/train.sh COMMIT POD` (`--agreement` for a pinned
guarantee changed outside `backends/flock/`). A change under `backends/flock/` also needs `check`'s agreement with upstream's verifier
(`lean-agreement`, on the build pinned in `backends/flock/verifier/upstream.json`; not for a `README.md` or `PROTOCOL.md`
alone, and skipped by name elsewhere): record it with
`uv run python tools/check/check.py --record --on POD`, which sends that build when the commit touches `backends/flock/`.
`check` reuses a pass (a test suite, a circuit-check target, a Lean package's audit) only while every input it read is
unchanged, and keeps each pod's Lean dependencies warm under their manifest and toolchain.
A pure move goes through the same `check`, but once `uv run python tools/check/move_check.py BASE HEAD` certifies it (only
byte-identical renames, and edits that rewrite moved paths or imports in path-bearing files) the lander may schedule it
first in a quiet window, alone in its train (Daniel, 4 Oct).

## Lean

- Soundness is one theorem over everything the compiled Lean verifier accepts, for arbitrary Boolean circuits, with no
  hypothesis on the circuit file or the statement's form. The verifier refuses any form the proof doesn't cover, and gains
  no new form ahead of its proof. Circuit inputs and outputs use one canonical format proved in general, so a new workload
  needs no verifier or proof change (ruling 2026-10-04, `.agents/skills/friction/SKILL.md`).
- The Lean packages are `backends/flock/verifier/lean/` (the executable verifier, no dependencies), its `level3/` (proofs
  about the executable's definitions, Mathlib) and `soundness/` (the protocol's proofs, ArkLib), and
  `protocols/pous/lean/` (POUS's guarantees, their proofs and grader, Mathlib), `protocols/pouw/lean/` (PoUW's games,
  named assumptions and the proofs behind its γ, Mathlib and core's), `protocols/network_warden/lean/` (the network
  warden's timing-channel guarantees and their proofs, Mathlib), `protocols/nci/lean/` (NCI's policy and guarantees, on
  core's), and `packages/verity/lean/` (the model every protocol states its guarantees in: circuits, partitions,
  declarations and games; and core's own proofs about `verity`'s definitions, such as `verity.ml.tc`'s step relation;
  Mathlib only). New Lean goes into the package whose dependencies it needs, in a subdirectory and namespace of its own;
  a core-only proof goes into core's package, never a backend's. A protocol's package may require core's and other
  protocols' packages, never a backend's. All packages share one toolchain and
  one Mathlib revision (`tests/test_lean_packages.py`); bump them together, in a PR of their own.
- Core's and NCI's packages are laid out under four names: `Protocol` (definitions), `Assumptions` (named `Prop`s) and
  `Guarantees` (each a `def G : Prop`) are the trusted text, importing only one another, core's trusted text and
  Mathlib (`layers` in `lean-audit.json`); `SecurityProofs` holds every proof and lemma, with `SecurityProofs.G : G` for
  each guarantee proved. The other packages move to this layout one at a time.
- Every Lake package has a `lean-audit.json`, and `check` audits every declaration (`tools/lean/audit.py`): only `propext`,
  `Classical.choice` and `Quot.sound`; no `sorry`, `axiom`, `native_decide` or kernel bypass; an assumption is a named
  `Prop` in the package's assumptions module, taken as a hypothesis; every `.lean` file is built, and every declaration
  is replayed through the kernel.
- A guarantee (README "Glossary"), the theorem a ledger, a table or a PR cites as proved, is listed in `lean-audit.json`,
  and `audit.py --update` records what it says: its statement and every definition the statement reads. No axiom check
  can catch a weakened definition, so when a change to those records lands on `main`, Daniel gets a DM showing it (each
  changed statement before and after, each changed definition as it is now). Nobody signs off and nothing waits on it.
  A guarantee that only moved to another module or name sends nothing when `--update --moved MOVES.json` (old names to
  new) prints it as the same statement under its old names.
- A named assumption or hand-proved lemma that a pinned dependency could prove, because its statement can be written in
  that dependency's constants, has an entry in the package's `upstream` watch list. The audit fails when the pinned
  dependency proves a theorem about it that nobody has acknowledged (`tools/lean/upstream.py`). One stated over the
  package's own definitions (PoUS's `LemmaA`) gets no entry, since no upstream theorem can mention them, and neither does
  one about a single number (PoUS's `PrimeP16448`), whose entry would match every theorem about primes. A document's
  "status upstream" cites that output; it is never written by hand.
- Two agents never share a writable Lake build directory (`.lake`): parallel Lean work uses separate checkouts.
- How to work on Lean, and how parallel Lean lanes avoid conflicts: `.agents/skills/lean-proofs/SKILL.md`.

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

Make ordinary reversible engineering decisions independently, and when a tool,
rule or step gets in your way, fix it at the source if the fix is clearly yours
rather than working around it. Surface changes to agreed semantics, acceptance
criteria, material spending, or destructive behavior. When you can't tell which
side a change is on, ask Daniel through your coordinator, with your
recommendation, and record his ruling as a line here or in a skill so nobody
asks twice. The rulings so far, and friction notes for what you can't fix now:
`.agents/skills/friction/SKILL.md`.

Two objectives are open, and Daniel wants help with them while we develop the
codebase and the research harness: the right amount of cross-agent chatter and
note-taking, enough to drive self-improvement and process wins and to catch
inefficiencies but no more; and the right balance of autonomy, teamwork and
asking Daniel. Neither is settled, and agents are expected to use their judgment
and help tune both. Each ruling and each friction note is how a balance gets
codified (`.agents/skills/friction/SKILL.md`).

A guard never fails open, and an agent VM can be suspended at any moment, so
nothing that bounds spend or preserves results may depend on one: pods bound
themselves (`research pods create --max-hours`, under a line of the notes repo's
`budgets.toml`), and runs preserve their results from the pod.

Never print a secret's value. List environment variable names with `compgen -e` (or `env -0 | cut -z -d= -f1 | tr '\0' '\n'`),
never `env | cut`, `env | grep` or `printenv | …`: a multi-line value's later lines print as if they were names. A secret
is stored as one line (a key as base64, `NAME_B64`), and `tools/agent-guard/install.sh` puts a redacting `env` on the VM.

Architectural rationale and human-facing reports belong in Notion;
experimental evidence belongs in artifact storage. Temporary working notes
need not become permanent documentation. Remove obsolete rules and machinery
rather than accumulating them.

## Where writing goes

- This repo: code, tests, small fixtures, and maintained current-truth docs only (`README.md`, `AGENTS.md`, `PROTOCOL.md`, `DESIGN.md`, `DISCREPANCIES.md`, `SKILL.md`, `CHANGELOG.md`, `*.template.md`).
- Lane reports, checkpoints, findings, drafts, friction notes: markdown under `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-<kind>-<slug>.md` with YAML front-matter (`id`, `campaign`, `lane`, `kind: report|finding|draft|friction`, `status`, `repo`, `origin`); cite them from here by id as `note:<id>`. Notes are records, never messages: a handoff, an order or an ask goes on Slack (below), or to your own Cursor workers as a follow-up.
- Evidence and generated output (explore JSON, plots, gate outputs, logs, run results, rendered tables and reports): the evidence store only, never Git, neither this repo nor the notes. `research run` publishes a run's outputs; anything else goes in with `research data put --kind K --file|--tree P --preserve`; cite it as `art:<id>`. Tables and reports are rendered on demand from code plus the store. An old `notes-asset:<path>` resolves through the notes README.
- Architecture decisions: Notion (the coordinator publishes); no `docs/adr/` in the repo.
- Evidence goes through `research run --tool … --campaign …`: every completed or failed run is auto-published as an Attempt in the evidence store (`~/.research/store`, `research data …`; contract in `tools/research/src/research/store/README.md`).
- Findings and claims about a run are labels, never edits to a shared table: `research data label <run id|art:…> <key> <value> --by <lane> --ref <run id>` (a red-team verdict is another label by the red-team; tables pick a policy, `store_tables.py --proof-class-by 'red-team-*'`).
- The overhead ledger is frozen history, in the store with the run results the frozen backends' scripts read (`verity_numerical.bench.frozen`, at their former paths `backends/numerical/reports/ledger/*.jsonl`, `backends/*/results*`); every ledger claim is a label there (`ledger_to_store.py`), and new rows are labels on the attempt they describe.
- Fixtures are registered in `fixtures/artifacts.json` (`{art id: path}`, `research data put --kind fixture/v1 --file …`); `tools/research/tests/test_repo_replicas.py` checks every entry hashes to its id. A tracked file is at most 256 KiB (`tests/test_repository.py`); a larger fixture's path is gitignored and `research data fetch-fixtures` fetches it (the store README §7.3).
- `tests/test_repository.py` rejects report-genre files (`*HANDOFF*`, `*REPORT*`, `*URGENT*`, `*DRAFT*`, `*TRIAGE*`, `*STATUS*`) and generated output (anything under `reports/`, `results/`, `renders/`, `previews/`, `plots/`, `evidence/`, `assets/` or `outputs/`, image, PDF and HTML renders, `<YYYYMMDDTHHMMZ>-*` stamped files), and `.gitignore` keeps them out of `git add`. In the notes, `research notes sync` and the steward's snapshot leave `renders/`, `campaigns/*/assets/` and `lanes/*/evidence/` out, and the steward's renders go to the store.

## Lanes (worker agents)

If your launch message names you a lane, `~/.research/notes/kb/LANE-CONTRACT.md` is your rulebook. Keep chatter low
(contract §3a): every message and every background shell you start reaches the coordinator and the human, so run long
jobs detached on the pod, write one-line checkpoints, and message someone only to unblock or to change their plan. Before
starting work another agent could plausibly be doing, read the notes' `kb/onboarding.md`: check `research notes approaches`
and claim yours at your first checkpoint (`research notes claim`).

Push your branch after every commit: `research notes push <you>` (never `main`, never `--force`). Only the research
coordinator merges `main`, from origin and through `research merge`; `research notes branches --fetch` lists open lanes' work
origin lacks.

Slack (computeverification.slack.com) is for handle holders (coordinators and service agents) and named humans only; a worker asks its
own coordinator. The coordination API (ask, announce, 👀/✅/❌, who subscribes, the wake recipe): `.agents/skills/using-slack/SKILL.md`.

## Skills

When writing or launching a job for `research run` (its cwd, its outputs, `--declared-output`, scratch), use
`.agents/skills/writing-runs/SKILL.md`: custody uploads only the declared outputs and the run's own records in full.

When creating or modifying a skill in `.agents/skills/`, use
`.agents/skills/authoring-skills/SKILL.md`.

GPU kernel work (PoUW kernels on sm_120: what's been tried, gates, timing, recording every attempt, vLLM hookup):
`.agents/skills/kernel-engineering/SKILL.md`.

Before adding, changing or retiring a process (a rule here or in the lane contract, a gate in `check` or `research merge`,
a required review, a file every PR must update), assess its effect on research velocity with
`.agents/skills/process-design/SKILL.md`.
