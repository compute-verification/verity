# Verity

Representing computations precisely, verifying claims about them, and
deriving compute/memory/network resource guarantees. The architecture and
component boundaries are in `README.md`; that document is settled, read it
first. Do not scaffold directories the README merely anticipates.

## Map and commands

- `verity/`: the core, one regular package: the root's `pyproject.toml` is the `verity` distribution. Import
  from the owning module (`from verity.primitives.circuits import Program`), never a
  convenience re-export at `verity`. Every directory under `verity/` is a package, and its tests sit beside it as a suite
  of its own (a tests-only `pyproject.toml`).
  - `verity.primitives.circuits`: types, definitions, canonical layout, evaluator, descriptor
    codec. Stdlib only; imports no other core module. The program format
    (descriptor v1, canonical JSON = RFC 8785, layout, scope resolution) is
    specified in `verity/primitives/circuits/PROTOCOL.md`, every rule pinned by
    `verity/primitives/circuits/tests/format_vectors.json`. The partition checker's
    references live here: `cut` (the query `Q_word` v1's canonical evaluator:
    each Call's computing graph, the cut, the width rule in bits; the algorithm
    is written in its docstring), `partition.validate_unit_cut` (the invariant)
    and `partition_object` (`verity/partition/v1` = program digest + query,
    `Q_word` v1 or the instance-granular `Q_template_instance(s)` v0:
    `evaluate`, `locate`, `validate`, `verify`; pinned by
    `partition_vectors.json` and `template_instance_vectors.json` beside the format's).
    `units` cuts each proof unit of a partition out as a standalone circuit,
    with the shape that units of equal circuits share.
    `annotations` is `verity/annotations/v0`: names, descriptions, formats and
    groupings keyed by descriptor ids, validated against a descriptor's
    structure, never inside a digest.
    `boolean` is the Boolean profile of descriptor v1 (PROTOCOL §9): the gates
    `And<n>_v1`, `Xor<n>_v1` (any n >= 2), `Not_v1`, the constant bits,
    `is_boolean` and the word view (`word_type`, `bit_type`); C-Flock's linear-form builder (`boolean.forms`), the
    multiplexer tree (`boolean.gather`) and `boolean.trace.definition` (a builder traced into And/Xor/Not gates, with
    `word_of` its word Definition). `library` is the library of hardware and protocol objects core owns, and
    `operations` the pinned list of operations with the guard that a program holds no data.
  - `verity.primitives.circuits.evaluation`: the one evaluator. The plain-integer reference
    (`evaluate`, the IR oracle, which also takes a table held in a registered
    input lazily, as a `RegisteredTable`), `evaluate_batch` over numpy arrays,
    `register_kernel`, `self_check`. Every registered kernel must be listed
    in `verity/primitives/circuits/tests/evaluation/test_evaluation.py`, which self-checks it.
    `bits` evaluates Boolean Definitions bit-sliced (`evaluate_bits`, and
    `evaluate_words` through the word view); `evaluate_batch` uses it for one
    without a kernel. numpy is a core dependency, kept out of the rest of `verity.primitives.circuits`,
    `verity.primitives.randomness` and `verity.claims` by `verity/tests/test_boundaries.py`.
  - `verity.primitives.randomness`: `derive(source, domain, context) -> Key`, the one
    challenge derivation, with exact samplers (`uniform`, `bernoulli`, `subset`)
    over counter-mode streams. Stdlib only, and it imports no other core module.
    Its byte formats are pinned by `verity/primitives/randomness/tests/vectors.json`.
  - `verity.primitives.crypto`: the hashes and the curve the commitments and protocols share (`blake3`, `turboshake`,
    `bls12_381`, `beacon`).
  - `verity.primitives.commitments`: see its module docstring; SHA-512 and the Merkle tree as Boolean gates are its
    `gates`.
  - `verity.protocols.profile.IntegrityProfile` records what verification
    established: the exhaustive facts, a verifier's law over named levels of a
    partition hierarchy (`verity.primitives.circuits.partition.Level`) and δ. Consumers read
    it with their own utilities; their readers stay out of core.
  - `verity.protocols`: one package per protocol, and each imports only the stdlib, the primitives `ALLOWED` in
    `verity/tests/test_boundaries.py`, itself and the protocols Daniel has allowed it (PoUW imports sampled proofs; its
    import of the catalog is the one exception to "core imports no other distribution").
    `verity.protocols.verification.sampled_proofs` is the one-stage
    audit (`one_stage`: registration, the verifier's draw over a partition's proof units, and the
    audit record with its profile); the two-stage law, its draws from the verifier's own
    randomness, verification plans and consumers are experimental's (`verity_experimental.sampled_proofs`). The Lean states it; its `README.md` signposts that
    Lean and lists what it does not state yet. `benchmarks/one_stage` runs the one-stage audit end to end with
    C-Flock's circuit prover (M0, `verity_flock.circuit`, run from the pinned tree
    that `--m0-pythonpath` names, not from the checkout) and the Lean
    verifier. `verity.protocols.accounting.space.pous` is proof of useful space: the reference
    codecs, setup and timed audit, its `PROTOCOL.md`, and its submission grader (`lean/`; the
    statements of its guarantees and their proofs are its area of `verity/Security/`). `verity.protocols.accounting.work.pouw`
    is proof of useful work: the `PoUWScheme` interface (`protocol`), its lifecycle
    (`audit`) and the named schemes (`schemes.get`: NCP, Pearl's FP8 scheme, Pearl-C), each with
    pinned vectors, `circuit`, `ncp-v2` as a Program that sampled proofs verifies, and its Lean
    in `verity/Security/` (`Pouw.SecurityProofs.EndToEnd` is the guarantee); see its `PROTOCOL.md`.
    `verity.protocols.accounting.communication.certifier` is network transparency's timing
    channel: the certifier, its audit and calibration, and its Lean in `verity/Security/`.
    NCI is novel communication intensity, the policy that throttles frontier training
    and leaves inference alone; so far only its Lean, in `verity/Security/`.
    Applications (vLLM) compose protocols.
  - `verity.claims`: the ids of security claims (guarantees, assumptions as `property/instance`
    such as `cr/sha-512`, models, setup) that Table 1 cites; the docs site imports them from the
    entities render. Stdlib only; imports no other core module.
    Other modules are created when code lands, not before.
- Outside `verity/`, a top-level directory is a grouping, not an import name; three hold a distribution of their own:
  - `catalog/` (`verity-catalog`, `verity_catalog`): public data the trusted base reads by digest.
    `verity_catalog.silicon` is the exact hardware semantics of the tensor-core accumulator step (stdlib only; its
    rule is `catalog/tests/test_boundaries.py`), the FP formats and gadgets on bits (`fp`, `fp4`, `tc_step`), and
    `relation`, which restates the step as witness columns + mul/linear/lookup predicates and counts them (`census()`);
    the catalog's tests replay the A100 / RTX 4090 / H100 captures under `verity_catalog/silicon/captures/` (see its
    README for provenance). `verity_catalog.definitions` holds their IR registrations (`prims`, `gemm`), the other IR
    primitives (`scalar`: integer, bit, select and comparison operations; `fp32`: FP32 add, mul, FMA and divide, their
    flush-to-zero forms, and BF16 element-wise arithmetic), and the word library as Boolean Definitions
    (`definitions.boolean`): the BF16 dot, coordinate and GEMM and the E4M3, E5M2,
    NVFP4 and MXFP4 steps and coordinates on bits (`gemm`: `DotBf16_v3`,
    `GemmCoordinate_v3`, `Gemm_v3`, `GemmCoordinateE4m3_v2`, `GemmCoordinateNvf4_v2`,
    `GemmCoordinateMxf4_v2`), attention's softmax operations and inverse sum on bits
    (`attention`, `softmax`), the element-wise FP32, BF16 and I32 operations on bits
    (`elementwise`), the bf16 widening on bits (`cast`), and the bf16 comparator,
    the selects, PTX max / min and the E4M3 cast on bits (`scalar`). `verity_catalog.targets` names the targets.
    `catalog/census/` is data only, meant to move to its own repository: `hardware.json`, `datatypes.json`, `networks.json`,
    `subcircuits.json`, `input_sets.json`, `workloads.json` (stable ids, sources), a JSON Schema for each under `schemas/`,
    and a README with the id scheme and sourcing rules. Nothing imports it; the renderer and benchmarks
    read it with `verity_numerical.bench.census` (a few lines), the docs site reads the same files, and
    core never reads it (`verity/tests/test_boundaries.py`): core takes such numbers as parameters.
  - `kernels/` (`verity-kernels`, `verity_kernels`): registered fast code, each kernel checked against its Definition
    (`verity_kernels.tc`: the numpy kernels of the catalog's Definitions, which register them lazily);
    `kernels/hash_gpu` is the frozen backends' CUDA hashing, imported by path.
  - `experimental/` (`verity-experimental`, `verity_experimental`): constructions under development or kept for study
    (`commitments.multiproof`, `private_circuits`, sampled proofs' law, plans and consumers); `verity` never imports it.
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
- `tools/cluster/`: `cluster`, the shared machines' description and policy (`cluster policy diff` names live drift), a pure
 per-node planner and the allocation ledger. Stdlib only, imports neither `verity` nor `research`. On vy-nebius-2 its agent
 runs live (`vy-cluster-agent.service`, its `README.md`).
- `backends/`, `frontends/`, `integrations/`, `benchmarks/`: created when a
  component has real code; there is no `frontends/` yet. `backends/flock`
  (`verity_flock`) is C-Flock, the backend in active development; its verifier
  of record is the Lean one under `backends/flock/verifier/` (see Lean below).
  A-GKR (`archive/gkr`), B-Ligero (`archive/direct`, `archive/ligero-verify`,
  `archive/ligerito-verify`) and D-SP1 (`archive/sp1`, whose `verity_sp1.proofs` holds single statements and how a
  backend is judged on them) are frozen: kept, and
  still in the benchmark tables, but not developed; so is the VOLE research
  backend (`archive/vole`). `benchmarks/numerical` holds the numerical references
  and the benchmark tooling that renders the tables. `archive/sp1/README.md` is
  D-SP1's how-to (build, identity, Python API, tests, recorded pod runs, and where proving may run).
  `integrations/vllm` (`verity_vllm`) is the vLLM integration.

Dependency direction (README "Development"): `primitives.circuits <- primitives.commitments -> primitives.crypto`,
`primitives.circuits <- protocols.profile <- the protocols`; `randomness`, `crypto` and `claims` import no core module;
core imports no other distribution (PoUW's import of the catalog aside), and everything else depends inward on the core
(`verity/tests/test_boundaries.py`). Backends must be testable
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
against `verity.primitives.circuits`, the Lean audit, and the verifier's cross-check against upstream's Rust verifier, at once on a multi-core
CPU pod, with caches that hit only on identical inputs. `--record` runs it through `research run` as an Attempt of the checked-out clean commit. Merge into `main`
only through `research merge <branch>`, which refuses without a passing `check` of that exact commit in the evidence store, or
when `main` has moved past it; `research merge --train A B C --on POD --project verity` lands several at once on one `check`
of the merged tree. Sharding one commit's `check` across pods (`check.py --shard K/N`, `--combine`) and the lander's train
check (`tools/check/train.sh`) are documented in their usage text. `research merge` also demands the steps
`merge_requires` (`tools/check/tool.py`) names for the paths a change touches, such as `lean-agreement` for
`backends/flock/`, and `check.py --record` sends what they need.
`check` reuses a pass (a test suite, a circuit-check target, a Lean package's audit) only while every input it read is
unchanged, and keeps each pod's Lean dependencies warm under their manifest and toolchain.
A pure move goes through the same `check`, but once `uv run python tools/check/move_check.py BASE HEAD` certifies it (only
byte-identical renames, edits that rewrite moved paths or imports in path-bearing files, and a Lean split's new
package copied from its source) the lander may schedule it
first in a quiet window, alone in its train (Daniel, 4 Oct).

## Lean

- Soundness is one theorem over everything the compiled Lean verifier accepts, for arbitrary Boolean circuits, with no
  hypothesis on the circuit file or the statement's form. The verifier refuses any form the proof doesn't cover, and gains
  no new form ahead of its proof. Circuit inputs and outputs use one canonical format proved in general, so a new workload
  needs no verifier or proof change (ruling 2026-10-04, `.agents/skills/friction/SKILL.md`).
- Lean is code or security analysis (Daniel's layout ruling, 5 Oct). Code stays with its component and never requires
  `verity/Security/`: C-Flock's executable verifier (`backends/flock/verifier/lean/`, `Flock` and `FlockVerify`, no
  dependencies), the PoUS grader (`verity/protocols/accounting/space/pous/lean/`, Lean only) and the certifier program
  (`verity/protocols/accounting/communication/certifier/lean/`, `NetworkCertifier`, Lean only, with its difftest). The analysis is
  `verity/Security/`, in three trees: `Definitions/` (each component's vocabulary) and `Specs/` (its named assumptions and
  guarantee statements), the package `security` (Mathlib, no ArkLib), and `Proofs/`, the nested package `security_proofs`
  (`srcDir = ".."`, root `Proofs`, ArkLib allowed), which requires `security` and holds every proof, C-Flock's level3 and
  soundness among them (`Proofs.Flock.*`). Each component has an area of its own in each tree (`Core`, `Nci`, `Pouw`,
  `Pous`, `NetworkCertifier`, `Flock`): `Definitions.Pouw.X`, `Specs.Pouw.Guarantees`, `Proofs.Pouw.X`; declarations keep their
  names (`Pouw.Guarantees.G`, `Pouw.SecurityProofs.G`). New Lean goes into its component's area of the tree it belongs
  to; `tools/move/lean.py --resolve MODULE` names where a module from before the move went. All packages share one
  toolchain and one Mathlib revision (`tests/test_lean_packages.py`); bump them together, in a PR of their own.
- `Definitions` and `Specs` are the spec (README "Glossary"), importing only one another and Mathlib (`layers` in
  `verity/Security/lean-audit.json`); `Proofs` holds every proof and lemma, with `SecurityProofs.G : G` for each
  guarantee proved. A theorem left in a trusted module is tolerated; a trusted module importing a proof or ArkLib is not.
  C-Flock's analysis is all under `Proofs.Flock` for now (its reads are exempt until its spec is extracted).
- Every Lake package has a `lean-audit.json`, and `check` audits every declaration (`tools/lean/audit.py`): only `propext`,
  `Classical.choice` and `Quot.sound`; no `sorry`, `axiom`, `native_decide` or kernel bypass; an assumption is a named
  `Prop` in the package's assumptions module, taken as a hypothesis; every `.lean` file is built, and every declaration
  is replayed through the kernel.
- A guarantee (README "Glossary"), a theorem something outside Lean relies on as proved, is listed in `lean-audit.json`,
  and `audit.py --update` records what it says in the lock: its statement and every definition the statement reads. No axiom check
  can catch a weakened definition, so when a change to those records lands on `main`, Daniel gets a DM showing it (each
  changed statement before and after, each changed definition as it is now). Nobody signs off and nothing waits on it.
  A guarantee that only moved to another module or name sends nothing when `--update --moved MOVES.json` (old names to
  new) prints it as the same statement under its old names.
- How to work on Lean, how parallel Lean lanes avoid conflicts (two agents never share a writable `.lake`), and which
  assumptions get an `upstream` watch entry (`tools/lean/upstream.py`): `.agents/skills/lean-proofs/SKILL.md`.

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

Remove obsolete rules and machinery rather than accumulating them.

## Where writing goes

- This repo: code, tests, small fixtures, and maintained current-truth docs only (`README.md`, `AGENTS.md`, `PROTOCOL.md`, `DESIGN.md`, `DISCREPANCIES.md`, `SKILL.md`, `CHANGELOG.md`, `*.template.md`). `tests/test_repository.py` enforces this section's file rules (report-genre names, generated output, the 256 KiB cap) and says where each offender goes; large fixtures: `.agents/skills/testing/SKILL.md`.
- Lane reports, checkpoints, findings, drafts, friction notes: markdown under `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-<kind>-<slug>.md` with YAML front-matter (`id`, `campaign`, `lane`, `kind: report|finding|draft|friction`, `status`, `repo`, `origin`); cite them from here by id as `note:<id>`. Notes are records, never messages: a handoff, an order or an ask goes on Slack (below), or to your own Cursor workers as a follow-up.
- Qualitative writing that anyone else may need goes in the research notes, where it can be shared, and data goes in the
  evidence store (Daniel, 4 Oct). A Cursor agent store is its agent's scratch: nothing another agent or a person needs
  lives only there, and nothing cites a `store:` path, which no other VM can read. Copy such a file into the notes or the
  store and cite its `note:` or `art:` id. `research notes sync --path lanes/<lane>` publishes notes to the notes' `main`
  from any VM with `RESEARCH_NOTES_TOKEN` set.
- Evidence and generated output (explore JSON, plots, gate outputs, logs, run results, rendered tables and reports): the evidence store only, never Git, neither this repo nor the notes. `research run` publishes every run, completed or failed, as an Attempt with its outputs (contract in `tools/research/src/research/store/README.md`); anything else goes in with `research data put --kind K --file|--tree P --preserve`; cite it as `art:<id>`. Tables and reports are rendered on demand from code plus the store. An old `notes-asset:<path>` resolves through the notes README.
- Architecture decisions: Notion (the coordinator publishes); no `docs/adr/` in the repo.
- Findings and claims about a run are labels, never edits to a shared table: `research data label <run id|art:…> <key> <value> --by <lane> --ref <run id>` (a red-team verdict is another label by the red-team; tables pick a policy, `store_tables.py --proof-class-by 'red-team-*'`).

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
