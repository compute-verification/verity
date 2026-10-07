# Verity

Verity represents AI datacenter computations exactly as Boolean circuits, proves claims about them, and derives
compute, memory and network accounting from those proofs. `README.md` has the architecture and the Glossary, whose terms
code, docs and output use. This file states what is true on `main` now: what each directory is, the design rules, and how
work lands. Each package's contents are in its own README and docstrings. Your checkout, and this file in your prompt, can
be older than `main` (a cloud VM boots from a snapshot), so when they differ, `git show origin/main:AGENTS.md` wins.

## Map and commands

Top-level directories (a grouping, not an import name, except `verity/`):

- `verity/`: the core library, the distribution `verity` (the root `pyproject.toml`).
  - `verity.ml`: circuits and their evaluator (`circuits`, format in `circuits/PROTOCOL.md`), commitments
    (`commitments`), shared hash code (`crypto`), and the one challenge derivation (`randomness`).
  - `verity.experimental.protocols`: the integrity profile (`profile`) and one package per protocol: sampled proofs
    (`verification.sampled_proofs`), PoUW (`accounting.work.pouw`), PoUS (`accounting.space.pous`) and the network
    certifier (`accounting.communication.certifier`). NCI is Lean only.
  - `verity.experimental.claims`: the ids of security claims.
  - `Security/`: the Lean every protocol states its guarantees in.

  The module list is `verity/README.md`. Import from the owning module (`from verity.ml.circuits import
  Program`); `verity` re-exports nothing. Each directory's tests sit beside it as their own suite.
- `verity/ml/circuits/` (`verity.ml.circuits`): what the trusted base reads by digest.
  - `silicon`: the exact hardware semantics.
  - `definitions`: the primitive circuits' Definitions.
  - `targets`.
  - `benchmarks/census/`: data only. Core never reads it; it takes such numbers as parameters.
- `verity/ml/kernels/` (`verity.ml.kernels`): fast code, each kernel checked against its Definition.
- `verity/experimental/` (`verity.experimental`): what is under development or kept for study (private circuits, multiproofs,
  sampled proofs' plans and consumers). `verity` never imports it.
- `verity/ml/flock/` (`verity_flock`): C-Flock, the proof system.
  - Its patches on upstream Flock, the CUDA prover, and the zero-knowledge prover (`live/`, `flock-circuit --zk`).
  - `verifier/`: the verifier of record, written in Lean, with upstream's Rust verifier as a cross-check.
- `integrations/vllm/` (`verity_vllm`): the vLLM integration. It builds each served workload's circuit, commits its
  values while serving, and proves.
- `benchmarks/`: reproducible workloads and measurements, among them `one_stage/`, the one-stage audit end to end.
  `benchmarks/numerical` is the table tooling of the archived proof systems.
- `tools/`:
  - `research`: runs, the evidence store, Slack, pods and deploys. It imports nothing from Verity.
  - `check`: the one check a merge requires.
  - `circuit_check`: the property checks every circuit passes.
  - `lean`: the Lean check (`audit.py` until lean renames it `lean_check.py`).
  - `cluster`: the shared machines' policy, planner and ledger. Stdlib only.
  - `agent-guard`: the VM's secret guard.
  - `move`: layout moves.
  - `tc_probe`, `tc_probe_fp4`, `native_peak`: GPU probes.
- `archive/`: proof systems and code that are kept but not developed: A-GKR, B-Ligero, D-SP1 and the VOLE prover, still in
  the benchmark tables, plus whatever a ruling retires. Its own rules are `archive/AGENTS.md`.
- `fixtures/`: `artifacts.json`, every replicated fixture by `art:` id, and the archived proof systems' fixtures.
- `tests/`: invariants of the whole tree.
- `infra/nebius/`: the console's units.

Code reaches the Nebius nodes only through `research deploy install` from a pushed commit, listed in
`tools/research/src/research/pods/nebius/deploy.toml`. Delete data you don't own on a node only with
`research retention gc` or `research retention rm --approved-by @OWNER`, never a bare `rm`.

Dependencies point inward, and `tests/test_boundaries.py` enforces them:

- Everything outside `verity/` may import the core. The core imports no other distribution, except that PoUW reads the
  catalog.
- Inside the core, each unit imports only what `ALLOWED` lists. `primitives.circuits`, `randomness`, `crypto` and
  `claims` import no other core module; no primitive imports a protocol.
- A protocol imports the primitives it's allowed, itself, and the protocols Daniel has allowed it (PoUW imports sampled
  proofs).

~~~text
uv sync --all-packages --extra torch-cpu  # the whole workspace, editable, with CPU-only torch (GPU work: a member's `torch` extra)
uv run tools/verity/check/suites.py              # every test suite whose inputs changed since it last passed; --quick skips slow tests
uv run --extra torch-cpu python tools/verity/check/check.py  # the check a merge requires; --record publishes it as an Attempt
~~~

## Design rules

Daniel's, current; the dated lines in `.agents/skills/friction/SKILL.md` say when each was decided.

- **Circuits and values.** A circuit holds only gates. Everything a program or a request brings, every intermediate and
  every output, is a value. A value is committed under a hiding leaf and touched only inside a zero-knowledge proof.
- **Everything zero-knowledge.** Every protocol is zero-knowledge, PoUS included, with no clear mode. Only four kinds of
  things are public, each with its reason:
  - protocol structure;
  - the protocol's outputs;
  - the verifier's coins and draws;
  - hiding commitments.

  Partitions are private, and their properties are public. A proof unit's outputs are at most 16 bits, a literal limit.
- **One proof service.** PoUW, PoUS and network accounting all use one proof service: the workers behind the firewall,
  the challenger, and the verifier.
  - A protocol states what is committed, which units need proofs, the public Program each must pass, and what it does
    with the outcome.
  - The service owns commitments, live coins and the record, the selection, proving, verifying and the integrity
    profile.
  - A protocol carries no policy parameter. It surfaces what its draw established (ε at δ), and the model's size is
    public.
- **One hash: SHA-512 with its whole output, everywhere.**
  - Commitments are hm96 rows in frame-v3-sha512 trees.
  - Signatures are hash-based (LMS, n = 64).
  - BLAKE3 leaves the served path for `archive/`. The one exception is Pearl's FP8 scheme, as Pearl defines it.
- **Live coins only.** Fiat–Shamir is out. There are no retries: a failed proof is a visible failure, and a retry never
  redraws. Outer proofs go through the firewall one at a time.
- **Auditor-trusted devices.** The network certifiers and the memory challenger sign what they see. What they sign is
  witness and never reaches the auditor in the clear. They run on Nebius, and trust in Nebius is a named assumption.
- **Soundness is one theorem** (see Lean below).

## Tests

Every package's tests live in its own `tests/` and run as one suite, in its own process (a `parallel` suite runs in
pytest-xdist workers), cached by input. The repository's own suite (`tests/`) holds only invariants of the whole tree. The
testing skill, `.agents/skills/testing/SKILL.md`, says how to run them, what a suite may read (`[tool.verity.tests]
inputs`), and the rules for slow tests, fixtures and dummy credentials.

No test depends on wall-clock time: no sleeps, clock readings or timeouts. Wait on the event itself or inject the clock.
`tests/test_no_wall_clock.py` enforces it, and its `ALLOWED` list, one reason per entry, holds the exceptions.

A test that flakes is deleted, not fixed or rerun. A test that adds more than about a minute to `check` says in its
docstring what it catches that a cheaper test can't.

## Circuits

Any change that adds or modifies a circuit (a Definition, a subcircuit template, a Boolean lowering) must pass
`circuit-check` (`uv run circuit-check <definition|template|row>`); include its report in the PR. A new Definition needs a
small binding in its module's sidecar (`<module>.circuit-check.py`, pins in `.circuit-check.json`; `circuit_check.sidecars`),
which the suite enforces.

## Checking and merging

There is no GitHub Actions CI. `check` (`tools/verity/check/check.py`) is the one check, run on a multi-core CPU pod. It covers:

- every test suite;
- `circuit-check --all`;
- C-Flock's Rust build and tests;
- the Lean verifier's build and its partition check;
- the Lean check;
- `merge_requires` (`tools/verity/check/tool.py`): the steps a change's paths demand, such as `lean-agreement`, the cross-check
  against upstream's Rust verifier, for `verity/ml/flock/` (`check.py --record` sends what it needs).

A pass of a test suite, a circuit-check target or a Lean package's Lean check is reused only while every input it read is
unchanged. `--record` runs `check` through `research run` as an Attempt of the checked-out clean commit.

Landing:

- `main` moves only through `research merge --push`. It refuses without a passing `check` of that exact commit, or when
  `main` has moved past it, and then pushes and sweeps.
- A plain `git push` would skip the spec DM, node 1's mirror and the post-train regression.
- PRs land in trains: ci stacks the tips, and the lander runs `research merge --train A B C --on POD --project verity`
  over one `check` of the merged tree.
- A train member needs its owner's `ready` label at its head, written by hand if need be:
  `research data label pr:N@SHA ready true --by LANE`. It needs no quick-tier pass, since the train's full `check` is the
  gate.
- A grant carries to a head whose own patch is byte-identical.
- A PR in a tip that is being checked keeps its head until the tip lands or ci asks for a change. ci fixes a conflict at
  the tip; a push to the PR would refuse every tip that pins its old head.
- A pure move that `tools/verity/check/move_check.py BASE HEAD` certifies goes first, alone in its train.

## Lean

- **Soundness is one theorem** over everything the compiled Lean verifier accepts, for arbitrary Boolean circuits, with
  no hypothesis on the circuit file or the statement's form. The verifier refuses any form the proof doesn't cover, and
  it gains no new form ahead of its proof. Circuit inputs and outputs use one canonical format, proved in general, so a
  new workload needs no verifier or proof change.
- **Lean is either code or security analysis.**
  - Code stays with its component and never requires `Security/`: C-Flock's executable verifier
    (`verity/core/service/flock/`, no dependencies), the PoUS grader (`verity/experimental/protocols/pous/lean/`) and
    the certifier program (`verity/experimental/protocols/certifier/lean/`, `NetworkCertifier`, with its
    difftest).
  - The analysis is `Security/`. Its spec is `Definitions/` (each component's vocabulary) and `Specs/` (its named
    assumptions and guarantee statements), the package `security` (Mathlib, no ArkLib). `Proofs/` is the nested package
    `security_proofs` (ArkLib allowed): every proof and lemma, with `SecurityProofs.G : G` for each guarantee.
  - Each component has its own area in each tree (`Core`, `Nci`, `Pouw`, `Pous`, `NetworkCertifier`, `Flock`).
  - The spec imports only itself and Mathlib (`layers` in `Security/lean-audit.json`). A theorem that must stay
    in a trusted module (a definition's proof field) may; the Lean check replays it like any proof. C-Flock's analysis is all
    in `Proofs.Flock` until its spec is extracted.
  - All packages share one toolchain and one Mathlib revision (`tests/test_lean_packages.py`). Bump them together, in a
    PR of their own.
- **The Lean check.** Every Lake package has a lock, `lean-audit.json` until lean renames it `lean-lock.json`, and `check` runs the Lean check on every declaration
  (`tools/verity/lean/check.py`):
  - only `propext`, `Classical.choice` and `Quot.sound`;
  - no `sorry`, `axiom`, `native_decide` or kernel bypass;
  - an assumption is a named `Prop`, taken as a hypothesis;
  - every file is built, and every declaration is replayed through the kernel.
- **The lock.** A guarantee (README "Glossary") is listed in `lean-audit.json`, and `audit.py --update` records its
  statement and every definition the statement reads; those records are the lock.
  - When a changed record lands on `main`, Daniel gets a DM showing it. That DM is the check: nobody signs off and nothing
    waits on it, and a statement reviewer is optional.
  - A guarantee that only moved sends nothing (`--update --moved MOVES.json`).
- **How to work on Lean**, how parallel Lean lanes avoid conflicts (two agents never share a writable `.lake`), and which
  assumptions get an `upstream` watch entry: the lean-proofs skill, `.agents/skills/lean-proofs/SKILL.md`.

## Working principles

Research velocity comes first (Daniel, 6 Oct). Be as generous as possible with what lands, and cut ceremony: propose
cutting any gate, review or step that slows landing without catching real errors, and fix what breaks later. Our machines
are research infrastructure, not a rehearsal of deployment, so don't mimic deployment settings.

Solve the current task with the smallest clear implementation. Reuse existing code and commands before adding
abstractions or procedures. Enforce deterministic requirements in code, with the fix in the failure message, rather than
in prose. Remove obsolete rules and machinery rather than accumulating them.

**A decision rolls out in the change that records it.** When Daniel rules, or a PR retires a term, a path or a procedure,
that PR rewrites every place that states the old way: `git grep` the old phrase across this file, the README and its
Glossary, the skills, READMEs, docstrings and CLI help. When code has to change first, the ruling's relay names the
owner and the PR, and the docs change with that PR. A stale statement found later is fixed on sight, by whoever finds
it. `tests/test_doc_paths.py` fails on a maintained doc that names a repository path that no longer exists.

Make reversible engineering decisions yourself. When a tool, rule or step gets in your way, fix it at the source if the
fix is clearly yours. Surface changes to agreed semantics, acceptance criteria, material spending or destructive
behavior. When you can't tell which side a change is on, ask Daniel through your coordinator, with your recommendation,
and record his ruling. The friction skill, `.agents/skills/friction/SKILL.md`, has the rulings so far and the friction
notes. Daniel tunes two balances through those rulings: how much cross-agent chatter and note-taking, and how much
autonomy versus asking him. On 6 Oct he set the second toward autonomy: be generous and cut ceremony.

A guard never fails open, and an agent VM can be suspended at any moment, so nothing that bounds spend or preserves
results may depend on one. Pods bound themselves (`research pods create --max-hours`, under a line of the notes repo's
`budgets.toml`), and runs preserve their results from the pod.

Never print a secret's value.

- List environment variable names with `compgen -e`, never `env | cut`, `env | grep` or `printenv | …`: a multi-line
  value's later lines print as if they were names.
- `.cursor/hooks.json` refuses any shell command that would print the environment or every process's full command line
  (`tools/agent-guard/shell_hook.py`).
- A secret is stored as one line (a key as base64, `NAME_B64`), and `tools/agent-guard/install.sh` puts a redacting `env`
  on the VM.

## Where writing goes

- **This repo** holds code, tests, small fixtures, and maintained current-truth docs only: `README.md`, `AGENTS.md`,
  `DESIGN.md`, `DISCREPANCIES.md`, `SKILL.md`, `CHANGELOG.md`, `*.template.md`, and `PROTOCOL.md`.
  - A protocol keeps a `PROTOCOL.md` only while it is experimental. Lean is the source of truth for a mature protocol,
    whose README is an outline, signposts and its core API, and the PR whose Lean covers it deletes the file.
  - `tests/test_repository.py` enforces the file rules: report-genre names, generated output, the 256 KiB cap. Large
    fixtures: the testing skill.
- **Lane reports, checkpoints, findings, drafts and friction notes** are Markdown under
  `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-<kind>-<slug>.md`, with YAML front-matter (`id`, `campaign`, `lane`,
  `kind: report|finding|draft|friction`, `status`, `repo`, `origin`). Cite them as `note:<id>`.
  - Notes are records, never messages. A handoff, an order or an ask goes on Slack, or to your own Cursor workers as a
    follow-up.
  - `research notes sync --path lanes/<lane>` publishes notes from any VM with `RESEARCH_NOTES_TOKEN` set.
  - A Cursor agent store is its agent's scratch: nothing another agent or a person needs lives only there, and nothing
    cites a `store:` path.
- **Evidence and generated output** (explore JSON, plots, gate outputs, logs, run results, rendered tables and reports)
  go in the evidence store only, never Git: neither this repo nor the notes.
  - `research run` publishes every run, completed or failed, as an Attempt with its outputs. The contract is
    `tools/research/src/research/store/README.md`.
  - Anything else goes in with `research data put --kind K --file|--tree P --preserve`. Cite it as `art:<id>`.
  - Tables and reports are rendered on demand from code plus the store.
- **Findings and claims about a run** are labels, never edits to a shared table:
  `research data label <run id|art:…> <key> <value> --by <lane> --ref <run id>`.
- **Architecture decisions** go in Notion, which the coordinator publishes.

## Lanes (worker agents)

If your launch message names you a lane, `~/.research/notes/kb/LANE-CONTRACT.md` is your rulebook.

- **Keep chatter low** (contract §3a). Every message, and every background shell you start, reaches the coordinator and
  the human. So run long jobs detached on the pod, write one-line checkpoints, and message someone only to unblock them
  or to change their plan.
- **Check what's being tried.** Before starting work another agent could plausibly be doing, read the notes'
  `kb/onboarding.md`, check `research notes approaches`, and claim yours at your first checkpoint
  (`research notes claim`).
- **Push your branch after every commit** (never `main`, never `--force`), except to a PR in a tip being checked
  (Checking and merging). Only the lander moves `main`, through
  `research merge`.
- **Slack is for handle holders** (coordinators and service agents) and named humans only, at computeverification.slack.com.
  A worker asks its own coordinator. The coordination API is `research msg`: the using-slack skill,
  `.agents/skills/using-slack/SKILL.md`.

## Skills

- Writing or launching a job for `research run` (its cwd, outputs, `--declared-output`, scratch):
  `.agents/skills/writing-runs/SKILL.md`.
- Testing: `.agents/skills/testing/SKILL.md`.
- Lean: `.agents/skills/lean-proofs/SKILL.md`.
- GPU kernel work (PoUW kernels on sm_120): `.agents/skills/kernel-engineering/SKILL.md`.
- When something gets in your way, and the rulings: `.agents/skills/friction/SKILL.md`.
- Before adding, changing or retiring a process (a rule here or in the lane contract, a gate in `check` or
  `research merge`, a required review, a file every PR must update), assess its effect on research velocity:
  `.agents/skills/process-design/SKILL.md`.
- Creating or modifying a skill: `.agents/skills/authoring-skills/SKILL.md`.
