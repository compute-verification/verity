# Verity

Verity represents AI datacenter computations exactly as Boolean circuits, proves claims about them, and derives
compute, memory and network accounting from those proofs. `README.md` has the architecture and the Glossary, whose terms
code, docs and output use. This file states what is true on `main` now: what each directory is, the design rules, and how
work lands. Each package's contents are in its own README and docstrings. Your checkout, and this file in your prompt, can
be older than `main` (a cloud VM boots from a snapshot), so when they differ, `git show origin/main:AGENTS.md` wins.

## Map and commands

Top-level directories (a grouping, not an import name, except `verity/`):

- `verity/`: the distribution `verity` (the root `pyproject.toml`), in two parts. The module list is `verity/README.md`.
  Import from the owning module (`from verity.ml.circuits import Program`); `verity` re-exports nothing. Each directory's
  tests sit beside it as their own suite.
  - `verity/core/` (`verity.core`): the verifiers, trusted code. `Security/Models/` states their guarantees, and one whose
    proof isn't in `Security/Proofs/` is open: that verifier's properties aren't proven yet.
    - `protocols`: the integrity profile (`profile`) and one package per protocol: sampled proofs (`sampled_proofs`, with
      its plans and consumers), PoUW (`pouw`), PoUS (`pous`), the bandwidth audit over PoUS's store (`bandwidth`), the
      remote bandwidth audit over a node's registered store (`remote_bandwidth`) and the network certifier
      (`certifier`), each a Python
      verifier with the code that runs it. A protocol's Lean code is its `lean/`. NCI is Lean only
      (`verity/core/protocols/network/nci/`).
    - `claims`: the ids of security claims.
    - The Lean ports in `primitives`, `devices`, `service` and `protocols`. A port is held to the Python it replaces by
      `difftest` (`verity_difftest`, with its Lake package `verity_difftest`).
    - `service/flock/`: C-Flock's executable verifier of record (no dependencies).
  - `verity/ml/` (`verity.ml`): Python and CUDA. These units are on the soundness path, because core's verifiers read
    them: `ml.circuits`, `ml.circuits.ops`, `ml.circuits.silicon`, `ml.circuits.targets`, `ml.kernels`,
    `ml.commitments`, `ml.crypto` and `ml.randomness` (`SOUNDNESS_PATH` in `tests/test_boundaries.py`, all that
    `ALLOWED` lets core reach). The rest, C-Flock's prover and the private circuits, can't affect soundness.
    - `circuits` (format in `circuits/PROTOCOL.md`): circuits and their evaluator (`evaluation`), the Boolean profile
      (`boolean`), what models compute (`ops`, the primitive circuits' Definitions), what the GPUs actually compute
      (`silicon`, the exact hardware semantics) and `targets`.
    - `kernels` (with `hash_gpu/`): fast code, each kernel checked against its Definition.
    - `commitments` (with Merkle multiproofs, `multiproof`), shared hash code (`crypto`), and the one challenge derivation
      (`randomness`).
    - `private_circuits`: the universal unit, one public Definition that runs any private circuit of a size class.
    - `flock/` (`verity_flock`, a distribution of its own, `flock/pyproject.toml` with its package in `flock/python/`):
      C-Flock, the proof system. Its patches on upstream Flock, the CUDA prover, the zero-knowledge prover (`live/`,
      `flock-circuit --zk`), and `verifier/`: the verifier's specification and the scripts that run it, with upstream's
      Rust verifier as a cross-check.
- `Security/`: the Lean every protocol states its guarantees in, and their proofs: `Definitions/`, `Models/` and `Proofs/`
  (Lean below). `Properties/` is `security_properties` (Lean below).
- `integrations/`: one directory per outside system. `integrations/vllm/` (`verity_vllm`) is the vLLM integration. It
  builds each served workload's circuit and commits its values while serving; `benchmarks/one_stage/` takes those values
  through C-Flock's prover and the Lean verifier.
- `benchmarks/`: the measuring side: reproducible workloads and measurements, among them `one_stage/`, the one-stage audit
  end to end, and the benchmark tools.
  - `benchmarks/census/`: data only. `verity` never reads it; it takes such numbers as parameters.
  - `benchmarks/tc_probe/`, `benchmarks/tc_probe_fp4/`, `benchmarks/native_peak/`: GPU probes.
  - `benchmarks/numerical/`: the table tooling of the archived proof systems.
- `tools/`:
  - `verity/`: the `verity` command (`cli/`) and the tools it runs: `check/` (the one check a merge requires),
    `circuit_check/` (the property checks every circuit passes) and `lean/` (the Lean check, `check.py`).
  - `research`: the run recorder and the run-store client: runs, the evidence store, Slack, pods and deploys. It imports
    nothing from Verity.
  - `agent-guard`: the VM's secret guard.
  - `move`: layout moves.
- `infra/`: our machines.
  - `cluster/` (`cluster`): the shared machines' policy, planner and ledger. Stdlib only.
  - `control/`: the control pod's boot script and loops.
  - `cursor/`: the coordinators, each a plain Cursor cloud agent: `launch_coordinator.py` creates and archives them from
    Daniel's laptop.
  - `nebius/`: the console's units.
  - `ops/` (`verity_ops`): `ops deploy` and `ops keeper`, on top of `research`.
- `archive/`: proof systems and code that are kept but not developed: A-GKR, B-Ligero, D-SP1 and the VOLE prover, still in
  the benchmark tables, plus whatever a ruling retires. Its own rules are `archive/AGENTS.md`.
- `fixtures/`: `artifacts.json`, every replicated fixture by `art:` id, and the archived proof systems' fixtures.
- `tests/`: invariants of the whole tree, the import rules among them.

Code reaches the Nebius nodes only through `research deploy install` from a pushed commit, listed in
`tools/research/src/research/pods/nebius/deploy.toml`. Delete data you don't own on a node only with
`research retention gc` or `research retention rm --approved-by @OWNER`, never a bare `rm`.

Dependencies point inward, and `tests/test_boundaries.py` enforces them:

- `verity.core` may import the units of `verity.ml` that `ALLOWED` lists for it, which puts them on the soundness path
  (`SOUNDNESS_PATH`). `verity.ml` never imports `verity.core`, apart from the files `ML_IMPORTS_CORE` names (C-Flock's
  tests and V*'s builders, and hm96's test of its claims).
- Everything outside `verity/` may import `verity`. `verity` imports no other distribution of the workspace.
- Inside `verity`, each unit imports only what `ALLOWED` lists. `ml.circuits`, `ml.randomness`, `ml.crypto` and
  `core.claims` import no other unit.
- A protocol imports the parts of `verity.ml` it's allowed, the profile, itself, and the protocols `ALLOWED` lists for it
  (PoUW and the network certifier import sampled proofs). PoUW is the one protocol that reads the primitive circuits'
  Definitions and silicon semantics (`ml.circuits.ops`, `silicon`, `targets`).
- Nothing outside `infra/` imports `infra/`, apart from the files `INFRA_IMPORTERS` names.

`verity` is the repository's one command (`tools/verity/cli/`): `verity check` runs every part of the check a merge
requires, `verity check tests|lean|circuits` runs only those parts, and `verity bench NAME` runs `benchmarks/<NAME>/`'s runner.
`verity --help` lists the rest.

~~~text
uv sync --all-packages --extra torch-cpu  # the whole workspace, editable, with CPU-only torch (GPU work: a member's `torch` extra)
uv run tools/verity/check/suites.py       # every test suite whose inputs changed since it last passed; --quick skips slow tests
uv run --extra torch-cpu verity check     # the check a merge requires; --record publishes it as an Attempt
~~~

## Design rules

Daniel's, current. `.agents/skills/friction/SKILL.md` has the dated rulings behind most of them.

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
  witness and never reaches the auditor in the clear. They are the auditor's hardware: each runs on a machine the prover
  doesn't control, never as a process on the prover's node, and trust in the device and its host provider is a named
  assumption (`trusted-nebius-host`). Nothing the prover runs or hosts is trusted.
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

There is no GitHub Actions CI. `check` (`verity check`, which runs `tools/verity/check/check.py`) is the one check, run on a
multi-core CPU pod. It covers:

- every test suite;
- `circuit-check --all`;
- C-Flock's Rust build and tests;
- the Lean verifier's build and its partition check;
- the Lean check;
- `merge_requires` (declared per job in `tools/verity/check/ci.toml`): the steps a change's paths demand, such as `lean-agreement`, the cross-check
  against upstream's Rust verifier, for C-Flock (`verity/ml/flock/`, and its Lean verifier in `verity/core/service/flock/`;
  `check.py --record` sends what it needs).

A pass of a test suite, a circuit-check target or a Lean package's Lean check is reused only while every input it read is
unchanged. `--record` runs `check` through `research run` as an Attempt of the checked-out clean commit.

Landing, as Daniel set it (7 Oct; the CI page in Notion): `main` always passes the whole `check`, and the check is to stay
fast by reusing a part's verdict whenever its inputs haven't changed. Each verdict is to be keyed on a hash of the part's
declared inputs (the read guard enforces them), the base image and the code that decides that verdict (a suite's runner and
harness, not every line of the check tool; Daniel, 9 Oct), so that a docs-only change reruns nothing, a deleted test reruns
its suite, and a layout move or a change to the runner reruns everything. The merge
queue runs the check on the queued group, and a nightly uncached check must agree; a disagreement means some part reads
something it doesn't declare, and it is fixed in the check, not by reverting anyone's change.

Check time is to come down by a lot, and anyone may do whatever it takes (Daniel, 11 Oct): a change to how `check` runs
lands on its own passing check without its file owner's review, and the owner is told. infra owns check time as an
outcome. Spend past a budget line, and moving a part out of what `main`'s passing check covers, go to Daniel. Hand
landing is automated, and ci owns that: from a ready mark to `main` moving, no agent acts unless the automation
refuses, and the lander handles what it refuses.

Daniel moved one class of part out (11 Oct, 8:53 PM PDT): a part whose cold run takes more than about 10 minutes may run
right after its train lands, on every landing that changed its inputs, and a failure there reverts the landing at once.
A reused verdict stays in the gate, and the Lean check always runs before landing. Citations and tables read only commits
whose trailing parts passed. ci and infra build this post-landing tier; until it exists, the whole `check` stays before
landing.

PRs land through ci's `research queue drive` (#1826), live since 12 Oct 06:00Z:

- Drive admits a PR once its owner marks it ready at its head: `research queue ready N --mark --by LANE`, or
  `research data label pr:N@SHA ready true --by LANE`. It needs no quick-tier pass, and a new head needs a new mark. The
  owner is the lane a `Lane NAME` line in the PR's body names, else the Cursor agent id the body carries. Marks written
  before `marked_after` in `tools/verity/check/queue.toml` don't count, and a `hold` label keeps a PR out.
  `research queue status` says what each PR waits for.
- No change waits on a per-head grant: not red-team's, the vLLM coordinator's or a statement reviewer's (Daniel, 8 Oct).
- Drive stacks admitted PRs on `next`, checks them in trains and lands the newest train whose check passed. What it
  refuses (a conflict, a failed check) goes to the lander, who ejects the PR that caused it; its owner is told, and
  marks its fixed head ready.
- A PR whose head is in a train being checked keeps that head until the train lands or fails: a new head takes it off
  `next` and cancels the checks of its train and every train above it.
- A landing that breaks `main` is reverted at once (`git revert -m 1`), and its owner relands with the fix.
- A pure move that `tools/verity/check/move_check.py BASE HEAD` (`verity check move BASE HEAD`) certifies lands alone, ahead
  of the queue, since drive doesn't order it: the lander lands it with `research merge` on its own passing check, and
  drive's next sync rebuilds `next` on it.

## Lean

- **Soundness is one theorem** (a rule, not a claim that `main` proves it): one theorem over everything the compiled
  Lean verifier accepts, for arbitrary Boolean circuits, with no hypothesis on the circuit file or the statement's form.
  The verifier must refuse any form the proof doesn't cover, and it gains no new form ahead of its proof. Circuit inputs
  and outputs are to use one canonical format, proved in general, so a new workload needs no verifier or proof change.
- **Lean is either code or security analysis.**
  - Code stays with its component and never requires `Security/`: C-Flock's executable verifier
    (`verity/core/service/flock/`, no dependencies), the PoUS grader (`verity/core/protocols/pous/lean/`) and
    the certifier program (`verity/core/protocols/certifier/lean/`, `NetworkCertifier`, with its
    difftest) and the NCI check (`verity/core/protocols/network/nci/lean/`, `NciCheck`, with its difftest).
  - The analysis is `Security/`. Its spec is `Definitions/` (each component's vocabulary) and `Models/` (its named
    assumptions and guarantee statements), the package `security` (Mathlib, no ArkLib). `Proofs/` is the nested package
    `security_proofs` (ArkLib allowed): every proof and lemma, with `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for
    each proved guarantee.
    `Properties/` is the nested package `security_properties`: properties of the code stated against its Lean functions
    (`Properties.<Protocol>`) and proved (`Properties.Proofs.<Protocol>`). It may require `security` and that code, never
    `security_proofs`.
  - A component's Lean goes in its own area of each tree it has Lean in (`Core`, `Nci`, `Pouw`, `Pous`,
    `NetworkCertifier`, `Flock`; `Accounting` is the device model the resource audits share, and their joint guarantee).
  - The spec imports only itself and Mathlib (`layers` in `Security/lean-audit.json`). A theorem that must stay
    in a trusted module (a definition's proof field) may; the Lean check replays it like any proof. C-Flock has no
    `Definitions/` area yet: its guarantees are `Models/Flock/Guarantees`, and the rest of its analysis is in `Proofs/Flock`
    until its spec is extracted.
  - All packages share one toolchain and one Mathlib revision (`tests/test_lean_packages.py`). Bump them together, in a
    PR of their own.
- **The Lean check.** Every Lake package has a lock, `lean-audit.json`, and `check` runs the Lean check on every declaration
  (`tools/verity/lean/check.py`):
  - only `propext`, `Classical.choice` and `Quot.sound`;
  - no `sorry`, `axiom`, `native_decide` or kernel bypass;
  - an assumption is a named `Prop`, taken as a hypothesis or read in a guarantee's statement, and listed by name in the
    lock with its own record;
  - every file is built, and every declaration is replayed through the kernel.
- **The lock.** A guarantee (README "Glossary") is listed in `lean-audit.json`, and `check.py --update` records its
  statement and every definition the statement reads; those records are the lock.
  - When a changed record lands on `main`, Daniel gets a DM showing it. That DM is the check: nobody signs off and nothing
    waits on it.
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

Make every decision yourself except keys and claims (Daniel, 8 Oct; spend became its lead's call 13 Oct). Access and
keys, and what a guarantee says or rests on, go to Daniel through your coordinator, with your recommendation; pods
still bound themselves under a `budgets.toml` line, and a budget line named in a standing ruling stands. A change that
crosses owners goes to the other owner, and the coordinator settles a disagreement. When a tool, rule or step gets in
your way, fix it at the source if the fix is clearly yours. Record each ruling you get. The friction skill,
`.agents/skills/friction/SKILL.md`, has the rulings so far and the friction notes.

A guard never fails open, and an agent VM can be suspended at any moment, so nothing that bounds spend or preserves
results may depend on one. Pods bound themselves (`research pods create --max-hours`, under a line of the notes repo's
`budgets.toml`), and runs preserve their results from the pod.

Never print a secret's value.

- List environment variable names with `compgen -e`, never `env | cut`, `env | grep`, `printenv | …` or tmux's
  `show-environment`: a multi-line value's later lines print as if they were names.
- `.cursor/hooks.json` refuses any shell command that would print the environment or every process's full command line
  (`tools/agent-guard/shell_hook.py`). Cloud agents read project hooks only, so on a branch older than the hook the
  environment's `start` runs origin/main's `tools/agent-guard/boot.sh`, which writes the hook into `/workspace` and runs
  `install.sh`.
- A secret is stored as one line (a key as base64, `NAME_B64`), and `tools/agent-guard/install.sh` puts a redacting `env`
  on the VM.

## Where writing goes

- **Work items** are Linear issues (Daniel, 14 Oct). The full brief — context, done-condition, pointers — lives in the
  issue, filed into the owning team at creation; each lead's subscription to its own team is the wake path, and pickup
  is the issue's state change. Slack carries conversation, broadcasts, alerts and Daniel's cards, never the brief.
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
- **Open work** (an ask that outlives its thread, what a handoff hands over) is an issue in the owning handle's Linear
  team (the using-slack skill).
- **Architecture decisions** and Daniel's decisions log go in Notion, which the admin keeps.

## Lanes (worker agents)

If your launch message names you a lane, `~/.research/notes/kb/LANE-CONTRACT.md` is your rulebook.

Each coordinator (a handle's holder in `.agents/skills/using-slack/registry.json`) is a plain Cursor cloud agent on the
one-repo environment "Verity compute-verification". It owns one vertical slice and runs its own workers as subagents.
There are no Cursor Projects (Daniel, 13 Oct). The admin holds `@top`: it opens handovers, keeps the roster, the
quiet-lane sweep and Daniel's decisions log, relays to Daniel, and announces and settles; it does no lane work.
Coordinators are launched, handed over and archived from Daniel's laptop with `infra/cursor/launch_coordinator.py`
(`infra/cursor/README.md`).

- **Reread the rules every night.** At the start of each night, reread this file and the rulings in
  `.agents/skills/friction/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md`: rules change during the day, and a lane
  that acts on a retired one waits on gates that no longer exist.
- **Keep chatter low** (contract §3a). Every message, and every background shell you start, reaches the coordinator and
  the human. So run long jobs detached on the pod, write one-line checkpoints, and message someone only to unblock them
  or to change their plan.
- **Check what's being tried.** Before starting work another agent could plausibly be doing, read the notes'
  `kb/onboarding.md`, check `research notes approaches`, and claim yours at your first checkpoint
  (`research notes claim`).
- **Work toward outcomes.** A lane's night is one or two outcomes, each with a done-condition anyone can check, which its
  lead proposed and Daniel approved.
- **Push your branch after every commit** (never `main`, never `--force`), except to a PR in a train being checked
  (Checking and merging). Only drive and the lander move `main`.
- **Landing is drive's.** Push your PR, mark it ready and go back to research; you hear back only if your PR fails. A
  lane works on at most two open PRs; one whose head is on `next` (queued or in a running check) doesn't count until
  its check fails.
- **Slack is for handle holders** (coordinators and service agents) and named humans only, at computeverification.slack.com.
  A worker asks its own coordinator. The coordination API is `research msg`: the using-slack skill,
  `.agents/skills/using-slack/SKILL.md`. Work for another lead goes in as a Linear issue into its team (Where writing
  goes); Slack is the conversation around it, not the brief.

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
