# Verity

Verity represents AI datacenter computations exactly as Boolean circuits, proves claims about them, and derives
compute, memory and network accounting from those proofs. `ARCHITECTURE.md` has the architecture and the Glossary, whose
terms code, docs and output use. This file states what holds on `main` now; each package's README and docstrings say what
it contains. Your checkout and your prompt can be older than `main`: `git show origin/main:AGENTS.md` wins.

## Map and commands

- `verity/`: the distribution `verity` (the root `pyproject.toml`; its module list is `verity/README.md`). Import from the
  owning module (`from verity.ml.circuits import Program`); `verity` re-exports nothing.
  - `verity/core/`: the verifiers, trusted code. `verity/core/protocols/` holds the integrity profile (`profile`) and one
    package per protocol: sampled proofs, PoUW, PoUS, the bandwidth audit, the remote bandwidth audit and the network
    certifier, each a Python verifier until its Lean (its `lean/`) is its verifier of record; NCI is Lean only. `claims`
    holds the security claims' ids, `service/flock/` C-Flock's Lean verifier of record and the ZK server
    (`Flock.Firewall`), `service/firewall/` the ZK server's release side, and `difftest` the Python held to each Lean.
  - `verity/ml/`: Python and CUDA: the provers, and Python that mirrors the Lean for speed. `circuits` (format in
    `circuits/PROTOCOL.md`; `ops`, the primitive circuits' Definitions; `silicon`, the GPUs' exact semantics),
    `kernels`, `commitments`, `crypto`, `randomness` (the one challenge derivation), `private_circuits`, `definitions`
    (the served workloads' Definitions), `protocols` (the circuits core's provers build and its verifiers don't read:
    `pouw`, `pous`, `certifier`) and `flock/` (`verity_flock`, its own distribution: C-Flock, the proof system, with its
    verifier's specification in `verifier/`). Until each protocol's verifier of record is Lean, the units its Python
    verifier reads are on the soundness path: `ml.circuits`, `ml.commitments`, `ml.crypto` and `ml.randomness`
    (`SOUNDNESS_PATH`, which only shrinks). The rest can't affect soundness.
- `Security/`: the Lean every guarantee is stated and proved in (Lean, below).
- `integrations/`: one directory per serving stack, holding only what attaches Verity to it. `integrations/vllm/`
  (`verity_vllm`) builds each served workload's circuit and commits its values while serving.
- `examples/`: one directory per example: its workload, placement, commands and doneness; a milestone is (example,
  run, proof) triples, which top keeps (`examples/README.md`).
- `benchmarks/`: the measuring side: reproducible workloads and measurements, among them `one_stage/`, the one-stage audit
  end to end, GPU probes, and `benchmarks/census/`, data that `verity` only takes as parameters.
- `tools/`: `tools/verity/` (the `verity` command, `check/`, `circuit_check/`, and `lean/`, the Lean check),
  `tools/research/` (runs, the evidence store, Slack, pods and deploys; it imports nothing from Verity),
  `tools/agent-guard/` (the VM's secret guard, OOM recorder and shell hook) and `tools/move/`.
- `infra/`: our machines: `infra/cluster/`, `infra/control/` (the control pod), `infra/cursor/` (launching
  coordinators), `infra/nebius/`, `infra/slack-events/` (comms's) and `infra/ops/`.
- `archive/`: kept but not developed: A-GKR, B-Ligero, D-SP1, the VOLE prover, and what a ruling retires
  (`archive/AGENTS.md`). `fixtures/artifacts.json`: every replicated fixture by `art:` id. `tests/`: invariants of the
  whole tree.

Dependencies point inward (`tests/test_boundaries.py`; every list it holds only shrinks). A verifier imports no
`verity.ml` code; until its protocol's verifier of record is Lean, a Python verifier keeps only the edges into `verity.ml`
it had on 18 Oct (`CORE_ML_EDGES`). `verity.ml` never imports `verity.core` (`ML_IMPORTS_CORE` names the exceptions).
Inside `verity` each unit imports only what `ALLOWED` lists, and `verity` imports no other distribution of the workspace.
Nothing outside `infra/` imports it (`INFRA_IMPORTERS`), and nothing under `verity/` imports an integration (`KNOWN` in
`tests/test_integration_boundaries.py`).

Code reaches the Nebius nodes only through `research deploy install` of a pushed commit
(`tools/research/src/research/pods/nebius/deploy.toml`). Delete data you don't own on a node only with
`research retention gc` or `research retention rm --approved-by @OWNER`.

`verity` is the one command: `verity check [tests|lean|circuits]`, `verity bench|example NAME`, `verity --help`.

~~~text
uv sync --all-packages --extra torch-cpu  # the whole workspace, editable, with CPU-only torch (GPU work: a member's `torch` extra)
research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py'  # suites whose inputs changed; --quick: no slow tests
research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run --extra torch-cpu verity check'  # the whole check
research status RUN  # check back on a run (`research run` returns once it starts); --wait: only a run under 20 min
~~~

## Design rules

Daniel's, current; their history is his decisions log in Notion, and the friction skill has the standing rulings stated
nowhere else.

- **Circuits and values.** A circuit holds only gates. Everything a program or a request brings, every intermediate and
  every output, is a value, committed under a hiding leaf and touched only inside a zero-knowledge proof.
- **Everything zero-knowledge.** Every protocol is, PoUS included. Only four kinds of things are public, each with its
  reason: protocol structure, the protocol's outputs, the verifier's coins and draws, and hiding commitments. Partitions
  are private and their properties public: a proof block's outputs fit its Definition's public size class, a power of two
  from 16 to 262,144 bits, and its output bits count in leakage (19 Oct, provisional; `UNIT_CLASS_MAX` = 16 restores the
  16-bit limit). A plain path, C-Flock without `--zk`, stays for a deployment whose values needn't be hidden (24 Oct): it
  works end to end beside the zero-knowledge paths, tests cover both, its guarantee is soundness alone, and `flock-verify`
  refuses a plain session past what the plain theorem covers (one table today).
- **One proof service** for PoUW, PoUS and network accounting: the workers behind the firewall, the challenger and the
  verifier. A protocol states what is committed, which units need proofs, the public Program each must pass and what it
  does with the outcome; the service owns commitments, live coins, the record, the selection, proving, verifying and the
  integrity profile. A protocol carries no policy parameter: it surfaces what its draw established (ε at δ), and the
  model's size is public.
- **One hash: SHA-512 with its whole output, everywhere.** Commitments are hm96 rows in frame-v3-sha512 trees, signatures
  are hash-based (LMS, n = 64), and BLAKE3 is `archive/`'s. Two named exceptions: Pearl's FP8 scheme as Pearl defines it,
  and Duo receipts, PoUW's per-step keyed u32 mul/xor/rotl folds over committed accumulator words, which bind under the
  named premise H-μ⁺ (the premise ledger's row R1, `note:20261016T0219Z-report-premise-ledger`), not yet in a lock.
- **Live coins only.** No Fiat–Shamir, except at two rates where live coins are impractical: the remote bandwidth
  audit's pass seeds inside a window, from the window's live challenge and the previous pass's root (12 Oct); and PoUW's
  per-matmul noise seeds, seed_j = SHA-512(tag ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j)), from the window's
  live salt (`pouw.salt`, #2637) and the operands' committed roots, which a drawn unit's proof derives. Grinding is a term
  of the guarantee: a steer that needs m noised coordinates succeeds with probability at most μ^m a try (18 Oct). No
  retries: a failed proof is a visible failure, and nothing redraws. Outer proofs go through the firewall as independent
  sessions: each staged statement is fixed before any session draws coins and gets exactly one session, with its own live
  coins in its own firewall pass; no statement reads another session's transcript or coins; every session is recorded and
  every verdict counts (24 Oct, provisional; https://app.notion.com/p/403399515d9e81298f49e4b6b78634e7).
- **Verifier cost: O(circuit description), never O(unrolled circuit)** (18 Oct). A verifier of record works in
  proportion to the descriptor it reads (its Definitions, nodes and references) and never walks every gate, batch member
  or scan iteration: it summarizes once per specialized Definition and multiplies through batches and scans
  (`NciCheck.Desc`). An unrolled walk survives only as a difftest's reference. `tests/test_verifier_cost.py` enforces it
  (the walks still open are its `ALLOWED`), and `benchmarks/verifier_cost/` measures it.
- **Auditor-trusted devices.** The network certifiers and memory challengers sign what they see; what they sign is
  witness and never reaches the auditor in the clear. Each is the auditor's hardware on a machine the prover doesn't
  control, never a process on the prover's node, and trust in it and its host is the named assumption `trusted-host`.
  Nothing the prover runs or hosts is trusted. "Network certifier" is its one name, never "warden" or "tap" (25 Oct).
- **The audited hardware is untrusted, and so is its vendor** (17 Oct). No guarantee rests on a GPU attesting its own
  SKU, firmware or clocks. Hardware constants (f_max, peak rates) come from the data sheet under two or three readings,
  and later from a separate physical calibration.
- **Lean is the definition and the verifier of record** (18 Oct). Python in `verity.ml` mirrors it for the prover's
  speed, held to it by difftest. Each protocol switches by the 8 Oct procedure (the friction skill); until then its
  Python verifier and the ml units it reads stay on `SOUNDNESS_PATH`, and no core → ml edge is added.
- **Soundness and zero knowledge in Lean; completeness may be Python** (18 Oct). What a guarantee says about a cheating
  prover is proved in Lean or rests on an auditor-trusted device; that honest serving passes may be checked in Python.
- **That a circuit computes what the model computes is completeness, never proved** (24 Oct). It decides only whether
  honest serving passes: every guarantee is about the registered circuit, one that passes circuit-check (25 Oct), and the
  developer validates the match empirically (circuit-check, hardware captures, served outputs). Nobody proves it, lists it
  as an assumption or counts it as a gap. Lean holds one description of the circuit, the parsed circuit file the verifier
  checks, and defines every other layer's circuit from it.
- **Integrations are frontends, and Verity is agnostic to them** (24 Oct). Verity's vocabulary is circuits, values,
  commitments, coins, proofs and verdicts; models, checkpoints, requests, batches and schedulers stay out of `verity/`, and
  those still in it (the certifier's advice format, vLLM's scheduler) move out. An integration hands Verity only: a
  circuit, partitioned into proof blocks, with its public layout of external I/O, registered by digest; for run-time
  choices, a registered compiler (a circuit with a public length cap) and the advice it compiles, a bit string Verity
  never interprets, whose length is charged; the values, committed in a `verity.ml.commitments` scheme and registered
  before any coin; and on a draw, the drawn units' openings, to the prover behind the firewall.
- **Results are per monitoring period, over all compute** (25 Oct). The time length of interest is the monitoring
  period (Glossary), not the window, the run or the demo. Every result, in a report, a
  table, a PR or a message to Daniel, is presented for monitoring periods of one week, one month and one year, and over
  all compute, in AND gates: the world's (*adversary resource model*), a datacenter's beside it, though a demo checks
  one. The datacenter's size (from `benchmarks/census/`) and the window's length are stated parameters, and a figure
  scaled from one machine says which parts are proved and which are scaled. Give each bound as its pooled total over the
  period and as a share of the compute it covers, beside the per-window figure, never instead of it: a per-window δ that
  looks weak is not the result.

## Tests

Each package's tests are one suite in its own `tests/`, run in its own process and cached by input; the root `tests/`
holds only invariants of the whole tree. How to run them, what a suite may read, and the rules for slow tests, fixtures
and dummy credentials: the testing skill.

An agent VM only edits (18 Oct): its 16 GB are shared by every agent on it, and one out-of-memory kill takes them all
down. Locally, run only git, reading and editing, messaging and the notes, one test file or `-k` selection under about a
minute and 1 GB, `lake env lean` on one file of a built tree, a Lean package with no Mathlib or ArkLib, and the lock fast
path. Everything else (every suite, `verity check`, circuit-check, any build of `Security`, cargo, nvcc, every e2e and GPU
job) runs through `research run --on auto --cwd source [--wait] -- CMD` (the writing-runs skill): an `agent-test` job on
the cluster queue (4 CPUs, 16 GiB, 60 min unless `--cpus`, `--mem-gb` or `--timeout` say otherwise), or a `research`
pool when one starts sooner. The shell hook refuses those locally and names the remote command; `AGENT_GUARD_LOCAL_OK=1`
in the command opts in.

No tool call blocks more than 20 minutes (19 Oct): every `--wait` stops there, cancels nothing and exits 79 naming the
command to check back with, and you end your turn. No test reads wall-clock time (no sleeps, clock readings or timeouts:
wait on the event or inject the clock; `tests/test_no_wall_clock.py`, its exceptions in `ALLOWED`). A test that flakes is
deleted, not rerun. A test that adds more than about a minute to `check` says in its docstring what it catches that a
cheaper test can't.

## Circuits

A change that adds or modifies a circuit (a Definition, a subcircuit template, a Boolean lowering) passes
`uv run circuit-check <definition|template|row>`, with its report in the PR. A new Definition gets a binding in its
module's sidecar (`<module>.circuit-check.py`, pins in `.circuit-check.json`), which the suite enforces.

## Checking and merging

There is no GitHub Actions CI. `check` (`verity check`, `tools/verity/check/check.py`) is the one check: every suite,
`circuit-check --all`, C-Flock's Rust build and tests, the Lean verifier's build and partition check, the Lean check, and
`lean-agreement` (against upstream's Rust verifier) when C-Flock or its Lean verifier changes.

A pass of a test suite, a circuit-check target or a Lean package's Lean check is reused only while every input it read is
unchanged. `check.py --record --on M --pool NAME`, from a checkout, records `check` as an Attempt of the clean commit
in M's NAME pool.

A part's verdict is reused while a hash of its declared inputs, the base image and its runner is unchanged (9 Oct), so a
docs-only change reruns nothing; a nightly uncached check must agree, and a disagreement is fixed in the check. PRs land
on an eyeball review, and the full check runs after landing (24 Oct). ci's `research queue drive` lands them
(`research.drive`'s docstring has the rules; `tools/verity/check/drive.toml` the settings):

- The worker that made a branch opens its PR, `research queue open --head BRANCH --sha SHA --title T --body-file F --by
  LANE` with `Lane LANE` in the body, and marks it ready at its head, `research queue ready N --mark --by LANE`; a push
  makes a new head, which needs a new mark and a new review. `research queue status` says what each PR waits for. A `hold` label, a conflict on `next` or a
  Markdown file over its cap keeps a PR out.
- Drive stacks admitted PRs on `next`, non-Lean changes first. Reviewers ci staffs, never the owner, record a verdict at
  the head (`research queue review N --land|--hold REASON --by LANE`); no other per-head grant exists (8 Oct). Drive
  fast-forwards `main` to the longest reviewed prefix whose merges are clean and whose `.py` files compile. A change to a
  `lean-audit.json` lock or to Lean that runs as code (a `.lean` outside `Security/`) also waits for a `lean-audit` that
  passed at or above it; Lean proofs alone land on the verdict (25 Oct).
- One run checks `main`'s tip after each landing. A red suite's culprit is rung and reverted 60 min later unless a fix is
  marked. Citations and tables read only commits whose after-landing run passed.
- A pure move that `verity check move BASE HEAD` certifies lands alone, ahead of the queue, through the lander. Only
  drive's App pushes `main` and `next`.
- Anyone may do whatever it takes to cut check time (11 Oct): a change to how `check` runs lands without its file owner's
  review, and the owner is told. infra owns check time and its spend, and ci owns landing: no agent acts between a ready
  mark and `main` moving unless the automation refuses.

## Lean

- **Soundness is one theorem.** Protocol soundness (`Verity.Soundness`) is to be one theorem over everything the compiled
  Lean verifier accepts, for arbitrary Boolean circuits, with no hypothesis on the circuit file or the statement's form;
  this is a requirement, not a claim that `main` proves it. The verifier refuses any form the proof doesn't cover and
  gains none ahead of its proof. Circuit inputs and outputs use one canonical format, proved in general, so a new
  workload needs no verifier or proof change.
- **The math is modular** (24 Oct). Each layer is proved once over the guarantee of the layer below: a session verifier's
  knowledge soundness for sampling, sampling's for an audit, an audit's for an accounting bound. Each configuration
  (zero-knowledge, plain, recursive) and each protocol instantiates those theorems, and a proof that redoes a layer is
  refactored onto the shared one.
- **One vocabulary** (25 Oct). A concept more than one guarantee reads (the circuit, commitments and hashes, coins and
  draws, probability, partitions and units, devices and signatures) has one definition, in `Core` or `Hash`, which every
  area and protocol reads; an area defines only its own, and no new Lean adds a second definition. lean owns
  consolidating today's duplicates.
- **Lean is code or security analysis.** Code stays with its component and never requires `Security/`: C-Flock's
  executable verifier and the ZK server (`verity/core/service/flock/`, with the release side in
  `verity/core/service/firewall/`), the PoUS grader, the certifier program, the NCI check and the remote bandwidth
  audit's verifier, each in its protocol's `lean/` with a difftest. Under `FC_FIREWALL=1` every byte that leaves is
  computed by `Flock.Firewall`; the ZK theorem (`Flock.zk_server_lean_loop`) still reads the released bytes through
  hypotheses not yet proved about the live path. The analysis is `Security/`, three trees (24 Oct): definitions
  (`Definitions/`, the package `security`, Mathlib only: the vocabulary, named assumptions and guarantee
  statements Daniel reads), properties (`Properties/`, `security_properties`: statements about the code's Lean
  functions) and proofs (`Proofs/`, `security_proofs`, ArkLib allowed: every proof, lemma and proof-only definition, with
  `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each proved guarantee). No statement or module of the definitions
  or properties trees reads the proofs tree (`trees` in `Security/lean-audit.json`, whose `except` is the cleanup's to-do). A component's Lean goes in its own area of each tree (`Core`, `Nci`,
  `Pouw`, `Pous`, `NetworkCertifier`, `Flock`, `Accounting`). All packages share one toolchain and one Mathlib revision,
  bumped together in a PR of their own.
- **The Lean check** (`tools/verity/lean/check.py`) builds every file and replays every declaration through the kernel:
  only `propext`, `Classical.choice` and `Quot.sound`; no `sorry`, `axiom`, `native_decide` or kernel bypass; an
  assumption is a named `Prop`, taken as a hypothesis or read in a guarantee's statement, listed by name in the lock.
- **The lock.** Each package's `lean-audit.json` lists its guarantees, and `check.py --update` records each one's
  statement, the definitions it reads and the named assumptions it takes. When a changed record lands on `main`, Daniel
  gets a DM showing it; that DM is the check, and nothing waits on it. A guarantee that only moved sends nothing
  (`--update --moved MOVES.json`).
- How to work on Lean (the edit loop, why two agents never share a writable `.lake`, `upstream` watch entries): the
  lean-proofs skill.

## Working principles

- **Research velocity comes first** (6 Oct). Be generous with what lands and cut ceremony: propose cutting any gate,
  review or step that slows landing without catching real errors, and fix what breaks later. Our machines are research
  infrastructure; don't mimic deployment settings.
- **Waiting is a bug** (25 Oct). A step you repeat that makes you wait more than about 10 minutes is friction even when it
  never fails: fix it at the source, or take it to its owner that day with its numbers (minutes a wait, waits a day).
  Push back the same way, through your coordinator, on a rule that costs velocity, Daniel's included. A slow loop's owner
  fixes it ahead of other work.
- Solve the task with the smallest clear implementation, reusing code and commands before adding any. Enforce
  deterministic requirements in code, with the fix in the failure message, rather than in prose. Remove obsolete rules and
  machinery rather than accumulating them.
- **The tree gets smaller over time** (Daniel, 26 Oct). Each lead owns its areas' size: make two interfaces or
  definitions of one thing one, make things extensible, delete dead and superseded code, and move data and logs to the
  evidence store. A PR that adds code says in its body what it deletes or unifies, or why nothing can go. top's hourly
  consolidation reading tracks `main`'s size.
- **Parked work gets done** (Daniel, 26 Oct). We have parallelism: a lead starts each Backlog or Todo issue worth doing on
  its own worker and closes the rest with a reason. Being busy is no reason to park one. Set more goals, not fewer.
- **A decision rolls out in the change that records it.** A PR that records a ruling, or retires a term, a path or a
  procedure, rewrites every place that states the old way: `git grep` this file, ARCHITECTURE.md, the skills, READMEs,
  docstrings and CLI help. When code must change first, the ruling's relay names the owner and the PR that changes the
  docs with it. Fix a stale statement on sight. `tests/test_doc_paths.py` fails on a doc naming a dead path.
- **Make every decision yourself except keys and claims** (8 Oct). Access, keys, and what a guarantee says or rests on go
  to Daniel through your coordinator, with your recommendation; spend is its lead's call (13 Oct). A change that crosses
  owners goes to the other owner, and the coordinator settles a disagreement. Fix what's in your way at the source when
  the fix is yours, and record each ruling you get (the friction skill).
- A guard never fails open, and an agent VM can be suspended at any moment, so nothing that bounds spend or preserves
  results depends on one: pods bound themselves (`research pods create --max-hours`, under a line of the notes repo's
  `budgets.toml`) and runs preserve their results from the pod. The lane whose budget line pays for a machine stops it
  when its work there ends (19 Oct).
- Never print a secret's value. List environment variable names with `compgen -e`, never `env`, `printenv` or tmux's
  `show-environment`. The shell hook (`.cursor/hooks.json`, `tools/agent-guard/shell_hook.py`) refuses a command that
  would print the environment or every process's command line; on a VM booted without it, run origin/main's
  `tools/agent-guard/boot.sh`. A secret is stored as one line (a key as base64, `NAME_B64`).

## Where writing goes

- **Work items** are Linear issues in the owning team (14 Oct), each holding its full brief: context, done-condition,
  pointers. A lead's subscription to its team is the wake path, and pickup is the issue's state change. Slack carries
  conversation, broadcasts, alerts and Daniel's cards, never the brief. Open work that outlives its thread is an issue too.
- **This repo** holds code, tests, small fixtures and maintained current-truth docs only (`README.md`, `AGENTS.md`,
  `ARCHITECTURE.md`, `DESIGN.md`, `DISCREPANCIES.md`, `SKILL.md`, `CHANGELOG.md`, `*.template.md`, `PROTOCOL.md`;
  `tests/test_repository.py`). A protocol keeps its `PROTOCOL.md` only while experimental: the PR whose Lean covers it
  deletes it, leaving a README of outline, signposts and core API.
- **Notes** (lane reports, checkpoints, findings, drafts, friction notes) are Markdown under
  `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-<kind>-<slug>.md` with YAML front-matter (`id`, `campaign`, `lane`,
  `kind`, `status`, `repo`, `origin`), cited as `note:<id>` and published with `research notes sync --path lanes/<lane>`.
  Notes are records, never messages: a handoff, an order or an ask goes on Slack, or to your own workers as a follow-up.
  A Cursor agent store is its agent's scratch: nothing another agent needs lives only there, and nothing cites it.
- **Writing for a person** is a Notion page, or cites only what they can open: PRs, repository files, run ids and Notion
  pages. The notes repo is private, so a `note:` id is for agents (19 Oct).
- **Evidence and generated output** go in the evidence store, never Git: `research run` publishes every run as an
  Attempt, and anything else goes in with `research data put --kind K --file|--tree P --preserve`, cited as `art:<id>`
  (`tools/research/src/research/store/README.md`). Tables and reports render on demand from code plus the store.
  Findings about a run are labels: `research data label <run id|art:…> <key> <value> --by <lane> --ref <run id>`.
- **Architecture decisions** and Daniel's decisions log are in Notion.

## Lanes (worker agents)

A lane's rulebook is `~/.research/notes/kb/LANE-CONTRACT.md`. Each coordinator, a handle's holder (`research msg
handles`), is a plain Cursor cloud agent on the environment "Verity compute-verification" that owns one vertical slice and
runs its own workers; there are no Cursor Projects (13 Oct). `@top` is the admin: handovers, the roster, the quiet-lane
sweep, Daniel's decisions log, relays to Daniel, announcing and settling, and consolidation (26 Oct: work, records and
notation converge, not multiply); it does no lane work. Launching, handing over and archiving coordinators:
`infra/cursor/README.md`.

- **Reread the rules each night**: this file and the rulings in the friction and lean-proofs skills.
- **Keep chatter low** (contract §3a). Every message and background shell reaches the coordinator and the human: run long
  jobs detached, and message someone only to unblock them or change their plan.
- **Workers run where they can't take you down**: as cloud subagents (`Task` with `environment: "cloud"`, which Daniel
  asked for once for every agent here, 18 Oct, so nobody asks first) or as detached `research run`s; a local subagent only
  for a short read-only lookup. A cloud worker boots from the environment's snapshot, so its brief is complete (the task,
  the done-condition, the deadline, "reread `git show origin/main:AGENTS.md`"), tells it to create
  `cursor/<lane>-<topic>-<hex>` from `origin/main` and push only there, and names an inbox branch it reads hourly
  (`git show origin/B:inbox/<agent-id>.md`); larger changes go in when you resume it. It opens its own PR, or reports its
  branch and SHA for its launcher to.
- **Reach a running subagent** on your VM through its next shell command, since it never rereads a shared file mid-run:
  `python3 tools/agent-guard/inbox.py send AGENT_ID --text '…'`. Agents on other VMs: through their coordinator on Slack.
- **Stay reachable** (19 Oct). Push and write a one-line checkpoint note at least every 60 minutes; when you'd wait on a
  run, end your turn naming its id and ETA, and your launcher resumes you. Agents keep no heartbeat: the control pod's
  agent watch flags a stalled or stopped agent and an out-of-memory kill. A worker resumed twice, whose base branch is
  gone, or silent past the watch's flag is relaunched fresh from its last note. Briefs ask for a checkpoint before long
  work and word red-team work neutrally.
- **Check what's being tried.** Before work another agent could plausibly be doing, read the notes' `kb/onboarding.md`
  and `research notes approaches`, and claim yours (`research notes claim`). A lane's night is one or two outcomes, each
  with a done-condition anyone can check, which its lead proposed and Daniel approved.
- **Work in a checkout no other agent uses** (your VM, or your own `git worktree` and venv), push only to branches you
  created or were handed, and push after every commit (never `main`, never `--force`). Before marking ready, run the Lean
  check of the packages you touched and the suites your change touches, `research run --on auto --cwd source -- bash -lc
  "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py --changed $(git rev-parse
  origin/main) --quick"`, plus a suite's slow tier when you changed its core. Prefer small PRs. You hear back only if
  yours is held, ejected or reverted.
- **Slack** (computeverification.slack.com) is for handle holders and named humans; a worker asks its own coordinator.
  The coordination API is `research msg` (the using-slack skill). Work for another lead goes in as a Linear issue in its
  team; Slack is the conversation around it.

## Skills

`.agents/skills/`: `writing-runs` (a job for `research run`), `testing`, `lean-proofs`, `kernel-engineering` (PoUW GPU
kernels on sm_120), `friction` (when something is in your way, and the rulings), `smell-check` (before you call something
served, end to end, proved or credited, or relay another lane's claim; the catalogue is the notes' `kb/smell-catalogue.md`),
`asking-daniel`, `process-design` (before adding, changing or retiring a process, its effect on velocity),
`using-slack` and `authoring-skills` (at most 16 KB a skill: current rules, not their history).
