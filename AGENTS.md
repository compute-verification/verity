# Verity

Verity represents AI datacenter computations exactly as Boolean circuits, proves claims about them, and derives
compute, memory and network accounting from those proofs. `ARCHITECTURE.md` has the architecture and the Glossary, whose
terms code, docs and output use. This file states what holds on `main` now; each package's README and docstrings say what
it contains. Your checkout and your prompt can be older than `main`: `git show origin/main:AGENTS.md` wins.

## Map and commands

- `verity/`: the distribution `verity` (the root `pyproject.toml`; its module list is `verity/README.md`). Import from the
  owning module (`from verity.ml.circuits import Program`); `verity` re-exports nothing.
  - `verity/core/`: the verifiers, trusted code. `verity/core/protocols/` holds the integrity profile (`profile`) and one
    package per protocol: sampled proofs, PoUW, proof of useful space (PoUS), the bandwidth audit, proof of useful
    bandwidth (PoUB) and network certification, each a Python verifier until its Lean verifier (its `lean/`) replaces
    it; the inference-only policy (`network/nci/`) is Lean only. `claims` holds the security claims' ids,
    `service/flock/` C-Flock's Lean verifier and the ZK server (`Flock.Firewall`), `service/firewall/` the ZK server's
    release side, and `difftest` the Python held to each Lean.
  - `verity/ml/`: Python and CUDA: the provers, and Python that mirrors the Lean for speed. `circuits` (format in
    `circuits/PROTOCOL.md`; `ops`, the primitive circuits' Definitions; `silicon`, the GPUs' exact semantics),
    `kernels`, `commitments`, `crypto`, `randomness` (the one challenge derivation), `private_circuits`, `definitions`
    (the served workloads' Definitions) and `flock/` (`verity_flock`, its own distribution: C-Flock, the proof
    system's prover, with its verifier's specification in `verifier/`). Until each protocol's verifier is Lean, the units its
    Python verifier reads are on the soundness path: `ml.circuits`, `ml.circuits.ops`, `ml.circuits.silicon`,
    `ml.kernels`, `ml.commitments`, `ml.crypto` and `ml.randomness` (`SOUNDNESS_PATH`, which only shrinks). The rest
    can't affect soundness.
- `Security/`: the Lean every guarantee is stated and proved in (Lean, below).
- `integrations/`: one directory per serving stack, holding only what attaches Verity to it. `integrations/vllm/`
  (`verity_vllm`) builds each served workload's circuit and commits its values while serving.
- `examples/`: one directory per example: its workload, placement, commands and doneness (`examples/README.md`).
- `benchmarks/`: the measuring side: reproducible workloads and measurements, among them `one_stage/`, the one-stage audit
  end to end, GPU probes, and `benchmarks/census/`, data that `verity` only takes as parameters.
- `tools/verity/`: the `verity` command, `check/` (the one check), `circuit_check/` and `lean/` (the Lean check).
- `fixtures/artifacts.json`: every replicated fixture by `art:` id. `tests/`: invariants of the whole tree. The archived
  proof systems (A-GKR, B-Ligero, D-SP1, the VOLE prover) were deleted in `2091252fe3`; `git checkout 2091252fe3^ --
  archive` restores them.
- The team's own tooling and machines, which nothing in `verity/` reads: `tools/research/` (runs, the evidence store,
  messaging, pods), `tools/agent-guard/`, `tools/move/` and `infra/` (the team skill).

Dependencies point inward (`tests/test_boundaries.py`; every list it holds only shrinks). A verifier imports no
`verity.ml` code; until its protocol's verifier is Lean, a Python verifier keeps only the edges into `verity.ml`
that `CORE_ML_EDGES` lists. `verity.ml` never imports `verity.core` (`ML_IMPORTS_CORE` names the exceptions).
Inside `verity` each unit imports only what `ALLOWED` lists, and `verity` imports no other distribution of the workspace.
Nothing outside `infra/` imports it (`INFRA_IMPORTERS`), and nothing under `verity/` imports an integration (`KNOWN` in
`tests/test_integration_boundaries.py`). `benchmarks/` is measurement only: nothing outside it imports its code, by
package name or by a path put on `sys.path` or loaded by file (`BENCHMARKS_IMPORTERS` names today's importers); a number a
protocol uses reaches it through a card file citing an `art:` id.

`verity` is the one command: `verity check [tests|lean|circuits]`, `verity bench|example NAME`, `verity --help`.

~~~text
uv sync --all-packages --extra torch-cpu        # the whole workspace, editable, with CPU-only torch (GPU work: a member's `torch` extra)
uv run tools/verity/check/suites.py             # the suites whose inputs changed; --quick: no slow tests
uv run --extra torch-cpu verity check           # the whole check
~~~

The team's agents run these remotely (the team skill).

## Design rules

The rules in force, each stated as itself; the friction skill has the standing rulings stated nowhere else.

- **Circuits and values.** A circuit holds only gates. Everything a program or a request brings, every intermediate and
  every output, is a value, committed under a hiding leaf and touched only inside a zero-knowledge proof.
- **Everything zero-knowledge.** Every protocol is, PoUS included. Only four kinds of things are public, each with its
  reason: protocol structure, the protocol's outputs, the verifier's coins and samples, and hiding commitments. Partitions
  are private and their properties public: a proof block's outputs fit its Definition's public size class, a power of two
  from 16 to 262,144 bits, and its output bits count in leakage (provisional; `UNIT_CLASS_MAX` = 16 restores the
  16-bit limit). A plain path, C-Flock without `--zk`, stays for a deployment whose values needn't be hidden: it
  works end to end beside the zero-knowledge paths, tests cover both, its guarantee is soundness alone, and `flock-verify`
  refuses a plain session past what the plain theorem covers (one table today).
- **One proof system** for PoUW, PoUS and network accounting: the workers behind the ZK server, the challenger and the
  verifier. A protocol states what is committed, which units need proofs, the public Program each must pass and what it
  does with the outcome; the proof system owns commitments, live coins, the record, the selection, proving, verifying and the
  integrity profile. A protocol carries no policy parameter: it surfaces what its sample established (ε at δ), and the
  model's size is public.
- **One hash: SHA-512 with its whole output, everywhere.** Commitments are hiding commitments (`hm96` rows) in frame-v3-sha512 trees, and signatures
  are hash-based (LMS, n = 64). Two named exceptions: Pearl's FP8 scheme as Pearl defines it,
  and Duo receipts, PoUW's per-step keyed u32 mul/xor/rotl folds over committed accumulator words, which bind under the
  named premise H-μ⁺, not yet in a lock.
- **Live coins only.** No Fiat–Shamir, except at two rates where live coins are impractical: PoUB's pass seeds
  inside a window, from the window's live challenge and the previous pass's root; and PoUW's
  per-matmul noise seeds, seed_j = SHA-512(tag ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j)), from the window's
  live salt (`pouw.salt`) and the operands' committed roots, which a sampled proof block's proof derives. Grinding is a term
  of the guarantee: a steer that needs m noised coordinates succeeds with probability at most μ^m a try. No
  retries: a failed proof is a visible failure, and nothing resamples. Outer proofs go through the ZK server as independent
  sessions: each staged statement is fixed before any session flips coins and gets exactly one session, with its own live
  coins in its own ZK server pass; no statement reads another session's transcript or coins; every session is recorded and
  every verdict counts (provisional).
- **Verifier cost: O(circuit description), never O(unrolled circuit).** A verifier works in
  proportion to the descriptor it reads (its Definitions, nodes and references) and never walks every gate, batch member
  or scan iteration: it summarizes once per specialized Definition and multiplies through batches and scans
  (`NciCheck.Desc`). An unrolled walk survives only as a difftest's reference. `tests/test_verifier_cost.py` enforces it
  (the walks still open are its `ALLOWED`), and `benchmarks/verifier_cost/` measures it.
- **Auditor-trusted devices.** The network certifiers and memory challengers sign what they see; what they sign is
  witness and never reaches the auditor in the clear. Each is the auditor's hardware on a machine the prover doesn't
  control, never a process on the prover's node, and trust in it and its host is the named assumption `trusted-host`.
  Nothing the prover runs or hosts is trusted. "Network certifier" is its one name, never "warden" or "tap".
- **The audited hardware is untrusted, and so is its vendor.** No guarantee rests on a GPU attesting its own
  SKU, firmware or clocks. Hardware constants (f_max, peak rates) come from the data sheet under two or three readings,
  and later from a separate physical calibration. A device's capacity is an assumed bound on its true peak compute
  capacity, and nothing else: no power certifier, power ceiling, energy floor or meter enters a protocol, premise or
  plan.
- **Lean is the definition and the verifier.** Python in `verity.ml` mirrors it for the prover's
  speed, held to it by difftest. Each protocol switches by the friction skill's switching ruling; until then its
  Python verifier and the ml units it reads stay on `SOUNDNESS_PATH`, and no core → ml edge is added.
- **Soundness and zero knowledge in Lean; completeness may be Python.** What a guarantee says about a cheating
  prover is proved in Lean or rests on an auditor-trusted device; that honest serving passes may be checked in Python.
- **That a circuit computes what the model computes is completeness, never proved.** It decides only whether
  honest serving passes: every guarantee is about the registered circuit, one that passes circuit-check, and the
  developer validates the match empirically (circuit-check, hardware captures, served outputs). Nobody proves it, lists it
  as an assumption or counts it as a gap. Lean holds one description of the circuit, the parsed circuit file the verifier
  checks, and defines every other layer's circuit from it.
- **Integrations are frontends, and Verity is agnostic to them.** Verity's vocabulary is circuits, values,
  commitments, coins, proofs and verdicts; models, checkpoints, requests, batches and schedulers stay out of `verity/`, and
  those still in it (the certifier's advice format, vLLM's scheduler) move out. An integration hands Verity only: a
  circuit, partitioned into proof blocks, with its public layout of external I/O, registered by digest; for run-time
  choices, a registered compiler (a circuit with a public length cap) and the advice it compiles, a bit string Verity
  never interprets, whose length is charged; the values, committed in a `verity.ml.commitments` scheme and registered
  before any coin; and on a sample, the sampled proof blocks' openings, to the prover behind the ZK server.
- **Results are per monitoring period, over all compute.** The time length of interest is the monitoring
  period (Glossary), not the window, the run or the demo. Every result, in a report, a
  table, a PR or a message, is presented for monitoring periods of one week, one month and one year, and over
  all compute, in AND gates: the world's (*adversary resource model*), a datacenter's beside it, though a demo checks
  one. The datacenter's size (from `benchmarks/census/`) and the window's length are stated parameters, and a figure
  scaled from one machine says which parts are proved and which are scaled. Give each bound as its pooled total over the
  period and as a share of the compute it covers, beside the per-window figure, never instead of it: a per-window δ that
  looks weak is not the result.

## Tests

Each package's tests are one suite in its own `tests/`, run in its own process and cached by input; the root `tests/`
holds only invariants of the whole tree. How to run them, what a suite may read, and the rules for slow tests, fixtures
and dummy credentials: the testing skill.

No test reads wall-clock time (no sleeps, clock readings or timeouts:
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
unchanged.

A part's verdict is reused while a hash of its declared inputs, the base image and its runner is unchanged, so a
docs-only change reruns nothing; a nightly uncached check must agree, and a disagreement is fixed in the check. PRs land
on review, and the full check runs after landing (how a PR lands: the team skill).

## Lean

- **Soundness is one theorem.** Protocol soundness (`Verity.Soundness`) is to be one theorem over everything the compiled
  Lean verifier accepts, for arbitrary Boolean circuits, with no hypothesis on the circuit file or the statement's form;
  this is a requirement, not a claim that `main` proves it. The verifier refuses any form the proof doesn't cover and
  gains none ahead of its proof. Circuit inputs and outputs use one canonical format, proved in general, so a new
  workload needs no verifier or proof change.
- **The math is modular.** Each layer is proved once over the guarantee of the layer below: the knowledge soundness of
  one run's verifier for sampling, sampling's for an audit, an audit's for an accounting bound. Each configuration
  (zero-knowledge, plain, recursive) and each protocol instantiates those theorems, and a proof that redoes a layer is
  refactored onto the shared one.
- **One vocabulary.** A concept more than one guarantee reads (the circuit, commitments and hashes, coins and samples,
  probability, partitions and proof blocks, devices and signatures) has one definition, in `Core` or `Hash`, which every
  area and protocol reads; an area defines only its own, and no new Lean or Python adds a second definition
  (`tests/test_lean_vocabulary.py`).
- **Lean is code or security analysis.** Code stays with its component and never requires `Security/`: C-Flock's
  executable verifier and the ZK server (`verity/core/service/flock/`, with the release side in
  `verity/core/service/firewall/`), the PoUS grader, the certifier program, the inference-only policy's check and
  PoUB's verifier, each in its protocol's `lean/` with a difftest. Under `FC_FIREWALL=1` every byte that leaves is
  computed by `Flock.Firewall`; the ZK theorem (`Flock.zk_server_lean_loop`) still reads the released bytes through
  hypotheses not yet proved about the live path. The analysis is `Security/`, three trees: definitions
  (`Definitions/`, the package `security`, Mathlib only: the vocabulary, named assumptions and guarantee
  statements a reader checks), properties (`Properties/`, `security_properties`: statements about the code's Lean
  functions) and proofs (`Proofs/`, `security_proofs`, ArkLib allowed: every proof, lemma and proof-only definition, with
  `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each proved guarantee). No statement or module of the definitions
  or properties trees reads the proofs tree (`trees` in `Security/lean-audit.json`, whose `except` is the cleanup's to-do). A subsystem's Lean goes in its own area of each tree (`Core`, `Nci`,
  `Pouw`, `Pous`, `NetworkCertifier`, `Flock`, `Accounting`). All packages share one toolchain and one Mathlib revision,
  bumped together in a PR of their own.
- **The Lean check** (`tools/verity/lean/check.py`) builds every file and replays every declaration through the kernel:
  only `propext`, `Classical.choice` and `Quot.sound`; no `sorry`, `axiom`, `native_decide` or kernel bypass; an
  assumption is a named `Prop`, taken as a hypothesis or read in a guarantee's statement, listed by name in the lock.
- **The lock.** Each package's `lean-audit.json` lists its guarantees, and `check.py --update` records each one's
  statement, the definitions it reads and the named assumptions it takes, so a changed statement is a changed record.
  A guarantee that only moved keeps its record (`--update --moved MOVES.json`).
- How to work on Lean (the edit loop, why two agents never share a writable `.lake`, `upstream` watch entries): the
  lean-proofs skill.

## Working principles

- Solve the task with the smallest clear implementation, reusing code and commands before adding any. Enforce
  deterministic requirements in code, with the fix in the failure message, rather than in prose. Remove obsolete rules and
  machinery rather than accumulating them.
- **The tree gets smaller over time.** Make two interfaces or definitions of one thing one, make things extensible,
  delete dead and superseded code, and move data and logs to the evidence store.
- **A decision rolls out in the change that records it.** A PR that records a ruling, or retires a term, a path or a
  procedure, rewrites every place that states the old way: `git grep` this file, ARCHITECTURE.md, GLOSSARY.md, the
  skills, READMEs, docstrings and CLI help. Fix a stale statement on sight. `tests/test_doc_paths.py` fails on a doc
  naming a dead path.
- Never print a secret's value. List environment variable names with `compgen -e`, never `env`, `printenv` or tmux's
  `show-environment`.

## Where writing goes

**The repository says what is true now and how to reproduce it. It is not a record.** A reader learns from it what each
part is and does, its contract in the Glossary's terms, the rules, and how to run each measurement (a benchmark's runner
and its parameters). What a run measured, what happened and who decided live outside it, and the tree doesn't cite them:

- **In the tree:** code, tests, Lean, circuits and their pins, the parameters and configuration code reads (with their
  public sources: a data sheet, the census), small fixtures beside their generators, and maintained current-truth docs
  (`README.md`, `AGENTS.md`, `ARCHITECTURE.md`, the root's `GLOSSARY.md`, `DESIGN.md`, `DISCREPANCIES.md`, `SKILL.md`,
  `CHANGELOG.md`, `*.template.md`, `PROTOCOL.md`; `tests/test_repository.py`). A protocol keeps its `PROTOCOL.md` only
  while experimental: the PR whose Lean covers it deletes it, leaving a README of outline, signposts and core API.
- **Not in the tree, and not cited from it** (code, comments, docstrings, Lean, READMEs, docs): measured results
  (numbers, tables, plots, overheads), reports and drafts, and the process's ids and names: run, `art:` and `note:` ids,
  Notion pages, Linear issues, Slack threads, PR numbers, agent ids, lane and worker names, dates, and attributions such
  as "(name, date)". A rule or a constraint is stated as itself. Only the skills, which describe the team's process,
  name its roles (a handle as an owner). Results are published from the store, never copied into the tree: a
  reader who finds no overhead figure in the repository has found it working as intended, and runs the benchmark or reads
  the published result.
- **Where those live:** results in the evidence store, why a line changed in git, and open work, decisions and reports
  outside the repository (the team skill).
- **An id a tool reads is data, not prose.** It goes in a declared data file the tool reads (`fixtures/artifacts.json`'s
  `art:` ids and digests, a lock's records, the handle registry, a table's runs of record), never in a comment, a
  docstring or a doc.

## Skills

`.agents/skills/`: `testing`, `lean-proofs`, `kernel-engineering` (PoUW GPU kernels on sm_120), `smell-check` (before you
call something served, end to end, proved or credited), `writing-runs` (a job for `research run`), `friction` (when
something is in your way, and the standing rulings), `process-design` and `authoring-skills` (at most 16 KB a skill).
The team's own agents start with `team` (machines, landing, lanes, and where work, notes and evidence live), then
`coordinating`, `using-slack` and `asking-daniel`.
