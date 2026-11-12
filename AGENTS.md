# Verity

Verity represents AI datacenter computations exactly as Boolean circuits, proves claims about them, and derives
compute, memory and network accounting from those proofs. `ARCHITECTURE.md` has the architecture and the Glossary, whose
terms code, docs and output use. This file states what holds on `main` now; each package's README and docstrings say what
it contains. Your checkout and your prompt can be older than `main`: `git show origin/main:AGENTS.md` wins.

## Map and commands

- `verity/`: the distribution `verity` (the root `pyproject.toml`; its module list is `verity/README.md`). Import from the
  owning module (`from verity.ml.circuits import Program`); `verity` re-exports nothing.
  - `verity/core/`: the verifiers, trusted code. `verity/core/protocols/` holds the integrity profile (`profile`) and one
    package per protocol: sampled proofs, PoUW, proof of useful space (PoUS), the bandwidth audit over PoUS's store,
    PoUB over a node's registered store and network certification, each a Python verifier until its Lean verifier (its
    `lean/`) replaces it; the inference-only policy (`network/nci/`) is Lean only. `claims` holds the security claims'
    ids, `service/flock/` Flock's Lean verifier and the ZK server (`Flock.Firewall`), `service/firewall/` the ZK server's
    release side, and `difftest` the Python held to each Lean.
  - `verity/ml/`: Python and CUDA: the provers, and Python that mirrors the Lean for speed. `circuits` (format in
    `circuits/PROTOCOL.md`; `ops`, the primitive circuits' Definitions; `silicon`, the GPUs' exact semantics),
    `kernels`, `commitments`, `crypto`, `randomness` (the one challenge derivation), `private_circuits`, `definitions`
    (the served workloads' Definitions), `protocols` (the circuits core's protocols' provers build, which their
    verifiers don't read) and `flock/` (`verity_flock`, its own distribution: Flock, the proof system's prover, with its
    verifier's specification in `verifier/`). Until each protocol's verifier is Lean, the units its Python verifier reads
    are on the soundness path: `ml.circuits`, `ml.commitments`, `ml.crypto` and `ml.randomness` (`SOUNDNESS_PATH`, which
    only shrinks); core reaches neither the catalog (`ml.circuits.ops`, `ml.circuits.silicon`) nor the kernels. The rest
    can't affect soundness.
- `Security/`: the Lean every guarantee is stated and proved in (Lean, below).
- `integrations/`: one directory per serving stack, holding only what attaches Verity to it. `integrations/vllm/`
  (`verity_vllm`) builds each served workload's circuit and commits its values while serving.
- `examples/`: one directory per demo: its workload, placement, commands and doneness (`examples/README.md`).
- `benchmarks/`: the measuring side: reproducible workloads and measurements, among them `one_stage/`, the one-stage audit
  end to end, GPU probes, and `benchmarks/census/`, data that `verity` only takes as parameters.
- `tools/verity/`: the `verity` command, `check/` (the one check), `circuit_check/` and `lean/` (the Lean check).
- `fixtures/artifacts.json`: every replicated fixture by `art:` id. `tests/`: invariants of the whole tree. The archived
  proof systems (A-GKR, B-Ligero, D-SP1, the VOLE prover) were deleted in `2091252fe3`; `git checkout 2091252fe3^ --
  archive` restores them.
- The team's own tooling and machines, which nothing in `verity/` reads: `tools/research/` (runs, the evidence store,
  messaging, pods), `tools/agent-guard/`, `tools/move/` and `infra/` (the team skill). They stay here and are never
  exported to the public repository; new infra goes here too. compute-verification/infra holds the website, the executor
  (its `infra/`) and its `.cursor`, nothing else.

Dependencies point inward (`tests/test_boundaries.py`; every list it holds only shrinks). A verifier imports no
`verity.ml` code; until its protocol's verifier is Lean, a Python verifier keeps only the edges into `verity.ml`
that `CORE_ML_EDGES` lists. `verity.ml` never imports `verity.core` (`ML_IMPORTS_CORE` names the exceptions).
Inside `verity` each unit imports only what `ALLOWED` lists, and `verity` imports no other distribution of the workspace.
Nothing outside `infra/` imports it (`INFRA_IMPORTERS`), and nothing under `verity/` imports an integration (`KNOWN` in
`tests/test_integration_boundaries.py`). `benchmarks/` is measurement only: nothing outside it imports its code, by
package name or by a path put on `sys.path` or loaded by file (`BENCHMARKS_IMPORTERS` names today's importers); a number a
protocol uses reaches it through a card file citing an `art:` id.

`verity` is the one command: `verity check [tests|lean|circuits]`, `verity bench|demo NAME`, `verity --help`.

~~~text
uv sync --all-packages --extra torch-cpu        # the whole workspace, editable, with CPU-only torch (GPU work: a member's `torch` extra)
uv run tools/verity/check/suites.py             # the suites whose inputs changed; --quick: no slow tests
uv run --extra torch-cpu verity check           # the whole check
~~~

The team's agents run these remotely (the team skill).

## Design rules

The rules in force, each stated as itself; the friction skill has the standing rulings stated nowhere else.

- **Circuits and values.** A circuit holds only gates. Everything a program or a request brings, every intermediate and
  every output, is a value, touched only inside a zero-knowledge proof. Only a proof block's inputs and outputs are
  committed, each under a hiding leaf; the values inside a proof block never are.
- **Everything zero-knowledge.** Every protocol is, PoUS included. Only four kinds of things are public, each with its
  reason: protocol structure, the protocol's outputs, the verifier's coins and samples, and hiding commitments. Partitions
  are private and their properties public: a proof block's outputs fit its Definition's public size class, a power of two
  from 16 to 262,144 bits, and its output bits count in leakage (provisional; `UNIT_CLASS_MAX` = 16 restores the
  16-bit limit). A plain path, Flock without `--zk`, stays for a deployment whose values needn't be hidden: it
  works end to end beside the zero-knowledge paths, tests cover both, its guarantee is soundness alone, and `flock-verify`
  refuses a plain session past what the plain theorem covers (one table today).
- **One proof system** for PoUW, PoUS and network certification: the workers behind the ZK server, the challenger and the
  verifier. A protocol states what is committed, which units need proofs, the public Program each must pass and what it
  does with the outcome; the proof system owns commitments, live coins, the record, the selection, proving, verifying and the
  integrity profile. A protocol carries no policy parameter: it surfaces what its sample established (ε at δ), and the
  model's size is public. These core protocols wrap SPOT[Flock], the sampled proof system, in the recursion layer and the
  ZK server. SPOT[Flock] on its own (the research release and its post) has none of that: an inference server serves and
  commits its values, a proof server replays the sampled requests and proves them, and the verifier (the challenger and
  the Lean verifier) checks them. The proof server talks only to the verifier, and bits sent to the verifier are not
  exfiltration. The demonstration adds the network certifier as a separate layer: it stamps each request's release order
  and bucket at ingress, which the certified tick admits by, so the scheduler's choice of batch is no longer advice.
  SPOT[Flock] reads that stamp as a certified input and never imports the certifier, which has its own code, guarantee
  and proof; the two compose only above both.
- **One hash: SHA-512 with its whole output, everywhere.** Commitments are hiding commitments (`hm96` rows, or keyed
  leaves: a subtree's digest SHA-512(ks ‖ salt ‖ M_top) under a window's secret key) in frame-v3-sha512 trees, and signatures
  are hash-based (LMS, n = 64). Two named exceptions: Pearl's FP8 scheme as Pearl defines it,
  and Duo receipts, PoUW's per-step keyed u32 mul/xor/rotl folds over committed accumulator words, which bind under the
  assumption H-μ⁺ (`Accounting.Assumptions.TS.OperandHMu`).
- **Live coins only.** No Fiat–Shamir, except at two rates where live coins are impractical: PoUB's pass seeds
  inside an execution phase, from its live challenge and the previous pass's root; and PoUW's
  per-matmul noise seeds, seed_j = SHA-512(tag ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j)), from the execution
  phase's live salt (`pouw.salt`) and the operands' committed roots, which a sampled proof block's proof derives. Grinding is a term
  of the guarantee: a steer that needs m noised coordinates succeeds with probability at most μ^m a try. No
  retries: a failed proof is a visible failure, and nothing resamples. In the core protocols, outer proofs go through the ZK server as independent
  sessions: each staged statement is fixed before any session flips coins and gets exactly one session, with its own live
  coins in its own ZK server pass; no statement reads another session's transcript or coins; every session is recorded and
  every verdict counts (provisional).
- **Output accounting: every output bit is explained, certified or charged.** What leaves (y, the certifier's record of
  every byte out, timing included) must be a function of what came in (x, certified), registered information (weights,
  configs, the compiler B, Definitions), advice (a, committed before the coins) and the wrong blocks: given those, at most
  one y is self-consistent (else the bound adds log₂ of how many are). Explanation: the circuit C = B(x, y, a) outputs y's
  serialization and the audit checks the equality, and every bit B reads from y is such an output. Prefix stability:
  changing y at or after step s doesn't change what C computes before s. Every block is a function of its committed
  inputs. Anything the prover chooses that reaches y or B is advice and is charged; anything else that leaves is refused
  by the egress spec.
- **Nondeterminism is quantified, never buried.** Every value a circuit reads is the output of one of its
  blocks, a certified input, a registered value, or advice charged at its width. A value the served hardware decides
  (an atomics order, a split choice) that no block computes is never a free witness checked by a relation: serving makes it
  canonical so a block computes it, or it is advice. A relation check stands in for computing only where its outputs are
  unique: a Lean lemma, once per Definition and for all its other inputs, that any two witnesses it accepts give the
  same value on every output wire. Its witness then carries no bits; until the lemma holds, it counts as free input at
  its width.
- **Verifier cost: O(circuit description), never O(unrolled circuit).** A verifier works in
  proportion to the descriptor it reads (its Definitions, nodes and references) and never walks every gate, batch member
  or scan iteration: it summarizes once per specialized Definition and multiplies through batches and scans
  (`NciCheck.Desc`). An unrolled walk survives only as a difftest's reference. `tests/test_verifier_cost.py` enforces it
  (the walks still open are its `ALLOWED`), and `benchmarks/verifier_cost/` measures it.
- **PoUW is mostly useful work.** A PoUW scheme leaves the useful work at least half of the honest developer's GPU
  effort, its commitments and checks included. The trivial PoUW, which hashes every product and leaves
  useful work near 1% of that effort, is ruled out as an answer; it stays on main only as a reference. Pearl+ is held
  to the same floor.
- **Auditor-trusted devices.** The network certifiers and memory challengers sign what they see; what they sign is
  witness and never reaches the auditor in the clear. Each is the auditor's hardware on a machine the developer doesn't
  control, never a process on the developer's node, and trust in it and its host is the named assumption `trusted-host`.
  Nothing the developer runs or hosts is trusted. "Network certifier" is its one name, never "warden" or "tap".
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
  before any coin (under a two-stage sampling design, a sampled replay block's interior between its two stages); and on a
  sample, the sampled proof blocks' openings, to the prover behind the ZK server.
- **Results are per monitoring period, over all compute.** The time length of interest is the monitoring
  period (Glossary), not the execution phase, the run or the demo. Every result, in a report, a
  table, a PR or a message, is presented for monitoring periods of one week, one month and one year, and over
  all compute, in AND gates: the world's (*adversary resource model*), a datacenter's beside it, though a demo checks
  one. The datacenter's size (from `benchmarks/census/`) and the execution phase's length are stated parameters, and a
  figure scaled from one machine says which parts are proved and which are scaled. Give each bound as its pooled total over
  the period and as a share of the compute it covers, beside the per-execution-phase figure, never instead of it: a
  per-execution-phase δ that looks weak is not the result.

## Tests

Each package's tests are one suite in its own `tests/`, run in its own process and cached by input; the root `tests/`
holds only invariants of the whole tree. How to run them, what a suite may read, and the rules for slow tests, fixtures
and dummy credentials: the testing skill.

No test reads wall-clock time (no sleeps, clock readings or timeouts:
wait on the event or inject the clock; `tests/test_no_wall_clock.py`, its exceptions in `ALLOWED`). A test that flakes is
deleted, not rerun. A test that adds more than about a minute to `check` says in its docstring what it catches that a
cheaper test can't.

Bookkeeping warns and a later PR fixes it: what can be put right after landing without any claim being false meanwhile. The root suite's bookkeeping tests (`BOOKKEEPING` in `tests/conftest.py`: lane text, retired terms, doc
paths, naming, stale allowlists, Markdown caps) and ruff's style findings warn. Ruff's errors, a crashed scan and
everything that can make a claim false or can't be undone still fail.

## Circuits

A change that adds or modifies a circuit (a Definition, a subcircuit template, a Boolean lowering) passes
`uv run circuit-check <definition|template|row>`, with its report in the PR. A new Definition gets a binding in its
module's sidecar (`<module>.circuit-check.py`, pins in `.circuit-check.json`), which the suite enforces.

## Checking and merging

There is no GitHub Actions CI. `check` (`verity check`, `tools/verity/check/check.py`) is the one check: every suite,
`circuit-check --all`, Flock's Rust build and tests, the Lean verifier's build and partition check, the Lean check, and
`lean-agreement` (against upstream's Rust verifier) when Flock or its Lean verifier changes. The runners report facts: each
suite, test, target and step's real verdict. What gates, what may stay red and what a red triggers is drive's config,
`tools/verity/check/policy.toml`: its `off_gate` jobs, and its allowances, each an id, an owner, a UTC expiry and a reason, under
which drive takes a failure as not red until it expires. `circuit-check` is off the merge gate until `circuit-check --all`
fits inside the rest of the check: a recorded check, `research merge` and the move of `main` leave it out, and it runs after
landing on `checked`'s tip, where a red holds the PRs touching its inputs until it passes.

A pass of a test suite, a circuit-check target or a Lean package's Lean check is reused only while every input it read is
unchanged.

A part's verdict is reused while a hash of its declared inputs, the base image and its runner is unchanged, so a
docs-only change reruns nothing; a nightly uncached check must agree, and a disagreement is fixed in the check. `main` holds
only fully checked commits: drive lands `next`'s longest reviewed prefix on `checked`, and its low-risk members (no Lean, lock or
soundness-path change: drive.toml's `lowrisk_excludes`) on their ready mark alone, runs the full check on its tip itself,
and moves `main` only to a commit of `checked` whose full check passed every step on the merge gate (how a PR lands: the
team skill). A red's culprit is the one landing whose change touches what its failure names (its files, the files declaring
a Lean declaration it names, else the failed suite's or step's declared inputs), never found by bisecting: its lane fixes it
forward, marking a fix within 30 min, or drive then reverts it on `checked`; a red naming several landings reverts none and
rings each one's lane. A ready mark admits a PR: the sync's own merge onto `next` (a conflict, drive.toml's `landing_gates`)
is the only check before it joins, and a body naming no live goal gets a warning, never a refusal. A change a `warn_gates`
gate flags (drive.toml; a Markdown file over its cap) lands too: drive tells its owner, who fixes it in their next PR. A red
ratchet-gate step never reverts (drive.toml's `warn_steps`): drive rings the culprit's owner, who fixes it forward in their
next PR. A Lean, lock or spec change lands on its verdict like any other: drive then runs lean-audit alone on `checked`'s
tip (its lock run), commits the lock records it regenerated there, and checks that commit, so `main` moves only to a commit
whose records are the regenerated ones and whose gated steps all passed on it. The lock run never fails on a lock's
bookkeeping: it repairs it in that commit, with a warning for each repair (The lock, under Lean).

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
- **Lean is code or security analysis.** Code stays with its subsystem and never requires `Security/`: Flock's
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
  only `propext`, `Classical.choice` and `Quot.sound`; no `sorry`, `axiom`, `native_decide` or kernel bypass; a
  premise is a named `Prop`, taken as a hypothesis or read in a guarantee's statement, listed by name in the lock: a
  named assumption (`assumptions`) is declared in `Security/Definitions/Assumptions.lean` or a module under it, and
  every other premise is open (`open`, with the Linear issue that discharges it and its due date), a list a PR's
  `--update` never adds to, unless it is part of what its guarantee says (`domain`, with a one-line why; an honest
  witness must conclude it). Each deployed guarantee (`deployed`) rests, outside the trust base (`trust.base`: SHA-512's
  collision resistance, uniform randomness, an honest verifier) and the premises a ruling's rule lets it alone rest on
  (`trust.extra`, each with its reason), only on the premises its `trust_gap` entry lists, which `--update` never adds
  to; the check prints each one's gap, the number Daniel drives to zero.
- **The lock.** Each package's `lean-audit.json` lists its guarantees, and `check.py --update` records each one's
  statement, the definitions it reads and the premises it takes, so a changed statement is a changed record.
  A guarantee that only moved keeps its record (`--update --moved MOVES.json`). Bookkeeping warns rather than fails
  after a landing. A PR's own check still fails on it. Drive's lock run repairs it in its lock commit and
  warns, naming each entry's owner, and the next PR fixes it. Bookkeeping is an unlisted or misfiled premise, a guarantee
  the witnesses rule holds to missing witnesses, a stale entry, a premise a `trust_gap` entry lists that its guarantee
  no longer rests on, a missing owner, or a cap of `unread-caps.json` or
  `tests/test_lean_packages.py`. A cap more than its room above its count only warns, in every check. The build, the replay, the axioms, `sorry`, `native_decide`, a kernel bypass and a
  malformed lock always fail. Lock files conflict on merge only where `tools/verity/lean/merge.py` can't vouch for the
  merge: a lock in a `format` newer than it knows, the worktree's copy merging in newer merge rules, or a merge that would
  drop most of a section. The registered driver runs `origin/main`'s copy wherever the repository has that ref.
- How to work on Lean (the edit loop, why two agents never share a writable `.lake`, `upstream` watch entries): the
  lean-proofs skill.

## Working principles

- Solve the task with the smallest clear implementation, reusing code and commands before adding any. Enforce
  deterministic requirements in code, with the fix in the failure message, rather than in prose. Remove obsolete rules and
  machinery rather than accumulating them.
- **Something broken becomes a benchmark.** A step that fails or costs far too much (an OOM, hours where minutes would
  do) gets a directory under `benchmarks/` with one or two numbers, a guard that a faster version is still correct, its
  baseline and a target. Several workers then drive the numbers down, each with its own approach, branch and `research
  run` compute. Each measured run carries a label with its numbers, and the best one that passes the guard lands.
- **The tree gets smaller over time.** Make two interfaces or definitions of one thing one, make things extensible,
  delete dead and superseded code, and move data and logs to the evidence store.
- **A decision rolls out in the change that records it.** A PR that records a ruling, or retires a term, a path or a
  procedure, rewrites every place that states the old way: `git grep` this file, ARCHITECTURE.md, GLOSSARY.md, the
  skills, READMEs, docstrings and CLI help. Fix a stale statement on sight. `tests/test_doc_paths.py` warns on a doc
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
  `CHANGELOG.md`, `*.template.md`, `PROTOCOL.md`; `tests/test_repository.py`). The root has no README: an agent starts at
  this file and `ARCHITECTURE.md`, and a directory keeps a README only where an agent couldn't find its way without one.
  A protocol keeps its `PROTOCOL.md` only while experimental: the PR whose Lean covers it deletes it, leaving a README of outline, signposts and core API.
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
`coordinating`, `using-slack`, `asking-daniel` and `writing-goals` (your goals in the plan Daniel reads).
