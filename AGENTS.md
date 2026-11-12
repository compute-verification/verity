# Verity

Verity represents AI datacenter computations exactly as Boolean circuits, proves claims about them, and derives
compute, memory and network accounting from those proofs. `ARCHITECTURE.md` has the architecture and the Glossary, whose
terms code, docs and output use. This file states what holds on `main` now; each package's README and docstrings say what
it contains. Your checkout and your prompt can be older than `main`: `git show origin/main:AGENTS.md` wins.

## Map and commands

The tree is in the layout below. The move that made it changed paths, not code, so a line marked "Target, not yet" says
what the layout's design has and the tree does not.

- `verity/`: the distribution `verity` (the root `pyproject.toml`; its module list is `verity/README.md`). Import from the
  owning module (`from verity.circuits import Program`); `verity` re-exports nothing.
  - `verity/protocols/`: one directory per protocol, the list flat: `flock`, `spot`, `recursion`, `pouw`, `pous`, `poub`,
    `certifier` and `inference_only`. Beside them, `remote` and `onsite` are configurations, not protocols: remote is
    what an auditor with no devices at the datacenter runs, and onsite what an auditor with devices there runs. Target,
    not yet: `compose/` (today's `remote/verifier/composition/`), a ninth protocol on top of the audits; `remote/` and
    `onsite/` under `verity/configurations/`, which no protocol imports; and `experimental/NAME/` for a whole directory
    no configuration lists. Inside a protocol, `verifier/` is trusted code (a Python verifier until its Lean verifier,
    its `lean/`, replaces it), `prover/` is what its prover builds and its verifier doesn't read, and `device/` is the
    auditor's device (the network certifier, the memory challenger, the challenger).
    `flock/` is the proof system: its Lean verifier and the ZK server (`flock/verifier/`, `Flock.Firewall`), its prover
    (`flock/prover/`: `verity_flock`, its own distribution) and the agreement with upstream's verifier
    (`flock/agreement/`). `spot/verifier/` is sampled proofs, with the integrity profile (`profile`). The inference-only
    policy's check (`inference_only/verifier/nci/`), `remote/` and `onsite/` are Lean only. `interface/` is the Lean
    interface each protocol's check implements. Code the move brought from `benchmarks/`, which no unit rule reads yet:
    `pouw/clear/`, `pouw/receipts/`, `poub/device/live_verifier/`, `poub/verifier/remote_session/`,
    `spot/device/live_challenger/` (the auditor host's challenger service), `inference_only/nci/` and
    `inference_only/rho/`.
  - `verity/circuits/`: the circuit language (format in `circuits/PROTOCOL.md`). Three base Lean packages, which Flock's
    verifier and the packages requiring it load, hold the shared Lean: `verity/circuits/base/` (bytes, hashes and
    parsing), `verity/circuits/lean/` (the circuit format) and `verity/commitments/lean/` (frame-v3 and the
    sampling of units). `verity/definitions/`: the circuits
    written in it: `ops` (the primitive circuits' Definitions), `silicon` (the GPUs' exact semantics), `kernels`,
    `private_circuits` (the universal circuit), the served workloads' Definitions, and the evaluator and a circuit's
    requirement in Lean (`evaluate`, `requirement`). Target, not yet: `verity/circuits/` also holds the circuits a
    protocol names, and the served kernels' circuits go with the adapter that serves them.
  - `verity/commitments/`: the commitment schemes, with `crypto`, `randomness` (the one challenge derivation), `lms`
    and `identity` (Lean), `attest` (an auditor's device's signed period roots) and `windows` (what an adapter hands
    the prover for each execution phase).
  - `verity/hardware/`: the registered hardware: `cards/` (one TOML data sheet per GPU, host or server model), `nodes/` (one per kind of node), `site` (its
    types), `registration` (the registration log) and `record` (a record entry). Target, not yet: the data sheets in a
    top-level `hardware/` that holds data only, and the types in one file of `verity/`.
  - `verity/configuration/` (not `verity/configurations/`): `wire` (the wire every party speaks), `deploy` (parties,
    channels, the configuration and the runtime that runs them) and `config/` (`verity_config`, its own distribution). Target, not yet: four files
    directly under `verity/` (what a party is, the registered hardware, what a protocol declares, the configuration)
    and `verity/runtime/`, which runs parties over channels.
  - `verity/claims/`: the security claims' ids.
- `security/`: the Lean every guarantee is stated and proved in (Lean, below).
- `adapters/`: one directory per serving system, holding only what attaches Verity to it: an adapter, which the Design
  rules below still call an integration. `adapters/vllm/` (`verity_vllm`) builds each served workload's circuit and
  commits its values while serving. Target, not yet: a torch adapter beside it, which the vLLM adapter sits on.
- `demos/`: one directory per demo: its workload, placement, commands and doneness (`demos/README.md`).
- `benchmarks/`: the measuring side: reproducible workloads and measurements, among them `one_stage/`, the one-stage audit
  end to end, GPU probes, and `benchmarks/census/`, data that `verity` only takes as parameters.
- `tools/verity/`: the `verity` command, `check/` (the one check), `circuit_check/` and `lean/` (the Lean check).
  `tools/difftest/`: the Python held to each Lean (`verity_difftest`, its own distribution) and the harness's Lake
  package, which each Lean difftest under `verity/` requires by path.
- `fixtures/artifacts.json`: every replicated fixture by `art:` id (target, not yet: no `fixtures/`, each file beside
  its reader). `tests/`: invariants of the whole tree. The tree keeps no archive directory: code that is retired is
  deleted, and history keeps every file. The archived proof systems (A-GKR, B-Ligero, D-SP1, the VOLE prover) were
  deleted in `2091252fe3`; `git checkout 2091252fe3^ -- archive` restores them. The H200 demo and the rest of what the
  layout set aside went with the directory; `git checkout 61f13c2275 -- archive` restores it as it last stood.
- The team's own tooling and machines, which nothing in `verity/` reads: `tools/research/` (runs, the evidence store,
  messaging, pods), `tools/agent-guard/`, `tools/move/` and `infra/` (the team skill). They stay here and are never
  exported to the public repository; new infra goes here too. compute-verification/infra holds the website, the executor
  (its `infra/`) and its `.cursor`, nothing else.

Dependencies point inward (`tests/test_boundaries.py`; every list it holds only shrinks). Its units keep the names they
had before the layout, in two sets: `core.*`, the verifiers' side, trusted code, and `ml.*`, the provers' side and
the Python that mirrors the Lean for speed; `PLACE` says where each unit's files are. A verifier imports no ml
code; until its protocol's verifier is Lean, a Python verifier keeps only the edges into ml that `CORE_ML_EDGES` lists,
and the units it reads are on the soundness path: `ml.circuits` (`verity/circuits/`), `ml.commitments`
(`verity/commitments/`), `ml.crypto` (its `crypto/`) and `ml.randomness` (its `randomness/`) (`SOUNDNESS_PATH`, which
only shrinks; `PLACE` gives the first two a few files outside those directories too); core reaches neither the catalog
(`ml.circuits.ops`, `ml.circuits.silicon`: `verity/definitions/ops/` and `silicon/`) nor the kernels. The rest can't
affect soundness. ml never imports core (`ML_IMPORTS_CORE` names the exceptions). Inside `verity` each unit imports only
what `ALLOWED` lists, and `verity` imports no other distribution of the workspace. Nothing outside `infra/` imports it
(`INFRA_IMPORTERS`), and nothing under `verity/` imports an adapter (`KNOWN` in `tests/test_integration_boundaries.py`).
A Lean package requires Flock's verifier only while `REQUIRES_FLOCK_VERIFIER` lists it, and the three base packages
require nothing above them (`BASE_LEAN`).
`benchmarks/` is measurement only: nothing outside it imports its code, by package name or by a path put on `sys.path`
or loaded by file (`BENCHMARKS_IMPORTERS` names today's importers); a number a protocol uses reaches it through a card
file citing an `art:` id. Target, not yet enforced: the layout's own rules. The runtime imports no protocol, and no
protocol or adapter imports it; `circuits/` imports nothing and `commitments/` only `circuits/`; no code imports
`security/` or the hardware data; inside `verity/protocols/`, flock is at the bottom, spot and recursion on it, the
audits on those, and compose on top; no protocol imports a configuration.

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
- **Lean is the definition and the verifier.** The provers' Python (the `ml.*` units) mirrors it for the prover's
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
  never interprets, whose length is charged; the values, committed in a `verity.commitments` scheme and registered
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
suite, test, target and step's real verdict. What gates and what may stay red is the landing policy,
`tools/verity/check/policy.toml`: its `off_gate` jobs, and its allowances, each an id, an owner, a UTC expiry and a reason, under
which the verified queue takes a failure as not red until it expires. `circuit-check` is off the merge gate until
`circuit-check --all` fits inside the rest of the check: a recorded check, `research merge` and the move of `main` leave it
out; a candidate's check still runs it, and its red there holds nothing.

A pass of a test suite, a circuit-check target or a Lean package's Lean check is reused only while every input it read is
unchanged.

A part's verdict is reused while a hash of its declared inputs, the base image and its runner is unchanged, so a
docs-only change reruns nothing; a nightly uncached check must agree, and a disagreement is fixed in the check. `main` holds
only fully checked commits: the verified queue's loop (`research queue vqueue`, ci's) merges the marked PRs that have a land
verdict, are low-risk or are certified moves onto `main` as a candidate, checks it (by default graph2's tasks, with lean-agreement and
ratchet-gate run beside them; under `--gate check`, every job in shards), and fast-forwards `main` only to a candidate with
no new failure: each gating task passed, failed only where an unexpired allowance of policy.toml at its base covers, or
fails on its base too. A failure the candidate's base shares blames none of its PRs;
any other's suspects are the PRs whose change touches what it names (its files, the files declaring a Lean declaration it
names), every PR when none does: a lone suspect is ejected, several are bisected, and in the same tick the candidate and
those above it are rebuilt without them. A candidate holding a Lean, lock or spec change runs lean-audit alone first (its records run) and is checked on the
commit of the lock records it regenerated, so `main` moves only to regenerated records. What drive.toml lets warn (its
`warn_gates` and `warn_steps`), a step off the gate and a lock's bookkeeping hold nothing, and the owner fixes it in their
next PR: a warn gate's failure is printed at the ready mark, a red off the gate that its base's run doesn't share is rung to
who acts on it under the loop's `--gate check` (to no one under graph2), and a lock's bookkeeping is told to `lean` and each
owner a warning names. How a PR lands, from its ready mark to `main`, and the loop's rule for each case: the team skill's
Landing.

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
- **Lean is code or security analysis.** Code stays with its subsystem and never requires `security/`: Flock's
  executable verifier and the ZK server (`verity/protocols/flock/verifier/`, with the release side in
  `verity/protocols/recursion/verifier/firewall/`), the PoUS grader, the certifier program, the inference-only policy's check and
  PoUB's verifier, each in its protocol's `lean/` with a difftest. Under `FC_FIREWALL=1` every byte that leaves is
  computed by `Flock.Firewall`; the ZK theorem (`Flock.zk_server_lean_loop`) still reads the released bytes through
  hypotheses not yet proved about the live path. The analysis is `security/`, three trees: definitions
  (`Definitions/`, the package `security`, Mathlib only: the vocabulary, named assumptions and guarantee
  statements a reader checks), properties (`Properties/`, `security_properties`: statements about the code's Lean
  functions) and proofs (`Proofs/`, `security_proofs`, ArkLib allowed: every proof, lemma and proof-only definition, with
  `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each proved guarantee). No statement or module of the definitions
  or properties trees reads the proofs tree (`trees` in `security/lean-audit.json`, whose `except` is the cleanup's to-do). A subsystem's Lean goes in its own area of each tree (`Core`, `Nci`,
  `Pouw`, `Pous`, `NetworkCertifier`, `Flock`, `Accounting`). All packages share one toolchain and one Mathlib revision,
  bumped together in a PR of their own.
- **The Lean check** (`tools/verity/lean/check.py`) builds every file and replays every declaration through the kernel:
  only `propext`, `Classical.choice` and `Quot.sound`; no `sorry`, `axiom`, `native_decide` or kernel bypass; a
  premise is a named `Prop`, taken as a hypothesis or read in a guarantee's statement, listed by name in the lock: a
  named assumption (`assumptions`) is declared in `security/Definitions/Assumptions.lean` or a module under it, and
  every other premise is open (`open`, with the Linear issue that discharges it and its due date), a list a PR's
  `--update` never adds to, unless it is part of what its guarantee says (`domain`, with a one-line why; an honest
  witness must conclude it). Each deployed guarantee (`deployed`) rests, outside the trust base (`trust.base`: SHA-512's
  collision resistance, uniform randomness, an honest verifier) and the premises a ruling's rule lets it alone rest on
  (`trust.extra`, each with its reason), only on the premises its `trust_gap` entry lists, which `--update` never adds
  to; the check prints each one's gap, the number Daniel drives to zero.
- **The lock.** Each package's `lean-audit.json` lists its guarantees, and `check.py --update` records each one's
  statement, the definitions it reads and the premises it takes, so a changed statement is a changed record, which a PR's
  own check regenerates rather than fails on and a candidate's records commit carries (`vqueue_records`).
  A guarantee that only moved keeps its record (`--update --moved MOVES.json`). Bookkeeping warns rather than fails
  in a candidate's records run. A PR's own check still fails on it. The records run repairs it in its records commit and
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
  `show-environment`, and an env file's with `python3 tools/agent-guard/envnames.py FILE...` (on another machine,
  `ssh HOST python3 - FILE... < tools/agent-guard/envnames.py`), never `cat`, `sed`, `cut` or `grep` on it or on a glob
  over its directory, which matches the keys beside it. Look at a key with `stat`, `sha256sum` or
  `openssl pkey -in FILE -pubout`.

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
`coordinating`, `using-slack`, `asking-daniel`, `writing-goals` (your goals in the plan Daniel reads) and `team-shapes`
(which team a piece of work gets, each shape with what its uses measured, and how any agent tries a different practice).
