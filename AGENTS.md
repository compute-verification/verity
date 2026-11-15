# Verity

Verity is a system by which an AI developer can make verifiable claims to external auditors about how it is using its
datacenters, without revealing its algorithms or data. It represents the computations exactly as Boolean circuits,
proves claims about them, and derives resource accounting (compute, network, memory) and workload classification from
those proofs. `ARCHITECTURE.md` has the architecture and the Glossary, whose terms code, docs and output use. A
directory's `AGENTS.md` adds what holds only there. Your checkout and your
prompt can be older than `main`: `git show origin/main:AGENTS.md` wins.

## Map and commands

`tests/tree.toml` is the layout of record, and `tests/test_tree.py` holds the tree to it.

- `verity/`: the distribution `verity`: what a party, a channel, a device, a protocol and a configuration are
  (`verity/party.py`, `channel.py`, `device.py`, `protocol.py`, `config.py`), the runtime (`verity/runtime/`), the
  circuits (`verity/circuits/`), the commitment schemes (`verity/commitments/`) and the protocols
  (`verity/protocols/`) by layer: proofs, resource accounting, workload classification, and on top the two
  deployment patterns, remote and onsite. `verity/AGENTS.md` has their layout and the import rules. `verity`
  re-exports nothing.
- `security/`: the Lean every guarantee is stated and proved in (Lean, below).
- `adapters/`: one directory per serving system, holding only what attaches Verity to it (`torch/`, and `vllm/` on
  it): an integration, in the Design rules.
- `hardware/`: the data sheets, TOML only (`cards/`, `nodes/`), which `verity/device.py` reads.
- `demos/`: one configuration file per demo, which `verity run` reads (`demos/README.md`).
- `benchmarks/`: the measuring side: reproducible workloads and measurements, and `benchmarks/census/`, data `verity`
  only takes as parameters.
- `tools/`: the `verity` command and the Lean check (`verity/`), the check (`check/`), the settings registry
  (`config/`), the Python held to each Lean (`difftest/`) and the layout's move maps (`move/`).
- `tests/`: invariants of the whole tree. `artifacts.json` registers every replicated fixture by `art:` id.
- The team's own machinery, which nothing in `verity/` reads: `infra/`, `tools/research/`, `tools/agent-guard/` (the
  team skill). They stay here until the move `tests/tree.toml` plans, and new infra goes here too; this repository is
  public whole. Retired code is deleted, never archived: history keeps every file.

Dependencies point inward: `tests/import_order.toml` lists which folder may import which, `tests/test_boundaries.py`
holds the tree to it, and `uv run tools/check/import_order.py` writes its tables. To allow an import, write it, run the
generator and say why in the pull request. No file under a `verifier/` loads one under a `prover/`. A place under a
`verifier/` or a `device/`, or one the listing marks `trusted`, is trusted code, and what it reaches outside the trusted
places is the soundness path, which only shrinks: `verity/circuits`, `verity/commitments`, `verity/commitments/crypto`
and `verity/commitments/randomness`; none reaches the catalog (`verity/circuits/ops/` and `silicon/`) or the kernels.
Nothing under `verity/` imports an adapter (`tests/test_integration_boundaries.py`). Nothing outside `benchmarks/`
imports its code; a number a protocol uses reaches it through a card file citing an `art:` id.

`verity` is the one command: `verity check [tests|lean|circuits]`, `verity bench NAME`,
`verity demo [NAME]`, `verity run CONFIG`, `verity verify proof ARGS`.

~~~text
uv sync --all-packages --extra torch-cpu   # the whole workspace, editable, CPU-only torch (GPU work: a member's `torch` extra)
uv run tools/check/suites.py               # the suites whose inputs changed; --quick: no slow tests
uv run --extra torch-cpu verity check      # the whole check
~~~

## Design rules

The rules in force, each stated as itself; the friction skill has the standing rulings stated nowhere else.

- **Circuits and values.** A circuit holds only gates. Everything a program or a request brings, every intermediate and
  every output, is a value, touched only inside a zero-knowledge proof. Only a proof block's inputs and outputs are
  committed, each under a hiding leaf; the values inside a proof block never are.
- **Everything zero-knowledge.** Every protocol is, PoUS included. Only four kinds of things are public, each with its
  reason: protocol structure, the protocol's outputs, the verifier's coins and samples, and hiding commitments. Partitions
  are private and their properties public: a proof block's outputs fit its Definition's public size class, a power of two
  from 16 to 262,144 bits, and its output bits count in leakage (provisional; `SIZE_CLASS_MAX` = 16 restores the
  16-bit limit). A plain path, Flock without `--zk`, stays for a deployment whose values needn't be hidden: it
  works end to end beside the zero-knowledge paths, tests cover both, its guarantee is soundness alone, and
  `verity verify proof` refuses a plain session past what the plain theorem covers (one table today).
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
  assumption H-μ⁺ (`Accounting.Assumptions.TS.OperandHMu`). Transport has one more: ML-KEM-1024, for a cableless remote
  party's sessions only, its key always XORed with a SHA-512 one (never below hash-only) and never on pair links
  (`ind-cca/ml-kem-1024`).
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
  (`InferenceOnlyCheck.Desc`). An unrolled walk survives only as a difftest's reference. `tests/test_verifier_cost.py` enforces it
  (the walks still open are its `ALLOWED`), and `benchmarks/verifier_cost/` measures it.
- **PoUW is mostly useful work.** A PoUW scheme leaves the useful work at least half of the honest developer's GPU
  effort, its commitments and checks included. The trivial PoUW, which hashes every product and leaves
  useful work near 1% of that effort, is ruled out as an answer; it stays on main only as a reference. Pearl+ is held
  to the same floor.
- **Auditor-trusted devices.** The network certifiers and memory challengers sign what they see; what they sign is
  witness and never reaches the auditor in the clear. Each is the auditor's hardware on a machine the developer doesn't
  control, never a process on the developer's node, and trust in it and its host is the named assumption `trusted-device`.
  Nothing the developer runs or hosts is trusted. "Network certifier" is its one name, never "warden" or "tap".
- **The audited hardware is untrusted, and so is its vendor.** No guarantee rests on a GPU attesting its own
  SKU, firmware or clocks. Hardware constants (f_max, peak rates) come from the data sheet under two or three readings,
  and later from a separate physical calibration. A device's capacity is an assumed bound on its true peak compute
  capacity, and nothing else: no power certifier, power ceiling, energy floor or meter enters a protocol, premise or
  plan.
- **Lean is the definition and the verifier.** The provers' Python (the listing's untrusted places) mirrors it for
  the prover's speed, held to it by difftest. Each protocol switches by the friction skill's switching ruling; until
  then its Python verifier and what it loads stay on the soundness path (the listing's trusted places), which gains no
  load.
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

There is no GitHub Actions CI. `check` (`verity check`, `tools/check/check.py`) is the one check: every suite,
`circuit-check --all`, Flock's Rust build and tests, the Lean verifier's build and partition check, the Lean check, and
`lean-agreement` when Flock or its Lean verifier changes. It reports each part's real verdict; what gates and what may
stay red is `tools/check/policy.toml` (`tools/check/README.md`). A part's pass is reused only while every input it read
is unchanged. Every commit of `main` is checked whole, before it lands or right after, and the verified queue moves
`main` only to a candidate with no new failure: the team skill's Landing.

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
  executable verifier and the ZK server (`verity/protocols/proofs/flock/verifier/`, with the release side in
  `verity/protocols/proofs/recursion/verifier/zk_server/`), the PoUS grader, the certifier program, the inference-only
  check and PoUB's verifier, each in its protocol's `lean/` with a difftest. Under `FC_ZK_SERVER=1` every byte
  that leaves is computed by `Flock.ZkServer`; the ZK theorem (`Flock.zk_server_lean_loop`) still reads the released
  bytes through hypotheses not yet proved about the live path. The analysis is `security/`, three trees: definitions
  (`Definitions/`, Mathlib only), properties (`Properties/`) and proofs (`Proofs/`, ArkLib allowed, with
  `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each proved guarantee); the first two never read the third
  (`trees` in `security/lean-audit.json`), and a subsystem's Lean goes in its own area of each (`security/README.md`).
  All packages share one toolchain and one Mathlib revision, bumped together in a PR of their own. A package requires
  Flock's verifier only while `REQUIRES_FLOCK_VERIFIER` lists it; the three base packages require nothing above them
  (`BASE_LEAN`).
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
  statement, the definitions it reads and the premises it takes, so a changed statement is a changed record. The build,
  the replay, the axioms, `sorry`, `native_decide`, a kernel bypass and a malformed lock always fail; what is bookkeeping,
  how a records run repairs it, and how lock files merge: `tools/verity/lean/README.md`.

## Working principles

- Solve the task with the smallest clear implementation, reusing code and commands before adding any. Enforce
  deterministic requirements in code, with the fix in the failure message, rather than in prose. Remove obsolete rules and
  machinery rather than accumulating them.
- **Add only what was asked for.** A design, a page, a schema, a goal or a message holds what solves the stated problem
  and nothing else. A field, count, label, option or caveat goes in only when it answers a question someone has asked:
  that the information is at hand, or might be useful later, is no reason. Offer an extra in one line, and don't build it in.
- **Something broken becomes a benchmark.** A step that fails or costs far too much (an OOM, hours where minutes would
  do) gets a directory under `benchmarks/` with one or two numbers, a guard that a faster version is still correct, its
  baseline and a target. The best measured version that passes the guard lands.
- **A slip made twice becomes a tool.** When you make the same kind of mistake a second time, or see that two agents
  have each made it, a note to be more careful is no fix: that day, change the tool so that kind of mistake can't
  happen, or fails with the fix in its message (a refusal in the shell hook, a default, a check), in a PR that names the
  slips it ends. Another owner's tool gets the PR all the same, and its owner is told.
- **The tree gets smaller over time.** Make two interfaces or definitions of one thing one, make things extensible,
  delete dead and superseded code, and move data and logs to the evidence store.
- **A decision rolls out in the change that records it.** A PR that records a ruling, or retires a term, a path or a
  procedure, rewrites every place that states the old way: `git grep` the `AGENTS.md` files, ARCHITECTURE.md,
  GLOSSARY.md, the skills, READMEs, docstrings and CLI help. Fix a stale statement on sight. `tests/test_doc_paths.py`
  warns on a doc naming a dead path.
- Never print a secret's value. List environment variable names with `compgen -e`, never `env`, `printenv` or tmux's
  `show-environment`, and an env file's with `python3 tools/agent-guard/envnames.py FILE...`, never `cat`, `sed`, `cut`
  or `grep` on it or on a glob over its directory. Look at a key with `stat`, `sha256sum` or `openssl pkey -in FILE
  -pubout`.

## Where writing goes

**The repository says what is true now and how to reproduce it. It is not a record.** A reader learns from it what each
part is and does, its contract in the Glossary's terms, the rules, and how to run each measurement (a benchmark's runner
and its parameters). What a run measured, what happened and who decided live outside it, and the tree doesn't cite them:

- **In the tree:** code, tests, Lean, circuits and their pins, the parameters and configuration code reads (with their
  public sources: a data sheet, the census), small fixtures beside their generators, and maintained current-truth docs
  under the names `tests/test_repository.py` allows. The root has no README: an agent starts at
  this file and `ARCHITECTURE.md`, and a directory keeps a README only where an agent couldn't find its way without one.
  A protocol keeps its `PROTOCOL.md` only while experimental: the PR whose Lean covers it deletes it, leaving a README of
  outline, signposts and core API.
- **Not in the tree, and not cited from it** (code, comments, docstrings, Lean, READMEs, docs): measured results
  (numbers, tables, plots, overheads), reports and drafts, and the process's ids and names: run, `art:` and `note:` ids,
  Notion pages, Linear issues, Slack threads, PR numbers, agent ids, lane and worker names, dates, and attributions such
  as "(name, date)". A rule or a constraint is stated as itself. Only the skills, which describe the team's process,
  name its roles (a handle as an owner). Results are published from the store, never copied into the tree: a
  reader who finds no overhead figure in the repository has found it working as intended, and runs the benchmark or reads
  the published result. Results live in the evidence store, why a line changed in git.
- **An id a tool reads is data, not prose.** It goes in a declared data file the tool reads (`artifacts.json`'s
  `art:` ids and digests, a lock's records, the handle registry, a table's runs of record), never in a comment, a
  docstring or a doc.

## Skills and nested files

Read the file before the work it names. Skills are in `.agents/skills/`.

| When you | Read |
| --- | --- |
| change code under `verity/`: a protocol, the circuits, the commitments, the runtime or an import | `verity/AGENTS.md` |
| write, move or run tests, or a suite fails its read guard | the testing skill |
| prove or state anything in Lean (the edit loop, why two agents never share a writable `.lake`) | the lean-proofs skill |
| write or tune a PoUW GPU kernel | the kernel-engineering skill |
| call something served, end to end, proved or credited | the smell-check skill |
| are one of the team's own agents | the team skill first, then the skills it names |
