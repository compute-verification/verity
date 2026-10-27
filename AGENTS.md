# Verity

Verity represents AI datacenter computations exactly as Boolean circuits, proves claims about them, and derives
compute, memory and network accounting from those proofs. `ARCHITECTURE.md` has the architecture and the Glossary, whose terms
code, docs and output use. This file states what is true on `main` now: what each directory is, the design rules, and how
work lands. Each package's contents are in its own README and docstrings. Your checkout, and this file in your prompt, can
be older than `main` (a cloud VM boots from a snapshot), so when they differ, `git show origin/main:AGENTS.md` wins.

## Map and commands

Top-level directories (a grouping, not an import name, except `verity/`):

- `verity/`: the distribution `verity` (the root `pyproject.toml`), in two parts. The module list is `verity/README.md`.
  Import from the owning module (`from verity.ml.circuits import Program`); `verity` re-exports nothing. Each directory's
  tests sit beside it as their own suite.
  - `verity/core/` (`verity.core`): the verifiers, trusted code. `Security/Definitions/` states their guarantees, and one whose
    proof isn't in `Security/Proofs/` is open: that verifier's properties aren't proven yet.
    - `protocols`: the integrity profile (`profile`) and one package per protocol: sampled proofs (`sampled_proofs`, with
      its plans and consumers), PoUW (`pouw`), PoUS (`pous`), the bandwidth audit over PoUS's store (`bandwidth`), the
      remote bandwidth audit over a node's registered store (`remote_bandwidth`) and the network certifier
      (`certifier`), each a Python
      verifier with the code that runs it, until its Lean is its verifier of record (Design rules). A protocol's Lean
      code is its `lean/`. NCI is Lean only (`verity/core/protocols/network/nci/`).
    - `claims`: the ids of security claims.
    - The Lean in `primitives`, `devices`, `service` and `protocols`. `difftest` (`verity_difftest`, with its Lake
      package `verity_difftest`) holds the Python that mirrors each one to the Lean.
    - `service/flock/`: C-Flock's executable verifier of record (no dependencies), and the ZK server's Lean
      (`Flock.Firewall`, the `flock-firewall` binary). `service/firewall/` (`core_firewall`) is the ZK server's release
      side and record.
  - `verity/ml/` (`verity.ml`): Python and CUDA: the provers, and Python that mirrors the Lean for the prover's speed.
    Until each protocol's verifier of record is Lean, these units are on the soundness path, because its Python verifier
    reads them: `ml.circuits`, `ml.circuits.ops`, `ml.circuits.silicon`, `ml.kernels`, `ml.commitments`,
    `ml.crypto` and `ml.randomness` (`SOUNDNESS_PATH` in `tests/test_boundaries.py`, all that
    `ALLOWED` lets core reach; it only shrinks). The rest, C-Flock's prover, the private circuits and `ml.definitions`,
    can't affect soundness.
    - `circuits` (format in `circuits/PROTOCOL.md`): circuits and their evaluator (`evaluation`), the Boolean profile
      (`boolean`), what models compute (`ops`, the primitive circuits' Definitions), what the GPUs actually compute
      (`silicon`, the exact hardware semantics) and `targets`.
    - `kernels`: fast code, each kernel checked against its Definition.
    - `commitments` (with Merkle multiproofs, `multiproof`), shared hash code (`crypto`), and the one challenge derivation
      (`randomness`).
    - `private_circuits`: the universal unit, one public Definition that runs any private circuit of a size class.
    - `definitions`: the served workloads' kernel Definitions (`registry`, twins in `numerics`), their Boolean versions,
      the proof-block cut (`word`), and their encoding and digest (`encoding`).
    - `flock/` (`verity_flock`, a distribution of its own, `flock/pyproject.toml` with its package in `flock/python/`):
      C-Flock, the proof system. Its patches on upstream Flock, the CUDA prover, the zero-knowledge prover (`live/`,
      `flock-circuit --zk`), and `verifier/`: the verifier's specification and the scripts that run it, with upstream's
      Rust verifier as a cross-check.
- `Security/`: the Lean every protocol states its guarantees in, and their proofs: `Definitions/`, `Models/` and `Proofs/`
  (Lean below). `Properties/` is `security_properties` (Lean below).
- `integrations/`: one directory per serving stack, holding only what attaches Verity to it. `integrations/vllm/`
  (`verity_vllm`) builds each served workload's circuit and commits its values while serving.
- `examples/`: one directory per example: its workload, placement, commands and level of doneness; a milestone is a set of
  (example, level) pairs, which top keeps (`examples/README.md`).
- `benchmarks/`: the measuring side: reproducible workloads and measurements, among them `one_stage/`, the one-stage audit
  end to end, and the benchmark tools.
  - `benchmarks/census/`: data only. `verity` never reads it; it takes such numbers as parameters.
  - `benchmarks/tc_probe/`, `benchmarks/tc_probe_fp4/`, `benchmarks/native_peak/`: GPU probes.
  - `benchmarks/numerical/`: the table tooling of the archived proof systems.
- `tools/`:
  - `verity/`: the `verity` command (`cli/`) and the tools it runs: `check/` (the one check, run on `main` after each landing),
    `circuit_check/` (the property checks every circuit passes) and `lean/` (the Lean check, `check.py`).
  - `research`: the run recorder and the run-store client: runs, the evidence store, Slack, pods and deploys. It imports
    nothing from Verity.
  - `agent-guard`: the VM's secret guard, its OOM recorder, and the hook that keeps builds and tests off agent VMs.
  - `move`: layout moves.
- `infra/`: our machines.
  - `cluster/` (`cluster`): the shared machines' policy, planner and ledger. Stdlib only.
  - `control/`: the control pod's boot script and loops.
  - `cursor/`: the coordinators, each a plain Cursor cloud agent: `launch_coordinator.py` creates and archives them from
    Daniel's laptop.
  - `nebius/`: the console's units.
  - `slack-events/`: the Slack events service, a Vercel project (root `infra/slack-events/service`) that routes people's
    posts and new members through `research msg`'s code. comms owns it.
  - `ops/` (`verity_ops`): `ops deploy` and `ops keeper`, on top of `research`.
- `archive/`: proof systems and code that are kept but not developed: A-GKR, B-Ligero, D-SP1 and the VOLE prover, still in
  the benchmark tables, plus whatever a ruling retires. Its own rules are `archive/AGENTS.md`.
- `fixtures/`: `artifacts.json`, every replicated fixture by `art:` id, and the archived proof systems' fixtures.
- `tests/`: invariants of the whole tree, the import rules among them.

Code reaches the Nebius nodes only through `research deploy install` from a pushed commit, listed in
`tools/research/src/research/pods/nebius/deploy.toml`. Delete data you don't own on a node only with
`research retention gc` or `research retention rm --approved-by @OWNER`, never a bare `rm`.

Dependencies point inward, and `tests/test_boundaries.py` enforces them:

- A verifier imports no `verity.ml` code (Design rules). Until its protocol's verifier of record is Lean, a Python
  verifier in `verity.core` keeps the edges into `verity.ml` it had on 18 Oct (`CORE_ML_EDGES`) and gains none, and
  what they reach (`SOUNDNESS_PATH`) only shrinks. `verity.ml` never imports `verity.core`, apart from the files
  `ML_IMPORTS_CORE` names (C-Flock's tests and V*'s builders, and hm96's test of its claims).
- Everything outside `verity/` may import `verity`. `verity` imports no other distribution of the workspace.
- Inside `verity`, each unit imports only what `ALLOWED` lists. `ml.circuits`, `ml.randomness`, `ml.crypto` and
  `core.claims` import no other unit.
- A protocol imports the parts of `verity.ml` it's allowed, the profile, itself, and the protocols `ALLOWED` lists for it
  (PoUW and the network certifier import sampled proofs). PoUW is the one protocol that reads the primitive circuits'
  Definitions and silicon semantics (`ml.circuits.ops`, `silicon`).
- Nothing outside `infra/` imports `infra/`, apart from the files `INFRA_IMPORTERS` names.
- Nothing under `verity/` imports an integration, apart from the sites `KNOWN` lists in
  `tests/test_integration_boundaries.py`, which only shrinks (Design rules, "Integrations are frontends").

`verity` is the repository's one command (`tools/verity/cli/`): `verity check` runs every part of the
check, `verity check tests|lean|circuits` runs only those parts, and `verity bench|example NAME` runs
`benchmarks/<NAME>/`'s or `examples/<NAME>/`'s runner. `verity --help` lists the rest.

~~~text
uv sync --all-packages --extra torch-cpu  # the whole workspace, editable, with CPU-only torch (GPU work: a member's `torch` extra)
research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py'  # suites whose inputs changed; --quick: no slow tests
research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run --extra torch-cpu verity check'  # the whole check; --record: an Attempt
research status RUN  # check back on a run (`research run` returns once it starts); --wait: only a run under 20 min
~~~

## Design rules

Daniel's, current. Their dates and history are in his decisions log in Notion; `.agents/skills/friction/SKILL.md` has
the standing rulings stated nowhere else.

- **Circuits and values.** A circuit holds only gates. Everything a program or a request brings, every intermediate and
  every output, is a value. A value is committed under a hiding leaf and touched only inside a zero-knowledge proof.
- **Everything zero-knowledge.** Every protocol is zero-knowledge, PoUS included. Only four kinds of things are public,
  each with its reason:
  - protocol structure;
  - the protocol's outputs;
  - the verifier's coins and draws;
  - hiding commitments.

  Partitions are private, and their properties are public. A proof block's outputs fit its size class, a public power of
  two from 16 to 262,144 bits and a property of its Definition, and its output bits count in leakage (Daniel, 19 Oct,
  provisional; `UNIT_CLASS_MAX` set to 16 restores the 16-bit limit).

  A plain path, C-Flock without `--zk`, stays beside the zero-knowledge ones, for a deployment whose values needn't be
  hidden (Daniel, 24 Oct). It has to work end to end, every zero-knowledge path keeps working, and tests cover both. Its
  guarantee is soundness alone, and `flock-verify` refuses a plain session past what the plain theorem covers (one table
  today).
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
  - BLAKE3 leaves the served path for `archive/`. There are two named exceptions:
    - Pearl's FP8 scheme, as Pearl defines it.
    - Duo receipts, PoUW's per-step receipts: keyed u32 mul/xor/rotl folds over committed accumulator words. They
      bind under the named premise H-μ⁺, the premise ledger's row R1
      (`note:20261016T0219Z-report-premise-ledger`), not yet in a lock.
- **Live coins only.** Fiat–Shamir is out, with two exceptions, each at a rate where live coins are impractical:
  - the remote bandwidth audit's pass seeds inside a window, derived from the window's live challenge and the previous
    pass's root (Daniel, 12 Oct);
  - PoUW's per-matmul noise seeds, seed_j = SHA-512(tag ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j)), from the
    window's live salt (`pouw.salt`, one per PoUW window from the certifier's draw, #2637) and the operands' committed
    roots, which a drawn unit's proof derives along with its own noise.
    Grinding is a term of the guarantee: a steer that needs m noised coordinates succeeds with probability at most
    μ^m a try (Daniel, 18 Oct).

  There are no retries: a failed proof is a visible failure, and a retry never redraws. Outer proofs go through the
  firewall as independent sessions: each staged statement is fixed before any session draws coins and gets exactly one
  session; each session draws its own live coins in its own firewall pass; no statement reads another session's
  transcript or coins; every session is recorded and every verdict counts (Daniel's provisional batch, 24 Oct;
  https://app.notion.com/p/403399515d9e81298f49e4b6b78634e7).
- **Verifier cost: O(circuit description), never O(unrolled circuit)** (Daniel, 18 Oct). A verifier of record may do
  work proportional to the descriptor it reads (its Definitions, nodes and references), never to the circuit it unrolls
  to: no walk over every gate, batch member or scan iteration. Summarize once per specialized Definition and multiply
  through batches and scans (`NciCheck.Desc`). An unrolled walk stays only as the reference a difftest holds the
  verifier to. `tests/test_verifier_cost.py` refuses one in `verity/core/`'s Lean, with the walks still open in its
  `ALLOWED`; `benchmarks/verifier_cost/` measures the cost.
- **Auditor-trusted devices.** The network certifiers and the memory challenger sign what they see. What they sign is
  witness and never reaches the auditor in the clear. They are the auditor's hardware: each runs on a machine the prover
  doesn't control, never as a process on the prover's node, and trust in the device and its host provider is a named
  assumption (`trusted-host`). Nothing the prover runs or hosts is trusted. A network certifier functions as an
  active warden, in network steganography's sense, but "network certifier" is its one name: never "warden" or "tap"
  (Daniel, 25 Oct; old names in `glossary/`).
- **The audited hardware is untrusted, and so is its vendor** (Daniel, 17 Oct: an international scheme may treat the chip
  designer as adversarial). No guarantee rests on a GPU attesting its own SKU, firmware or clocks. Hardware constants
  (f_max, peak rates) come from the data sheet for now, quoted under two or three readings rather than one main line,
  and later from a separate physical calibration.
- **Lean is the definition and the verifier of record** (Daniel, 18 Oct). Python in `verity.ml` mirrors it, for the
  prover's speed, and difftest holds that Python to the Lean. A verifier imports no `verity.ml` code. Each protocol still
  switches by the 8 Oct procedure (the friction skill): until it does, its Python verifier and the ml units it reads stay
  on `SOUNDNESS_PATH`, which only shrinks, and no core → ml import edge is added (`tests/test_boundaries.py`).
- **Soundness is one theorem** (see Lean below).
- **Soundness and zero knowledge in Lean; completeness may be Python** (Daniel, 18 Oct). What a guarantee says about a
  cheating prover, soundness and zero knowledge, is proved in Lean or rests on an auditor-trusted device. That honest
  serving passes (completeness: the Commit's gates, pre-flights, other prover-side checks) may be checked in Python.
- **That a circuit computes what the model computes is completeness, never proved** (Daniel, 24 Oct). Whether a
  Definition, its silicon semantics included, matches what the served model computes on its hardware decides only whether
  honest serving passes. No guarantee rests on it (exfiltration security, infiltration security, work soundness), since
  each is stated about the registered circuit. The developer validates it empirically: circuit-check, the hardware
  captures, the served outputs. Nobody proves it, lists it as an assumption, or counts it as an open gap in a guarantee.
  A circuit that is registered and passes circuit-check is the one its guarantees are about (Daniel, 25 Oct). Lean holds
  one description of it, the parsed circuit file the verifier checks: any other layer's circuit is defined from that
  file, so no proof, premise or open gap relates two descriptions of one circuit.
- **Integrations are frontends, and Verity is agnostic to them** (Daniel, 24 Oct). Verity's vocabulary is circuits,
  values, commitments, coins, proofs and verdicts. It has no notion of models, checkpoints, requests, batches or
  schedulers, and none enters `verity/`. An integration hands Verity only:
  - a circuit, partitioned into proof blocks, with its public layout (which gates are the system's external I/O, such as
    the network frames the certifiers see), registered by digest;
  - when the circuit depends on choices made at run time, a registered compiler (itself a circuit, with a public length
    cap) and the advice it compiles: a bit string Verity never interprets, whose length is charged;
  - the circuit's values, committed in a `verity.ml.commitments` scheme and registered with the circuit before any coin;
  - on a draw, the drawn units' openings, to the prover behind the firewall.

  What the circuit means, and how it was built, is the integration's; that it computes what is served is completeness
  (above). Any serving stack plugs in the same way without changing `verity/`. Nothing under `verity/` imports an
  integration (the import rules, above), and the serving notions `verity/` still holds (the certifier's advice format,
  vLLM's scheduler, among them) move out to the integration.
- **Results are per monitoring period, over all compute** (Daniel, 25 Oct). The time length of interest is the
  *monitoring period* (`ARCHITECTURE.md`, Glossary), not the window, the run or the demo. Every result, in a report, a
  table, a PR or a message to Daniel, is presented for monitoring periods of one week, one month and one year, and over
  all compute: the world's (*adversary resource model*), a datacenter's beside it, though a demo checks one. The
  datacenter's size (from
  `benchmarks/census/`) and the window's length are stated parameters, and a figure scaled from one machine says which
  parts are proved and which are scaled. Give each bound as its pooled total over the period and as a share of the
  compute it covers, beside the per-window figure, never the per-window figure alone: a per-window δ that looks weak is
  not the result.

## Tests

Every package's tests live in its own `tests/` and run as one suite, in its own process (a `parallel` suite runs in
pytest-xdist workers), cached by input. The repository's own suite (`tests/`) holds only invariants of the whole tree. The
testing skill, `.agents/skills/testing/SKILL.md`, says how to run them, what a suite may read (`[tool.verity.tests]
inputs`), and the rules for slow tests, fixtures and dummy credentials.

An agent VM does no work beyond editing (Daniel, 18 Oct): it is 16 GB, shared by every agent on it, and one
out-of-memory kill takes all of them down. Locally, run only git (`git merge-tree` included), reading and editing,
messaging and the notes, one named test file or `-k` selection of under about a minute and 1 GB, `lake env lean` on one
file in an already-built tree, a Lean package with no Mathlib or ArkLib, and the lock fast path. Everything else runs
remotely through `research run --on auto --cwd source [--wait] -- CMD` (the shipped tree needs its own `uv sync`), on the cluster queue as an `agent-test` job (4 CPUs, 16 GiB,
60 min unless `--cpus`, `--mem-gb` or `--timeout` say otherwise), outside the check slots, or in vy-nebius-cpu-1's `research`
pool when it starts sooner there (`research/auto.py`): every suite (`suites.py`),
`verity check` and `check.py`, circuit-check (24 GB a process), any build of `Security` or `Security/Proofs`, cargo, nvcc, every e2e run
and every GPU job (the writing-runs skill). The shell hook refuses those commands locally and names the remote one;
`AGENT_GUARD_LOCAL_OK=1` in the command is the explicit opt-in.

No tool call blocks more than 20 minutes (Daniel, 19 Oct): `research run --wait`, `ask-daniel --wait` and `queue ready --on`
stop there, cancel nothing, exit 79 naming the command to check back with (`research status RUN`), and you end your turn;
the shell hook refuses a longer `sleep`.

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
- `lean-agreement`, the cross-check against upstream's Rust verifier, when a change touches C-Flock (`verity/ml/flock/`, or
  its Lean verifier in `verity/core/service/flock/`; `merge_requires` in `tools/verity/check/ci.toml`): drive's after-landing
  run sends the upstream build then.

A pass of a test suite, a circuit-check target or a Lean package's Lean check is reused only while every input it read is
unchanged. `--record` runs `check` through `research run` as an Attempt of the checked-out clean commit.

Landing (Daniel, 24 Oct 06:05Z): PRs land on an eyeball review, and the full check runs after landing. A part's verdict is
reused only while a hash of its declared inputs (the read guard enforces them), the base image and the code that decides it (a
suite's runner and harness, not every line of the check tool; Daniel, 9 Oct) is unchanged, so a docs-only change reruns
nothing, a deleted test reruns its suite, and a layout move or a change to the runner reruns everything. A nightly uncached
check must agree; a disagreement means some part reads something it doesn't declare, and it is fixed in the check, not by
reverting anyone's change.

Check time is to come down by a lot, and anyone may do whatever it takes (Daniel, 11 Oct): a change to how `check` runs
lands without its file owner's review, and the owner is told. infra owns check time as an outcome, and its spend is
infra's call (13 Oct). Landing is automated, and ci owns that: from a ready mark and a review to `main` moving, no agent acts
unless the automation refuses, and the lander handles what it refuses.

PRs land through ci's `research queue drive`, with `land = "review"` in `tools/verity/check/drive.toml`:

- Drive admits a PR once its owner marks it ready at its head: `research queue ready N --mark --by LANE`, or
  `research data label pr:N@SHA ready true --by LANE`. A new head needs a new mark. The owner is the lane a `Lane NAME` line
  in the PR's body names, else the Cursor agent id the body carries. Marks written before `marked_after` in
  `tools/verity/check/queue.toml` don't count, and a `hold` label keeps a PR out, as does a merge onto `next` that conflicts
  or puts a Markdown file over its cap (drive.toml's `landing_gates`). `research queue status` says what each PR
  waits for.
- Drive stacks admitted PRs on `next`, one merge commit each, non-Lean first (`nonlean_first`, drive.toml): a sync puts
  a PR whose own change is no Lean change below every Lean change it doesn't contain, in `--first`'s order and then
  admission's, and never moves a member it keeps. Reviewers, agents ci staffs and never the PR's owner, read `next`
  bottom up: `research queue review` lists the members above `main` with no land verdict at their head (number, head,
  owner, files, whether it waits for a Lean check), and `research queue review N --land [--ref NOTE] --by LANE` or
  `--hold REASON --by LANE` records one, a `reviewed` label on `pr:N@HEAD`. A verdict by the owner, its handle or its
  ready mark's author doesn't count, and a new head needs a new one. No change waits on any other per-head grant: not
  red-team's, the vLLM coordinator's or a statement reviewer's (Daniel, 8 Oct).
- Each tick, drive fast-forwards `main` to the top of the longest prefix of `next` whose every member has a land verdict
  and no hold. Before it lands, in minutes: the merge onto `next` is clean, and every `.py` file the landing leaves
  compiles (drive ejects the PR that broke one). A PR that changes a `lean-audit.json` lock, or Lean that runs as code
  (a `.lean` file outside `Security/`), waits for its Lean check: a `lean-audit` that passed on a commit of `next` at or
  above it, which drive launches on the reviewed prefix's top, and after a failure there on the PR's own commit; only a
  failure at its own commit goes to its owner. Any recorded run whose `lean-audit` passed there counts too. Lean proofs
  alone land on the verdict and are checked after landing (Daniel, 25 Oct). The top stops below a PR that came onto
  `next` in the same tick. While a check runs, a Lean change above it gets one stacked (at most `max_lean_checks`
  running, drive.toml); the lower one runs on. Each check stops at a cut point, after the first ceil(n / f) of the n
  Lean changes it would hold (f: the checks still free), so a failure holds only what's above it. A PR containing an
  unmarked PR of the same owner, or one its owner withdrew, carries that change and doesn't wait for it.
- After each landing one run checks `main`'s tip with every check step (`train.sh MAIN MACHINES --deferred JOBS
  --no-regenerate`, with `--no-cache` while `after_landing_no_cache = true`), and nothing lands before one such run of `main`
  itself has ended, its baseline (or ci records `main` green on the control pod: `/data/infra/drive.sh queue baseline SHA
  --by ci`). A red suite's culprit, the landing since its last pass its failure touches, is rung and reverted (`git
  revert -m 1`) 60 min later unless a fix is marked; `main` 3 h unverified rings @ci and @top hourly (drive.py).
  Citations and tables read only commits whose after-landing run passed.
- A pure move that `tools/verity/check/move_check.py BASE HEAD` (`verity check move BASE HEAD`) certifies lands alone, ahead
  of the queue: the lander lands it with `research merge --push` on its own passing check, from the control pod
  (`/data/infra/drive.sh merge ...`), and drive's next sync rebuilds `next` on it. Only drive's App pushes `main` and `next`
  (GitHub's rulesets, 23 Oct): `research` refuses those pushes off the pod, and an eject, on the pod or off it, only withdraws
  the mark, for drive's next tick to rebuild `next`.
- `land = "check"`, drive's default, lands trains on their own check instead (`research.drive`'s docstring).

## Lean

- **Soundness is one theorem** (a rule, not a claim that `main` proves it): one theorem over everything the compiled
  Lean verifier accepts, for arbitrary Boolean circuits, with no hypothesis on the circuit file or the statement's form.
  The verifier must refuse any form the proof doesn't cover, and it gains no new form ahead of its proof. Circuit inputs
  and outputs are to use one canonical format, proved in general, so a new workload needs no verifier or proof change.
- **The math is modular** (Daniel, 24 Oct). Each layer of an end-to-end guarantee is proved once, over the guarantee of
  the layer below as its hypothesis: a session verifier's knowledge soundness for sampling, sampling's for an audit, an
  audit's for an accounting bound. Every configuration (zero-knowledge, plain, recursive) and every protocol instantiates
  those theorems. A new configuration adds an instance, not a proof, and a proof that redoes a layer for one
  configuration is refactored onto the shared one.
- **One vocabulary** (Daniel, 25 Oct). A concept that more than one guarantee reads (the circuit, commitments and
  hashes, coins and draws, probability, partitions and units, devices and signatures) has one definition in the spec,
  in `Core` or `Hash`, and every area and protocol reads that one. An area defines only what is its own, and no new Lean
  adds a second definition of a shared concept. A protocol is stated over the shared definitions and the layer theorems
  above, so adding one needs no new foundation, and guarantees of different protocols compose on them with no
  translation lemma between two descriptions of one thing. lean owns consolidating the duplicates that exist now.
- **Lean is either code or security analysis.**
  - Code stays with its component and never requires `Security/`: C-Flock's executable verifier
    (`verity/core/service/flock/`, no dependencies), the ZK server (`Flock.Firewall` in that package, run as
    `flock-firewall`, and its release side `core_firewall`, `verity/core/service/firewall/`, with its difftest), the
    PoUS grader (`verity/core/protocols/pous/lean/`) and
    the certifier program (`verity/core/protocols/certifier/lean/`, `NetworkCertifier`, with its
    difftest), the NCI check (`verity/core/protocols/network/nci/lean/`, `NciCheck`, with its difftest) and the remote
    bandwidth audit's verifier (`verity/core/protocols/remote_bandwidth/lean/`, `RemoteAudit`, with its difftest).
  - The ZK server is Lean, and its Rust path is dropped: under `FC_FIREWALL=1` every byte that leaves is computed by
    `Flock.Firewall`, and a recursive session's release by `core_firewall`; `flock-circuit` (Rust) computes the prover
    workers' values, can only refuse more, and relays (live `PROTOCOL.md` §10.4). The ZK theorem
    (`Flock.zk_server_lean_loop`) reads the released bytes only through named hypotheses not yet proved about the live
    path, so the ZK server is not yet proved.
  - The analysis is `Security/`. Its spec is `Definitions/` (each component's vocabulary) and `Models/` (its named
    assumptions and guarantee statements), the package `security` (Mathlib, no ArkLib). `Proofs/` is the nested package
    `security_proofs` (ArkLib allowed): every proof and lemma, with `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for
    each proved guarantee.
    `Properties/` is the nested package `security_properties`: properties of the code stated against its Lean functions
    (`Properties.<Protocol>`) and proved (`Proofs.Properties.<Protocol>`). It may require `security` and that code, never
    `security_proofs`.
  - A component's Lean goes in its own area of each tree it has Lean in (`Core`, `Nci`, `Pouw`, `Pous`,
    `NetworkCertifier`, `Flock`; `Accounting` is the device model the resource audits share, and their joint guarantee).
  - The spec imports only itself and Mathlib (`layers` in `Security/lean-audit.json`). A theorem that must stay
    in a trusted module (a definition's proof field) may; the Lean check replays it like any proof. C-Flock's
    `Definitions/Flock` holds what has been extracted so far (`Definitions/Flock.lean` lists it): its
    guarantees are `Definitions/Flock/Guarantees`, and the rest of its analysis is in `Proofs/Flock` until its spec is
    extracted.
  - `Security/` is three trees (Daniel, 24 Oct): definitions (what Daniel reads: vocabulary, named assumptions,
    guarantee statements), properties (statements about the code, reading only the definitions and that code) and
    proofs (every proof, lemma and proof-only definition). The Lean check refuses a guarantee or property whose
    statement reads a definition of the proofs tree, and a module of the definitions or properties tree that imports
    one (`trees` in `Security/lean-audit.json` names the trees; its `except`, one reason each, is the cleanup's to-do).
  - All packages share one toolchain and one Mathlib revision (`tests/test_lean_packages.py`). Bump them together, in a
    PR of their own.
- **The Lean check.** Every Lake package has a lock, `lean-audit.json`, and `check` runs the Lean check on every declaration
  (`tools/verity/lean/check.py`):
  - only `propext`, `Classical.choice` and `Quot.sound`;
  - no `sorry`, `axiom`, `native_decide` or kernel bypass;
  - an assumption is a named `Prop`, taken as a hypothesis or read in a guarantee's statement, and listed by name in the
    lock with its own record;
  - every file is built, and every declaration is replayed through the kernel.
- **The lock.** A guarantee (ARCHITECTURE.md "Glossary") is listed in `lean-audit.json`, and `check.py --update` records its
  statement, every definition the statement reads, and each named assumption it takes with a digest of it and what it
  reads, wherever its module lies; those records are the lock.
  - When a changed record lands on `main`, Daniel gets a DM showing it. That DM is the check: nobody signs off and nothing
    waits on it.
  - A guarantee that only moved sends nothing (`--update --moved MOVES.json`).
- **How to work on Lean**, how parallel Lean lanes avoid conflicts (two agents never share a writable `.lake`), and which
  assumptions get an `upstream` watch entry: the lean-proofs skill, `.agents/skills/lean-proofs/SKILL.md`.

## Working principles

Research velocity comes first (Daniel, 6 Oct). Be as generous as possible with what lands, and cut ceremony: propose
cutting any gate, review or step that slows landing without catching real errors, and fix what breaks later. Our machines
are research infrastructure, not a rehearsal of deployment, so don't mimic deployment settings.

**Waiting is a bug** (Daniel, 25 Oct). A step you repeat (a build, a run, a check, a queue) that makes you wait more than
about 10 minutes each time is friction even when it never fails. Fix it at the source, or take it to its owner that day
with its numbers: minutes a wait, waits a day. The same goes for a rule or a ruling, Daniel's included: when one costs
velocity, push back through your coordinator with the numbers and the change you'd make. The owner of a slow loop fixes
it ahead of its other work.

Solve the current task with the smallest clear implementation. Reuse existing code and commands before adding
abstractions or procedures. Enforce deterministic requirements in code, with the fix in the failure message, rather than
in prose. Remove obsolete rules and machinery rather than accumulating them.

**A decision rolls out in the change that records it.** When Daniel rules, or a PR retires a term, a path or a procedure,
that PR rewrites every place that states the old way: `git grep` the old phrase across this file, ARCHITECTURE.md and its
Glossary, the skills, READMEs, docstrings and CLI help. When code has to change first, the ruling's relay names the
owner and the PR, and the docs change with that PR. A stale statement found later is fixed on sight, by whoever finds
it. `tests/test_doc_paths.py` fails on a maintained doc that names a repository path that no longer exists.

Make every decision yourself except keys and claims (Daniel, 8 Oct; spend became its lead's call 13 Oct). Access and
keys, and what a guarantee says or rests on, go to Daniel through your coordinator, with your recommendation; pods
still bound themselves under a `budgets.toml` line, and a budget line named in a standing ruling stands. A change that
crosses owners goes to the other owner, and the coordinator settles a disagreement. When a tool, rule or step gets in
your way, fix it at the source if the fix is clearly yours. Record each ruling you get. The friction skill,
`.agents/skills/friction/SKILL.md`, has the standing rulings this file doesn't state and the friction note format.

A guard never fails open, and an agent VM can be suspended at any moment, so nothing that bounds spend or preserves
results may depend on one. Pods bound themselves (`research pods create --max-hours`, under a line of the notes repo's
`budgets.toml`), and runs preserve their results from the pod. The lane whose budget line pays for a machine owns its
use, and stops it when its work there ends; `research run --on` starts it again, and infra's idle stop is only a backstop
(Daniel, 19 Oct).

Never print a secret's value.

- List environment variable names with `compgen -e`, never `env | cut`, `env | grep`, `printenv | …` or tmux's
  `show-environment`: a multi-line value's later lines print as if they were names.
- `.cursor/hooks.json` refuses any shell command that would print the environment or every process's full command line
  (`tools/agent-guard/shell_hook.py`). Cloud agents read project hooks only, so on a branch older than the hook,
  origin/main's `tools/agent-guard/boot.sh` writes the hook into `/workspace`, runs `install.sh`, starts the VM's OOM
  recorder and puts `research` on PATH. The environment's `start` runs it after the GitHub helper (since 18 Oct, INF-71),
  and every coordinator's launch brief (`infra/cursor/launch_coordinator.py`) runs it first too, for a VM booted from an
  older environment version.
- A secret is stored as one line (a key as base64, `NAME_B64`), and `tools/agent-guard/install.sh` puts a redacting `env`
  on the VM.

## Where writing goes

- **Work and its status** are Linear issues in the owner's team, each in a project and milestone; the projects, grouped
  by Goal, are the campaign map (Daniel, 14 and 26 Oct). Slack carries rings and conversation, never the brief or status.
  How: the coordinating skill.
- **This repo** holds code, tests, small fixtures, and maintained current-truth docs only: `README.md`, `AGENTS.md`,
  `ARCHITECTURE.md`, `DESIGN.md`, `DISCREPANCIES.md`, `SKILL.md`, `CHANGELOG.md`, `*.template.md`, and `PROTOCOL.md`.
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
- **Writing for a person** (an explanation, a one-pager, an answer in someone's chat) is a Notion page, or cites only
  what they can open: PRs, repository files, run ids and Notion pages. The notes repo is private, so a `note:` id is for
  agents (Daniel, 19 Oct).
- **Evidence and generated output** (explore JSON, plots, gate outputs, logs, run results, rendered tables and reports)
  go in the evidence store only, never Git: neither this repo nor the notes.
  - `research run` publishes every run, completed or failed, as an Attempt with its outputs. The contract is
    `tools/research/src/research/store/README.md`.
  - Anything else goes in with `research data put --kind K --file|--tree P --preserve`. Cite it as `art:<id>`.
  - Tables and reports are rendered on demand from code plus the store.
- **Findings and claims about a run** are labels, never edits to a shared table:
  `research data label <run id|art:…> <key> <value> --by <lane> --ref <run id>`.
- **Architecture decisions** and Daniel's decisions log go in Notion.

## Lanes (worker agents)

If your launch message names you a lane, `~/.research/notes/kb/LANE-CONTRACT.md` is your rulebook.

Each coordinator (a handle's holder, which its inbox route records: `research msg handles`) is a plain Cursor cloud agent on the
one-repo environment "Verity compute-verification". It owns one vertical slice and runs its own workers as subagents.
There are no Cursor Projects (Daniel, 13 Oct). The admin holds `@top`: it opens handovers, keeps the roster, the
quiet-lane sweep and Daniel's decisions log, relays to Daniel, and announces and settles; it does no lane work. It owns
consolidation (Daniel, 26 Oct): work, records and notation converge, not multiply. Coordinators are launched, handed
over and archived as `infra/cursor/README.md` says.

- **Reread the rules every night.** At the start of each night, reread this file and the rulings in
  `.agents/skills/friction/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md`: rules change during the day, and a lane
  that acts on a retired one waits on gates that no longer exist.
- **Keep chatter low** (contract §3a). Every message, and every background shell you start, reaches the coordinator and
  the human. So run long jobs detached on the pod, write one-line checkpoints, and message someone only to unblock them
  or to change their plan.
- **Workers run where they can't take you down.** Launch each worker as a cloud subagent (`Task` with
  `environment: "cloud"`; Daniel's ruling, 18 Oct 10:20 AM PDT) or as a detached `research run`. A local subagent is only
  for a short read-only lookup. A coordinator's foreground local Task is refused (#2510), and heavy work never runs on an
  agent VM (the shell hook refuses it).
- **No step waits long.** No tool call blocks for more than about 20 minutes: launch detached, end the step, and check
  back on the run.
- **Workers restart from notes.** Each brief asks for a checkpoint note before long work and every 3 to 4 hours, and
  words red-team work neutrally. A worker resumed twice, or whose base branch was deleted, is relaunched fresh from its
  last note.
- **Outages are caught outside you.** The control pod's agent watch (INF-69) flags a run stopped mid-turn, a stalled or
  ERROR agent and an out-of-memory kill, and frees what it can. Agents keep no heartbeat of their own.
- **Reach a running subagent through its next shell command:** `python3 tools/agent-guard/inbox.py send AGENT_ID --text
  '…'`, where AGENT_ID is the `bc-…` id its Task call returned. A subagent never rereads a file mid-run, so don't put
  ownership changes or new requirements in a shared file and hope. The shell hook refuses the agent's next command once,
  with your message as the refusal's text, and the agent re-runs it. Agents on other VMs: through their coordinator on
  Slack.
- **Use cloud workers freely** (Daniel, 18 Oct, 9:59 PM PDT: "It is totally fine to use cloud workers"). This is the
  standing explicit request that Cursor's Task tool wants before `environment: "cloud"`, for every agent in this repo, so
  nobody asks Daniel or a coordinator first; spend is the lead's call. Use one when the work is long or heavy, or your VM
  is full. How a cloud worker differs from a local subagent:
  - It boots its own VM from the environment's snapshot, with the environment's secrets. It does not reliably get its
    own branch: with any `cloud_base_branch`, workers have committed onto the launcher's branch and onto each other's
    commits. So its brief says: "create a new branch `cursor/<lane>-<topic>-<hex>` from `origin/main` and push only to
    it, never to the launcher's branch, a PR head or `main`."
  - Its brief is complete: the task, the done-condition, the deadline, and "reread `git show origin/main:AGENTS.md`",
    since its snapshot can be older than `main`. Nothing reaches it mid-turn, so the brief names an inbox branch it reads
    at each hourly checkpoint (`git fetch -q origin B && git show origin/B:inbox/<agent-id>.md`), which gets changes to
    it within the hour. Larger changes go in when you resume it (Task's `resume`, with the `bc-…` id the launch
    returned) after its turn ends.
  - It stays reachable as below: when it would wait on a run, it ends its turn naming the run id and ETA, and its
    launcher resumes it. A running turn holds one of the team's concurrent cloud-agent slots.
  - It opens its own PR from its VM (`research queue open`), and syncs its notes like any worker. Where that fails, it
    reports its branch and SHA and the launcher opens the PR for it. Its final message goes to whoever launched it.
- **Check what's being tried.** Before starting work another agent could plausibly be doing, read the notes'
  `kb/onboarding.md`, check `research notes approaches`, and claim yours at your first checkpoint
  (`research notes claim`).
- **Work toward outcomes.** A lane's night is one or two outcomes, each with a done-condition anyone can check, which its
  lead proposed and Daniel approved.
- **Push your branch after every commit** (never `main`, never `--force`). A push gives a PR a new head, which needs a new
  ready mark and a new review (Checking and merging). Only drive and the lander move `main`.
- **Work in a checkout no other agent uses:** your own VM, or your own `git worktree add` (with its own venv) on a shared
  VM, removed when your task ends. Never check out or switch a branch in a tree another agent uses, and push only to
  branches you created or were handed.
- **Stay reachable** (Daniel, 19 Oct: silent agents cost whole nights). Push and write a one-line checkpoint in your
  note at least every 60 minutes. No tool call blocks for more than 20 minutes: launch runs detached, and when you'd
  otherwise wait on one, end your turn naming the run id and its ETA, so your coordinator resumes you when it ends. A
  coordinator replaces a worker that stays silent past the agent watch's flag with a fresh agent briefed from its note,
  and never resumes a local worker a third time.
- **Landing is drive's.** The worker that made a branch opens its PR (`research queue open --head BRANCH --sha SHA
  --title T --body-file F --by LANE`, from any VM, with `Lane LANE` in the body), marks it ready and goes back to
  research. A reviewer ci staffs writes its verdict, a lock or code Lean change waits for its Lean check, and the full check
  runs after landing and reverts a PR that breaks `main`; you hear back only if yours is held, ejected or reverted. Before
  marking, run the Lean check of the packages you touched and the suites your change touches, `research run --on auto --cwd
  source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py
  --changed $(git rev-parse origin/main) --quick"` (main's SHA, expanded in your checkout: a shipped tree may hold no
  `origin/main`), and a suite's slow tier when you changed that suite's core. Prefer small PRs, which review lands soonest.
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
- When Daniel or a lead asks for a smell check, before you call something served, end to end, proved or credited, and when
  you relay another lane's claim: `.agents/skills/smell-check/SKILL.md`. The catalogue of smells found so far is the
  notes' `kb/smell-catalogue.md`.
- Asking Daniel for a decision (a page in the Asks for Daniel Notion database, and its `#ask-daniel` card):
  `.agents/skills/asking-daniel/SKILL.md`.
- Before adding, changing or retiring a process (a rule here or in the lane contract, a gate in `check` or
  `research merge`, a required review, a file every PR must update), assess its effect on research velocity:
  `.agents/skills/process-design/SKILL.md`.
- Creating or modifying a skill (at most 16 KB: current rules, not their history): `.agents/skills/authoring-skills/SKILL.md`.
- Filing, tracking and reporting work in Linear, and how long each message is: `.agents/skills/coordinating/SKILL.md`.
