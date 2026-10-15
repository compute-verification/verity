---
name: lean-proofs
description: >
  How to work on Lean in Verity: where a new Lean file or package goes, the inner loop (lake env lean on one file,
  scratch probes with #print axioms, exact? and trace_state, profiling a slow proof), building and auditing with
  tools/verity/lean/check.py, stating assumptions as named Props, freezing and listing the guarantees a ledger or PR cites in
  lean-audit.json, moving them without a DM to Daniel (--moved), watching ArkLib and other dependencies for proofs
  of what we assume (tools/verity/lean/upstream.py), keeping parallel Lean lanes out of each other's files and build
  directories, and Daniel's Lean rulings (guarantees proved under their assumptions, Lean as the source of truth for
  them). Use when adding or changing Lean proofs, definitions or packages (verity/core/service/flock, Security
  and its Proofs, the PoUS grader in verity/core/protocols/pous/lean), when a Lean build or proof is slow,
  editing a lean-audit.json or an ASSUMPTIONS.md ledger, bumping a Lean dependency, or preparing a Lean PR.
---

# Lean proofs

The rules are in `AGENTS.md` ("Lean"); the Lean check's reference is `tools/verity/lean/README.md`. This is the workflow.

## Where the work goes

- **Code or analysis** (`AGENTS.md` "Lean"). Code the verifier runs, and anything the V[B] generator or a reference
  prover runs: `verity/core/service/flock/` (no dependencies); the PoUS grader and the certifier program stay with their
  protocols, each in its `lean/`. Everything else is
  security analysis, under `Security/`: a definition or statement in `Definitions/<Area>/` or
  `Models/<Area>/{Assumptions,Guarantees}` (package `security`, Mathlib only), a proof in `Proofs/<Area>/` (package
  `security_proofs` in `Security/Proofs/`, which may use ArkLib). C-Flock's level3 and soundness are
  `Proofs/Flock/Level3/` and `Proofs/Flock/Soundness/`. A property of code, stated against its Lean functions, is
  `Properties.<Protocol>`, proved in `Properties.Proofs.<Protocol>` (package `security_properties`, which may require
  `security` and the code, never `security_proofs`). No new package: an area is a directory in each tree.
- **Own a subdirectory and namespace** (`Proofs/Flock/Soundness/Knowledge/`, namespace `FlockSoundness.Knowledge`), with
  one aggregator module that imports its files. The area's root imports the aggregator once, when the area is created,
  so parallel lanes don't edit the same root file.
- **Parallel proof work runs against fixed statements** (Daniel, 8 Oct). Before any worker starts, the final theorem's
  statement and each open piece's statement are on the branch; each worker proves its piece against that statement, and
  integration builds the final theorem against them continuously. What isn't closed by the deadline is a gap list on the
  branch: exact lemma names with no `sorry` outside them, each with a size and an owner.
- **Definitions a guarantee reads go in `Definitions/` or `Models/`**, not in a proof file. The Lean check lists where they are
  (`reads` in `Security/lean-audit.json`); keep that list short.
- **Build in your own checkout.** Two agents never share a writable `.lake`: concurrent builds overwrite each other's
  `.olean` files, and a Lean check then reads what another agent's build left. A second Lean task on the same machine gets
  its own worktree. Seed its dependencies from a checkout with the same `lake-manifest.json` and `lean-toolchain`:
  `mkdir -p WT/PKG/.lake && cp -a --reflink=auto PKG/.lake/packages WT/PKG/.lake/`. Copy, never symlink, because Lake
  rewrites `.olean` files in place. After the copy, `lake build` compiles only the package's own modules.
- **A reviewer or red-team subagent works in its own worktree** (`git worktree add`), never by `git checkout` in a tree
  another agent is using: a checkout there switches that agent's files under it.

## Build, iterate, check

~~~text
bash tools/verity/lean/setup.sh PKG    once per machine: elan, the toolchain, Mathlib's cache
cd PKG && lake build                   once per checkout; incremental from then on
lake env lean Pkg/Area/File.lean       each edit: elaborate one file against the built imports
lake build Pkg.Area.File               after editing a file others import, so they see the change
python tools/verity/lean/check.py PKG  once, before pushing (the kernel replay of Security/Proofs is the slow part)
~~~

Build on a CPU machine; `Security/Proofs` (package `security_proofs`, on ArkLib) is the long one. Never `lake
update`, and never change a `require`, a revision or `lean-toolchain` in a feature PR. Its build of C-Flock's soundness
proofs peaks at 15–24 GB of memory: on a machine with less than 24 GB, add a swapfile first, and give a queued job
`--mem-gb 32`.

To check your records, or get the `--update` printout of what changed (what Daniel's DM carries), without a build of
your own, use the fast path on a shared machine. Don't run `check.py --build --update` from a fresh tree there: that took
about 22 minutes unpinned on the check cores.

~~~text
research run --on vy-nebius-1 --project verity --source . --cwd source --declared-output 'lean-audit/*' -- \
  python3 tools/verity/lean/lean_changed.py --records PKG --update
~~~

It waits for one of the node's Lean slots of pool `build` (`lean_slot.py`; a node whose slot file lists no `build` pool
refuses it, exit 2, even with an `any` pool), never behind merge checks or hour-long audits, rebuilds only what changed since
that slot's last run, and audits without the kernel replay: on C-Flock's soundness with nothing to rebuild that was 6.5 minutes, and a
slot's first run, a cold build, 31. The record is in the run's `lean-audit/audit-*/`, with `review.txt` when it changed; commit
the record. It then checks, without `--update`, every package whose guarantees read PKG's build (`Security/Properties` for
`Security`), each in `lean-audit/requirer-<path>/`. When one fails, your change moved what its guarantees read: if you meant
it, run `--records` on that package with `--update` and commit its record too, since check audits every package whose inputs
changed. It vouches for nothing: its result is of kind `lean-fast/v1`, which `check`'s lean-audit step and `research merge`
refuse.

To regenerate a record on a pod, run `research run --on POD --project verity --source . --cwd clone -- python3
tools/verity/lean/check.py --build --update PKG` with no wrapper around it (vy-nebius-1 has no `python`, exit 127). It exits 0 when the Lean check passes, whether or not the
record changed, and the record is among the run's outputs, at `lean-audit/audit-TAG/lean-audit.json`. `--cwd clone` builds
in the run's own clone of the shipped tree, removed when the run ends; with `--cwd source` every run of the commit builds in
the one shared tree, and a cancelled duplicate there deleted `PKG/.lake/packages` under a live Lean check.

While proving, check one file at a time, and leave the whole package and the Lean check for the end:

- **`lake env lean FILE`** prints that file's errors and warnings, and writes nothing. The `Core` area's slowest file takes about 10 s
  this way. A file that imports what you changed sees the change only after `lake build` of the changed module.
- **Ask Lean in a scratch file** outside the repository, run from the package directory with `lake env lean
  /tmp/scratch/Probe.lean`. It takes well under a second against built imports:

  ~~~lean
  import Proofs.Core.TC.AmpereSound
  #print axioms Verity.TC.ampere_step_hw     -- a sorry shows as sorryAx, long before the Lean check runs
  example (a b : Nat) : a + b = b + a := by exact?
  ~~~

  Probe what you build on, not only what you wrote: a section-level `variable` whose type goes through a local
  notation can elaborate to `sorry` with no error or warning, and the silent `sorryAx` then shows up as opaque
  timeouts in proofs that use the binder (proofs' worker, 14 Oct: five failed pod runs before a `#print axioms`
  probe caught it). A theorem reading such variables can also be dropped silently — no error, no `sorry`, the
  declaration simply absent from the built module, so a probe failing with "unknown constant" on a theorem you
  wrote is this, not a typo (same night: a theorem vanished through two runs before a bisect found it; #1923's
  audit hardening fails the package on both). Put such binders on the theorem itself, not in a section `variable`.
- **See a goal** by putting `trace_state` before a `sorry`: the goal prints with the file's output. The editor's language
  server shows it too.
- **Search before you prove:** `exact?` or `apply?` on a scratch `example` (both work without Mathlib), then `rg` for the
  name or shape across every package's `.lean` files, then [Loogle](https://loogle.lean-lang.org) for Mathlib by type.
  A lemma another package already proves is imported or moved to a shared place, not proved again.
- **When a declaration is slow,** find out where the time goes before raising `maxHeartbeats`:
  - `set_option profiler true in` before the declaration prints its elaboration times;
  - in a Mathlib package, `#count_heartbeats in` (from `import Mathlib.Util.CountHeartbeats`) prints the heartbeats it
    used.

  Then fix the proof: use `simp only [...]` outside terminal calls, split out a lemma, or shrink a `decide`. If an override
  stays, give its reason in a `--` comment ending its line or just above it; the audit lists every override with its
  reason and its module's build time (`heartbeats` in its report) and refuses none. Keep files to a few hundred declarations, so that Lake builds them in parallel and an
  edit re-elaborates little.
- **Push after every commit.** Work that lives only in a VM's worktree is lost when the VM is suspended.

The replay refuses `X.congr_simp is declared by a replayed module and by one it imports`. `simp` realizes a structure
constructor's congruence lemma (`ZeroHom.mk.congr_simp`) in your module when it rewrites under an unfolded bundled map,
and Mathlib has realized it already. Don't unfold a `→+` or `→*` inside `simp`: prove its `_apply` lemma by `rfl` and
simp with that.

## Assumptions and guarantees

- **An assumption is a named `Prop`** in its area's `Models/<Area>/Assumptions` (C-Flock's in
  `Proofs/Flock/Soundness/Assumptions`, `FlockSoundness.Assumptions`, until its spec is extracted), taken as a
  hypothesis by each theorem that needs it (`(hCR : Assumptions.SHA512CRExpected …)`), and listed by name in the lock's
  `assumptions` with its owner (`check.py --update` writes its record). Its docstring opens with its
  claim id (`uniform/io-getrandombytes`, from `verity.core.claims`) and gives its rationale. Lean is the source of truth for
  what we assume, so a ledger points to the docstring rather than restating it. Never an `axiom`, never a `sorry` on
  `main`, never `native_decide`, never an option that skips the kernel. A statement you can't prove yet is an
  assumption, or a theorem in a lane branch, not on `main`.
- **Freeze the statement, then prove it.** In your lane branch, state the guarantee on a `sorry` stub, check it against
  the paper or the spec, list it in `lean-audit.json` and run `--update`. The record holds the statement, so it passes once
  the proof lands, and the record shows if the statement moved on the way. Never make a
  guarantee easier: don't strengthen an input relation, weaken an output relation, shrink the challenge space, add an
  unmotivated hypothesis, or move a verifier check into an assumption. That list is also what Daniel looks for when a change reaches him.
- **List what you cite.** When a ledger row, a table or a PR description cites a guarantee as proved, add its theorem
  (`SecurityProofs.G` under the four names) to `guarantees` and run `python tools/verity/lean/check.py --update PKG`. Commit the
  record with the proof.
- **Watch upstream for what you assume.** A new named assumption or a lemma you prove by hand that a pinned dependency
  (Mathlib, ArkLib, VCVio, CompPoly) could prove gets an entry under `upstream.watch` in the package's `lean-audit.json`.
  The entry lists the dependency's constants that the statement is about. An assumption stated over the package's own
  definitions gets none: no upstream theorem can mention them, so the entry could never fire. Nor does one about a single
  number (PoUS's `PrimeP16448`): its entry would fire on every theorem about primes. The Lean check fails when the
  pinned dependency proves an unacknowledged theorem about them. Read the theorem, say in the PR whether it discharges the assumption or replaces the proof, then run
  `python tools/verity/lean/upstream.py PKG --ack`. At a dependency bump, also run `upstream.py PKG --lean` (the exact pass).
  At a toolchain bump, also re-read `lean_io_get_random_bytes` in the new toolchain's `src/runtime/io.cpp`. A3
  (`Security/Proofs/Flock/Soundness/ASSUMPTIONS.md`) cites what it reads, and `tests/test_lean_packages.py` fails until the citation names the
  new version.
  Cite the output in `ASSUMPTIONS.md`'s upstream status rather than writing one.
- **A changed record is a changed claim.** If `--update` changes a record or `reads` for a guarantee you didn't mean to
  change, your edit changed what that guarantee says: undo it, or say so in the PR and update the ledger.
- **A changed record reaches Daniel.** The axiom checks and the replay pass a weakened definition (a vacuous theorem is
  still a theorem), so paste what `--update` printed into the PR. When it lands on `main`, Daniel gets a DM showing the
  change, and that DM is the check: nobody signs off and nothing waits on it. A statement reviewer is optional: ask one
  when you want a second reading of the printout, but don't hold the merge for it.
- **A move is not a change.** When you rename guarantees or move what they read (a package moving to the four names), write
  the moves as a JSON object of old names to new ones, a declaration or a namespace each, and run
  `check.py --update --moved MOVES.json PKG` on the record as it was before the move. A guarantee that says what its
  record says, renamed or lifted, and the definitions the same under their old names, each get one line; the rest prints
  in full. Moving a definition to another module needs no map.
- **A lift needs no map.** `theorem X (xs) : e` becoming `theorem X (xs) : G as`, with `def G (ys) : Prop := e` in a
  `Guarantees` module, prints as "lifted into G" on a plain `--update` when `G` unfolded once is `e` (the same type hash)
  and the theorem keeps its named assumptions as binders (tools/verity/lean/README.md, "A lift is not a change"). Daniel's DM
  still lists the changed record.
- **One shape.** A property takes its inputs, one hypothesis that its function returned an outcome, and premises; the
  rest is its conclusion. Each guarantee's entry under `shape.properties` in `lean-audit.json` names its `function` (a
  constant or several) and that `outcome`: `accept` by default, `undecided` or `reject` where the property is about that
  outcome, `none` for a function that always returns (a device's `step`, a constant), which takes only the input; two
  acceptances of one function (a binding property) are one, of the pair. A premise is a hypothesis whose head is a
  constant of a module under `shape.premises` (under its leading `∀`s); the acceptance applies the function in its own
  body, under no hypothesis of its own; a `Prop` field of a structure an input ranges over, or a family of them
  (`Vs : Fin S → VStmt`) holds, counts as a hypothesis (`Nonempty` too),
  and a conjunction as its conjuncts; when they are about different functions, the entry lists `conjuncts`, one
  `{function, outcome}` each. A probability statement has its acceptance in the event instead: with its constant under
  `shape.events` (`FlockSoundness.Game.prob`), a conjunct whose events are each the acceptance and then what fails
  (`prob (fun o => A o ∧ B o) g σ ≤ …`, every conjunct before the last an acceptance) takes premises only, and every
  other hypothesis is refused; a compound failure goes under one `¬` or definition. Every package is checked. A guarantee that doesn't fit yet is under `shape.frozen`
  with one sentence on why, starting with its category (`restatable: …`, `frozen (hash): …`, `lemma`, `completeness`,
  `falsified`); the check fails once it fits, until the entry goes. A named conjecture is a premise under
  `shape.conjectures` (open with a falsification test that runs, its `test` a pytest node id `PATH::NAME`, or
  falsified, with why); a property resting on one
  lists it under `rests_on`, and a falsified one takes every property resting on it off the list. Every `def G : Prop` of a `Guarantees` module (or a module
  under `shape.statements`) is the statement of a guarantee the lock lists, or is under `shape.frozen` until it is. Adding an entry is a claims change the
  landing DM shows Daniel: rather restate the guarantee, or move what isn't a property (an iff's converse, an honest
  case) to `Proofs/` as completeness.

## Before the PR

- `python tools/verity/lean/check.py PKG` passes for every package you touched, and `pytest verity/ml/flock/tests
  tools/verity/lean/tests tools/verity/check/tests/test_lean_*.py tests/test_lean_packages.py` passes.
- The PR says which guarantees changed, if any, and why; the ledger's status rows match `lean-audit.json`.
- Merge `main` into your branch rather than rebasing someone else's work; conflicts in a ledger are resolved by keeping
  both sides' rows. Run `python tools/verity/lean/merge.py --install` once per clone: `lean-audit.json` then merges without a
  rebuild, and only a conflict it refuses needs `check.py --update` on the merged tree.
- Record your lock at your own head, for what your change says, and then mark the PR ready. Don't merge a moving `main`
  in again just to re-record. When the train's check fails only because a record's `guarantees`, `reads` or
  `assumptions` differ (another PR changed a definition yours reads), the lander's `tools/verity/check/retip.py`
  regenerates them at the train tip. Re-record yourself only what it can't regenerate: a conflict `merge.py` refuses, any
  other key (`shape`, a move's map), or a failure in another step.

## Rulings so far

These are Daniel's rulings, newest first.

- 2026-10-08 (5:59 PM PDT): a named conjecture can be a premise, in a section of the list of its own, and a claim that
  rests on one is listed as conditional, apart from the rest. Sort first: a statement that models how a specific device
  computes, such as `TTOutPearlC` (falsified by what the hardware does, and dependent on G), is an assumption about the
  world, scoped by device and checked by measurement. Only genuine math goes in the conjectures section, and a statement
  that mixes the two is split into the device model and the math that remains. Each conjecture carries a falsification
  test that runs, not only a status, and one about finite objects is decided by computation instead. Incompressibility
  of fixed data is false in Lean, since a decompressor can hard-code the data (the trap collision resistance has), so it
  is stated over random inputs or as a reduction. A claim resting on a falsified premise says nothing: `Gamma16384` and
  `Gamma8192` come off the guarantee list, and `TTOutPearlCDevRev1OfGranted` is stated for G=0 only, each device setting
  getting its own claim. Until the shape check unfolds `def G : Prop` bodies, no past "no assumptions" result of the
  audit counts as checked.
- 2026-10-08 (5:11 PM PDT): every property in `Properties/` has one shape: for every input, if the function it's about
  accepts, the conclusion holds, given premises from the list. Its only arguments are the input, the acceptance and
  those premises, so the check enforces the premises rule with no judgment call. A premise is an assumption about the
  world that the function can't see, such as `DeviceIntact`. Where the code can see something, it checks it instead: a
  device that sees a late frame refuses to mark the record complete, and `HonestTiming` is no premise. A condition that
  picks out what the conclusion talks about, such as "this event is an egress tick", goes inside the conclusion; a name,
  such as a parameters digest, becomes a definition; any other condition on the input is a scope condition, which the
  function refuses, so it follows from acceptance. Completeness theorems aren't claims: they live in `Proofs/`, their
  honest-case hypotheses as written, each with one concrete instance where that case holds, which shows the soundness
  property it pairs with isn't vacuous. Properties that don't fit yet are on a frozen list, one file per property with
  its reason. The list only shrinks, and adding to it is a claims change.
- 2026-10-08 (4:30 PM PDT): for a commit-then-coin step, the break tied to the run is the fork form: two runs of the
  prover's own strategy that share a commitment and hash `x ≠ y` to one SHA-512 digest along their paths to it. A single
  run never holds the collision, since a cheater opens one table per run. The `∃` ranges over the strategy's
  continuations, visibly in the statement, never over arbitrary strings. `RecursiveSound` takes this form. Its bound is
  exact for a strategy that never collides on a fork and says nothing about one that collides on a single pair, so it
  is never called tighter than a bound under `cr/sha-512`. The quantitative forking version comes next: a strategy that
  beats the bound by δ yields, from two runs off one commitment, a collision with probability about δ².
- 2026-10-08 (4:30 PM PDT): a property's hypotheses come only from the named premises in `Security/Definitions/`
  (physical premises, and VBridge until it's proved). The Lean check refuses any other, and adding a premise messages
  Daniel like any `Definitions/` change. A theorem whose hypotheses can't all hold for the real system says nothing.
- 2026-10-08 (8:25 AM PDT): a guarantee never assumes a hash or a signature. Its conclusion is the property or a break
  tied to the run: for SHA-512, `P ∨ ∃ x ∈ run.hashInputs, ∃ y ∈ run.hashInputs, x ≠ y ∧ sha512 x = sha512 y`, over the
  inputs the run itself hashed; for LMS, a signature in the run that verifies on a message missing from the device's
  signing log; for Merkle binding, a collision among the inputs hashed along the two openings' paths. Every `∃` in the
  break is bounded by something the run carries. "Some collision exists", or an extractor free to ignore the run, is
  vacuous: collisions exist by pigeonhole, and `Classical.choice` is an allowed axiom, so the axiom check can't catch it.
  No advantage bounds, which would need a probability model. An input condition becomes a refusal in the verifier, not a
  hypothesis (the Lean check refuses NaN-coded noise, as Python's verifier does); that the honest draw never produces
  such an input is a completeness fact, measured rather than proved.
- 2026-10-03 (6:46 PM PDT): a change to what a guarantee says needs no reviewer and no approval: when it lands on `main`,
  Daniel gets a DM showing it. "Pin" is retired: say guarantee. (Since 8 Oct no change waits on a per-head grant at all.)
- 2026-10-01: a pinned theorem is called a **guarantee** (security guarantee): a proved `Prop`, assumptions implying a
  conclusion, where an assumption is a named `Prop` (README "Glossary").
- 2026-10-01: a guarantee that only moved, renamed or restated as a `Guarantees` definition, needs no statement reviewer
  when `--update --moved` prints it as the same statement under its old names, with every definition it reads the same.
  Since 2026-10-03 no change needs one; `--moved` now decides only whether Daniel's DM lists it.
- 2026-10-01: a Lean protocol package may require another protocol's package (NCI's requiring sampled proofs', say),
  because composing their guarantees needs it. The requires stay acyclic, and a protocol never requires a backend. Such a
  `path` require changes no revision, so a feature PR may add it. Python's protocols import only `verity`, themselves
  and the protocols `ALLOWED` in `tests/test_boundaries.py` lists (PoUW → sampled proofs, the ruling of
  2026-10-03 in `.agents/skills/friction/SKILL.md`). Its Lean half is superseded by the Lean layout move (2026-10-06):
  there are no protocol packages, only areas of `Security` and its Proofs.
- 2026-10-01: there are guarantees and assumptions, and a guarantee is **proved under its assumptions**; there is no
  third status. Some assumptions are better than others: SHA-512's collision resistance (`cr/sha-512`) is a better one
  than a fact about our own circuit that we mean to prove (`HmRowComputes`). So what a guarantee assumes is explicit and
  hard to break, a table, a ledger or a PR cites a guarantee with its assumptions, and proving an assumption removes it
  from every guarantee that took it. Since 2026-10-08 a hash or a signature is not an assumption at all: the guarantee
  ends in a break tied to the run (above).
- 2026-09-30: Lean is the one source of truth for the assumptions we make. Views of them (the console, the tables) are
  derived from Lean and the Lean check's output, and no hand-kept ledger is generated alongside.
- 2026-09-30: probability and games shared across packages are our own minimal definitions over Mathlib's `PMF`, not
  VCVio's `OracleComp`; soundness bridges to VCVio where it needs it.
