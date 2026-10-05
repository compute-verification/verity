---
name: lean-proofs
description: >
  How to work on Lean in Verity: where a new Lean file or package goes, the inner loop (lake env lean on one file,
  scratch probes with #print axioms, exact? and trace_state, profiling a slow proof), building and auditing with
  tools/lean/audit.py, stating assumptions as named Props, freezing and listing the guarantees a ledger or PR cites in
  lean-audit.json, moving them without a statement review (--moved), watching ArkLib and other dependencies for proofs
  of what we assume (tools/lean/upstream.py), keeping parallel Lean lanes out of each other's files and build
  directories, and Daniel's Lean rulings (guarantees proved under their assumptions, Lean as the source of truth for
  them). Use when adding or changing Lean proofs, definitions or packages (backends/flock/verifier/lean, level3,
  soundness, verity/lean, verity/protocols/accounting/space/pous/lean, verity/protocols/accounting/communication/warden/lean), when a Lean build or proof is
  slow, editing a lean-audit.json or an ASSUMPTIONS.md ledger, bumping a Lean dependency, or preparing a Lean PR.
---

# Lean proofs

The rules are in `AGENTS.md` ("Lean"); the audit's reference is `tools/lean/README.md`. This is the workflow.

## Where the work goes

- **Pick the package by its dependencies.** Code the verifier runs, and anything the V[B] generator or a reference
  prover runs: `backends/flock/verifier/lean/` (no dependencies). Facts about the executable's definitions:
  `level3/` (Mathlib). C-Flock's guarantees (soundness, knowledge soundness, zero knowledge, audits): `soundness/`
  (ArkLib).
  Proofs about core's own definitions (`verity_catalog.silicon` and the like), and the model every protocol states its guarantees
  in (circuits, partitions, declarations, games): `verity/lean/` (Mathlib only), since core depends on no
  backend. A protocol's guarantees: its own package under `protocols/<name>/lean/`, which may require core's and other
  protocols' packages. A new package only when the dependencies differ; give it a `lean-audit.json` and the shared `lean-toolchain`.
- **Own a subdirectory and namespace** (`FlockSoundness/Knowledge/`, `FlockSoundness.Knowledge`), with one aggregator
  module that imports its files. The package root imports the aggregator once, when the area is created, so parallel
  lanes don't edit the same root file.
- **Definitions a guarantee reads go in definition files** (`Protocol/`, `Assumptions`, `Guarantees/` in the
  packages laid out under the four names, see `AGENTS.md`; `Model/`, `Accounting/`, `Assumptions.lean` elsewhere), not in
  a proof file. The audit lists where they are (`reads` in `lean-audit.json`); keep that list short.
- **Build in your own checkout.** Two agents never share a writable `.lake`: concurrent builds overwrite each other's
  `.olean` files, and an audit then reads what another agent's build left. A second Lean task on the same machine gets
  its own worktree. Seed its dependencies from a checkout with the same `lake-manifest.json` and `lean-toolchain`:
  `mkdir -p WT/PKG/.lake && cp -a --reflink=auto PKG/.lake/packages WT/PKG/.lake/`. Copy, never symlink, because Lake
  rewrites `.olean` files in place. After the copy, `lake build` compiles only the package's own modules.

## Build, iterate, audit

~~~text
bash tools/lean/setup.sh PKG                  once per machine: elan, the toolchain, Mathlib's cache
cd PKG && lake build                          once per checkout; incremental from then on
lake env lean Pkg/Area/File.lean              each edit: elaborate one file against the built imports
lake build Pkg.Area.File                      after editing a file others import, so they see the change
python tools/lean/audit.py PKG                once, before pushing (seconds; level3's replay takes ~3 min)
~~~

Build on a CPU machine; the three packages take about nine minutes from scratch on four cores. Never `lake update`, and
never change a `require`, a revision or `lean-toolchain` in a feature PR.

To check your records, or get the `--update` printout of what changed (what Daniel's DM carries), without a build of
your own, use the fast path on a shared machine. Don't run `audit.py --build --update` from a fresh tree there: that took
about 22 minutes unpinned on the check cores.

~~~text
research run --on vy-nebius-1 --project verity --source . --cwd source --declared-output 'lean-audit/*' -- \
  python3 tools/lean/lean_changed.py --records PKG --update
~~~

It waits for one of the node's Lean slots of pool `build` (`lean_slot.py`; a node whose slot file lists no `build` pool
refuses it, exit 2, even with an `any` pool), never behind merge checks or hour-long audits, rebuilds only what changed since
that slot's last run, and audits without the kernel replay: on `soundness/` with nothing to rebuild that was 6.5 minutes, and a
slot's first run, a cold build, 31. The record is in the run's `lean-audit/audit-*/`, with `review.txt` when it changed; commit
the record. It vouches for nothing: its result is of kind `lean-fast/v1`, which `check`'s lean-audit step and `research merge`
refuse.

To regenerate a record on a pod, run `research run --on POD --project verity --source . --cwd clone -- python
tools/lean/audit.py --build --update PKG` with no wrapper around it. It exits 0 when the audit passes, whether or not the
record changed, and the record is among the run's outputs, at `lean-audit/audit-TAG/lean-audit.json`. `--cwd clone` builds
in the run's own clone of the shipped tree, removed when the run ends; with `--cwd source` every run of the commit builds in
the one shared tree, and a cancelled duplicate there deleted `PKG/.lake/packages` under a live audit.

While proving, check one file at a time, and leave the whole package and the audit for the end:

- **`lake env lean FILE`** prints that file's errors and warnings, and writes nothing. Core's slowest file takes about 10 s
  this way. A file that imports what you changed sees the change only after `lake build` of the changed module.
- **Ask Lean in a scratch file** outside the repository, run from the package directory with `lake env lean
  /tmp/scratch/Probe.lean`. It takes well under a second against built imports:

  ~~~lean
  import Verity.SecurityProofs.TC.AmpereSound
  #print axioms Verity.TC.ampere_step_hw     -- a sorry shows as sorryAx, long before the audit runs
  example (a b : Nat) : a + b = b + a := by exact?
  ~~~

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
  stays, give its reason beside it. Keep files to a few hundred declarations, so that Lake builds them in parallel and an
  edit re-elaborates little.
- **Push after every commit.** Work that lives only in a VM's worktree is lost when the VM is suspended.

The replay refuses `X.congr_simp is declared by a replayed module and by one it imports`. `simp` realizes a structure
constructor's congruence lemma (`ZeroHom.mk.congr_simp`) in your module when it rewrites under an unfolded bundled map,
and Mathlib has realized it already. Don't unfold a `→+` or `→*` inside `simp`: prove its `_apply` lemma by `rfl` and
simp with that.

## Assumptions and guarantees

- **An assumption is a named `Prop`** in the package's assumptions module (`FlockSoundness.Assumptions`), taken as a
  hypothesis by each theorem that needs it (`(hCR : Assumptions.SHA512CRExpected …)`). Its docstring opens with its
  claim id (`uniform/io-getrandombytes`, from `verity.claims`) and gives its rationale. Lean is the source of truth for
  what we assume, so a ledger points to the docstring rather than restating it. Never an `axiom`, never a `sorry` on
  `main`, never `native_decide`, never an option that skips the kernel. A statement you can't prove yet is an
  assumption, or a theorem in a lane branch, not on `main`.
- **Freeze the statement, then prove it.** In your lane branch, state the guarantee on a `sorry` stub, check it against
  the paper or the spec, list it in `lean-audit.json` and run `--update`. The record holds the statement, so it passes once
  the proof lands, and the record shows if the statement moved on the way. Never make a
  guarantee easier: don't strengthen an input relation, weaken an output relation, shrink the challenge space, add an
  unmotivated hypothesis, or move a verifier check into an assumption. That list is also what Daniel looks for when a change reaches him.
- **List what you cite.** When a ledger row, a table or a PR description cites a guarantee as proved, add its theorem
  (`SecurityProofs.G` under the four names) to `guarantees` and run `python tools/lean/audit.py --update PKG`. Commit the
  record with the proof.
- **Watch upstream for what you assume.** A new named assumption or a lemma you prove by hand that a pinned dependency
  (Mathlib, ArkLib, VCVio, CompPoly) could prove gets an entry under `upstream.watch` in the package's `lean-audit.json`.
  The entry lists the dependency's constants that the statement is about. An assumption stated over the package's own
  definitions gets none: no upstream theorem can mention them, so the entry could never fire. Nor does one about a single
  number (PoUS's `PrimeP16448`): its entry would fire on every theorem about primes. The audit fails when the
  pinned dependency proves an unacknowledged theorem about them. Read the theorem, say in the PR whether it discharges the assumption or replaces the proof, then run
  `python tools/lean/upstream.py PKG --ack`. At a dependency bump, also run `upstream.py PKG --lean` (the exact pass).
  At a toolchain bump, also re-read `lean_io_get_random_bytes` in the new toolchain's `src/runtime/io.cpp`. A3
  (`soundness/ASSUMPTIONS.md`) cites what it reads, and `tests/test_lean_packages.py` fails until the citation names the
  new version.
  Cite the output in `ASSUMPTIONS.md`'s upstream status rather than writing one.
- **A changed record is a changed claim.** If `--update` changes a record or `reads` for a guarantee you didn't mean to
  change, your edit changed what that guarantee says: undo it, or say so in the PR and update the ledger.
- **A changed record reaches Daniel.** The axiom checks and the replay pass a weakened definition (a vacuous theorem is
  still a theorem), so paste what `--update` printed into the PR. When it lands on `main`, Daniel gets a DM showing the
  change; nobody signs off and nothing waits on it.
- **A move is not a change.** When you rename guarantees or move what they read (a package moving to the four names), write
  the moves as a JSON object of old names to new ones, a declaration or a namespace each, and run
  `audit.py --update --moved MOVES.json PKG` on the record as it was before the move. A guarantee that says what its
  record says, renamed or lifted, and the definitions the same under their old names, each get one line; the rest prints
  in full. Moving a definition to another module needs no map.
- **A lift needs no map.** `theorem X (xs) : e` becoming `theorem X (xs) : G as`, with `def G (ys) : Prop := e` in a
  `Guarantees` module, prints as "lifted into G" on a plain `--update` when `G` unfolded once is `e` (the same type hash)
  and the theorem keeps its named assumptions as binders (tools/lean/README.md, "A lift is not a change"). It needs no
  statement reviewer, but Daniel's DM still lists the changed record.

## Before the PR

- `python tools/lean/audit.py PKG` passes for every package you touched, and `pytest backends/flock/tests
  tools/lean/tests tests/test_lean_packages.py` passes.
- The PR says which guarantees changed, if any, and why; the ledger's status rows match `lean-audit.json`.
- Merge `main` into your branch rather than rebasing someone else's work; conflicts in a ledger are resolved by keeping
  both sides' rows. Run `python tools/lean/merge.py --install` once per clone: `lean-audit.json` then merges without a
  rebuild, and only a conflict it refuses needs `audit.py --update` on the merged tree.

## Rulings so far

These are Daniel's rulings, newest first.

- 2026-10-03 (6:46 PM PDT): a change to what a guarantee says needs no reviewer and no approval: when it lands on `main`,
  Daniel gets a DM showing it. Approval may come back for chosen packages (a grant in `tools/check/queue.toml`). "Pin" is
  retired: say guarantee.
- 2026-10-01: a pinned theorem is called a **guarantee** (security guarantee): a proved `Prop`, assumptions implying a
  conclusion, where an assumption is a named `Prop` (README "Glossary").
- 2026-10-01: a guarantee that only moved, renamed or restated as a `Guarantees` definition, needs no statement reviewer
  when `--update --moved` prints it as the same statement under its old names, with every definition it reads the same.
- 2026-10-01: a Lean protocol package may require another protocol's package (NCI's requiring sampled proofs', say),
  because composing their guarantees needs it. The requires stay acyclic, and a protocol never requires a backend. Such a
  `path` require changes no revision, so a feature PR may add it. Python's protocols import only `verity`, themselves
  and the protocols `EDGES` in `tests/test_protocol_boundaries.py` lists (PoUW → sampled proofs, the ruling of
  2026-10-03 in `.agents/skills/friction/SKILL.md`).
- 2026-10-01: there are guarantees and assumptions, and a guarantee is **proved under its assumptions**; there is no
  third status. Some assumptions are better than others: SHA-512's collision resistance (`cr/sha-512`) is a better one
  than a fact about our own circuit that we mean to prove (`HmRowComputes`). So what a guarantee assumes is explicit and
  hard to break, a table, a ledger or a PR cites a guarantee with its assumptions, and proving an assumption removes it
  from every guarantee that took it.
- 2026-09-30: Lean is the one source of truth for the assumptions we make. Views of them (the console, the tables) are
  derived from Lean and the audit's output, and no hand-kept ledger is generated alongside.
- 2026-09-30: probability and games shared across packages are our own minimal definitions over Mathlib's `PMF`, not
  VCVio's `OracleComp`; soundness bridges to VCVio where it needs it.
