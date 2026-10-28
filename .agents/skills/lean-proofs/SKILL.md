---
name: lean-proofs
description: >
  How to work on Lean in Verity: where a new Lean file or package goes, the edit loop (verity lean try FILE, lake env lean
  in the file's package; scratch probes with #print axioms, exact? and trace_state; profiling a slow proof), building and auditing with
  tools/verity/lean/check.py and its fast path, stating assumptions as named Props, freezing and listing guarantees in
  lean-audit.json, moving them without a landing DM (--moved), the shape of a property, watching dependencies for
  proofs of what we assume (tools/verity/lean/upstream.py), keeping parallel Lean lanes out of each other's files and
  build directories, and the standing Lean rulings. Use when adding or changing Lean proofs, definitions or
  packages (verity/core/service/flock, Security and its Proofs and Properties, a protocol's lean/), when a Lean build
  or proof is slow or runs out of memory, editing a lean-audit.json or an ASSUMPTIONS.md ledger, bumping a Lean
  dependency, or preparing a Lean PR.
---

# Lean proofs

The rules are AGENTS.md "Lean"; the Lean check's reference is `tools/verity/lean/README.md`. This is the workflow.

## Where the work goes

- **Code or analysis**: AGENTS.md "Lean" says which tree and package. No new package: an area is a directory in each
  tree.
- **Own a subdirectory and namespace** with one aggregator module, imported once by the area's root, so parallel lanes
  don't edit one root file.
- **Parallel proof work runs against fixed statements**: the final theorem's statement and each open piece's statement
  are on the branch before anyone proves a piece. What isn't closed by the deadline is a gap list of exact lemma names, each
  with a size and an owner, with no `sorry` outside them.
- **Definitions a guarantee or property reads go in `Definitions/`**, not in a proof file (`reads` in the
  lock lists where they are; keep it short). The Lean check refuses a statement that reads the proofs tree, and a
  definitions or properties module that imports it (`trees` in `Security/lean-audit.json`); a lemma's statement over
  proof-only definitions is not a guarantee.
- **Build in your own checkout**: two agents never share a writable `.lake`, since two Lake builds in one tree break
  each other. A second Lean task on one machine gets its own worktree, its dependencies copied (never symlinked) from a
  checkout with the same manifest and toolchain: `mkdir -p WT/PKG/.lake && cp -a --reflink=auto PKG/.lake/packages
  WT/PKG/.lake/`. A reviewing or attacking subagent runs as a cloud agent in its own checkout.

## Build, iterate, check

- **The edit loop is `verity lean try FILE`**: `lake env lean FILE` in the Lake package that holds FILE, against its
  built imports (`bash tools/verity/lean/setup.sh PKG`, then `lake build` there, once), exiting with lean's status. It
  never writes a record. After editing a file others import, `lake build` its module, then try an importer. It also runs
  on a project machine with `--on` (the README).
- **Build on a CPU machine.** Never `lake update`, and never change a `require`, a revision or `lean-toolchain` in a
  feature PR. The full check, `python tools/verity/lean/check.py PKG`, is the train's; run it by hand only on a machine
  of your own. It refuses a package too big for the machine, naming where to run it; a queued Security audit takes
  `--kind lean-audit`.
- **The fast path is for the lock and the mark**, once `verity lean try` passes: your records and the `--update` printout
  the landing DM carries, without a build of your own (`lean_changed.py`'s docstring).

  ~~~text
  research run --on vy-nebius-cpu-1 --pool research --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- \
    python3 tools/verity/lean/lean_changed.py --records PKG --update
  ~~~

  In cpu-1's research pool the runner takes the check slot: don't wrap it in `slot.py`. On vy-nebius-1 a bare run is
  refused: wrap it in n1's Lean-record pool, `env CHECK_SLOTS=/workspace/research/locks/leanrec/slots python3
  tools/verity/check/slot.py --by LANE -- python3 tools/verity/lean/lean_changed.py --records PKG --update`. Commit the
  record from `lean-audit/audit-*/` and, when it changed, `lean-audit/record/` (with `review.txt`). It also audits every
  package whose guarantees read PKG's build (`lean-audit/requirer-<path>/`): run `--records --update` on one that fails.
  Its result, `lean-fast/v1`, no merge check accepts. It repairs a slot tree another run left mid-way, so never delete or
  copy one by hand. A spec's `proved_in: P's build did not pass` points at P's own report.
- **A full record on a pod**: `research run --on POD --project verity --source . --cwd clone -- python3
  tools/verity/lean/check.py --build --update PKG`, unwrapped (vy-nebius-1 has no `python`); it exits 0 when the check
  passes, the record at `lean-audit/audit-TAG/lean-audit.json`. Use `--cwd clone`: runs of one commit share `--cwd
  source`'s tree.
- **Ask Lean in a scratch file** outside the repository, from the package directory: `lake env lean
  /tmp/scratch/Probe.lean` with `#print axioms X` (a `sorry` shows as `sorryAx`), or `example … := by exact?`. Probe what
  you build on too: a section `variable` typed through a local notation can elaborate to `sorry` silently, or drop a
  theorem with no error ("unknown constant" on your own theorem); put such binders on the theorem. See a goal with
  `trace_state` before a `sorry`.
- **Search before you prove**: `exact?` or `apply?`, `rg` across every package's `.lean` files, then
  [Loogle](https://loogle.lean-lang.org). A lemma another package proves is imported or moved, not proved again.
- **A slow declaration**: measure (`set_option profiler true in`; `#count_heartbeats in` in a Mathlib package), then fix
  the proof: `simp only [...]` outside terminal calls, a split-out lemma, a smaller `decide`. An override that stays has
  its reason in a `--` comment on or above it. Keep files to a few hundred declarations.
- **`X.congr_simp is declared by a replayed module and by one it imports`**: `simp` realized a structure constructor's
  congruence lemma Mathlib had already. Don't unfold a `→+` or `→*` in `simp`; prove its `_apply` lemma by `rfl` and simp
  with that.

## Assumptions and guarantees

- **An assumption is a named `Prop`** in its area's `Definitions/<Area>/Assumptions` (C-Flock's in
  `Proofs/Flock/Soundness/Assumptions` until its spec is extracted), taken as a hypothesis by each theorem that needs it
  and listed by name in the lock's `assumptions` with its owner. Its docstring opens with its claim id
  (`verity.core.claims`) and gives its rationale; a ledger points to it. `--update` lists every premise a guarantee
  takes that this repository defines, and every assumption a premise mentions, with a digest, so strengthening a
  premise is a record change. AGENTS.md "Lean" has the rest.
- **Statement first, then an attack on it, then an ask for each new premise, then the proof.** State the guarantee on
  a `sorry` stub in your lane branch, list it and run `--update`, so the record shows if it moves. Never make a guarantee
  easier: no stronger input relation, weaker output relation, smaller challenge space, unmotivated hypothesis, or
  verifier check moved into an assumption.
- **List what you cite**: a guarantee a ledger, table or PR cites as proved is in `guarantees`, its record committed with
  the proof.
- **Watch upstream for what you assume**: a named assumption or hand proof a pinned dependency could prove gets an
  `upstream.watch` entry (the README's `upstream.py` section). When the check fails on a new upstream theorem, read it,
  say in the PR whether it discharges the assumption, and `--ack` it; at a dependency bump run `upstream.py PKG --lean`,
  at a toolchain bump `tests/test_lean_packages.py` says what to re-read.
- **A changed record is a changed claim**: one `--update` changed that you didn't mean to means your edit changed what
  that guarantee says. Undo it, or say so in the PR, and paste the printout into the PR.
- **A move is not a change**: map old names to new (declarations or namespaces) in a JSON file and run `check.py --update
  --moved MOVES.json PKG` on the record as it was. A lift (`theorem X : e` to `theorem X : G`, `G` unfolding to `e`)
  needs no map.
- **One shape** (`shape` in `tools/verity/lean/README.md`). A property takes its inputs, one hypothesis that its function
  returned an outcome (`shape.properties`), and premises (heads under `shape.premises`, or a conjecture); the rest is its
  conclusion. Binder order and a hypothesis `x.f = y.f` between two inputs it accepts don't matter, so don't restate for
  either. A probability statement has its acceptance in the event (`shape.events`). What doesn't fit yet is under
  `shape.frozen`, its reason opening with its category, and fails once it fits; a conjecture under `shape.conjectures` is
  open with a falsification test that runs, or falsified, taking every property resting on it off the list. Every `def G
  : Prop` of a `Guarantees` module is a listed guarantee or frozen. Adding a frozen entry is a claims change: rather
  restate the guarantee, or move what isn't a property to `Proofs/`.

## Before the PR

- `check.py PKG` passes for every package you touched, and so does `pytest tools/verity/lean/tests
  tools/verity/check/tests/test_lean_*.py tests/test_lean_packages.py`.
- The PR says which guarantees changed and why; the ledger's status rows match `lean-audit.json`.
- Merge `main` in rather than rebasing, with `python tools/verity/lean/merge.py --install` run once per clone so
  `lean-audit.json` merges without a rebuild; only a conflict it refuses needs `check.py --update` on the merged tree.
- Record your lock at your own head, then mark (`ready` refuses a changed lock no records run of that head passed); a
  passing Lean check, a clean merge of `main` over a passing head, or a commit of exactly the `record/lean-audit.json` a
  passing run kept needs no new run. A train regenerates records that differ only because another PR moved what yours
  reads; re-record only what it can't (a refused conflict, `shape`, a move's map).

## Standing rulings

The Lean rulings in force, each stated as itself; who set each one, when and why, are in the decisions log, outside the
repository.

- A guarantee is a proved `Prop` under named assumptions, with no third status; proving an assumption removes it from
  every guarantee that took it. Lean is the one source of truth for what we assume.
- A guarantee never assumes a hash or a signature. It ends in a break tied to the run: for SHA-512,
  `P ∨ ∃ x ∈ run.hashInputs, ∃ y ∈ run.hashInputs, x ≠ y ∧ sha512 x = sha512 y`; for LMS, a verifying signature on a
  message missing from the device's log; for Merkle binding, a collision along the two openings' paths. Every `∃` is
  bounded by what the run carries. No assumed advantage bounds, except PoUW's named random-oracle premises for SHA-512
  (`StraightLineExtraction`, `SeedExtractedAt`), listed in the lock. An input condition is a verifier refusal, not a
  hypothesis.
- One adversary class: every prover strategy τ, no budget class.
  `Pr[accept ∧ the claim fails on E^τ's values] ≤ δ + f(Pr[B^τ exhibits a SHA-512 collision])` (+ LMS forgery), with
  `E` a fixed named extractor and `B` a finder built from τ in about τ's time (`XplurZC`, `outCZC`). Collision
  resistance is applied to `B` alone, outside the theorem.
- A claim reads `E`'s transcript, never a witness the prover picks, and a break counts only collisions the run or `E`
  exhibits. For a commit-then-coin step the break is the fork form (two runs of the prover's strategy sharing a
  commitment); the knowledge form (`EndToEndHidden`, `Claims.KS`) is preferred.
- Every security guarantee has an honest lemma (its hypotheses hold at the honest deployed instance) and a nontrivial
  one (its conclusion fails for a verifier that accepts everything), in the lock as a claim's `meets` and `fails` are;
  failing either, it says nothing. `check.py` holds them to it (`witnesses`): both lemmas build their instance, taking
  only listed assumptions; an exclusion's nontrivial lemma is a circuit where the other premises hold and one named
  premise and the bound fail.
- A property's premises are assumptions about the world the function can't see; what the code can see, it checks. A
  condition picking out what the conclusion talks about goes inside the conclusion. Completeness theorems are not claims:
  they live in `Proofs/` with one concrete instance where the honest case holds.
- A device model (falsifiable by measurement, scoped by device) is an assumption, not a conjecture; a statement mixing
  the two is split. A conjecture about finite objects is decided by computation. Incompressibility of fixed data is false
  in Lean, so it is stated over random inputs or as a reduction. A claim resting on a falsified premise says nothing.
- Probability and games are our own minimal definitions over Mathlib's `PMF`; soundness bridges to VCVio where needed.
- What a physical region reads is not a premise: it follows from the actual circuit's regions and causality
  (`width_into_region_le`; `Links`, `Covered` in `Security/Definitions/Accounting/Physical.lean`). A bound per
  server-window is stated on `Accounting.Placed` with `WindowsAt`.
