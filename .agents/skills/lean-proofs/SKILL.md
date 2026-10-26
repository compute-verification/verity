---
name: lean-proofs
description: >
  How to work on Lean in Verity: where a new Lean file or package goes, the edit loop (verity lean try FILE on a node's
  warm tree; scratch probes with #print axioms, exact? and trace_state; profiling a slow proof), building and auditing with
  tools/verity/lean/check.py and its fast path, stating assumptions as named Props, freezing and listing guarantees in
  lean-audit.json, moving them without a DM to Daniel (--moved), the shape of a property, watching dependencies for
  proofs of what we assume (tools/verity/lean/upstream.py), keeping parallel Lean lanes out of each other's files and
  build directories, and Daniel's standing Lean rulings. Use when adding or changing Lean proofs, definitions or
  packages (verity/core/service/flock, Security and its Proofs and Properties, a protocol's lean/), when a Lean build
  or proof is slow or runs out of memory, editing a lean-audit.json or an ASSUMPTIONS.md ledger, bumping a Lean
  dependency, or preparing a Lean PR.
---

# Lean proofs

The rules are in `AGENTS.md` ("Lean"); the Lean check's reference is `tools/verity/lean/README.md`. This is the workflow.

## Where the work goes

- **Code or analysis**: AGENTS.md "Lean" says which tree and package. No new package: an area is a directory in each
  tree.
- **Own a subdirectory and namespace** with one aggregator module; the area's root imports the aggregator once, so
  parallel lanes don't edit the same root file.
- **Code the verifier runs costs O(description)**: summarize per Definition and multiply through batch members and scan
  iterations, never walk the unrolled circuit (`tests/test_verifier_cost.py` refuses the walk). A gate-by-gate version
  may stay as the reference the description-sized one is difftested against.
- **Parallel proof work runs against fixed statements**: the final theorem's statement and each open piece's statement
  are on the branch before any worker starts. What isn't closed by the deadline is a gap list of exact lemma names, each
  with a size and an owner, with no `sorry` outside them.
- **Definitions a guarantee or property reads go in `Definitions/` or `Models/`**, not in a proof file (`reads` in the
  lock lists where they are; keep it short). The Lean check refuses a statement that reads the proofs tree, and a
  definitions or properties module that imports it (`trees` in `Security/lean-audit.json`); a lemma's statement over
  proof-only definitions is not a guarantee.
- **Build in your own checkout.** Two agents never share a writable `.lake`. A second Lean task on one machine gets its
  own worktree, its dependencies copied (never symlinked) from a checkout with the same manifest and toolchain:
  `mkdir -p WT/PKG/.lake && cp -a --reflink=auto PKG/.lake/packages WT/PKG/.lake/`. A reviewer or red-team subagent is
  a cloud worker in its own checkout.

## Build, iterate, check

~~~text
verity lean try PKG/Area/File.lean     each edit: File's imports built, then `lake env lean` on it, on a node's warm tree
lean_changed.py --records PKG --update the fast path (below), once the proofs elaborate: the lock and the mark
python tools/verity/lean/check.py PKG  the full check, which the train's check runs; by hand only on a machine of your own
~~~

- **The edit loop is `verity lean try FILE`** (FILE from the repository's root). It ships your checkout, uncommitted
  edits and new files included, through `research run` (default vy-nebius-cpu-1's `research` pool; `--on NODE [--pool P]`
  for another node with `build` Lean slots, e.g. `--on vy-nebius-cpu-5 --pool proofs`), takes a `build` slot's tree there, the one `--records` keeps warm, builds
  only what changed among FILE's imports, runs `lake env lean FILE`, and prints Lean's output and the wall time, exiting
  with lean's status. It never writes a record. After editing a file others import, try the importer: its imports'
  build rebuilds the edited file.

- **Where it runs.** Build on a CPU machine. Never `lake update`, and never change a `require`, a revision or
  `lean-toolchain` in a feature PR. `check.py` refuses an ArkLib package below 24 GiB and a whole-Security
  audit below 160 GiB, and prints where to run it instead; a queued Security audit takes `--kind lean-audit`.
- **The fast path is for the lock and the mark, not the edit loop**: it gives your records, and the `--update` printout
  Daniel's DM carries, without a build of your own, once `verity lean try` passes on your files:

  ~~~text
  research run --on vy-nebius-cpu-1 --pool research --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- \
    python3 tools/verity/lean/lean_changed.py --records PKG --update
  ~~~

  On a node whose slot file has a `build` pool it takes one of its slots, rebuilds only what changed in that slot's tree,
  waits for another run's hold on a shared tree, and audits without the kernel replay. Where the slot file lists no `build`
  pool, or with `--dev-cache` (a slot tree whose dependencies are fetched cold), it takes the `--dev-cache` path instead:
  `check.py --dev-cache --build --no-replay` on the run's clone, its build restored from the published cache (Security:
  about 40 minutes); it prints which path it took. In cpu-1's research pool the pool's runner takes the check slot, so don't wrap it in
  `slot.py` there. On vy-nebius-1 a bare run is refused: wrap it in n1's Lean-record pool, `env
  CHECK_SLOTS=/workspace/research/locks/leanrec/slots python3 tools/verity/check/slot.py --by LANE -- python3
  tools/verity/lean/lean_changed.py --records PKG --update` (4 slots; the rest wait in slot.py's line). The
  record is in `lean-audit/audit-*/` and, when it changed, `lean-audit/record/` (with `review.txt`); commit it. It then checks every package whose
  guarantees read PKG's build (`lean-audit/requirer-<path>/`); when one fails, run `--records --update` on it too. Its
  result is `lean-fast/v1`, which no merge check accepts. A slot tree another run left mid-way (cancelled, timed out,
  killed for memory) is repaired before the build, with the reason printed, so don't delete or copy a slot tree by hand.
  A spec's failure `proved_in: P's build did not pass` points at P's own report, where its build failed.
  A package whose records passed on the same Lean inputs on that node (or whose Lean check passed there) is skipped, and
  so is one where HEAD only merged `main`, taking every Lean input of it whole from main's side and its lock from
  `merge.py`: that lock is a candidate the train's Lean check rebuilds. With nothing left the run builds nothing and
  takes no slot (`path: reused`); `--rebuild` runs it anyway.
- **A full record on a pod**: `research run --on POD --project verity --source . --cwd clone -- python3
  tools/verity/lean/check.py --build --update PKG`, with no wrapper (vy-nebius-1 has no `python`). It exits 0 when the
  check passes; the record is at `lean-audit/audit-TAG/lean-audit.json`. Use `--cwd clone`: runs of one commit share
  `--cwd source`'s tree, and two Lake builds there break each other.
- **Ask Lean in a scratch file** outside the repository, from the package directory: `lake env lean
  /tmp/scratch/Probe.lean` with `#print axioms X` (a `sorry` shows as `sorryAx`), or `example … := by exact?`. Probe what
  you build on too: a section `variable` whose type goes through a local notation can elaborate to `sorry` silently, or
  drop a theorem from the module with no error ("unknown constant" on a theorem you wrote is this). Put such binders on
  the theorem itself.
- **See a goal** with `trace_state` before a `sorry`.
- **Search before you prove**: `exact?` or `apply?`, then `rg` across every package's `.lean` files, then
  [Loogle](https://loogle.lean-lang.org). A lemma another package proves is imported or moved, not proved again.
- **When a declaration is slow**, measure first (`set_option profiler true in`; `#count_heartbeats in` in a Mathlib
  package), then fix the proof: `simp only [...]` outside terminal calls, a split-out lemma, a smaller `decide`. An
  override that stays has its reason in a `--` comment on or above its line (the audit lists every override). Keep
  files to a few hundred declarations.
- **`X.congr_simp is declared by a replayed module and by one it imports`**: `simp` realized a structure constructor's
  congruence lemma that Mathlib realized already. Don't unfold a `→+` or `→*` inside `simp`; prove its `_apply` lemma by
  `rfl` and simp with that.

## Assumptions and guarantees

- **An assumption is a named `Prop`** in its area's `Models/<Area>/Assumptions` (C-Flock's in
  `Proofs/Flock/Soundness/Assumptions` until its spec is extracted), taken as a hypothesis by each theorem that needs it
  and listed by name in the lock's `assumptions` with its owner. Its docstring opens with its claim id
  (`verity.core.claims`) and gives its rationale; a ledger points to the docstring. Never an `axiom`, a `sorry` on
  `main`, `native_decide` or a kernel bypass. Every premise a guarantee takes whose head is defined in a package of this
  repository is listed, wherever it is defined, and so is every assumption a premise mentions (`A ∧ B`, `¬ A`, `∃ x, A x`,
  a field of a structure a binder ranges over); `--update` adds it with its guarantees' owner.
- **Statement first, then red team, then a card to Daniel for each new premise, then the proof.** State the guarantee
  on a `sorry` stub in your lane branch, list it, and run `--update`, so the record holds the statement and shows if it
  moves. Never make a guarantee easier: no stronger input relation, weaker output relation, smaller challenge space,
  unmotivated hypothesis, or verifier check moved into an assumption.
- **List what you cite.** A guarantee a ledger, table or PR cites as proved is in `guarantees`; commit its record with
  the proof.
- **Watch upstream for what you assume.** A named assumption or hand proof that a pinned dependency (Mathlib, ArkLib,
  VCVio, CompPoly) could prove gets an `upstream.watch` entry listing the dependency's constants it is about (none for
  one over our own definitions, or about a single number). When the check fails on a new upstream theorem, read it, say
  in the PR whether it discharges the assumption, and run `tools/verity/lean/upstream.py PKG --ack`. At a dependency
  bump also run `upstream.py PKG --lean`; at a toolchain bump `tests/test_lean_packages.py` says what to re-read.
- **A changed record is a changed claim.** If `--update` changes a record you didn't mean to, your edit changed what that
  guarantee says: undo it, or say so in the PR. Paste the printout into the PR; when it lands, Daniel's DM shows it, and
  that DM is the check. Nothing waits on a reviewer.
- **A move is not a change.** Renames and moves go in a JSON map of old names to new (declarations or namespaces):
  `check.py --update --moved MOVES.json PKG` on the record as it was. A lift (`theorem X : e` to `theorem X : G` with
  `G` unfolding to `e`) needs no map.
- **One shape** (`shape` in `tools/verity/lean/README.md`). A property takes its inputs, one hypothesis that its function returned an outcome
  (`shape.properties`: `function` and `outcome`, `accept` by default), and premises (heads under `shape.premises`, or a
  conjecture); the rest is its conclusion. Binder order and a hypothesis `x.f = y.f` between two inputs it accepts don't
  matter: the check reads the statement reordered and with that field bound once, so don't restate for either. A
  probability statement has its acceptance in the event (`shape.events`).
  What doesn't fit yet is under `shape.frozen` with a reason starting with its category (`restatable`, `frozen (hash)`,
  `lemma`, `completeness`, `falsified`); the check fails once it fits. A named conjecture is under `shape.conjectures`,
  open with a falsification test that runs (a pytest node id) or falsified with why; a falsified one takes every
  property resting on it off the list. Every `def G : Prop` of a `Guarantees` module is a listed guarantee or frozen.
  Adding a frozen entry is a claims change: rather restate the guarantee, or move what isn't a property to `Proofs/`.

## Before the PR

- `check.py PKG` passes for every package you touched, and `pytest tools/verity/lean/tests
  tools/verity/check/tests/test_lean_*.py tests/test_lean_packages.py` passes.
- The PR says which guarantees changed and why; the ledger's status rows match `lean-audit.json`.
- Merge `main` into your branch rather than rebasing. Run `python tools/verity/lean/merge.py --install` once per clone, so
  `lean-audit.json` merges without a rebuild; only a conflict it refuses needs `check.py --update` on the merged tree.
- Record your lock at your own head, then mark: `ready` refuses a changed lock no records run of the head passed on
  its own bytes (`--no-preflight` too). A passing Lean check, a clean merge of `main` over a passing head, or a commit
  of exactly the `record/lean-audit.json` a passing run kept needs no new run.
  A train regenerates records that differ only because another PR moved what yours reads; re-record yourself only
  what it can't (a refused conflict, `shape`, a move's map).

## Standing rulings (Daniel)

The dated text is note:lean/20261020T1000Z-report-lean-proofs-rulings-archive.

- A guarantee is a proved `Prop` under named assumptions; there is no third status, and proving an assumption removes it
  from every guarantee that took it. Lean is the one source of truth for what we assume.
- A guarantee never assumes a hash or a signature. It ends in a break tied to the run: for SHA-512,
  `P ∨ ∃ x ∈ run.hashInputs, ∃ y ∈ run.hashInputs, x ≠ y ∧ sha512 x = sha512 y`; for LMS, a verifying signature on a
  message missing from the device's log; for Merkle binding, a collision along the two openings' paths. Every `∃` is
  bounded by what the run carries. No assumed advantage bounds, except PoUW's named random-oracle premises for SHA-512
  (`StraightLineExtraction`, `SeedExtractedAt`; ASK-16), each a named `Prop` the lock lists. An input
  condition is a verifier refusal, not a hypothesis.
- One adversary class (26 Oct): every prover strategy τ, no budget class.
  `Pr[accept ∧ the claim fails on E^τ's values] ≤ δ + f(Pr[B^τ exhibits a SHA-512 collision])` (+ LMS forgery), with
  `E` a fixed named extractor and `B` a finder built from τ in about τ's time (`XplurZC`, `outCZC`). Collision
  resistance is applied to `B` alone, outside the theorem.
- A claim reads `E`'s transcript, never a witness the prover picks, and a break counts only collisions the run or `E`
  exhibits (R4′). For a commit-then-coin step the break is the fork form (two runs of the prover's strategy sharing a
  commitment), and the knowledge form (`EndToEndHidden`, `Claims.KS`) is preferred.
- Every security guarantee has an honest lemma (its hypotheses hold at the honest deployed instance) and a nontrivial
  one (its conclusion fails for a verifier that accepts everything), in the lock as a claim's `meets` and `fails`
  are. Failing either, it says nothing (LEAN-66 enforces it).
- A property's premises are assumptions about the world the function can't see; what the code can see, it checks. A
  condition picking out what the conclusion talks about goes inside the conclusion. Completeness theorems are not claims:
  they live in `Proofs/` with one concrete instance where the honest case holds.
- A device model (falsifiable by measurement, scoped by device) is an assumption, not a conjecture; a statement mixing
  the two is split. A conjecture about finite objects is decided by computation. Incompressibility of fixed data is
  false in Lean, so it is stated over random inputs or as a reduction. A claim resting on a falsified premise says
  nothing.
- Probability and games are our own minimal definitions over Mathlib's `PMF`; soundness bridges to VCVio where needed.
- What a physical region reads is not a premise: it follows from the actual circuit's regions and causality
  (`width_into_region_le`; `Links`, `Covered` in `Security/Definitions/Accounting/Physical.lean`). A bound per
  server-window is stated on `Accounting.Placed` with `WindowsAt` (ASK-141).
