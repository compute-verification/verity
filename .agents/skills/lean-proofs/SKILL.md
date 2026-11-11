---
name: lean-proofs
description: >
  How to work on Lean in Verity: where Lean goes, the edit loop (verity lean try, its axiom probe and --siblings;
  scratch probes with #print axioms, exact?, trace_state; a slow proof), records, the fast path and check.py,
  stating a guarantee (counted extractors, what proved means), assumptions as named Props, listing, freezing and
  moving guarantees in lean-audit.json, a property's shape, upstream.py watches, parallel Lean lanes, and the standing
  Lean rulings. Use when adding or changing Lean proofs, definitions or packages (verity/protocols/flock/verifier, Security
  and its Proofs and Properties, a protocol's lean/), when a Lean build or proof is slow or runs out of memory, editing
  a lean-audit.json or an ASSUMPTIONS.md ledger, bumping a Lean dependency, or preparing a Lean PR.
---

# Lean proofs

The rules are AGENTS.md "Lean"; the Lean check's reference is `tools/verity/lean/README.md`. This is the workflow.

## Where the work goes

- **Code or analysis**: AGENTS.md "Lean" says which tree and package. No new package: an area is a directory in each
  tree.
- **Own a subdirectory and namespace** with one aggregator module, imported once by the area's root, so parallel lanes
  don't edit one root file.
- **Parallel proof work runs against fixed statements**, the final theorem's and each open piece's, on the branch before
  anyone proves a piece. What isn't closed by the deadline is a gap list of exact lemma names, each with a size and an
  owner, with no `sorry` outside them.
- **Definitions a guarantee or property reads go in `Definitions/`**, not in a proof file; a lemma's statement over
  proof-only definitions is not a guarantee.
- **Every module is on Lean's module system** (`tests/test_lean_packages.py`): a new file starts `module`, imports with
  `public import M`, and opens `@[expose] public section` after its header; `python3 tools/verity/lean/modules.py FILE`
  writes that. A proof's body is private, so an edit to it rebuilds no importer. Fix by hand what else the module
  system refuses, as the codemod's docstring says: a tactic's helper is `meta`; `import all` only of a core module,
  never an in-tree one; `:= (rfl)` for an exported `rfl` through an unexposed definition; `modules.py --option
  backward.privateInPublic FILE` where a public declaration reads a private one.
- **Build in your own checkout**: two agents never share a writable `.lake`, since two Lake builds in one tree break
  each other. A second Lean task on one machine copies its dependencies (never symlinks) from a checkout with the same
  manifest and toolchain: `mkdir -p WT/PKG/.lake && cp -a --reflink=auto PKG/.lake/packages WT/PKG/.lake/`. A
  reviewing or attacking subagent is a cloud agent in its own checkout.

## Build, iterate, check

- **The edit loop is `verity lean try FILE`** (the README): `lake env lean FILE` in FILE's package against its built
  imports (`bash tools/verity/lean/setup.sh PKG`, then `lake build` there, once), with lean's status. Lean's exit 0
  isn't proof: the last line, the axiom probe's, names each theorem of FILE whose axioms go beyond `propext`,
  `Classical.choice` and `Quot.sound` (`sorryAx` too). After editing a file others import, `lake build` its module, then
  try an importer. Try one that adds or renames declarations with `--siblings --on auto` too: a name another module
  declares otherwise clashes first at the package root, an hour into a records run.
- **Build on a CPU machine.** Never `lake update` or change an external dependency (a git `require` or its revision,
  Mathlib, ArkLib, `lean-toolchain`) in a feature PR; a `path` require between in-repo packages that AGENTS.md "Lean"
  allows may go in one, in its own commit with the regenerated `lake-manifest.json`, the requires acyclic. The full
  check, `check.py PKG`, runs in the merge queue's check; run it by hand only on a machine of your own.
- **The fast path is for the lock and the mark**, once `verity lean try` passes: it regenerates PKG's records and its
  requirers', with the `--update` printout the landing DM carries, without a build of your own (`lean_changed.py`'s
  docstring).

  ~~~text
  research run --on auto --mem-gb 32 --max-min 120 --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- \
    python3 tools/verity/lean/lean_changed.py --records PKG --update
  ~~~

  `--on auto` places it: never name a node, and never launch one run on two nodes to race them (cancel a duplicate).
  Don't wrap it in `slot.py`: where it lands, the runner takes the slot. Commit each record it kept under
  `lean-audit/record/` (its log says over what). Its result, `lean-fast/v1`, no merge check accepts. It repairs a slot
  tree another run left mid-way: never delete or copy one.
- **A full record**: `research run --on auto --mem-gb 32 --max-min 180 --project verity --source . --cwd clone --
  python3 tools/verity/lean/check.py --build --update PKG`, unwrapped, leaves it at
  `lean-audit/audit-TAG/lean-audit.json`; below a package's memory floor check.py refuses and prints the line to run.
- **Ask Lean in a scratch file** outside the repository, from the package directory (`lake env lean
  /tmp/scratch/Probe.lean`): `#print axioms X`, `example … := by exact?`, or
  `trace_state` before a `sorry`. Probe what you build on too: a section `variable` typed through a local notation can
  elaborate to `sorry` silently, or drop a theorem with no error ("unknown constant" on your own theorem); put such
  binders on the theorem.
- **Search before you prove**: `exact?` or `apply?`, `rg` across every package's `.lean` files, then
  [Loogle](https://loogle.lean-lang.org). A lemma another package proves is imported or moved, not proved again.
- **A slow declaration**: measure (`set_option profiler true in`; `#count_heartbeats in` in a Mathlib package), then fix
  the proof: `simp only [...]` outside terminal calls, a split-out lemma, a smaller `decide`. An override that stays has
  its reason in a `--` comment on or above it.
- **`X.congr_simp is declared by a replayed module and by one it imports`**: `simp` realized a structure constructor's
  congruence lemma Mathlib had already. Don't unfold a `→+` or `→*` in `simp`; prove its `_apply` lemma by `rfl` and simp
  with that.

## Stating a guarantee

- **Statement first, then an attack on it, then an ask for each new premise, then the proof.** State it on a `sorry`
  stub, list it and run `--update`, so the record shows if it moves. Never make a guarantee easier: no stronger input
  relation, weaker output relation, smaller challenge space, unmotivated hypothesis, or verifier check moved into an
  assumption.
- **Extractors and simulators are counted programs**, `Claims.Extractor` (`RP.eval` counts its SHA-512 calls and
  reruns) and `Claims.Simulator` (`SP.eval`), fixed before the prover and the witness; never a function, free fields
  or `Classical.choose`. hm96 hides statistically, so a free extractor opens each commitment to an honest trace
  and the guarantee says nothing. A cost is what the program evaluates, never a field. `tests/test_lean_vocabulary.py`
  flags a second `Extractor` or `Simulator`, not a renamed one.
- **The verifier derives what soundness reads**: a set soundness depends on (ports to register, sampled proof blocks,
  bound roots) is computed in Lean from public registered data. A file `verity.ml` or an integration writes only
  supplies values checked against it; else the guarantee covers the set the writer chose.
- **"Proved" is four checks**: it elaborates; it and each lemma you added use no axiom past the three AGENTS.md
  allows (the probe's line, or `#print axioms`); a full `check.py` replayed it through the kernel (the fast path
  doesn't); an agent other than its prover red-teamed its statement. When not all hold, say which: "elaborates, rests
  on `X`'s sorry", "proved, not replayed".

## Assumptions and guarantees

- **An assumption** (AGENTS.md "Lean") is declared in `Definitions/Assumptions.lean`, the lock's `assumptions_module`
  (a module or a list of them: `security/Properties` also lists `Properties/Assumptions.lean`, for one stated over the
  code, which the spec can't import), or in a module under it when the public export copies its area (`Assumptions/Flock.lean`, since the root module
  reads Nci); an area's other premises live in its `Definitions/<Area>/Assumptions` (Flock's in
  `Proofs/Flock/Soundness/Assumptions` for now), each owner in the lock. Its docstring opens with its claim id
  (`verity.claims`) and gives its rationale; a ledger points to it. `--update` digests every in-repo premise and
  the assumptions it mentions, so strengthening one changes a record.
- **List what you cite**: a guarantee a ledger, table or PR cites as proved is in `guarantees`, its record committed with
  the proof.
- **Watch upstream for what you assume**: a named assumption or hand proof a pinned dependency could prove gets an
  `upstream.watch` entry; on a new hit, say in the PR whether it discharges the assumption, and `--ack` it (the README's
  `upstream.py`).
- **A changed record is a changed claim**: an `--update` change you didn't mean is your edit changing what that
  guarantee says. Undo it, or say so in the PR and paste the printout.
- **A move is not a change**: map old names to new (declarations or namespaces) in `MOVES.json` for `check.py --update
  --moved MOVES.json PKG` on the record as it was (the fast path's `--records PKG --update --moved MOVES.json` takes it
  to PKG's requirers too); a lift (`theorem X : e` to `theorem X : G`, `G` unfolding to `e`) needs no map.
- **One shape** (`shape` in the README): a property takes its inputs, one hypothesis that its function returned an
  outcome, and premises (heads under `shape.premises`, or a conjecture); the rest, in any binder order, is its
  conclusion, a probability's acceptance inside the event. What doesn't fit yet is under `shape.frozen`, its reason
  opening with its category, and fails once it fits; a conjecture (`shape.conjectures`) has a falsification test that
  runs, and once falsified takes every property resting on it off the list. Every `def G : Prop` of a `Guarantees`
  module is listed or frozen; adding a frozen entry is a claims change, so rather restate the guarantee or move what
  isn't a property to `Proofs/`.

## Standards

The checks in `tests/test_lean_packages.py` and `tests/test_retired_terms.py` print their fixes. Two they don't
hold: Mathlib's names, and no binder `X` or `Y` where ArkLib is imported (its `notation3 "X"` and `"Y"` make each a
token, and the parse error lands far from the binder).

## Before the PR

- The PR says which guarantees changed and why; the ledger's status rows match `lean-audit.json`.
- Fetch and merge `main`, never rebase, with `python tools/verity/lean/merge.py --install` run once per clone so
  `lean-audit.json` merges without a rebuild, by `origin/main`'s copy of the driver; each `merge.py:` warning names an
  entry it took from one side, so read them. It conflicts only where it can't vouch for the merge (a newer lock `format`,
  the worktree's copy merging in newer merge rules, a merge that drops most of a section), and says why: do what it says.
- Record your lock at your own head, then mark; a passing Lean check, a clean merge of `main` over a passing head, or a
  commit of exactly the records a passing run kept needs no new run. Once your PR lands, drive's lock run regenerates
  records another PR moved; re-record only what it can't (a hand-written entry a merge warning says took theirs, `shape`, a
  move's map).

## Standing rulings

Their history is in the decisions log.

- A guarantee is a proved `Prop` under named assumptions, with no third status; proving an assumption removes it from
  every guarantee that took it. Lean is the one source of truth for what we assume.
- Named assumptions are the few real ones (a hash's collision resistance, the auditor's devices), in the lock's
  `assumptions_module`; any other premise is `domain` or `open` (the README).
- A guarantee never assumes a hash or a signature. It ends in a break tied to the run: for SHA-512,
  `P ∨ ∃ x ∈ run.hashInputs, ∃ y ∈ run.hashInputs, x ≠ y ∧ sha512 x = sha512 y`; for LMS, a verifying signature on a
  message missing from the device's log; for Merkle binding, a collision along the two openings' paths. Every `∃` is
  bounded by what the run carries. No assumed advantage bounds but PoUW's named random-oracle premises for SHA-512
  (`StraightLineExtraction`, `SeedExtractedAt`, in the lock) and the sampled audits' per-finder bound below. An input
  condition is a verifier refusal, not a hypothesis.
- One adversary class: every prover strategy τ, no budget class.
  `Pr[accept ∧ the claim fails on E^τ's values] ≤ δ + f(Pr[B^τ exhibits a SHA-512 collision])` (+ LMS forgery), with
  `E` a fixed named extractor and `B` a finder built from τ in about τ's time (`XplurZC`, `outCZC`). Collision
  resistance is applied to `B` alone, outside the theorem.
- Soundness is stated for the system as deployed: for recursion, challenges fixed at the start, hidden in a commitment
  and revealed round by round, resting on its hiding. The sampled audits' (`SpotSound`, from Flock's live
  chain) may take the generic SHA-512 collision bound at each prover-built finder (`TableCRZ`, `LinkCRZC`), its budget
  tied to the prover's cost (`finder_calls`), not at one combined finder. Zero knowledge is multiplicative, `real ≤
  2·sim + ε` (a leak of at most one bit plus the counted stop-position bits), reported as weekly, monthly and yearly
  totals.
- A claim reads `E`'s transcript, never a witness the prover picks, and a break counts only collisions the run or `E`
  exhibits. For a commit-then-coin step the break is the fork form (two runs of the prover's strategy sharing a
  commitment); the knowledge form (`EndToEndHidden`, `Claims.KS`) is preferred.
- A PoUW scheme gets new Lean only once its honest overhead γ ≤ 10 at both operating points, as the benchmarks'
  overhead front door measures it. Measure first, prove second.
- Every security guarantee has an honest lemma (its hypotheses hold at the honest deployed instance) and a nontrivial
  one (its conclusion fails for a verifier that accepts everything), in the lock as a claim's `meets` and `fails`;
  failing either, it says nothing. Both build their instance (`witnesses` in `check.py`), taking
  only listed assumptions; an exclusion's nontrivial lemma is a circuit where the other premises hold and one assumption and the bound fail, or, for an exclusion with no premise, one where a tighter bound fails. `nontrivial` may
  be a list: each lemma past the first has that exclusion form, whatever the guarantee's kind, and drops a premise no
  earlier one drops (a `Wf` negation, when `Wf` is a premise).
- A property's premises are assumptions about the world the function can't see; what the code can see, it checks. A
  condition picking out what the conclusion talks about goes inside the conclusion. Completeness theorems are not
  claims: they live in `Proofs/` with one concrete instance where the honest case holds.
- A device model (falsifiable by measurement, scoped by device) is an assumption, not a conjecture; a statement mixing
  the two is split. A conjecture about finite objects is decided by computation. Incompressibility of fixed data is
  false in Lean: state it over random inputs or as a reduction. A claim resting on a falsified premise says nothing.
- Probability and games are our own minimal definitions over Mathlib's `PMF`; soundness bridges to VCVio where needed.
- What a physical region reads is not a premise: it follows from the circuit's regions and causality
  (`width_into_region_le`; `Links`, `Covered` in `security/Definitions/Accounting/Physical.lean`). A bound per
  server block is stated on `Accounting.Placed` with `WindowsAt`.
- Verifier code compares `Json` by a match the kernel reduces, never by `BEq Json`'s `==`, a `partial def` no proof can
  read. A string is compared by `Spot.jsonStrIs`, which `Properties.Pouw.LiveSalt.jsonStrIs_eq` reads back.
  Replacing such an `==` or a partial reading with a total one that means the same (it refuses on every parse failure
  and never defaults) changes no claim: the lane makes the change and tells the code's owner.
