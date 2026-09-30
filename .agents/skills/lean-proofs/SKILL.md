---
name: lean-proofs
description: >
  How to work on Lean in Verity: where a new Lean file or package goes, building and auditing it with
  tools/lean/audit.py, stating assumptions as named Props, freezing and pinning the theorems a ledger or PR
  cites in lean-audit.json, watching ArkLib and other dependencies for proofs of what we assume
  (tools/lean/upstream.py), and keeping parallel Lean lanes out of each other's files and build directories.
  Use when adding or changing Lean proofs, definitions or packages (backends/flock/verifier/lean, level3,
  soundness), editing a lean-audit.json or an ASSUMPTIONS.md ledger, bumping a Lean dependency, or preparing
  a Lean PR.
---

# Lean proofs

The rules are in `AGENTS.md` ("Lean"); the audit's reference is `tools/lean/README.md`. This is the workflow.

## Where the work goes

- **Pick the package by its dependencies.** Code the verifier runs, and anything the V[B] generator or a reference
  prover runs: `backends/flock/verifier/lean/` (no dependencies). Facts about the executable's definitions:
  `level3/` (Mathlib). Protocol theorems (soundness, knowledge soundness, zero knowledge, audits): `soundness/` (ArkLib).
  A new package only when the dependencies differ; give it a `lean-audit.json` and the shared `lean-toolchain`.
- **Own a subdirectory and namespace** (`FlockSoundness/Knowledge/`, `FlockSoundness.Knowledge`), with one aggregator
  module that imports its files. The package root imports the aggregator once, when the area is created, so parallel
  lanes don't edit the same root file.
- **Definitions a pinned statement reads go in definition files** (`Model/`, `Accounting/`, `Assumptions.lean`), not in a
  proof file. The audit lists where they are (`reads` in `lean-audit.json`); keep that list short.
- **Build in your own checkout.** Two agents never share a writable `.lake`: concurrent builds overwrite each other's
  `.olean` files, and an audit then reads what another agent's build left. A second Lean task on the same machine gets
  its own worktree.

## Build and audit

~~~text
bash tools/lean/setup.sh PKG                       once per machine: elan, the toolchain, Mathlib's cache
cd PKG && lake build
python tools/lean/audit.py PKG                     what check runs, for one package (seconds; level3's replay takes ~3 min)
~~~

Build on a CPU machine; the three packages take about nine minutes from scratch on four cores. Never `lake update`, and
never change a `require`, a revision or `lean-toolchain` in a feature PR.

## Assumptions and pins

- **An assumption is a named `Prop`** in the package's assumptions module (`FlockSoundness.Assumptions`), taken as a
  hypothesis by each theorem that needs it (`(hCR : Assumptions.SHA512ExpectedTimeCR …)`), and stated in the ledger
  (`ASSUMPTIONS.md`). Never an `axiom`, never a `sorry` on `main`, never `native_decide`, never an option that skips
  the kernel. A statement you can't prove yet is a hypothesis, or a theorem in a lane branch, not on `main`.
- **Freeze the statement, then prove it.** In your lane branch, state the theorem on a `sorry` stub, check it against the
  paper or the spec, pin it and run `--update`, and have the statement reviewed before the proof. The record holds the
  statement, so it passes once the proof lands, and the pin shows if the statement moved on the way. Never make a theorem
  easier: don't strengthen an input relation, weaken an output relation, shrink the challenge space, add an unmotivated
  hypothesis, or move a verifier check into an assumption. That list is also what the statement reviewer checks.
- **Pin what you cite.** When a ledger row, a table or a PR description says a theorem is proved, add its name under
  `pins` and run `python tools/lean/audit.py --update PKG`. Commit the record with the proof.
- **Watch upstream for what you assume.** A new named assumption, or a lemma you prove by hand that ArkLib, VCVio or
  CompPoly could prove, gets an entry under `upstream.watch` in the package's `lean-audit.json`. The entry lists the
  constants its statement is about. The audit fails when the pinned dependency proves an unacknowledged theorem about
  them. Read the theorem, say in the PR whether it discharges the assumption or replaces the proof, then run
  `python tools/lean/upstream.py PKG --ack`. At a dependency bump, also run `upstream.py PKG --lean` (the exact pass).
  At a toolchain bump, also re-read `lean_io_get_random_bytes` in the new toolchain's `src/runtime/io.cpp`. A3
  (`soundness/ASSUMPTIONS.md`) cites what it reads, and `tests/test_lean_packages.py` fails until the citation names the
  new version.
  Cite the output in `ASSUMPTIONS.md`'s upstream status rather than writing one.
- **A changed record is a changed claim.** If `--update` changes a record or `reads` for a theorem you didn't mean to
  change, your edit changed what that theorem says: undo it, or say so in the PR and update the ledger.
- **A changed record needs a named statement reviewer.** The axiom checks and the replay pass a weakened definition
  (a vacuous theorem is still a theorem), so paste what `--update` printed into the PR, and ask someone other than the
  author to read it. The merge handoff names that reviewer.

## Before the PR

- `python tools/lean/audit.py PKG` passes for every package you touched, and `pytest backends/flock/tests
  tools/lean/tests tests/test_lean_packages.py` passes.
- The PR says which pins changed, if any, and why; the ledger's status rows match the pins.
- Merge `main` into your branch rather than rebasing someone else's work; conflicts in a ledger are resolved by keeping
  both sides' rows. Run `python tools/lean/merge.py --install` once per clone: `lean-audit.json` then merges without a
  rebuild, and only a conflict it refuses needs `audit.py --update` on the merged tree.
