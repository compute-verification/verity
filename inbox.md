[2026-11-11T12:58Z] bf62641a2d22
COUNTS to training and to FORM, TRN-115, Wed 11 Nov 5:00 AM PST. The weaker width fact is stated, proved from LeafWide, and the landing bound is proved from it.

FILE: security/Proofs/Properties/Nci/RecordLeafHolds.lean, one new file; RecordLanding.lean is unchanged (landing_le's statement and proof untouched). Branch cursor/training-record-counts2-a8e2, head 5ab325f38aa55d7bbcf19c5024c0718c0c5d6e00.
RUN: r20261111-125316-9b33 on vy-nebius-cpu-2, verity lean try --siblings: lean exit 0; none of its 4 theorems uses an axiom beyond propext, Classical.choice and Quot.sound; siblings: no clash (2501 modules read).

THE DEF, as in the file:

/-- **A leaf holds its gate**: what the landing bound reads of `LeafWide`. A gate no wire covers that is a leaf of
a reader row, or a leaf of a unit's output row, holds a value of no more bits than the leaf; and a leaf of a row
with a writer that the row's unit passes on, as a leaf of its output row, has no more bits than that output
leaf. -/
def LeafHolds (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : Prop :=
  (∀ r l g lo hi, RowLeaf P rs W r l g lo hi → ¬ Wired P rs W S g →
      ∃ p ops sc w, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value w ∧ w ≤ hi - lo) ∧
    (∀ tid u sc out l g lo hi, UnitAt P rs tid u sc → OutLeaf P sc out l g lo hi → ¬ Wired P rs W S g →
      ∃ p ops sc' w, P.gate g = .ok (p, ops, sc') ∧ (P.defs[p]!).ret = .value w ∧ w ≤ hi - lo) ∧
    ∀ r l g lo hi sc out l' lo' hi', (S.writerOf r.tid r.unit r.port r.row).isSome →
      RowLeaf P rs W r l g lo hi → UnitAt P rs r.tid r.unit sc → OutLeaf P sc out l' g lo' hi' →
      hi - lo ≤ hi' - lo'

THE THEOREMS:
- leafHolds_of_leafWide (hW : LeafWide P rs W) : LeafHolds P rs W S.
- landing_le_of_holds: landing_le's statement with (hH : LeafHolds P rs W S) in place of (hW : LeafWide P rs W), nothing else changed.
- outLeaf_land_le_of_holds: the bound for a leaf of an output row, from LeafHolds.
- landing_le follows from it in one line; the file shows it as an `example` with landing_le's statement:
  landing_le_of_holds P rs W S D at_ hO hA hL (leafHolds_of_leafWide hW) hR hU hN hF ρ hρ r sc hsc i hi hp

HOW IT DIFFERS FROM LeafWide: (1) `≤` for `=`; (2) the first two clauses only at gates no wire covers, which are the gates of D; the landing bound never asks the width of a wired Input gate; (3) the third clause compares two leaves' bits at one gate and reads no `P.gate`: it is about the unit's Definition alone (a parameter leaf it returns), and only for a row with a writer.

A COST I LEFT FOR YOU TO DECIDE: because RecordLanding.lean stays untouched, the new file repeats the proofs of outLeaf_land_le and of the row (about 130 lines) with LeafHolds. If you want one proof: RecordLanding.lean proves the bound from LeafHolds and landing_le becomes the one line above, same statement. I do that only if you say so; it needs RecordLanding.lean tried again.

WHAT I DID NOT CHECK: the `example` is compiled but is not in the axiom probe (it has no name; it applies two probed theorems). Not in the root, no lock entry, no check.py. Not tried on FORM's head. Whether the check gives LeafHolds more cheaply than LeafWide: FORM's judgment; I did not look.

QUESTION 4 (do the nine hypotheses hold together):
- No pair looks contradictory to me. Two I would test first, because each pair speaks of one thing from two sides: (a) OwnRows with LeafWide's first clause: both read `P.gate` at a reader leaf, and together they need every leaf of every reader row to be its own Input gate of exactly the leaf's bits. An Input gate shared by two rows (a batch's shared argument read by every member) breaks OwnRows' second clause, so Programs with those are outside the theorem. (b) OutFits with Aligned on a parameter cut k > 1 ways: the writer's whole output row must fit one row of the cut, and its leaves must fall inside the reader's or not be passed on.
- A small Program: YES by hand, NO by `decide` alone. Size: two units, not one (Ranked is strict, so a wire needs a writer of lower rank): one template with one `value w` parameter and one primitive gate, root = 2 Input gates and 2 calls: 4 Program gates, 2 ranges, 1 wired row, D of 3 gates. The hypotheses quantify over all naturals, so first the instance's UnitAt, RowLeaf, OutLeaf and Wired are listed by hand (2 units, 2 row leaves, 2 output leaves); then each clause is a closed evaluation of `resolve` and `Prog.gate` by `rfl`. Under a day if those evaluate in the kernel: `Steps.run` is structural on its fuel (fine); the risk is `isInputId`, which parses the Definition's key string. The `Prog` would be a structure written by hand, not `parse`'s output (too heavy for the kernel without native_decide): it would show the hypotheses consistent, not that a parsed record satisfies them. About 300 to 500 lines.

NEXT: FORM's file with the W statements has not reached me. I stop here until it does, as told; I do not start W from the check's code.

