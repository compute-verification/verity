import Proofs.Core.NaiveStrassen

/-!
Scratch (CMP-400 plan, never landed): Theorem 1 of naive tiled against Strassen and its size form with the atom table
as a parameter, today's two listed statements as their corollaries, and the witnesses a listed general statement would
need.

The general forms are `main`'s `Packed.beats_of_check` and `Packed.upTo_of_check` at the kernel check's result. Their
hypothesis on a table is `Packed.Dom`: every atom of the table, wherever it holds, weighs at least the cheapest atom
of the reference table that holds there. The reference table is still the constant `rtx6000Sheet`, which the grid and
the kernel table are computed at.
-/

namespace Verity.Accel.NaiveStrassen

/-- **Theorem 1 at any atom table** within the reference's reach, to depth 12. -/
def NaiveBeatsStrassenAny : Prop := ∀ fs : List Atom, Packed.Dom fs → NaiveBeatsStrassen fs 12

/-- **Theorem 1 in size at any atom table** within the reference's reach, at `c = 4096`. -/
def NaiveBeatsStrassenAnySize : Prop := ∀ fs : List Atom, Packed.Dom fs → NaiveBeatsStrassenUpTo fs 4096

/-- A table within the reference's reach has no free atom: every atom holds at `(0, 0)`, where the reference's
cheapest atom weighs `1` (b1). So the size form needs no second hypothesis. -/
theorem weight_pos_of_dom {fs : List Atom} (hD : Packed.Dom fs) : ∀ f ∈ fs, 1 ≤ f.weight8 := by
  intro f hf
  have h0 : f.holds 0 0 = true := by
    rcases f with ⟨x, c, _ | ⟨k, w⟩⟩ <;> first | simp [Atom.holds] | decide
  have h1 : Packed.atomMin rtx6000Sheet 0 0 ≤ f.weight8 := hD f hf 0 0 h0
  have e : Packed.atomMin rtx6000Sheet 0 0 = 1 := by decide
  omega

theorem naiveBeatsStrassenAny : NaiveBeatsStrassenAny :=
  fun _ hD => Packed.beats_of_check Packed.checkAll_eq_true hD

theorem naiveBeatsStrassenAnySize : NaiveBeatsStrassenAnySize :=
  fun _ hD => Packed.upTo_of_check Packed.checkAll_eq_true hD (weight_pos_of_dom hD)

/-! ## Today's two listed statements, from the general ones and one fact about each table -/

theorem served_conj :
    NaiveBeatsStrassen h200Sheet 12 ∧ NaiveBeatsStrassen h200Measured 12 ∧ NaiveBeatsStrassen rtx6000Sheet 12 ∧
      NaiveBeatsStrassen rtx6000Measured 12 :=
  ⟨naiveBeatsStrassenAny _ Packed.dom_h200Sheet, naiveBeatsStrassenAny _ Packed.dom_h200Measured,
    naiveBeatsStrassenAny _ Packed.dom_rtx6000Sheet, naiveBeatsStrassenAny _ Packed.dom_rtx6000Measured⟩

theorem servedSize_conj :
    NaiveBeatsStrassenUpTo h200Sheet 4096 ∧ NaiveBeatsStrassenUpTo h200Measured 4096 ∧
      NaiveBeatsStrassenUpTo rtx6000Sheet 4096 ∧ NaiveBeatsStrassenUpTo rtx6000Measured 4096 :=
  ⟨naiveBeatsStrassenAnySize _ Packed.dom_h200Sheet, naiveBeatsStrassenAnySize _ Packed.dom_h200Measured,
    naiveBeatsStrassenAnySize _ Packed.dom_rtx6000Sheet, naiveBeatsStrassenAnySize _ Packed.dom_rtx6000Measured⟩

/-- The listed statement, as a corollary. -/
theorem served_of_general : Verity.Guarantees.NaiveBeatsStrassenServed := served_conj

/-- The listed size statement, as a corollary. -/
theorem servedSize_of_general : Verity.Guarantees.NaiveBeatsStrassenServedSize := servedSize_conj

/-! ## The witnesses a listed general statement would need -/

/-- **Honest**: the hypothesis holds at a closed table (the H200's data sheet's), and a fast schedule is valid on
it at depth 1 and attains the bound, so the statement says something there. -/
theorem any_honest :
    Packed.Dom h200Sheet ∧ winogradOnce.valid h200Sheet schemes 0 1 256 256 = true ∧ winogradOnce.fast = true ∧
      winogradOnce.cost 1 = 8 ^ 2 + 32 :=
  ⟨Packed.dom_h200Sheet, by decide, by decide, by decide⟩

/-- **Honest, closed at the reference table alone**, for a lock in which the cards' own tables sit beside the proof:
the same facts at `rtx6000Sheet`, which the hypothesis names. -/
theorem any_honest_ref :
    Packed.Dom rtx6000Sheet ∧ winogradOnce.valid rtx6000Sheet schemes 0 1 256 256 = true ∧
      winogradOnce.fast = true ∧ winogradOnce.cost 1 = 8 ^ 2 + 32 :=
  ⟨Packed.dom_rtx6000Sheet, by decide, by decide, by decide⟩

/-- **Honest, size form**: the hypotheses hold at the same table and schedule at `l = b = n = 1`. -/
theorem anySize_honest :
    Packed.Dom h200Sheet ∧ winogradOnce.valid h200Sheet schemes 0 1 256 256 = true ∧ winogradOnce.fast = true ∧
      1 ≤ 1 ∧ 1 ≤ 2 ^ 1 * 1 ∧ 1 ≤ 4096 * 1 ∧ 8 * 1 ^ 3 < 1 ^ 3 * winogradOnce.cost 1 :=
  ⟨Packed.dom_h200Sheet, by decide, by decide, by decide, by decide, by decide, by decide⟩

/-- One Winograd at depth 1 on a free atom. -/
def freeOnce : Sched := .scheme winograd fun _ => .atom ⟨0, 4097, none⟩

/-- **Nontrivial**: the conclusion is not a tautology. At a table with a free atom, Winograd once costs nothing. -/
theorem any_nontrivial : ¬ NaiveBeatsStrassen [(⟨0, 4097, none⟩ : Atom)] 12 := by
  intro h
  have h1 := h 1 (by decide) freeOnce (by decide) (by decide)
  exact absurd h1 (by decide)

/-- **Nontrivial, size form**. -/
theorem anySize_nontrivial : ¬ NaiveBeatsStrassenUpTo [(⟨0, 4097, none⟩ : Atom)] 4096 := by
  intro h
  have h1 := h 1 1 1 freeOnce (by decide) (by decide) (by decide) (by decide) (by decide)
  exact absurd h1 (by decide)

/-- The free table is outside the hypothesis, as it must be. -/
theorem free_not_dom : ¬ Packed.Dom [(⟨0, 4097, none⟩ : Atom)] := fun hD =>
  any_nontrivial (naiveBeatsStrassenAny _ hD)

end Verity.Accel.NaiveStrassen

#print axioms Verity.Accel.NaiveStrassen.naiveBeatsStrassenAny
#print axioms Verity.Accel.NaiveStrassen.naiveBeatsStrassenAnySize
#print axioms Verity.Accel.NaiveStrassen.served_of_general
#print axioms Verity.Accel.NaiveStrassen.servedSize_of_general
#print axioms Verity.Accel.NaiveStrassen.any_honest
#print axioms Verity.Accel.NaiveStrassen.any_honest_ref
#print axioms Verity.Accel.NaiveStrassen.anySize_honest
#print axioms Verity.Accel.NaiveStrassen.any_nontrivial
#print axioms Verity.Accel.NaiveStrassen.anySize_nontrivial
