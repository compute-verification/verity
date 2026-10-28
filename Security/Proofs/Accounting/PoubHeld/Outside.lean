import Definitions.Accounting.Assumptions.PoubOutside
import Definitions.Accounting.Assumptions.PoubSketch
import Proofs.Accounting.PoubHeld

/-!
# Statement C in full against the receipts' sketch bound and its derivation

With no device outside the node computing on a registered block during a credited pass (`NoOutsideCompute`), a block
on the node or computed on outside it at a tick of the pass was on the node (`present_of_outside`). So:

* `SketchBoundAt` and `NoOutsideCompute` give `SketchBound` (`sketchBound_of_noOutside`), and `SketchBound` gives
  `SketchBoundAt` at every record of outside computation (`sketchBoundAt_of_sketchBound`);
* the sketch model's `Present` is `PresentAt`, the same with outside computation allowed, under C over the model's
  passes (`present_of_noOutside`, `presentAt_of_present`);
* the provider's lockdown (`ProviderLockdown`) over an interval holding every credited pass gives `NoOutsideCompute`
  (`noOutsideCompute_of_lockdown`).
-/

namespace Accounting.SecurityProofs.Poub

open Accounting.Poub.Held Accounting.Poub.Held.Assumptions

section Trace

variable {Ω : Type} {T : Trace Ω} {O : Outside T}

/-- Under `NoOutsideCompute`, a block on the node or computed on outside it at a tick of a credited pass of an
accepting outcome was on the node then. -/
theorem present_of_outside (hO : NoOutsideCompute T O) {ω : Ω} (ha : T.accepts ω) {p t : ℕ} (hp : p < T.passes)
    (h1 : T.tick ω p ≤ t) (h2 : t < T.tick ω (p + 1)) {j : Fin T.blocks} (hj : j ∈ T.present ω t ∪ O.blocks ω t) :
    j ∈ T.present ω t := by
  rcases Finset.mem_union.1 hj with h | h
  · exact h
  · rw [(hO ω ha p hp t h1 h2).1] at h
    simp at h

/-- The provider's lockdown over `[applied, lifted)` grounds `NoOutsideCompute` when every credited pass of every
accepting outcome lies inside that interval. -/
theorem noOutsideCompute_of_lockdown {applied lifted : ℕ} (hL : ProviderLockdown T O applied lifted)
    (inside : ∀ ω, T.accepts ω → ∀ p < T.passes, applied ≤ T.tick ω p ∧ T.tick ω (p + 1) ≤ lifted) :
    NoOutsideCompute T O := fun ω ha p hp t h1 h2 =>
  hL ω ha t ((inside ω ha p hp).1.trans h1) (lt_of_lt_of_le h2 (inside ω ha p hp).2)

variable [Fintype Ω]

/-- `SketchBoundAt` and `NoOutsideCompute` give `SketchBound`. -/
theorem sketchBound_of_noOutside {δ : ℝ} (hk : SketchBoundAt T O δ) (hO : NoOutsideCompute T O) :
    SketchBound T δ :=
  holdsMono (fun ω ha h p hp j hj => by
    obtain ⟨t, h1, h2, ht⟩ := h p hp j hj
    exact ⟨t, h1, h2, present_of_outside hO ha hp h1 h2 ht⟩) hk

/-- `SketchBound` gives `SketchBoundAt` at every record of outside computation. -/
theorem sketchBoundAt_of_sketchBound {δ : ℝ} (hk : SketchBound T δ) (O : Outside T) : SketchBoundAt T O δ :=
  holdsMono (fun _ _ h p hp j hj => by
    obtain ⟨t, h1, h2, ht⟩ := h p hp j hj
    exact ⟨t, h1, h2, Finset.mem_union_left _ ht⟩) hk

end Trace

section Sketch

open Accounting.Poub.Sketch Accounting.Poub.Sketch.Assumptions

variable {β Hs : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

/-- `Present` with outside computation allowed: on every accepting outcome, a block the kept candidate's device of pass
`j` obtained whole was in node memory or on a chip, or computed on by a device outside the node, at some tick of the
pass. -/
def PresentAt (A : Session β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)) (O : Outside T) : Prop :=
  ∀ ω, T.accepts ω → ∀ j : Fin P, ∀ t : Fin N, A.Whole ω j (A.kept ω j) t →
    ∀ b : Fin T.blocks, (b : ℕ) = t →
      ∃ τ, T.tick ω (j : ℕ) ≤ τ ∧ τ < T.tick ω ((j : ℕ) + 1) ∧ b ∈ T.present ω τ ∪ O.blocks ω τ

variable {A : Session β Hs P N m h S L q ks G} {T : Trace (Outcome β Hs N m)}

/-- `Present` gives `PresentAt` at every record of outside computation. -/
theorem presentAt_of_present (hP : Present A T) (O : Outside T) : PresentAt A T O :=
  fun ω ha j t hw b hb => by
    obtain ⟨τ, h1, h2, h3⟩ := hP ω ha j t hw b hb
    exact ⟨τ, h1, h2, Finset.mem_union_left _ h3⟩

/-- `PresentAt` and `NoOutsideCompute` give `Present`, where the model's passes are among the trace's credited ones. -/
theorem present_of_noOutside {O : Outside T} (hle : P ≤ T.passes) (hO : NoOutsideCompute T O)
    (hP : PresentAt A T O) : Present A T :=
  fun ω ha j t hw b hb => by
    obtain ⟨τ, h1, h2, h3⟩ := hP ω ha j t hw b hb
    exact ⟨τ, h1, h2, present_of_outside hO ha (lt_of_lt_of_le j.2 hle) h1 h2 h3⟩

end Sketch

end Accounting.SecurityProofs.Poub
