import Definitions.Nci.Capability
import Proofs.Core.Formalism

/-!
# Local bounds: what a block holds

* `holds_mono`: more gates hold more;
* `exists_selection`: the items a family of products reads, each among what its own gates hold, have a selection held by
  any gates containing those products;
* `Committed.recompute_congr`: a proof block's recomputation, on what it holds, reads only `X`'s values on its
  unregistered inputs;
* `Circuit.bits_congr`: two runs agreeing on a bundle's gates give it the same bits.
-/

namespace Nci.Capability

open Verity Finset

theorem holds_mono {C : Circuit} {S T : Finset (Fin C.N)} (h : S ⊆ T) : holds C S ⊆ holds C T := by
  intro a ha
  simp only [holds, Circuit.into, mem_union, mem_filter, mem_univ, true_and] at ha ⊢
  rcases ha with ha | ⟨-, g, hg, hag⟩
  · exact Or.inl (h ha)
  · by_cases haT : a ∈ T
    · exact Or.inl haT
    · exact Or.inr ⟨haT, g, h hg, hag⟩

theorem exists_selection {C : Circuit} {ι α : Type} [DecidableEq ι] (X : Items C ι) (Q : Finset α) (f : α → ι)
    (m : α → Finset (Fin C.N)) (S : Finset (Fin C.N)) (hread : ∀ q ∈ Q, ∃ c ∈ X.cells (f q), c ⊆ holds C (m q))
    (hm : ∀ q ∈ Q, m q ⊆ S) :
    ∃ σ : ι → Finset (Fin C.N), Selects X (Q.image f) σ ∧ ∀ i ∈ Q.image f, σ i ⊆ holds C S := by
  have h : ∀ i ∈ Q.image f, ∃ c ∈ X.cells i, c ⊆ holds C S := by
    intro i hi
    obtain ⟨q, hq, rfl⟩ := mem_image.1 hi
    obtain ⟨c, hc, hcs⟩ := hread q hq
    exact ⟨c, hc, hcs.trans (holds_mono (hm q hq))⟩
  choose! σ hσ using h
  exact ⟨σ, fun i hi => (hσ i hi).1, fun i hi => (hσ i hi).2⟩

end Nci.Capability

namespace Verity

open Finset

theorem Circuit.bits_congr (C : Circuit) {X Y : C.Transcript} {l : List (Fin C.N)} (h : ∀ g ∈ l, X g = Y g) :
    C.bits X l = C.bits Y l := by
  induction l with
  | nil => rfl
  | cons g l ih =>
    simp only [Circuit.bits, List.flatMap_cons] at ih ⊢
    rw [h g List.mem_cons_self, ih fun x hx => h x (List.mem_cons_of_mem g hx)]

theorem Committed.recompute_congr (R : Committed) {X Y : R.d.D.Transcript} {u : Fin R.B}
    (h : ∀ g ∈ R.d.unregistered (R.P.gates u), X g = Y g) :
    ∀ g ∈ Nci.Capability.holds R.d.D (R.P.gates u), R.recompute X u g = R.recompute Y u g := by
  intro g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  intro hg
  by_cases hθ : g ∈ R.d.θ
  · rw [recompute, R.d.D.run_fixed (b := R.val g) (by simp [hθ]), recompute,
      R.d.D.run_fixed (b := R.val g) (by simp [hθ])]
  · by_cases hu : R.P.unit g = some u
    · rw [recompute, R.d.D.run_free (by simp [hθ, hu]), recompute, R.d.D.run_free (by simp [hθ, hu])]
      refine R.d.D.op_congr g _ _ fun x hx => ih x (R.d.D.reads_lt g x hx) ?_
      by_cases hxu : x ∈ R.P.gates u
      · exact mem_union_left _ hxu
      · exact mem_union_right _ (mem_filter.2 ⟨mem_univ _, hxu, g, R.P.mem_gates.2 hu, hx⟩)
    · have hgI : g ∈ R.d.D.into (R.P.gates u) := by
        rcases mem_union.1 hg with hg | hg
        · exact absurd (R.P.mem_gates.1 hg) hu
        · exact hg
      rw [recompute, R.d.D.run_fixed (b := X g) (by simp [hθ, hu]), recompute,
        R.d.D.run_fixed (b := Y g) (by simp [hθ, hu])]
      exact h g (mem_sdiff.2 ⟨hgI, hθ⟩)

end Verity
