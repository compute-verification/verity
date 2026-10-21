import Proofs.Nci.Cover.Paths

/-!
# The cover theorem: what a block pays for the products it finishes

`whole_use_entry`: a block that computes whole a counted product of use `j` is handed a gate of `A j`. The product's
gate `g` reads the certificate's activation `a ∈ A j`, so `a ∈ I`; were `I` handed no gate of `A j`, it would compute
every gate of `j`'s upstream products (`BackwardClosure`), and `UpdateHardness` would charge it `b D > k` bits.
`ActivationCharge` counts these gates, distinct across uses since the regions are disjoint and none is in `θ`.

`finished_sdiff_whole_card_le`: each counted product whose output gate `I` computes but not all of whose gates has a
gate `x ∉ computed I`, which reaches the output inside the product (`reachesIn_output`), so the path meets an input of
`I` among the product's gates (`Entry`): a non-input gate, so not in `θ`, and distinct across products since they are
disjoint.
-/

namespace Nci.Cover.SecurityProofs

/-- A block that computes whole a counted product of use `j` is handed a gate of `j`'s activation region. -/
theorem whole_use_entry {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D : ℕ} {b : ℝ} (hb : 0 < b) (hH : Assumptions.UpdateHardness M s b)
    (hU : M.Upstream A D) (hk : (k : ℝ) < b * D) {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) :
    ∀ j ∈ (M.whole I).image Prod.fst, (C.inputs I ∩ A j).Nonempty := by
  sorry

theorem ActivationCharge : Guarantees.ActivationCharge := by
  sorry

/-- The counted products a subcircuit finishes without computing them whole are at most its inputs outside `θ`. -/
theorem finished_sdiff_whole_card_le {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    {I : Finset (Fin C.N)} (hI : C.Subcircuit I) :
    (M.finished I \ M.whole I).card ≤ (C.inputs I \ C.θ).card := by
  sorry

end Nci.Cover.SecurityProofs
