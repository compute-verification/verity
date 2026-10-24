import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Proofs.Nci.Cover.Paths

/-!
# The cover theorem: what a block pays for the products it finishes

`whole_use_entry`: a block that computes whole a counted product of use `j` is handed a gate of `A j`. The product's
gate `g` reads the certificate's activation `a ∈ A j`, so `a ∈ I`; were `I` handed no gate of `A j`, it would compute
the output gate of each of `j`'s upstream products (`backwardClosure`), and so each of their gates, which reaches the
output inside the product and the region (`reachesIn_output`, `entry`), and `UpdateHardness` would charge it `b D > k`
bits.
`activationCharge` counts these gates, distinct across uses since the regions are disjoint and none is in `θ`.

`finished_sdiff_whole_card_le`: each counted product whose output gate `I` computes but not all of whose gates has a
gate `x ∉ computed I`, which reaches the output inside the product (`reachesIn_output`), so the path meets an input of
`I` among the product's gates (`entry`): a non-input gate, so not in `θ`, and distinct across products since they are
disjoint.
-/

namespace Nci.Cover.SecurityProofs

/-- A block that computes whole a counted product of use `j` is handed a gate of `j`'s activation region. -/
theorem whole_use_entry {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D : ℕ} {b : ℝ} (hb : 0 < b) (hH : Assumptions.UpdateHardness M s b)
    (hU : M.Upstream A D) (hk : (k : ℝ) < b * D) {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) :
    ∀ j ∈ (M.whole I).image Prod.fst, (C.inputs I ∩ A j).Nonempty := by
  intro j hj
  obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hj
  obtain ⟨hpP, hpI⟩ := Finset.mem_filter.1 hp
  obtain ⟨U, hUP, hUA, hDU, hcert⟩ := hU.upstream p.1 (Finset.mem_image_of_mem Prod.fst hpP)
  obtain ⟨g, hg, a, ha, hreach⟩ := hcert p hpP rfl
  obtain ⟨haPred, -⟩ := Finset.mem_inter.1 ha
  by_contra hne
  rw [Finset.not_nonempty_iff_eq_empty] at hne
  have haI : a ∈ I := pred_subset_of_mem_computed (hpI hg) haPred
  have hUc : ∀ q ∈ U, M.m q ⊆ C.computed I := by
    intro q hq x hx
    have hoI := backwardClosure C I (A p.1) a (M.o q) haI hne (hreach q hq)
    obtain ⟨π, hπ, hπm⟩ := reachesIn_output C (M.m q) (M.o q) (M.m_outputs q (hUP hq)) x hx
    by_contra hxc
    obtain ⟨g', hg'π, hg'I⟩ := entry C I π x (M.o q) hπ (computed_subset C I hoI) hxc
    have hg' : g' ∈ C.inputs I ∩ A p.1 := Finset.mem_inter.2 ⟨hg'I, hUA q hq (hπm g' hg'π)⟩
    rw [hne] at hg'
    exact Finset.notMem_empty g' hg'
  have hpay := hH I hI.1 U hUP hUc
  have hcap : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  have hD : (D : ℝ) ≤ (U.image Prod.snd).card := by exact_mod_cast hDU
  have := mul_le_mul_of_nonneg_left hD hb.le
  linarith

/-- **Activation charge**: under `UpdateHardness` and the upstream certificate at depth `D` with `k < b D`, an inference
block computes whole counted products of at most `#(inputs I \ θ)` uses. -/
theorem activationCharge {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D : ℕ} {b : ℝ} (hb : 0 < b) (hH : Assumptions.UpdateHardness M s b)
    (hU : M.Upstream A D) (hk : (k : ℝ) < b * D) {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) :
    ((M.whole I).image Prod.fst).card ≤ (C.inputs I \ C.θ).card := by
  have hWuses : (M.whole I).image Prod.fst ⊆ M.uses := by
    intro j hj
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hj
    exact Finset.mem_image_of_mem Prod.fst (Finset.mem_filter.1 hp).1
  have hdisj : (((M.whole I).image Prod.fst : Finset J) : Set J).PairwiseDisjoint fun j => C.inputs I ∩ A j :=
    (hU.disjoint.subset (Finset.coe_subset.2 hWuses)).mono fun j => Finset.inter_subset_right
  refine (Finset.card_le_card_biUnion hdisj (whole_use_entry M A hb hH hU hk hI)).trans (Finset.card_le_card ?_)
  intro x hx
  obtain ⟨j, hj, hx⟩ := Finset.mem_biUnion.1 hx
  obtain ⟨hxi, hxA⟩ := Finset.mem_inter.1 hx
  exact Finset.mem_sdiff.2 ⟨hxi, Finset.disjoint_left.1 (hU.disjoint_θ j (hWuses hj)) hxA⟩

/-- The counted products a set of gates finishes without computing them whole are at most its inputs outside `θ`. -/
theorem finished_sdiff_whole_card_le {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    {I : Finset (Fin C.N)} :
    (M.finished I \ M.whole I).card ≤ (C.inputs I \ C.θ).card := by
  have hPall : ∀ p ∈ M.finished I \ M.whole I, p ∈ M.Pall := fun p hp =>
    M.P_subset (Finset.mem_filter.1 (Finset.mem_sdiff.1 hp).1).1
  have hT : ∀ p ∈ M.finished I \ M.whole I, (M.m p ∩ C.inputs I).Nonempty := by
    intro p hp
    obtain ⟨hpf, hpw⟩ := Finset.mem_sdiff.1 hp
    obtain ⟨hpP, hoI⟩ := Finset.mem_filter.1 hpf
    obtain ⟨x, hxm, hxc⟩ := Finset.not_subset.1 fun h => hpw (Finset.mem_filter.2 ⟨hpP, h⟩)
    obtain ⟨π, hπ, hπm⟩ := reachesIn_output C (M.m p) (M.o p) (M.m_outputs p (hPall p hp)) x hxm
    obtain ⟨g, hgπ, hgI⟩ := entry C I π x (M.o p) hπ (computed_subset C I hoI) hxc
    exact ⟨g, Finset.mem_inter.2 ⟨hπm g hgπ, hgI⟩⟩
  have hdisj : ((M.finished I \ M.whole I : Finset (J × L)) : Set (J × L)).PairwiseDisjoint
      fun p => M.m p ∩ C.inputs I :=
    (M.m_disjoint.subset fun p hp => hPall p hp).mono fun p => Finset.inter_subset_left
  refine (Finset.card_le_card_biUnion hdisj hT).trans (Finset.card_le_card ?_)
  intro x hx
  obtain ⟨p, hp, hx⟩ := Finset.mem_biUnion.1 hx
  obtain ⟨hxm, hxi⟩ := Finset.mem_inter.1 hx
  refine Finset.mem_sdiff.2 ⟨hxi, fun hxθ => ?_⟩
  obtain ⟨y, hy⟩ := M.m_computed p (hPall p hp) x hxm
  rw [C.θ_input x hxθ] at hy
  exact Finset.notMem_empty y hy

end Nci.Cover.SecurityProofs
