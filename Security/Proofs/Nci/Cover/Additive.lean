import Proofs.Nci.Cover.Area
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Finset.Prod
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The cover theorem: the additive bound

Under `ConditionalUpdateHardness` a block's whole products pay for their uses inside the activation regions
(`whole_use_entry`, one gate of `A j` per use) and for their updated parameters outside them, so `#J + b #L ≤ k`, and
`4 b #J #L ≤ (#J + b #L)² ≤ k²`. The rest are at most `k`, as in the area bound.

`whole_uses_params_le` is the split, for uses that cost `c` gates of their region each, and `finished_le_of_charge` the
rest of the count; the width bound (`Width`) reuses both with `c = f`.
-/

namespace Nci.Cover.SecurityProofs

/-- `ConditionalUpdateHardness` implies `UpdateHardness`. -/
theorem updateHardness_of_conditional {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s : ℕ} {b : ℝ} (h : Assumptions.ConditionalUpdateHardness M A s b) :
    Assumptions.UpdateHardness M s b := by
  intro S hs B hB hBS
  refine (h S hs B hB hBS).trans ?_
  exact_mod_cast Finset.card_le_card (Finset.sdiff_subset_sdiff le_rfl Finset.subset_union_left)

/-- **The split of a block's inputs**: under `ConditionalUpdateHardness` and the certificate's disjoint regions, if each
use of a block's whole products is handed at least `c` gates of its region, the uses and the updated parameters of
those products together cost at most the block's inputs outside `θ`: `c #J + b #L ≤ #(inputs I \ θ)`. The uses are
paid inside the regions, disjoint and outside `θ`, and the parameters outside them. -/
theorem whole_uses_params_le {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D : ℕ} {b c : ℝ} (hCH : Assumptions.ConditionalUpdateHardness M A s b)
    (hU : M.Upstream A D) {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I)
    (hc : ∀ j ∈ (M.whole I).image Prod.fst, c ≤ (C.inputs I ∩ A j).card) :
    c * ((M.whole I).image Prod.fst).card + b * ((M.whole I).image Prod.snd).card ≤ (C.inputs I \ C.θ).card := by
  have hWP : M.whole I ⊆ M.P := Finset.filter_subset _ _
  have hJ : (M.whole I).image Prod.fst ⊆ M.uses := Finset.image_subset_image hWP
  have hL : b * ((M.whole I).image Prod.snd).card ≤ (C.inputs I \ (C.θ ∪ M.uses.biUnion A)).card :=
    hCH I hI.1.le (M.whole I) (hWP.trans M.P_subset) fun p hp => (Finset.mem_filter.1 hp).2
  have hT : ((M.whole I).image Prod.fst).biUnion (fun j => C.inputs I ∩ A j) ⊆
      (C.inputs I \ C.θ) ∩ M.uses.biUnion A := by
    intro x hx
    obtain ⟨j, hj, hx⟩ := Finset.mem_biUnion.1 hx
    obtain ⟨hxI, hxA⟩ := Finset.mem_inter.1 hx
    refine Finset.mem_inter.2 ⟨Finset.mem_sdiff.2 ⟨hxI, fun hθ => ?_⟩, Finset.mem_biUnion.2 ⟨j, hJ hj, hxA⟩⟩
    exact Finset.disjoint_left.1 (hU.disjoint_θ j (hJ hj)) hxA hθ
  have hTcard : (((M.whole I).image Prod.fst).biUnion (fun j => C.inputs I ∩ A j)).card =
      ∑ j ∈ (M.whole I).image Prod.fst, (C.inputs I ∩ A j).card := by
    refine Finset.card_biUnion fun j hj j' hj' hne => ?_
    exact (hU.disjoint (hJ hj) (hJ hj') hne).mono Finset.inter_subset_right Finset.inter_subset_right
  have hcJ : c * ((M.whole I).image Prod.fst).card ≤ ((C.inputs I \ C.θ) ∩ M.uses.biUnion A).card := by
    calc c * ((M.whole I).image Prod.fst).card
        = ∑ _j ∈ (M.whole I).image Prod.fst, c := by rw [Finset.sum_const, nsmul_eq_mul, mul_comm]
      _ ≤ ∑ j ∈ (M.whole I).image Prod.fst, ((C.inputs I ∩ A j).card : ℝ) := Finset.sum_le_sum hc
      _ = (((M.whole I).image Prod.fst).biUnion (fun j => C.inputs I ∩ A j)).card := by
        rw [hTcard, Nat.cast_sum]
      _ ≤ ((C.inputs I \ C.θ) ∩ M.uses.biUnion A).card := by exact_mod_cast Finset.card_le_card hT
  have hsplit : C.inputs I \ (C.θ ∪ M.uses.biUnion A) = (C.inputs I \ C.θ) \ M.uses.biUnion A := by
    ext x
    simp only [Finset.mem_sdiff, Finset.mem_union, not_or]
    tauto
  have hsum : (((C.inputs I \ C.θ) ∩ M.uses.biUnion A).card : ℝ) +
      ((C.inputs I \ (C.θ ∪ M.uses.biUnion A)).card : ℝ) = (C.inputs I \ C.θ).card := by
    rw [hsplit]
    exact_mod_cast Finset.card_inter_add_card_sdiff _ _
  linarith

/-- **AM–GM on the whole products**: `x + y ≤ k` with `x, y ≥ 0` gives `4 x y ≤ k²`. -/
theorem four_mul_le_sq {x y k : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x + y ≤ k) : 4 * x * y ≤ k * k := by
  nlinarith [sq_nonneg (x - y)]

/-- **Intensity from a per-use charge**: under `ConditionalUpdateHardness` and the certificate, if each use of a block's
whole products is handed at least `c ≥ 0` gates of its region, the block finishes at most `k² / (4 b c) + k` counted
products: `4 b c · #(finished I) ≤ k² + 4 b c k`. The whole ones lie in `J × L` with `c #J + b #L ≤ k`
(`whole_uses_params_le`), so `4 b c #J #L ≤ k²`; the rest are at most `k` (`finished_sdiff_whole_card_le`). -/
theorem finished_le_of_charge {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D : ℕ} {b c : ℝ} (hb : 0 < b) (hc0 : 0 ≤ c)
    (hCH : Assumptions.ConditionalUpdateHardness M A s b) (hU : M.Upstream A D) {I : Finset (Fin C.N)}
    (hI : C.InferenceBlock s k I) (hc : ∀ j ∈ (M.whole I).image Prod.fst, c ≤ (C.inputs I ∩ A j).card) :
    4 * b * c * (M.finished I).card ≤ k * k + 4 * b * c * k := by
  have hK : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  have hsplit := whole_uses_params_le M A hCH hU hI hc
  have hJL : 4 * (c * ((M.whole I).image Prod.fst).card) * (b * ((M.whole I).image Prod.snd).card) ≤ k * k :=
    four_mul_le_sq (by positivity) (by positivity) (hsplit.trans hK)
  have hW : ((M.whole I).card : ℝ) ≤ ((M.whole I).image Prod.fst).card * ((M.whole I).image Prod.snd).card := by
    exact_mod_cast (Finset.card_le_card Finset.subset_product).trans (Finset.card_product _ _).le
  have hrest : ((M.finished I \ M.whole I).card : ℝ) ≤ k :=
    (by exact_mod_cast finished_sdiff_whole_card_le M (I := I) : ((M.finished I \ M.whole I).card : ℝ) ≤ _).trans hK
  have hfin : ((M.finished I).card : ℝ) ≤ (M.finished I \ M.whole I).card + (M.whole I).card := by
    exact_mod_cast Finset.card_le_card_sdiff_add_card
  have hbc : 0 ≤ 4 * b * c := by positivity
  have h1 := mul_le_mul_of_nonneg_left hfin hbc
  have h2 := mul_le_mul_of_nonneg_left hrest hbc
  have h3 := mul_le_mul_of_nonneg_left hW hbc
  have h4 : 4 * b * c * (((M.whole I).image Prod.fst).card * ((M.whole I).image Prod.snd).card : ℝ) =
      4 * (c * ((M.whole I).image Prod.fst).card) * (b * ((M.whole I).image Prod.snd).card) := by ring
  linarith

theorem AdditiveIntensity : Guarantees.AdditiveIntensity := by
  intro C J L _ _ M A s k D b hb hCH hU hk I hI
  have h := finished_le_of_charge M A (c := 1) hb zero_le_one hCH hU hI fun j hj => by
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hj
    have hne := whole_use_entry M A hb (updateHardness_of_conditional M A hCH) hU hk hI p.1
      (Finset.mem_image_of_mem _ hp)
    exact_mod_cast hne.card_pos
  simpa only [mul_one] using h

theorem AdditiveCoverBound : Guarantees.AdditiveCoverBound := by
  intro C J L _ _ M A s k D b 𝓘 hb hCH hU hk h𝓘 hcov
  exact cover_sum M hcov (by positivity) fun I hI => AdditiveIntensity C M A s k D b hb hCH hU hk I (h𝓘 I hI)

end Nci.Cover.SecurityProofs
