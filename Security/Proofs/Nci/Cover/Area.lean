import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.GroupWithZero.Basic
import Mathlib.Data.Finset.Prod
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Proofs.Nci.Cover.Charge

/-!
# The cover theorem: a block's products, by area

`intensity`: the counted products a block computes whole lie in `J × L`, with `#J ≤ k` (`activationCharge`) and
`b #L ≤ k` (`UpdateHardness` on them), so `b #whole ≤ k²`; the rest are at most `k` (`finished_sdiff_whole_card_le`).
It is this vocabulary's form of the per-block step of the covering bound at `c_w = b`, `c_a = 1`
(`Nci.Guarantees.CoveringBoundTwoCosts`), with the upstream certificate in place of activation-chaining, and the bridge
to a compliant declaration (`CoverTraining`) sums it. `cover_sum` adds a per-block
bound over a cover: every counted product's output gate is a non-input gate, so some block finishes it.
-/

namespace Nci.Cover.SecurityProofs

/-- A product's output gate is one of its gates. -/
theorem o_mem {C : Circuit} {J L : Type} (M : MulBlocks C J L) {p : J × L} (hp : p ∈ M.Pall) : M.o p ∈ M.m p := by
  have ho : M.o p ∈ C.outputs (M.m p) := by
    rw [M.m_outputs p hp]
    exact Finset.mem_singleton_self _
  exact (mem_outputs.1 ho).1

/-- A block that computes a counted product whole finishes it. -/
theorem whole_subset_finished {C : Circuit} {J L : Type} [DecidableEq J] (M : MulBlocks C J L)
    (I : Finset (Fin C.N)) : M.whole I ⊆ M.finished I := by
  intro p hp
  obtain ⟨hpP, hpI⟩ := Finset.mem_filter.1 hp
  exact Finset.mem_filter.2 ⟨hpP, hpI (o_mem M (M.P_subset hpP))⟩

/-- **Intensity**: under `UpdateHardness` and the upstream certificate with `k < b D`, an inference block finishes at
most `k · k / b + k` counted products: `b · #(finished) ≤ k² + b k`. -/
theorem intensity {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D : ℕ} {b : ℝ} (hb : 0 < b) (hH : Assumptions.UpdateHardness M s b)
    (hU : M.Upstream A D) (hk : (k : ℝ) < b * D) {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) :
    b * (M.finished I).card ≤ k * k + b * k := by
  have hsplit := Finset.card_sdiff_add_card_eq_card (whole_subset_finished M I)
  have hJ := (activationCharge M A hb hH hU hk hI).trans hI.2
  have hL := hH I hI.1 (M.whole I) (fun p hp => M.P_subset (Finset.mem_filter.1 hp).1)
    (fun p hp => (Finset.mem_filter.1 hp).2)
  have hprod : (M.whole I).card ≤ ((M.whole I).image Prod.fst).card * ((M.whole I).image Prod.snd).card := by
    rw [← Finset.card_product]
    exact Finset.card_le_card fun p hp =>
      Finset.mem_product.2 ⟨Finset.mem_image_of_mem _ hp, Finset.mem_image_of_mem _ hp⟩
  have hrest := (finished_sdiff_whole_card_le M (I := I)).trans hI.2
  have hcap : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  have e1 : ((M.finished I).card : ℝ) = (M.finished I \ M.whole I).card + (M.whole I).card := by
    exact_mod_cast hsplit.symm
  have e2 : ((M.whole I).card : ℝ) ≤ ((M.whole I).image Prod.fst).card * ((M.whole I).image Prod.snd).card := by
    exact_mod_cast hprod
  have e3 : (((M.whole I).image Prod.fst).card : ℝ) ≤ k := by exact_mod_cast hJ
  have e4 : ((M.finished I \ M.whole I).card : ℝ) ≤ k := by exact_mod_cast hrest
  have hwhole : b * (M.whole I).card ≤ k * k :=
    calc b * (M.whole I).card
        ≤ b * (((M.whole I).image Prod.fst).card * ((M.whole I).image Prod.snd).card) :=
          mul_le_mul_of_nonneg_left e2 hb.le
      _ = ((M.whole I).image Prod.fst).card * (b * ((M.whole I).image Prod.snd).card) := by ring
      _ ≤ k * k := mul_le_mul e3 (hL.trans hcap) (by positivity) (Nat.cast_nonneg _)
  have hpart := mul_le_mul_of_nonneg_left e4 hb.le
  rw [e1, mul_add]
  linarith

/-- A bound `c · #(finished I) ≤ X` on every block of a cover bounds the counted products: `c · #P ≤ #𝓘 · X`. -/
theorem cover_sum {C : Circuit} {J L : Type} (M : MulBlocks C J L)
    {𝓘 : Finset (Finset (Fin C.N))} (h𝓘 : C.Covers 𝓘) {c X : ℝ} (hc : 0 ≤ c)
    (hX : ∀ I ∈ 𝓘, c * (M.finished I).card ≤ X) : c * M.P.card ≤ 𝓘.card * X := by
  classical
  have hsub : M.P ⊆ 𝓘.biUnion fun I => M.finished I := by
    intro p hp
    have hpAll := M.P_subset hp
    obtain ⟨I, hI, hoI⟩ := h𝓘 (M.o p) (M.m_computed p hpAll _ (o_mem M hpAll))
    exact Finset.mem_biUnion.2 ⟨I, hI, Finset.mem_filter.2 ⟨hp, hoI⟩⟩
  have hcard : (M.P.card : ℝ) ≤ ∑ I ∈ 𝓘, ((M.finished I).card : ℝ) := by
    exact_mod_cast (Finset.card_le_card hsub).trans Finset.card_biUnion_le
  calc c * M.P.card ≤ c * ∑ I ∈ 𝓘, ((M.finished I).card : ℝ) := mul_le_mul_of_nonneg_left hcard hc
    _ = ∑ I ∈ 𝓘, c * ((M.finished I).card : ℝ) := Finset.mul_sum _ _ _
    _ ≤ ∑ _I ∈ 𝓘, X := Finset.sum_le_sum hX
    _ = 𝓘.card * X := by rw [Finset.sum_const, nsmul_eq_mul]

end Nci.Cover.SecurityProofs
