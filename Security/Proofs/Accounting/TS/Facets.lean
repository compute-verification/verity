import Definitions.Accounting.Guarantees.TS

/-!
# The tensor–scalar model's facet guarantees

Lemma 1 and Corollaries D1 and D2 (EPS1-SPEC §1.1, §2.5).
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS

/-- Lemma 1. -/
theorem TimeBound : Accounting.Guarantees.TS.TimeBound := by
  intro K R _ _ _ G P h
  obtain ⟨-, hcyc, hclk, hfmax, hr⟩ := h
  rw [div_le_iff₀ (hclk.trans_le hfmax), Machine.bound]
  refine ciSup_le fun r => ?_
  cases hs : G.smClocked r
  · linarith [(hr r).2 hs]
  · calc G.load P.n r ≤ P.cycles := (hr r).1 hs
      _ ≤ P.time * G.fmax := by
        rw [Run.time, div_mul_eq_mul_div, le_div_iff₀ hclk]
        exact mul_le_mul_of_nonneg_left hfmax hcyc

/-- The certificate on one facet prices counts at that facet's load. -/
theorem price_onFacet {K R : Type} [Fintype K] [Fintype R] [DecidableEq R] (G : Machine K R) (r : R)
    (n : K → ℝ) : G.price (onFacet r) n = G.load n r := by
  simp [Machine.price, onFacet]

/-- Corollary D1, one binding facet. -/
theorem BindingFacetD1 : Accounting.Guarantees.TS.BindingFacetD1 := by
  intro K R T L U F X _ _ _ _ _ _ _ _ Z H r hload hpos hw hphi hchi
  unfold Model.epsMachine Model.epsCert Model.epsRoute
  rw [hw, hphi, hchi, price_onFacet, hload, div_self hpos.ne']
  simp

/-- Corollary D1, the vector route's threshold. -/
theorem VectorThresholdD1 : Accounting.Guarantees.TS.VectorThresholdD1 := by
  intro K R T L U F X _ _ _ _ _ _ _ Z t phiDel r hβ hω
  simp only [Model.tensorDemandDel, Model.demand]
  rw [div_le_iff₀ hω, div_le_iff₀ (mul_pos hβ hω)]
  constructor <;> intro h <;> linarith

/-- Corollary D2. -/
theorem RoutesOffD2 : Accounting.Guarantees.TS.RoutesOffD2 := by
  intro K R ι _ _ _ G n rs cap N s κ B₀ _ hs hκ hB₀ hN
  have hle : ∀ r, G.load n r ≤ G.bound n := fun r => le_ciSup (Set.finite_range _).bddAbove r
  have hD : 0 < 1 + ∑ u, κ u * s u := by
    linarith [Finset.sum_nonneg fun u (_ : u ∈ Finset.univ) => mul_nonneg (hκ u) (hs u)]
  have hsum : ∑ u, N u * s u ≤ (∑ u, κ u * s u) * G.bound n := by
    rw [Finset.sum_mul]
    refine Finset.sum_le_sum fun u _ => ?_
    have hNu : N u ≤ κ u * G.bound n := (hN u).trans (mul_le_mul_of_nonneg_left (hle (cap u)) (hκ u))
    nlinarith [hs u]
  rw [div_le_iff₀ hD]
  linarith [hle rs]

end Accounting.SecurityProofs.TS
