import Models.Accounting.Guarantees.TS

/-!
# The tensor–scalar model's facet guarantees (stubs)

Lemma 1 and Corollaries D1 and D2 (EPS1-SPEC §1.1, §2.5), each a `sorry` until a prover takes it (gap list M1, M7–M9:
note:20261017T0801Z-draft-lean-stmt).
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
  sorry

/-- Corollary D2. -/
theorem RoutesOffD2 : Accounting.Guarantees.TS.RoutesOffD2 := by
  sorry

end Accounting.SecurityProofs.TS
