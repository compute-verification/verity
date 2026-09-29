import PouwProofs.PearlC.Reduction
import Mathlib.Tactic.NormNum

/-! Pearl-C's headline accounting in exact rationals, and γ from `TT_OUT(1/400)` at 8192³ and 16384³. -/

namespace Pouw.PearlC

open Assumptions

theorem wref8192 : wrefOf Params.headline sh8192 = 530 / 529 * creditOf Params.headline sh8192 := by
  norm_num [wrefOf, creditOf, Params.headline, sh8192]

theorem wref16384 : wrefOf Params.headline sh16384 = 1042 / 1041 * creditOf Params.headline sh16384 := by
  norm_num [wrefOf, creditOf, Params.headline, sh16384]

/-- The reduction at one shape: a protocol with Pearl-C's accounting on the one-shape domain has
`W_ref ≤ Ω·credit` for `Ω` given by `hω`. -/
theorem wref_le_at {Q R S : Type} (P : Protocol Q R S) (sh : Shape) (ω : ℚ)
    (hω : wrefOf Params.headline sh = ω * creditOf Params.headline sh) (hacc : PearlCAccounting P Params.headline) :
    ∀ U : Workload, U.InDomain P (pearlCDomainAt Params.headline sh) →
      ∀ u < U.layout.N, P.Wref U.layout u ≤ (ω : ℝ) * P.credit U.layout u := by
  intro U hU u hu
  obtain ⟨hc, hw⟩ := hacc U.layout u
  rw [hc, hw, hU.2.1.2 u hu, hω]
  push_cast
  exact le_refl _

theorem pearlCGamma8192 : PearlCGamma8192 := by
  intro Q R S _ _ _ _ CM P ε hacc hTT
  have key := gammaFromTTOut CM P _ (1 / 400) ((530 / 529 : ℚ) : ℝ) ε (by norm_num) (by norm_num) (by norm_num)
    (wref_le_at P sh8192 _ wref8192 hacc) hTT
  have e : ((929 / 212000 : ℚ) : ℝ) = 1 - (1 - 1 / 400) / ((530 / 529 : ℚ) : ℝ) := by
    push_cast
    norm_num
  rw [e]
  exact key

theorem pearlCGamma16384 : PearlCGamma16384 := by
  intro Q R S _ _ _ _ CM P ε hacc hTT
  have key := gammaFromTTOut CM P _ (1 / 400) ((1042 / 1041 : ℚ) : ℝ) ε (by norm_num) (by norm_num) (by norm_num)
    (wref_le_at P sh16384 _ wref16384 hacc) hTT
  have e : ((1441 / 416800 : ℚ) : ℝ) = 1 - (1 - 1 / 400) / ((1042 / 1041 : ℚ) : ℝ) := by
    push_cast
    norm_num
  rw [e]
  exact key

theorem chainOnlyGamma8192 : ChainOnlyGamma8192 := by
  norm_num [ChainOnlyGamma8192, gammaOf, wrefOf, creditChainOnly, creditOf, Params.headline, sh8192]

end Pouw.PearlC
