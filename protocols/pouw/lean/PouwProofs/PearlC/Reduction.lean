import Pouw.PearlC.Statements
import PouwProofs.PearlC.Prob
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-! The reduction from `TT_OUT(γ₀)` to the γ-gap property: the winning event implies TT_OUT's event pointwise. -/

namespace Pouw.PearlC

open Finset Assumptions

theorem gammaFromTTOut : GammaFromTTOut := by
  intro Q R S _ _ _ _ CM P D γ₀ Ω ε hγ0 hγ1 hΩ hW hTT U hU T hT q hq A hA
  classical
  refine le_trans (pr_mono ?_) (hTT.2 U hU T hT q hq A hA)
  intro x hwin
  unfold Wins at hwin
  have hKsub : correctSet P U x.1 x.2 (A.transcript U x.1 x.2) ⊆ range U.layout.N := by
    unfold correctSet
    exact filter_subset _ _
  have hsum : ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), P.Wref U.layout u ≤
      Ω * ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), P.credit U.layout u := by
    rw [mul_sum]
    apply sum_le_sum
    intro u hu
    exact hW U hU u (mem_range.mp (hKsub hu))
  have hden : 1 - (1 - (1 - γ₀) / Ω) = (1 - γ₀) / Ω := by ring
  rw [hden, div_div_eq_mul_div] at hwin
  have hscale : T * Ω / (1 - γ₀) = Ω * (T / (1 - γ₀)) := by ring
  rw [hscale] at hwin
  exact lt_of_mul_lt_mul_left (lt_of_lt_of_le hwin hsum) hΩ.le

end Pouw.PearlC
