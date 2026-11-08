module

public import Definitions.Core.Prob
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
# Independent coins

`piLaw ν` is the law of independent draws, coordinate `i` drawn from `ν i`: the weight of `x` is `∏ i, ν i (x i)`.
`iid μ n` is `n` independent draws from `μ`, and `prodLaw ρ ν` two independent draws of different types. An event's
probability under any of them is `prPMFReal` (`Definitions.Core.Prob`).
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

/-- Independent draws, coordinate `i` from `ν i`. -/
noncomputable def piLaw {ι Ω : Type} [Fintype ι] [DecidableEq ι] [Fintype Ω] (ν : ι → PMF Ω) : PMF (ι → Ω) :=
  PMF.ofFintype (fun x => ∏ i, ν i (x i)) (by
    rw [← Fintype.prod_sum]
    exact Finset.prod_eq_one fun i _ =>
      (tsum_fintype (L := SummationFilter.unconditional Ω) (fun ω => ν i ω)).symm.trans (ν i).tsum_coe)

/-- `n` independent draws from `μ`. -/
noncomputable abbrev iid {Ω : Type} [Fintype Ω] (μ : PMF Ω) (n : ℕ) : PMF (Fin n → Ω) :=
  piLaw fun _ => μ

/-- Two independent draws, `w` from `ρ` and `x` from `ν`. -/
noncomputable def prodLaw {W X : Type} [Fintype W] [Fintype X] (ρ : PMF W) (ν : PMF X) : PMF (W × X) :=
  PMF.ofFintype (fun z => ρ z.1 * ν z.2) (by
    have hρ : ∑ w, ρ w = 1 :=
      (tsum_fintype (L := SummationFilter.unconditional W) (fun w => ρ w)).symm.trans ρ.tsum_coe
    have hν : ∑ x, ν x = 1 :=
      (tsum_fintype (L := SummationFilter.unconditional X) (fun x => ν x)).symm.trans ν.tsum_coe
    rw [Fintype.sum_prod_type]
    simp only [← Finset.mul_sum, hν, mul_one, hρ])

end Verity.Game
