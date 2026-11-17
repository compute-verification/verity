module

public import Definitions.Core.Prob
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-! Independent coins: the product distributions `piDistribution`, `iid` and `prodDistribution`. -/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

noncomputable def piDistribution {ι Ω : Type} [Fintype ι] [DecidableEq ι] [Fintype Ω] (ν : ι → PMF Ω) : PMF (ι → Ω) :=
  PMF.ofFintype (fun x => ∏ i, ν i (x i)) (by
    rw [← Fintype.prod_sum]
    exact Finset.prod_eq_one fun i _ =>
      (tsum_fintype (L := SummationFilter.unconditional Ω) (fun ω => ν i ω)).symm.trans (ν i).tsum_coe)

noncomputable abbrev iid {Ω : Type} [Fintype Ω] (μ : PMF Ω) (n : ℕ) : PMF (Fin n → Ω) :=
  piDistribution fun _ => μ

noncomputable def prodDistribution {W X : Type} [Fintype W] [Fintype X] (ρ : PMF W) (ν : PMF X) : PMF (W × X) :=
  PMF.ofFintype (fun z => ρ z.1 * ν z.2) (by
    have hρ : ∑ w, ρ w = 1 :=
      (tsum_fintype (L := SummationFilter.unconditional W) (fun w => ρ w)).symm.trans ρ.tsum_coe
    have hν : ∑ x, ν x = 1 :=
      (tsum_fintype (L := SummationFilter.unconditional X) (fun x => ν x)).symm.trans ν.tsum_coe
    rw [Fintype.sum_prod_type]
    simp only [← Finset.mul_sum, hν, mul_one, hρ])

end Verity.Game
