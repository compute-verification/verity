module

public import Definitions.Core.Coins
public import Proofs.Core.Prob
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Basic.ENNReal.BigOperators

/-! The weights of independent coins. -/

@[expose] public section

namespace Verity.Game

variable {Ω : Type} [Fintype Ω]

theorem piDistribution_apply {ι : Type} [Fintype ι] [DecidableEq ι] (ν : ι → PMF Ω) (x : ι → Ω) :
    piDistribution ν x = ∏ i, ν i (x i) := rfl

theorem piDistribution_toReal {ι : Type} [Fintype ι] [DecidableEq ι] (ν : ι → PMF Ω) (x : ι → Ω) :
    (piDistribution ν x).toReal = ∏ i, (ν i (x i)).toReal := by
  rw [piDistribution_apply, ENNReal.toReal_prod]

theorem iid_toReal (μ : PMF Ω) {n : ℕ} (cs : Fin n → Ω) : (iid μ n cs).toReal = ∏ i, (μ (cs i)).toReal :=
  piDistribution_toReal _ cs

theorem iid_cons_toReal (μ : PMF Ω) {n : ℕ} (c : Ω) (cs : Fin n → Ω) :
    (iid μ (n + 1) (Fin.cons c cs)).toReal = (μ c).toReal * (iid μ n cs).toReal := by
  rw [iid_toReal, iid_toReal, Fin.prod_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]

theorem prodDistribution_toReal {W X : Type} [Fintype W] [Fintype X] (ρ : PMF W) (ν : PMF X) (w : W) (x : X) :
    (prodDistribution ρ ν (w, x)).toReal = (ρ w).toReal * (ν x).toReal :=
  ENNReal.toReal_mul

end Verity.Game
