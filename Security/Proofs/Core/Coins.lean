module

public import Definitions.Core.Coins
public import Proofs.Core.Prob
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Basic.ENNReal.BigOperators

/-!
# Independent coins: their weights

`piLaw ν` weighs `x` by `∏ i, ν i (x i)` (`piLaw_apply`, `piLaw_toReal`), so `iid μ n` weighs a path of coins by the
product of theirs (`iid_toReal`), and the first coin splits off (`iid_cons_toReal`). `prodLaw ρ ν` weighs `(w, x)` by
`ρ w · ν x` (`prodLaw_toReal`).
-/

@[expose] public section

namespace Verity.Game

variable {Ω : Type} [Fintype Ω]

theorem piLaw_apply {ι : Type} [Fintype ι] [DecidableEq ι] (ν : ι → PMF Ω) (x : ι → Ω) :
    piLaw ν x = ∏ i, ν i (x i) := rfl

theorem piLaw_toReal {ι : Type} [Fintype ι] [DecidableEq ι] (ν : ι → PMF Ω) (x : ι → Ω) :
    (piLaw ν x).toReal = ∏ i, (ν i (x i)).toReal := by
  rw [piLaw_apply, ENNReal.toReal_prod]

theorem iid_toReal (μ : PMF Ω) {n : ℕ} (cs : Fin n → Ω) : (iid μ n cs).toReal = ∏ i, (μ (cs i)).toReal :=
  piLaw_toReal _ cs

theorem iid_cons_toReal (μ : PMF Ω) {n : ℕ} (c : Ω) (cs : Fin n → Ω) :
    (iid μ (n + 1) (Fin.cons c cs)).toReal = (μ c).toReal * (iid μ n cs).toReal := by
  rw [iid_toReal, iid_toReal, Fin.prod_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]

theorem prodLaw_toReal {W X : Type} [Fintype W] [Fintype X] (ρ : PMF W) (ν : PMF X) (w : W) (x : X) :
    (prodLaw ρ ν (w, x)).toReal = (ρ w).toReal * (ν x).toReal :=
  ENNReal.toReal_mul

end Verity.Game
