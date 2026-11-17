module

public import Definitions.Spot.Exfiltration
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.EReal.Inv

/-! The cheating pool with the halt `Δ` windows after the first caught window's commitment, so a cheater keeps skipping work up to the halt. -/

@[expose] public section

namespace Population.Exfil

variable {Ω : Type}

noncomputable def Strategy.lagLeak (A : Strategy Ω) : (n Δ : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _, _ => 0
  | _ + 1, 0, _, _ => 0
  | n + 1, Δ + 1, h, cs => A.leak h + A.lagLeak n Δ (cs 0 :: h) (Fin.tail cs)

/-- With `j = 0` that detection has happened, and only the `Δ` windows before the halt remain. -/
noncomputable def Strategy.cumUpToLag [DecidableEq Ω] (A : Strategy Ω) (Δ : ℕ) : (n j : ℕ) → List Ω → (Fin n → Ω) → ℝ
  | 0, _, _, _ => 0
  | n + 1, 0, h, cs => A.lagLeak (n + 1) Δ h cs
  | n + 1, j + 1, h, cs => A.leak h + if cs 0 ∈ A.esc h then A.cumUpToLag Δ n (j + 1) (cs 0 :: h) (Fin.tail cs)
      else A.cumUpToLag Δ n j (cs 0 :: h) (Fin.tail cs)

noncomputable def Strategy.cumLag [DecidableEq Ω] (A : Strategy Ω) (Δ n : ℕ) (h : List Ω) (cs : Fin n → Ω) : ℝ :=
  A.cumUpToLag Δ n 1 h cs

end Population.Exfil
