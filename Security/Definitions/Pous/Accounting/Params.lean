import Mathlib.Algebra.Order.Floor.Semiring
import Mathlib.Data.Rat.Floor
import Definitions.Pous.Model.Audit

/-!
# The problem statement's parameters, and the requirement at them
-/

namespace Pous

open scoped ENNReal

namespace Params

/-- Retention bound `ρ = 18/19`. -/
def ρ : ℚ := 18 / 19

/-- Soundness `δ = 1/100`. -/
def δ : ℚ := 1 / 100

/-- Space: `|C| + |pp| ≤ 1.05·|W|`. -/
def expansion : ℚ := 21 / 20

/-- Security parameter `λ = 128`. Preprocessing is unbounded here, so `λ` enters only through `εMax`. -/
def secParam : ℕ := 128

/-- The largest `ε_crypto` a construction may claim: `2^-λ`. **Provisional:** the problem statement names `λ = 128`
but does not bound `ε_crypto`, so this bound is not yet confirmed. -/
noncomputable def εMax : ℝ≥0∞ := 2⁻¹ ^ secParam

end Params

variable {M : IdealModel}

/-- The state bound `⌊ρ·|C|⌋` in bits. -/
def Scheme.stateBound (sch : Scheme M) (ρ : ℚ) : ℕ :=
  ⌊ρ * (sch.B * sch.ℓ : ℕ)⌋₊

/-- **The requirement** at the problem statement's parameters, for a construction that chose its reveal mode, its
`k`, its `ε_crypto ≤ εMax`, and was given `(D, Q)` by the deadline: correct, `|C| ≥ |W|`, within the space bound,
and every `⌊ρ|C|⌋`-bit adversary is accepted with probability at most `δ + ε`. -/
def Meets (sch : Scheme M) (reveal : Reveal) (D Q k : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
    AuditSecure sch reveal (sch.stateBound Params.ρ) D Q k (ENNReal.ofReal Params.δ) ε

end Pous
