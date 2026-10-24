import Definitions.Pous.P2.Cost

/-! The proof-only part of `Definitions.Pous.P2.Cost`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pous

open scoped ENNReal

variable {B w p : ℕ} [NeZero p]

/-- **The requirement over the ring bound**: `Meets`' conjuncts, with `AuditSecureOn` in place of `AuditSecure`. -/
def MeetsOn (sch : Scheme (ringRO B w p)) (reveal : Reveal) (D Q L k : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
    AuditSecureOn sch reveal (sch.stateBound Params.ρ) D Q L k (ENNReal.ofReal Params.δ) ε

/-- **The requirement with `lam` bits of slack per block**: `Meets`, except that the adversary's state is
`⌊ρ·|C|⌋ + lam·B` bits. -/
def MeetsWithSlack {M : IdealModel} (sch : Scheme M) (reveal : Reveal) (lam D Q k : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
    AuditSecure sch reveal (sch.stateBound Params.ρ + lam * sch.B) D Q k (ENNReal.ofReal Params.δ) ε

end Pous
