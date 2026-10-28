import Definitions.Pous.Model.Graded
import Definitions.Pous.Accounting.Params

/-!
# The requirement with grace `t`

`GradedMeets`: `Meets`'s clauses with the graded audit (`Security/Definitions/Pous/Model/Graded.lean`) in place
of the sequential one. It is the requirement of `Pous.Guarantees.BandGradedMeetsFamily`.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- **The requirement with grace `t`**: `Meets` with the graded audit in place of the sequential one. At `t = 0` it is
`Meets … .sequential` (`PousGraded.gradedMeets_zero_iff`, for `D ≤ D_late` and `Q ≤ Q_late`). -/
def GradedMeets (sch : Scheme M) (D Q D_late Q_late k t : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
    GradedAuditSecure sch (sch.stateBound Params.ρ) D Q D_late Q_late k t (ENNReal.ofReal Params.δ) ε

end Pous
