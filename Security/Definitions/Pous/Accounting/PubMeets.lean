import Definitions.Pous.Model.PubEncoder
import Definitions.Pous.Accounting.Params

/-!
# The requirement with a public encoder

`PubMeets`, copied verbatim from `lean/submissions/public-encoder/PubEncoderModel.lean`: `Meets`'s clauses, the
check's completeness, and the public-encoder audit bound at the problem statement's parameters.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- **The requirement with a public encoder**: `Meets`'s clauses, the check's completeness, and the public-encoder
audit bound at the problem statement's parameters. -/
def PubMeets (sch : Scheme M) (V : SetupCheck sch) (reveal : Reveal) (D Q k : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧ V.Complete ∧
    PubAuditSecure sch V reveal (sch.stateBound Params.ρ) D Q k (ENNReal.ofReal Params.δ) ε

end Pous
