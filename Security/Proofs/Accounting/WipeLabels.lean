import Definitions.Accounting.Guarantees.WipeLabels

/-!
# The deployed wipe labels' PRF advantage (proof)

`Accounting.SecurityProofs.WipeLabelsAdvantage`: `WipeLabelsPrf` is `Pr[real] ≤ Pr[ideal] + ε`, and the advantage is the
truncated difference `Pr[real] - Pr[ideal]`.
-/

namespace Accounting.SecurityProofs

theorem WipeLabelsAdvantage : Accounting.Guarantees.WipeLabelsAdvantage :=
  fun _ _ _ _ _ _ _ h => tsub_le_iff_left.mpr h

end Accounting.SecurityProofs
