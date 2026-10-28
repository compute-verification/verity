import Properties.Nci.Protocol
import Proofs.Properties.Nci.BaselineSteps

/-!
# NCI's core protocol's property, proved

`ProtocolAccepts` (`Properties.Nci.Protocol`) by unfolding `NciCheck.Protocol.check` past its `names` refusal, with the policy check's verdict
from `BaselineSteps.evaluate_ok`: `NciCheck.Baseline.evaluate` returns the verdict `NciCheck.Desc.evaluate` gives.
-/

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)

theorem ProtocolAccepts : Properties.ProtocolAccepts := by
  intro p c e pub prof h
  unfold NciCheck.Protocol.check at h
  split at h
  · cases h
  split at h
  · cases h
  · rename_i out hb
    split at h
    · rename_i hc
      cases h
      exact ⟨out, hb, (BaselineSteps.evaluate_ok _ _ _ hb).1, hc, rfl⟩
    · cases h

end NciCheck.Properties.SecurityProofs
