import Properties.Nci.Protocol
import Proofs.Properties.Nci.BaselineSteps

/-!
# NCI's core protocol's property, proved

`ProtocolAccepts` (`Properties.Nci.Protocol`) by unfolding `NciCheck.Protocol.check` past its `names` check, with the policy check's verdict
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

theorem unparsed_refused :
    (NciCheck.Baseline.evaluate
      (NciCheck.Protocol.instanceOf ⟨.null, 1, #[], Json.arr #[], ""⟩ ⟨Json.arr #[], Json.arr #[], Json.arr #[]⟩)
      (NciCheck.Protocol.trustedOf ⟨.null, 1, #[], Json.arr #[], ""⟩ ⟨.null, "", .empty⟩)).toBool = false := by
  decide +kernel

/-- **`ProtocolAccepts` fails without `NciCheck.Protocol.check`.** At a case whose α is no fraction, which a verifier that
accepted every case would accept, the check of record gives no verdict at all: its conclusion's first part fails. -/
theorem ProtocolAccepts_nontrivial :
    ∀ out, ¬ NciCheck.Baseline.evaluate
      (NciCheck.Protocol.instanceOf ⟨.null, 1, #[], Json.arr #[], ""⟩ ⟨Json.arr #[], Json.arr #[], Json.arr #[]⟩)
      (NciCheck.Protocol.trustedOf ⟨.null, 1, #[], Json.arr #[], ""⟩ ⟨.null, "", .empty⟩) = .ok out := by
  intro out h
  have := unparsed_refused
  rw [h] at this
  simp [Except.toBool] at this

end NciCheck.Properties.SecurityProofs
