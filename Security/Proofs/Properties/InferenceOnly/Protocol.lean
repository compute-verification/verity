import Properties.InferenceOnly.Protocol
import Proofs.Properties.InferenceOnly.ThetaSteps

/-!
# NCI's core protocol's property, proved

`ProtocolAccepts` (`Properties.InferenceOnly.Protocol`) by unfolding `InferenceOnlyCheck.Protocol.check` past its `names` check, with the policy check's verdict
from `ThetaSteps.evaluate_ok`: `InferenceOnlyCheck.Theta.evaluate` returns the verdict `InferenceOnlyCheck.Desc.evaluate` gives.
-/

namespace InferenceOnlyCheck.Properties.SecurityProofs

open Lean (Json)

theorem ProtocolAccepts : Properties.ProtocolAccepts := by
  intro p c e pub prof h
  unfold InferenceOnlyCheck.Protocol.check at h
  split at h
  · cases h
  split at h
  · cases h
  · rename_i out hb
    split at h
    · rename_i hc
      cases h
      exact ⟨out, hb, (ThetaSteps.evaluate_ok _ _ _ hb).1, hc, rfl⟩
    · cases h

theorem unparsed_refused :
    (InferenceOnlyCheck.Theta.evaluate
      (InferenceOnlyCheck.Protocol.instanceOf ⟨.null, 1, #[], Json.arr #[], ""⟩ ⟨Json.arr #[], Json.arr #[], Json.arr #[]⟩)
      (InferenceOnlyCheck.Protocol.trustedOf ⟨.null, 1, #[], Json.arr #[], ""⟩ ⟨.null, "", .empty⟩)).toBool = false := by
  decide +kernel

/-- **`ProtocolAccepts` fails without `InferenceOnlyCheck.Protocol.check`.** At a case whose α is no fraction, which a verifier that
accepted every case would accept, the check of record gives no verdict at all: its conclusion's first part fails. -/
theorem ProtocolAccepts_nontrivial :
    ∀ out, ¬ InferenceOnlyCheck.Theta.evaluate
      (InferenceOnlyCheck.Protocol.instanceOf ⟨.null, 1, #[], Json.arr #[], ""⟩ ⟨Json.arr #[], Json.arr #[], Json.arr #[]⟩)
      (InferenceOnlyCheck.Protocol.trustedOf ⟨.null, 1, #[], Json.arr #[], ""⟩ ⟨.null, "", .empty⟩) = .ok out := by
  intro out h
  have := unparsed_refused
  rw [h] at this
  simp [Except.toBool] at this

end InferenceOnlyCheck.Properties.SecurityProofs
