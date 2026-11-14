module

public import Proofs.Properties.Verity.Sampled.HonestSession.Instance

/-!
# Honest witness for deployed remote soundness

The deployed theorem's premises hold together at a deployment that registers a real circuit and public file and runs
an accepted proving session over them.
-/

@[expose] public section

namespace Verity.Sampled

/-- **`remote_soundness_deployedReady`'s honest lemma**: one honest deployment runs the concrete accepted proving
session, registers its circuit and public file, and satisfies every premise of the deployed remote theorem. -/
theorem remote_soundness_deployedReady_honest :
    Instance.HonestSession.inst.SessionsReady ∧ Instance.HonestSession.inst.RegisteredFiles ∧
      Instance.HonestSession.inst.SessionRegistered .remote ∧
      Instance.HonestSession.inst.base.PouwSound .remote ∧
      Instance.HonestSession.inst.base.InferenceOnlySound .remote :=
  ⟨Instance.HonestSession.sessions_ready, Instance.HonestSession.registered_files,
    Instance.HonestSession.session_registered, Instance.HonestSession.target_top .remote .pouw,
    Instance.HonestSession.target_top .remote .inferenceOnly⟩

end Verity.Sampled
