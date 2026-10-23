import Proofs.Flock.SessionLoop.Statement

/-!
# Proof of C-Flock's zero knowledge at the session loop's own frames

`Flock.SecurityProofs.SessionLoopZK : Flock.Guarantees.SessionLoopZK` is `Claims.ZKServer.Flock.zk_server_lean_ofLoop`
at the deployed relation and at the hiding gap of hm96-sha512's pinned key (`CROnly.hm96Hiding_gap`, finite as a real).
-/

namespace Flock.SecurityProofs

open FlockSoundness.Discharge

/-- **C-Flock's zero knowledge at the session loop's own frames**, at every statement the loop serves. -/
theorem SessionLoopZK : Flock.Guarantees.SessionLoopZK := by
  intro T vs t ht hS hV hD hc hsc hT L₀ real wk x w hR hServe hRt hL0 hLk hF hE hEf Θ Vs θ hk Ev
  have h := Claims.ZKServer.Flock.zk_server_lean_ofLoop _root_.Flock.Readable.relation T vs t ht
    (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hS hV hD hc hsc hT L₀ real wk
    x w hR hServe hRt hL0 hLk hF hE hEf Vs θ hk Ev
  rwa [ENNReal.toReal_ofReal (FlockSoundness.CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash)] at h

end Flock.SecurityProofs
