import Proofs.Flock.SessionLoop.Statement
import Proofs.Flock.Claims.ZK.FrameUnrefused

/-!
# Proof of C-Flock's zero knowledge at the session loop's own frames

`Flock.SecurityProofs.SessionLoopZK : Flock.Guarantees.SessionLoopZK` is `Claims.ZKServer.Flock.zk_server_lean_ofLoop`
at the deployed relation and at the hiding gap of hm96-sha512's pinned key (`CROnly.hm96Hiding_gap`, finite as a real),
its `FinalRoundEq` from `FinishIs`, the loop's frames refusing every coin fault (`LeanRelease.frameUnrefused_ofLoop`)
and the tables admitting unrefused coins (`LeanRelease.finalRoundEq_ofLoop`). `SessionLoopZKDeployed` is it at the
deployed tables, which admit unrefused coins (`LeanRelease.tablesAdmitUnrefused_dataE`).
-/

namespace Flock.SecurityProofs

open FlockSoundness.Discharge

/-- **C-Flock's zero knowledge at the session loop's own frames**, at every statement the loop serves, at tables that
admit unrefused coins. -/
theorem SessionLoopZK : Flock.Guarantees.SessionLoopZK := by
  intro T hA vs t ht hS hV hD hc hsc hT L₀ real wk x w hR hServe hRt hL0 hLk hFin hE hEf Θ Vs θ hk Ev
  have h := Claims.ZKServer.Flock.zk_server_lean_ofLoop _root_.Flock.Readable.relation T vs t ht
    (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hS hV hD hc hsc hT L₀ real wk
    x w hR hServe hRt hL0 hLk
    (Claims.ZKServer.LeanRelease.finalRoundEq_ofLoop T vs L₀ real _ wk x w hA
      (Claims.ZKServer.LeanRelease.frameUnrefused_ofLoop T L₀ real _ wk x w hServe) hFin) hE hEf Vs θ hk Ev
  rwa [ENNReal.toReal_ofReal (FlockSoundness.CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash)] at h

/-- **C-Flock's zero knowledge at the session loop's own frames, at the deployed tables**: `SessionLoopZK` at
`Masked.dataE`, whose tables admit unrefused coins. -/
theorem SessionLoopZKDeployed : Flock.Guarantees.SessionLoopZKDeployed :=
  SessionLoopZK _ Claims.ZKServer.LeanRelease.tablesAdmitUnrefused_dataE

end Flock.SecurityProofs
