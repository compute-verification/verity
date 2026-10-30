import Proofs.Flock.SessionLoop.Statement
import Proofs.Flock.Claims.ZK.FrameUnrefused
import Proofs.Flock.SessionLoop.Link
import Proofs.Flock.Claims.ZK.FinishSwFrame
import Proofs.Flock.Claims.ZK.FinishSwRep
import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.SessionLoop.FinCommits
import Proofs.Flock.SessionLoop.FinPoints
import Proofs.Flock.SessionLoop.FinRounds
import Proofs.Flock.Claims.ZK.FinishBridge
import Proofs.Flock.Claims.ZK.FinishTrunc
import Proofs.Flock.SessionLoop.FinRounds
import Proofs.Flock.Claims.ZK.FinHmkLive
import Proofs.Flock.Claims.ZK.FinishFwRep
import Proofs.Flock.Claims.ZK.FinTau
import Proofs.Flock.Claims.ZK.L0PinLoop
import Proofs.Flock.Claims.ZK.L0PinCut
import Proofs.Flock.Claims.ZK.SimCommit
import Proofs.Flock.Claims.ZK.FinishSwVals
import Proofs.Flock.Claims.ZK.FinishClaimsX
import Proofs.Flock.Claims.ZK.FrameFree
import Proofs.Flock.Claims.ZK.PiopFree
import Proofs.Flock.Claims.ZK.FinishLigEnd
import Proofs.Flock.Claims.ZK.RepCutFw
import Proofs.Flock.Claims.ZK.FinishSwMkAny
import Proofs.Flock.Claims.ZK.FinishFwL0

/-!
# Proof of C-Flock's zero knowledge at the session loop's own frames

`Flock.SecurityProofs.SessionLoopZK : Flock.Guarantees.SessionLoopZK` is `Claims.ZKServer.Flock.zk_server_lean_ofLoop`
at the deployed relation and at the hiding gap of hm96-sha512's pinned key (`CROnly.hm96Hiding_gap`, finite as a real),
at the loop's own tables (`Loop.realOf`, whose tables are `Count.RealTabs`: `Loop.realTabs_realOf`), its `FinalRoundEq`
from `FinishIs`, the loop's frames refusing every coin fault (`LeanRelease.frameUnrefused_ofLoop`) and the tables
admitting unrefused coins (`LeanRelease.finalRoundEq_ofLoop`). `SessionLoopZKDeployed` is it at the deployed tables,
which admit unrefused coins (`LeanRelease.tablesAdmitUnrefused_dataE`), at the loop's own link and `Finish`
(`LeanRelease.ofLoopLink`),
whose `LinkRoundEq` holds in the masked scope where the loop draws the tape's level 0
(`LeanRelease.linkRoundEq_ofLoopLink`, from `Loop.Src`'s level-0 half, `Loop.Level0Src.linkLevel0`).
-/

namespace Claims.ZKServer.Loop

/-- **The loop's own tables are `RealTabs`**: each is `tabOfLevel0` at a level 0 `Level0.ofDraws` drew. -/
theorem realTabs_realOf {R : Type}
    (src : Msg → Flock.Wit → R → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) {s0 : _} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) : Count.RealTabs sts (realOf src x w ρ) := by
  intro k tr hr h
  unfold realOf at h
  rw [hs] at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i a _
    obtain ⟨g, d, hi⟩ := a
    dsimp only at h
    split at h
    · cases h
    · rename_i l hl
      split at h
      · cases h
      · rename_i t ht
        cases h
        exact ⟨g, d, l, hl, ht⟩

end Claims.ZKServer.Loop

namespace Flock.SecurityProofs

open FlockSoundness.Discharge

/-- **C-Flock's zero knowledge at the session loop's own frames**, at every statement the loop serves, at tables that
admit unrefused coins. -/
theorem SessionLoopZK : Flock.Guarantees.SessionLoopZK := by
  intro T hA vs t ht hS hV hD hc hsc hT L₀ src real hreal wk x w hR hServe hL0 hLk hFin hE hEf Θ Vs θ hk Ev
  have h := Claims.ZKServer.Flock.zk_server_lean_ofLoop _root_.Flock.Readable.relation T vs t ht
    (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hS hV hD hc hsc hT L₀ real wk
    x w hR hServe (fun ρ _ _ _ hs => hreal ▸ Claims.ZKServer.Loop.realTabs_realOf src x w ρ hs) hL0 hLk
    (Claims.ZKServer.LeanRelease.finalRoundEq_ofLoop T vs L₀ real _ wk x w hA
      (Claims.ZKServer.LeanRelease.frameUnrefused_ofLoop T L₀ real _ wk x w hServe) hFin) hE hEf Vs θ hk Ev
  rwa [ENNReal.toReal_ofReal (FlockSoundness.CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash)] at h

/-- **C-Flock's zero knowledge at the session loop's own frames and link, at the deployed tables**: `SessionLoopZK` at
`Masked.dataE`, whose tables admit unrefused coins, and the release `withFiles (withLink L₀ src) src`, whose
`LinkRoundEq` is `linkRoundEq_ofLoopLink`. -/
theorem SessionLoopZKDeployed : Flock.Guarantees.SessionLoopZKDeployed := by
  intro vs t ht hS hV hD hc hsc hT L₀ src real hreal wk x w hR hServe hL0 hScope hSrc hFin hE hEf Θ Vs θ hk Ev
  exact SessionLoopZK _ Claims.ZKServer.LeanRelease.tablesAdmitUnrefused_dataE vs t ht hS hV hD hc hsc hT
    (Claims.ZKServer.LeanRelease.withFiles (Claims.ZKServer.LeanRelease.withLink L₀ src) src) src real hreal wk x w hR hServe hL0
    (Claims.ZKServer.LeanRelease.linkRoundEq_ofLoopLink vs L₀ src real hScope hR hSrc.1.linkLevel0) hFin hE hEf Vs θ hk
    Ev

end Flock.SecurityProofs
