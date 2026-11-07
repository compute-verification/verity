module

public import Proofs.Flock.SessionLoop.Statement
public import Proofs.Flock.SessionLoop.TakenVals
public import Proofs.Flock.SessionLoop.FinalC
public import Proofs.Flock.SessionLoop.TakesTau
public import Proofs.Flock.SessionLoop.Inner
public import Proofs.Flock.SessionLoop.InnerWitness
public import Proofs.Flock.SessionLoop.LiveInner
public import Proofs.Flock.Claims.ZK.FrameUnrefused
public import Proofs.Flock.SessionLoop.Link
public import Proofs.Flock.Claims.ZK.FinishSwFrame
public import Proofs.Flock.Claims.ZK.FinishSwRep
public import Proofs.Flock.SessionLoop.FinishParts
public import Proofs.Flock.SessionLoop.FinCommits
public import Proofs.Flock.SessionLoop.FinPoints
public import Proofs.Flock.SessionLoop.FinRounds
public import Proofs.Flock.SessionLoop.TablesSide
public import Proofs.Flock.SessionLoop.Level0SrcSat
public import Proofs.Flock.Claims.ZK.FinishBridge
public import Proofs.Flock.Claims.ZK.FinishTrunc
public import Proofs.Flock.Claims.ZK.FinHmkLive
public import Proofs.Flock.Claims.ZK.FinishFwRep
public import Proofs.Flock.Claims.ZK.FinTau
public import Proofs.Flock.Claims.ZK.L0PinLoop
public import Proofs.Flock.Claims.ZK.L0PinCut
public import Proofs.Flock.Claims.ZK.SimCommit
public import Proofs.Flock.Claims.ZK.FinishSwVals
public import Proofs.Flock.Claims.ZK.FinishClaimsX
public import Proofs.Flock.Claims.ZK.FrameFree
public import Proofs.Flock.Claims.ZK.PiopFree
public import Proofs.Flock.Claims.ZK.FinishLigEnd
public import Proofs.Flock.Claims.ZK.RepCutFw
public import Proofs.Flock.Claims.ZK.FinishSwMkAny
public import Proofs.Flock.Claims.ZK.FinishFwL0
public import Proofs.Flock.SessionLoop.FinishIsLoop
public import Proofs.Flock.SessionLoop.PiopState
public import Proofs.Flock.SessionLoop.CutPiopLive
public import Proofs.Flock.SessionLoop.CutInnerLive
public import Proofs.Flock.SessionLoop.InnerSlot
public import Proofs.Flock.SessionLoop.InnerUnrefused
public import Proofs.Flock.SessionLoop.LiveInnerAt
public import Proofs.Flock.SessionLoop.Phase
public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.SessionLoop.SampledAudit

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
(`LeanRelease.linkRoundEq_ofLoopLink`, from `Loop.Src`'s level-0 half, `Loop.Level0Src.linkLevel0`), and whose `FinishIs`
holds there at `LoopInnerRound0`, `Loop.Src`, `LiveRoundEq` and `SwCausalOn` (`Loop.finishIsOn`). That level-0 half is
satisfiable at every statement (`Loop.level0Src_tapeSrc`, `Proofs.Flock.SessionLoop.Level0SrcSat`). The switched
causality is asked only at `Loop.SwOk`'s coins, and the loop's stop rule computes only there
(`LeanRelease.stopsOk_swOk`: its frames refuse a zerocheck outer coin `1`, `LeanRelease.rfOf_outer`).

`SessionLoopZKDeployed` is over every worker. Its premises on the worker (`LiveRoundEq`, `FillRoundEq`) read it in the
algebraic phase only (`Proofs.Flock.SessionLoop.Phase`): a worker that, until the algebraic phase's last coins, reads
only the nonce, what the workers see of the statement's tables (`RefSession.seenOf`, what `vs` reads) and the answers as
coins (`Loop.PhaseBlind`) is `Loop.phased` at an algebraic phase's worker, whatever it reads after them
(`Loop.PhaseBlind.exists_phased`), and the premises at it are the premises at that worker as a pad-blind one
(`Loop.liveRoundEq_phased`, `Loop.fillRoundEq_phased`).
-/

@[expose] public section

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

/-- **Every worker that reads only what the workers see is a pad-blind one** (`Loop.blind`): one that returns the same
at two tapes alike there (`Masked.seenOf`) and two histories whose answers carry the same coins (`CoinLayer.rd`). -/
theorem exists_blind (L : LeanRelease Flock.Wit
    ((ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Nonce ×
      (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Rand))
    {wk : L.toSrv.Worker} (hb : L.toSrv.Blind ZKMal.Masked.seenOf ZKMal.CoinLayer.rd wk) : ∃ bw, blind L bw = wk :=
  hb.exists_lift ZKMal.Masked.seenOf_surjective

end Claims.ZKServer.Loop

namespace Flock.SecurityProofs

open FlockSoundness.Discharge

/-- **C-Flock's zero knowledge at the session loop's own frames**, at every statement the loop serves, at tables that
admit unrefused coins. -/
theorem SessionLoopZK : Flock.Guarantees.SessionLoopZK := by
  intro T hA vs t ht hS hV hD hc hsc hT L₀ src real hreal wk x w hR hServe hL0 hLk hFin hE hEf Θ Vs θ hk Ev
  have h := Claims.ZKServer.Flock.zk_server_lean_ofLoop _root_.Flock.Readable.relation T vs t ht
    (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hS hV hD hc hsc hT L₀ real
    (Claims.ZKServer.LeanRelease.stopsOk_swOk T L₀ real) wk x w hR hServe
    (fun ρ _ _ _ hs => hreal ▸ Claims.ZKServer.Loop.realTabs_realOf src x w ρ hs) hL0 hLk
    (Claims.ZKServer.LeanRelease.finalRoundEq_ofLoop T vs L₀ real _ wk x w hA
      (Claims.ZKServer.LeanRelease.frameUnrefused_ofLoop T L₀ real _ wk x w hServe) hFin) hE hEf Vs θ hk Ev
  rwa [ENNReal.toReal_ofReal (FlockSoundness.CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash)] at h

/-- **C-Flock's zero knowledge at the session loop's own frames and link, at the deployed tables**: `SessionLoopZK` at
`Masked.dataE`, whose tables admit unrefused coins and hold the tables' side but `SwCausalOn` (`Loop.tables_dataE`,
`complete_dataE`, `dummyReg_dataE`, `causal_dataE`, `tapeOK_dataE`), and the release `withFiles (withLink L₀ src) src`,
whose `LinkRoundEq` is `linkRoundEq_ofLoopLink` and whose `FinishIs` is `Loop.finishIsOn`. -/
theorem SessionLoopZKDeployed : Flock.Guarantees.SessionLoopZKDeployed := by
  intro vs t ht hsc L₀ src real hreal wk x w hR hServe hL0 hScope hSrc hE hEf Θ Vs θ hk Ev
  subst hreal
  exact SessionLoopZK _ Claims.ZKServer.LeanRelease.tablesAdmitUnrefused_dataE vs t ht Claims.ZKServer.Loop.tables_dataE
    Claims.ZKServer.Loop.complete_dataE Claims.ZKServer.Loop.dummyReg_dataE Claims.ZKServer.Loop.causal_dataE hsc
    Claims.ZKServer.Loop.tapeOK_dataE
    (Claims.ZKServer.LeanRelease.withFiles (Claims.ZKServer.LeanRelease.withLink L₀ src) src) src _ rfl wk x w hR hServe
    hL0 (Claims.ZKServer.LeanRelease.linkRoundEq_ofLoopLink vs L₀ src _ hScope hR hSrc.1.linkLevel0)
    (Claims.ZKServer.Loop.finishIsOn vs L₀ src wk x w hScope hR hServe hL0 hSrc hE hsc) hE hEf Vs θ hk Ev

end Flock.SecurityProofs
