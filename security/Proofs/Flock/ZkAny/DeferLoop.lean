module

public import Proofs.Flock.SamplingGeneric.FlockZk
public import Proofs.Flock.SessionLoop

/-! The deployed loop's deferred server is zero knowledge per outcome, with no factor `2` (`Loop.zkOut_loopDeployed`). -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

noncomputable abbrev deferL (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)) : LeanRelease Flock.Wit Tp :=
  LeanRelease.ofLoopLink L₀ src (realOf src)

def DeferRel (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (wk : (deferL L₀ src).toSrv.Worker) : Rel Flock.Wit := fun x w =>
  Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w ∧ Serves x ∧
    _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w ∧
    Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x ∧ Loop.Src L₀ src x w ∧
    LeanRelease.LiveRoundEq finS vs (deferL L₀ src) CoinOpen.shaH wk x w ∧
    LeanRelease.FillRoundEq finS vs (deferL L₀ src) CoinOpen.shaH wk x w

theorem liveRel_deployed (vs : finS.SwVals) (hsc : finS.SwCausal vs) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (wk : (deferL L₀ src).toSrv.Worker) (x : Msg) (w : Flock.Wit) (h : DeferRel vs L₀ src wk x w) :
    SamplingGeneric.FlockZk.LiveRel _root_.Flock.Readable.relation Masked.dataE vs (deferL L₀ src) wk progOf freshOf
      x w := by
  obtain ⟨hR, hServe, hL0, hScope, hSrc, hE, hEf⟩ := h
  have hLk : LeanRelease.LinkRoundEq finS vs (deferL L₀ src) x w :=
    LeanRelease.linkRoundEq_ofLoopLink vs L₀ src _ hScope hR hSrc.1.linkLevel0
  have hFin : LeanRelease.FinishIs finS vs (deferL L₀ src) CoinOpen.shaH wk x w :=
    finishIs vs L₀ src wk x w hScope hR hServe hL0 hSrc hE hsc
  have hF : LeanRelease.FinalRoundEq finS vs (deferL L₀ src) CoinOpen.shaH wk x w :=
    LeanRelease.finalRoundEq_ofLoop Masked.dataE vs (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src)
      (realOf src) _ wk x w LeanRelease.tablesAdmitUnrefused_dataE
      (LeanRelease.frameUnrefused_ofLoop Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src)
        (realOf src) _ wk x w hServe) hFin
  exact ⟨hR, LeanRelease.ofLoop_progReads Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src)
    (realOf src) x, hLk, hF, hE, hEf⟩

theorem zkOut_loopDeployed (vs : finS.SwVals) (t : ℕ) (ht : 0 < t) (hS : finS.Tables)
    (hV : finS.Complete (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse))
    (hD : finS.DummyReg) (hc : finS.Causal) (hsc : finS.SwCausal vs) (hT : finS.TapeOK)
    (L₀ : LeanRelease Flock.Wit Tp) (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (wk : (deferL L₀ src).toSrv.Worker) :
    ZKOut (fun _ => Tp) (fun _ => (deferL L₀ src).toSrv.Chk) CoinOpen.shaH
      (SamplingGeneric.FlockZk.rounds _root_.Flock.Readable.relation)
      (fun w _ => (deferL L₀ src).toSrv.proverD (deferL L₀ src).fillOf
        (SamplingGeneric.FlockZk.rounds _root_.Flock.Readable.relation) wk w)
      (DeferRel vs L₀ src wk)
      (outD (deferL L₀ src).toSrv.stop (SamplingGeneric.FlockZk.rounds _root_.Flock.Readable.relation))
      (SamplingGeneric.FlockZk.simOut _root_.Flock.Readable.relation Masked.dataE (deferL L₀ src) progOf freshOf t)
      (SamplingGeneric.FlockZk.errOut Masked.dataE vs (deferL L₀ src) progOf freshOf t
        (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash))) :=
  fun x w hR =>
    SamplingGeneric.FlockZk.server_zkOut _root_.Flock.Readable.relation Masked.dataE vs t ht
      (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hS hV hD hc hsc hT
      (LeanRelease.withFrames Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src))
      (deferL L₀ src) rfl wk progOf freshOf x w (liveRel_deployed vs hsc L₀ src wk x w hR)

end Loop

end Claims.ZKServer
