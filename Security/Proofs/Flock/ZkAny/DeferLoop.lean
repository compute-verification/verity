module

public import Proofs.Flock.SamplingGeneric.FlockZk
public import Proofs.Flock.SessionLoop

/-!
# The deployed loop's deferred server, zero knowledge per outcome

`SessionLoopZKDeployed` bounds the deferred server at the session loop's own frames, link and `Finish` by twice the
simulator that guesses whether it stopped. Here the same server, at the same premises, is zero knowledge per outcome
(`Claims.ZKOut`) at its outcome (`ZKServer.outD`: the verdict, and whether it stopped at its `Finish`), with no factor
`2`: `FlockZk.server_zkOut` at the loop's release, with its live path's premises from `SessionLoopZKDeployed`'s and
hm96-sha512's hiding gap at the pinned key.

* `Loop.deferL`: `SessionLoopZKDeployed`'s release at the loop's own tables on `src`.
* `Loop.DeferRel`: `SessionLoopZKDeployed`'s premises at a statement and witness, at a worker.
* `Loop.liveRel_deployed`: at those premises the live path's (`FlockZk.LiveRel`) hold at the loop's release, from the
  link (`linkRoundEq_ofLoopLink`), the `Finish` (`Loop.finishIs`, `finalRoundEq_ofLoop`) and the loop's frame states
  (`ofLoop_progReads`).
* **`Loop.zkOut_loopDeployed`**: the deferred server at the loop's release is `ZKOut` at `DeferRel`, by
  `FlockZk.simOut`, at `FlockZk.errOut` at the pinned key's hiding gap: `SessionLoopZKDeployed`'s error without its
  factor `2`.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **`SessionLoopZKDeployed`'s release** at the loop's own tables on `src`. -/
noncomputable abbrev deferL (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)) : LeanRelease Flock.Wit Tp :=
  LeanRelease.ofLoopLink L₀ src (realOf src)

/-- **`SessionLoopZKDeployed`'s premises at `(x, w)`**, at the worker `wk`: the relation, a statement the loop serves,
the loop's inner round 0, the masked verifier's scope, the loop's inputs the tape's, and the live and stand-in rounds'
bytes the session's. -/
def DeferRel (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (wk : (deferL L₀ src).toSrv.Worker) : Rel Flock.Wit := fun x w =>
  Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w ∧ Serves x ∧
    _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w ∧
    Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x ∧ Loop.Src L₀ src x w ∧
    LeanRelease.LiveRoundEq finS vs (deferL L₀ src) CoinOpen.shaH wk x w ∧
    LeanRelease.FillRoundEq finS vs (deferL L₀ src) CoinOpen.shaH wk x w

/-- **The live path's premises at `SessionLoopZKDeployed`'s**: at `DeferRel` and causal switched values, the loop's
release meets `FlockZk.LiveRel` at the loop's frame states: its frames are the transport's (`ofLoop_progReads`), its
link the reference's (`linkRoundEq_ofLoopLink`), and its `Finish` the session's (`Loop.finishIs`,
`finalRoundEq_ofLoop`). -/
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

/-- **The deployed loop's deferred server is zero knowledge per outcome**: at `SessionLoopZKDeployed`'s premises on the
session (`Tables`, `Complete`, `DummyReg`, `Causal`, `SwCausal`, `TapeOK`) and `t > 0`, the deferred server at the
loop's own frames, link and `Finish` with the worker `wk` is `ZKOut` at `DeferRel`: per outcome (`outD`: the verdict,
and whether it stopped at its `Finish`), an event of its transcript and that outcome is at most the event at the
switched prover's simulator given the outcome (`FlockZk.simOut`), plus `8·n_L·δ₁ + 8·t·n_L·δ₁ + 2·3⁻ᵗ +
4·t·3ᵗ·Pr[Fork]` at hm96-sha512's hiding gap `δ₁` at the pinned key (`FlockZk.errOut`), `SessionLoopZKDeployed`'s error
without its factor `2`. `FlockZk.server_zkOut` at the loop's release, at `liveRel_deployed`. -/
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
