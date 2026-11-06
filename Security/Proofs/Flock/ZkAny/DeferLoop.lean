module

public import Proofs.Flock.ZkAny.DeferOut
public import Proofs.Flock.SessionLoop

/-!
# The deployed loop's deferred server, zero knowledge per outcome

`SessionLoopZKDeployed` bounds the deferred server at the session loop's own frames, link and `Finish` by twice the
simulator that guesses whether it stopped. Here the same server, at the same premises, is zero knowledge per outcome
(`Claims.ZKOut`) at its stop bit, with no factor `2`: `ZkServer.zkOut_deferred` at the switched prover's `ZKmalR`
(`Flock.zk_deployZk_sw`) and the item-stream refinement `SessionLoopZKDeployed`'s chain proves from its premises.

* `Loop.deferL`, `deferK`, `deferS`, `deferSim`, `deferEps`: `SessionLoopZKDeployed`'s release at the loop's own tables
  on `src`, its last round, the session with the release's stops, its simulator and its error but for the fork.
* `Loop.DeferRel`: `SessionLoopZKDeployed`'s premises at a statement and witness, at a worker.
* `Loop.itemsRefine_deployed`: at those premises the release refines the switched prover's item stream
  (`LeanRelease.ItemsRefine`), from the chain `Flock.zk_server_lean_stops` … `SessionLoopZKDeployed` reads.
* **`Loop.zkOut_loopDeployed`**: the deferred server at the loop's release is `ZKOut` at `DeferRel`, its stop bit
  public, by `leaky1` told the bit (`leakyOut`), at `SessionLoopZKDeployed`'s error without the factor `2`.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **`SessionLoopZKDeployed`'s release** at the loop's own tables on `src`. -/
noncomputable abbrev deferL (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)) : LeanRelease Flock.Wit Tp :=
  LeanRelease.ofLoopLink L₀ src (realOf src)

/-- **The deferred server's last round**: one past the deployed verifier's, the `Finish`. -/
noncomputable abbrev deferK : Msg → ℕ := fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1

/-- **The session the deferred server at `L` is held to**: `finS` with the stops of `L`'s frames. -/
noncomputable abbrev deferS (L : LeanRelease Flock.Wit Tp) : RefSession :=
  finS.withStops (LeanRelease.rfOf finS L progOf freshOf readCoins) L.stop

/-- **The switched prover's simulator** at `deferS L`, `SessionLoopZKDeployed`'s before `leaky1`. -/
noncomputable abbrev deferSim (t : ℕ) (L : LeanRelease Flock.Wit Tp) : Simulator :=
  ((deferS L).refP.withSim fun x _ c => (deferS L).coinModel x c).simulatorE t (t * 3 ^ t)

/-- **The switched prover's error but for the fork**, at hm96-sha512's hiding gap at the pinned key. -/
noncomputable abbrev deferEps (t : ℕ) (L : LeanRelease Flock.Wit Tp) (x : Msg) : ℝ :=
  8 * (deferS L).nLeaves x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
    8 * t * (deferS L).nLeaves x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t

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

/-- **The item-stream refinement at `SessionLoopZKDeployed`'s premises**: at `DeferRel` and causal switched values, the
loop's release refines the switched prover's items at `deferS`, from the link (`linkRoundEq_ofLoopLink`), the `Finish`
(`Loop.finishIs`, `finalRoundEq_ofLoop`), the loop's `Hello`, coin check, requests and last round (`loop_helloIs`,
`checkIs_coinLayer`, `loop_reqsFollow`, `loop_last`) and its frame stops (`stopsAt_of_readsCoins`), as
`Flock.zk_server_lean_stops` builds them. -/
theorem itemsRefine_deployed (vs : finS.SwVals) (hsc : finS.SwCausal vs) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (wk : (deferL L₀ src).toSrv.Worker) (x : Msg) (w : Flock.Wit) (h : DeferRel vs L₀ src wk x w) :
    (deferL L₀ src).ItemsRefine CoinOpen.shaH ((deferS (deferL L₀ src)).swP vs).prover deferK wk x w := by
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
  have hlast : (deferL L₀ src).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE (LeanRelease.withFrames Masked.dataE
      (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src)) x
  have hRd : LeanRelease.ReadsCoins finS (deferL L₀ src) progOf freshOf readCoins x :=
    ⟨LeanRelease.ofLoop_progReads Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src) x,
      LeanRelease.loop_coinsRead loopNr Masked.dataE (LeanRelease.withFrames Masked.dataE
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src)) x⟩
  have hHe : LeanRelease.HelloEq finS (deferL L₀ src) x :=
    LeanRelease.helloEq_of_helloIs loopNr Masked.dataE (deferL L₀ src) x
      (LeanRelease.loop_helloIs loopNr Masked.dataE (LeanRelease.withFrames Masked.dataE
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src)) x)
  have hC : LeanRelease.CheckIs finS (deferL L₀ src) CoinOpen.shaH x :=
    LeanRelease.checkIs_coinLayer loopNr Masked.dataE (deferL L₀ src) x
      (LeanRelease.loop_helloParses loopNr Masked.dataE (LeanRelease.withFrames Masked.dataE
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src)) x)
      (LeanRelease.loop_reqsFollow Masked.dataE (LeanRelease.withFrames Masked.dataE
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (realOf src)) x)
  have hst := LeanRelease.stopsAt_of_readsCoins finS (deferL L₀ src) CoinOpen.shaH progOf freshOf readCoins x hRd hC
  have hO := LeanRelease.openingsRefused_of_checkIs finS (deferL L₀ src) CoinOpen.shaH x hC
  have hn : deferK x = refK x + 2 := deployZk_last_eq _root_.Flock.Readable.relation x
  exact ⟨LeanRelease.helloRefine_withStops finS _ vs (deferL L₀ src) CoinOpen.shaH x w hHe,
    LeanRelease.ownRefine_withStops finS _ vs (deferL L₀ src) CoinOpen.shaH deferK wk x w hn hlast hst hO hLk hF,
    LeanRelease.liveRefine_withStops finS _ vs (deferL L₀ src) CoinOpen.shaH wk x w hlast hst hO hE,
    LeanRelease.fillRefine_withStops finS _ vs (deferL L₀ src) CoinOpen.shaH deferK wk x w hn.le hlast hst hO hEf,
    LeanRelease.stopRefine_withStops finS _ vs (deferL L₀ src) CoinOpen.shaH deferK wk x w hn.le hst⟩

/-- **The deployed loop's deferred server is zero knowledge per outcome**: at `SessionLoopZKDeployed`'s premises on the
session (`Tables`, `Complete`, `DummyReg`, `Causal`, `SwCausal`, `TapeOK`) and `t > 0`, the deferred server at the
loop's own frames, link and `Finish` with the worker `wk` is `ZKOut` at `DeferRel` with its stop bit public: per stop
bit, an event of its transcript and that stop bit is at most the event at the switched prover's simulator told the bit
(`leakyOut`), plus `8·n_L·δ₁ + 8·t·n_L·δ₁ + 2·3⁻ᵗ + 4·t·3ᵗ·Pr[Fork]`, `SessionLoopZKDeployed`'s error without its
factor `2`. -/
theorem zkOut_loopDeployed (vs : finS.SwVals) (t : ℕ) (ht : 0 < t) (hS : finS.Tables)
    (hV : finS.Complete (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse))
    (hD : finS.DummyReg) (hc : finS.Causal) (hsc : finS.SwCausal vs) (hT : finS.TapeOK)
    (L₀ : LeanRelease Flock.Wit Tp) (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (wk : (deferL L₀ src).toSrv.Worker) :
    ZKOut (fun _ => Tp) (fun _ => (deferL L₀ src).toSrv.Chk) CoinOpen.shaH deferK
      (fun w _ => (deferL L₀ src).toSrv.proverD (deferL L₀ src).fillOf deferK wk w) (DeferRel vs L₀ src wk)
      (stopBit (deferL L₀ src).toSrv.stop deferK)
      (leakyOut (deferSim t (deferL L₀ src)) deferK (deferL L₀ src).toSrv.stop)
      (fun x w _ Vs θ _ => deferEps t (deferL L₀ src) x +
        4 * t * 3 ^ t * (Fork CoinOpen.shaH (((deferS (deferL L₀ src)).swP vs).prover w) Vs θ x).Pr CoinOpen.shaH) := by
  have hRf := LeanRelease.stopsCausal_rfOf finS (deferL L₀ src) progOf freshOf readCoins
  have hZK := (Claims.ZKMal.Flock.zk_deployZk_sw CoinOpen.shaH _root_.Flock.Readable.relation
    (deferS (deferL L₀ src)) vs t ht (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)
    ENNReal.ofReal_ne_top (RefSession.withStops_tables finS _ _ hS) (RefSession.withStops_complete finS _ _ hV)
    (RefSession.withStops_dummyReg finS _ _ hD) (RefSession.withStops_causal finS _ _ hRf hc)
    (RefSession.withStops_swCausal finS _ _ vs hRf hsc) (RefSession.withStops_tapeOK finS _ _ hT)).2
  rw [ENNReal.toReal_ofReal (FlockSoundness.CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash)] at hZK
  intro x w hR Θ Vs θ hk o Ev
  exact ZkServer.zkOut_deferred (deferL L₀ src).toSrv (deferL L₀ src).fillOf CoinOpen.shaH deferK
    ((deferS (deferL L₀ src)).swP vs).prover _ _ _ _ hZK wk x w
    ⟨hR.1, LeanRelease.refD _ _ _ _ wk x w (itemsRefine_deployed vs hsc L₀ src wk x w hR)⟩ Θ Vs θ hk o Ev

end Loop

end Claims.ZKServer
