module

public import Proofs.Flock.SessionLoop

/-!
# `SessionLoopZKDeployed` at the loop's own inner rounds

Past an opened rep's scheduled rounds the loop's frame releases the inner proof's rounds itself
(`LeanRelease.withInner`), and there `LoopInnerRound0` holds at every finish (`LoopInnerRound0Own`). At that release
`SessionLoopZKDeployed` takes no `LoopInnerRound0`:

* `Flock.Guarantees.SessionLoopZKDeployedInner`: `SessionLoopZKDeployed`'s statement at the release
  `withInner Masked.dataE L₁ real fin`, at every `L₁` and finish rounds `fin`, without `LoopInnerRound0`.
* `Flock.SecurityProofs.SessionLoopZKDeployedInner`: `SessionLoopZKDeployed` there, its `LoopInnerRound0` from
  `LoopInnerRound0Own`.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

/-- **C-Flock's zero knowledge at the session loop's own frames and inner rounds**: `SessionLoopZKDeployed`'s statement
at the release with the loop's own rounds of the inner proof (`withInner Masked.dataE L₁ real fin`), at every `L₁` and
finish rounds `fin`, without its premise `LoopInnerRound0`, which holds there. -/
def _root_.Flock.Guarantees.SessionLoopZKDeployedInner : Prop :=
  ∀ (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (t : ℕ), 0 < t →
    (CoinLayer.session Loop.loopNr Masked.dataE).SwCausalOn (Loop.SwOk Masked.dataE) vs →
    ∀ (L₁ : LeanRelease Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (fin : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        List (Msg × Msg) → Array ByteArray)
      (L₀ : LeanRelease Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → V (Array F128 × ByteArray × Array ByteArray))
      (real : Msg → Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)),
    real = Loop.realOf src → L₀ = LeanRelease.withInner Masked.dataE L₁ real fin →
    ∀ (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit),
    Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w → Loop.Serves x →
    Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x → Loop.Src L₀ src x w →
    LeanRelease.LiveRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w →
    LeanRelease.FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (LeanRelease.ofLoopLink L₀ src real) CoinOpen.shaH wk x w →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ),
    Vs.k x = (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop),
    (LeanRelease.ofLoopLink L₀ src real).ProgChecks x w ∧
    Pr (fun ω : ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk =>
        Ev ((LeanRelease.ofLoopLink L₀ src real).toSrv.timed x
          (runStar CoinOpen.shaH ((LeanRelease.ofLoopLink L₀ src real).toSrv.proverD (LeanRelease.ofLoopLink L₀ src real).fillOf
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1) wk w) Vs x ω θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoopLink L₀ src real).toSrv.stop).C x =>
          Ev ((LeanRelease.ofLoopLink L₀ src real).toSrv.timed x ((leaky1 ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1)
            (LeanRelease.ofLoopLink L₀ src real).toSrv.stop).out CoinOpen.shaH Vs θ x c))) +
          (8 * ((CoinLayer.session Loop.loopNr Masked.dataE).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE)
              (LeanRelease.ofLoopLink L₀ src real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoopLink L₀ src real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
            8 * t * ((CoinLayer.session Loop.loopNr Masked.dataE).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE)
              (LeanRelease.ofLoopLink L₀ src real) Loop.progOf Loop.freshOf Loop.readCoins)
              (LeanRelease.ofLoopLink L₀ src real).stop).nLeaves x *
              FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t *
            (Fork CoinOpen.shaH ((((CoinLayer.session Loop.loopNr Masked.dataE).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (LeanRelease.ofLoopLink L₀ src real) Loop.progOf
                  Loop.freshOf Loop.readCoins) (LeanRelease.ofLoopLink L₀ src real).stop).swP vs).prover w) Vs θ x).Pr
              CoinOpen.shaH)

end Claims.ZKServer

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer

/-- **`SessionLoopZKDeployed` at the loop's own inner rounds**: its `LoopInnerRound0` at
`withInner Masked.dataE L₁ real fin` is `LoopInnerRound0Own`'s. -/
theorem SessionLoopZKDeployedInner : Flock.Guarantees.SessionLoopZKDeployedInner := by
  intro vs t ht hsc L₁ fin L₀ src real hreal hL₀ wk x w hR hServe hScope hSrc hE hEf Θ Vs θ hk Ev
  subst hL₀
  exact SessionLoopZKDeployed vs t ht hsc _ src real hreal wk x w hR hServe
    (LoopInnerRound0Own Masked.dataE L₁ real fin x w) hScope hSrc hE hEf Vs θ hk Ev

end Flock.SecurityProofs
