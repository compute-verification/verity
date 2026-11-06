module

public import Proofs.Flock.ZkAny.RelayLoopStart
public import Proofs.Flock.ZkAny.RelayWorker

/-!
# The relay's model frames at the deployed worker

`relayPre_run` at the release over `LeanRelease.withoutCross`, whose `cross` passes every value as Rust's checks are the
worker's, and at the deployed worker (`LeanRelease.deployedWk`): `CrossPasses` holds by definition
(`LoopModel.Deployed.crossPasses`), and the deployed worker answers the link and the `Finish` with the firewall's own
message (`LoopModel.link_deployed`, `LoopModel.finish_deployed`).

* **`relayPre_deployed`**: so the relay's model frames are the model's messages, encoded, with no premise on Rust's
  checks or on the worker's link and `Finish`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V SessionSpec Stmt)

/-- **The relay's model frames are the model's messages, encoded, at the deployed worker**: `relayPre_run` at the
release over `L₁.withoutCross`, where the model's worker is the deployed worker at a device `dv` (`hdw`). -/
theorem relayPre_deployed
    {L₁ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {fin : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M (LeanRelease.withFiles (LeanRelease.withLink
      (LeanRelease.withInner ZKMal.Masked.dataE L₁.withoutCross (Loop.realOf src) fin) src) src) (Loop.realOf src) s0
      spec sts)
    (hsrc : M.src = src M.x M.w M.ρ) (hfin : M.fin = fin M.x M.w M.ρ) (hS : Loop.Level0Src src M.x M.w)
    (hlive : ZKMal.Masked.Live M.x) {Θ : Type} {H : Msg → Msg}
    {Vs : VStar Θ} {θ : Θ}
    (hτ : M.τ = (runStar H (M.L.toSrv.proverD M.L.fillOf M.nN M.wk M.w) Vs M.x (M.ρ, M.κ) θ).msgs)
    (hk : 0 < Vs.k M.x) {dv : M.L.toSrv.Worker} (hdw : M.wk = M.L.deployedWk dv)
    {rv : List (Msg × Msg) → Option Msg} (hwd : WorkerDeterministic M rv)
    {sid : Nat} {nonce : ByteArray} (hn : nonce = ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩)
    {ins : List (ByteArray × Env)} (hs : ∀ p ∈ ins, p.2.setup = .ok (spec, sts))
    (hb : ∀ p ∈ ins, ∀ why, (p.2.barrier (some why)).isSome) {σ : LoopAt} (hc : σ.c = 0)
    (hnew : ∀ p ∈ σ.R.reps, p.sid ≠ sid) {aud : List ByteArray → Option ByteArray}
    (hR : RelayServes M rv aud sid spec sts nonce σ ins)
    (hA : AuditorAnswers M (modelSlots {} {} (pick true {} {} ins))) :
    RelayPre (relayTrace true ins) (encOf spec sts M.msgs) :=
  relayPre_run hD hsrc hfin hS hlive hD.crossPasses hτ hk (LoopModel.link_deployed hdw)
    (LoopModel.finish_deployed hdw) hwd hn hs hb hc hnew hR hA

end Claims.ZKServer.Relay
