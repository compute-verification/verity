module

public import Proofs.Flock.ZkAny.RelayLoopDeployed
public import Proofs.Flock.ZkAny.RelayLoopFinish

/-!
# `EnvPre` at the deployed loop

`envPre_loop` gives `EnvPre` from `LoopP` given `SlotCase` at the `Hello`, the `Commit`, each frame round and the
`Finish`, and `J` kept at every other slot. Here they are at the invariant `LoopJ` on the loop's own tables:

* `LoopModel.Deployed.last_eq`: the deployed model's last round is two past the setup's plan.
* **`envPre_deployed`**: where each of the run's messages is the deferred release's, its first the `Hello` at the drawn
  nonce, the worker's link the firewall's own, and `RoundFacts` holds at each request at a frame round's slot as the
  loop answers it (`hRF`), `EnvPre` holds wherever `LoopP` at `LoopJ` does.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed planOf)
open _root_.Flock (V SessionSpec Stmt)

/-- **The deployed model's last round is two past the setup's plan**: the `Hello`, the link and each of the plan's
rounds, the `Finish` the last. -/
theorem LoopModel.Deployed.last_eq {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts) :
    M.L.last M.x = (planOf spec sts).size + 2 := by
  rw [hD.rel]
  show Loop.lastOf M.x = _
  rw [Loop.lastOf_eq, Loop.refK_of_setup hD.setup, Loop.rounds_eq hD.sizes, ← Loop.plan_size]
  omega

/-- **`EnvPre` at the deployed loop**: at the loop's own tables on `src` (`LoopModel.Deployed` at `Loop.realOf src`),
the loop handed `src`'s bytes (`hsrc`) whose level 0 are drawn from the tape (`Loop.Level0Src`) at a statement the
masked session proves (`hlive`), where each of the run's messages is the deferred release's (`LoopModel.Own`), its
first the `Hello` at the nonce the relay drew (`h0`), the worker's link the firewall's own (`hwk`), and `RoundFacts`
holds at each request at a frame round's slot as the loop answers it (`hRF`), `EnvPre` holds wherever `LoopP` at
`LoopJ` does: `LoopJ` is kept at every slot that reads no model message (`loopKept_ctl_all`), at the `Hello`, at the
`Commit` (below the `Finish`, the last round two past the plan) and at each frame round (`loopKept_round_deployed`),
and at the `Finish` the session's stop is whether an earlier round failed (`LoopJ.stop`). -/
theorem envPre_deployed {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M L₀ (Loop.realOf src) s0 spec sts) (hsrc : M.src = src M.x M.w M.ρ)
    (hS : Loop.Level0Src src M.x M.w) (hlive : ZKMal.Masked.Live M.x) (hOwn : M.Own) {sid : Nat}
    {nonce : ByteArray} (h0 : M.msgs.headD [] = _root_.Flock.Firewall.Hello.msg spec nonce)
    (hwk : M.wk M.x M.w M.ρ (M.hist 1) 0 = some (M.L.ownMsg M.x M.w M.ρ (M.hist 1), false))
    (hRF : ∀ {stream : String} {n : Nat}, ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt)
      (i r : Nat) (dom : ByteArray) (v : Claims.Msg), slot st pl = some (.round stream n) →
      LoopFacts sid spec seen st pl σ → LoopJ M sid spec sts seen st σ → σ.c < M.τ.length →
      M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
      InnerSrc M sid i r σ → (∃ o, e.round = .ok o) → RoundFacts M sid spec seen stream n i r dom v σ) :
    ∀ (es : List Env) (seen : List String) (st : State) (pl : Placed) (ms : List Claims.Msg),
      LoopP M (LoopJ M sid spec sts) sid spec sts nonce seen st pl es ms →
        EnvPre spec seen (schedSteps true st pl es) ms := by
  have hlast := hD.last_eq
  have hnext := hD.next
  exact envPre_loop loopKept_ctl_all (slotCase_hello h0 loopKept_hello)
    (slotCase_commit hOwn (by omega) (loopKept_commit hD.setup (by omega) hwk))
    (fun _ _ => slotCase_round hOwn hRF (loopKept_round_deployed hD hsrc hS hlive hRF))
    (slotCase_finish hOwn fun _ _ _ _ hsl _ hJ => by rw [hJ.stop, slot_finish hsl, Bool.true_and])

end Claims.ZKServer.Relay
