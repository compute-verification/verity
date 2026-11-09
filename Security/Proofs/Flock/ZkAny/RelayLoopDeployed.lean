module

public import Proofs.Flock.ZkAny.RelayLoopRep
public import Proofs.Flock.ZkAny.RelayLoopL0
public import Proofs.Flock.SessionLoop
import Proofs.Flock.Claims.ZK.Reference

/-!
# `LoopJ` kept at a frame round of the deployed loop

* `LoopModel.Deployed.helloChk`, `LoopModel.Deployed.req`, `LoopModel.Deployed.frameAnswer_one`: the deployed loop's
  release checks the setup's `Hello` at the transport's nonce, requests the loop's openings, and takes the coins after
  the `Hello`.
* **`loopKept_round_deployed`**: `loopKept_round` at the loop's own tables on `src` (`Loop.realOf src`), its `hcur`
  being `repJ_round` with the loop's tables `RealTabs` (`Loop.realTabs_realOf`) and the model's samples the tables the
  loop's level 0 give (`realAt_h0`), given `RoundFacts` at each request at the slot as the loop answers it (`hA`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed helloOf)
open _root_.Flock (V SessionSpec Stmt)

/-- **The deployed loop's release checks the setup's `Hello` at the transport's nonce** (`Loop.helloChkAt`). -/
theorem LoopModel.Deployed.helloChk {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    {nonce : ByteArray} (hn : nonce = ⟨(ZKMal.CoinLayer.nb M.ρ.1).toArray⟩) :
    ∀ hl, helloOf spec sts nonce = some hl → M.L.helloChk M.x M.ρ = .ok hl := by
  intro hl hh
  subst hn
  rw [hD.rel]
  show Loop.helloChkAt M.x M.ρ.1 = .ok hl
  unfold Loop.helloChkAt
  rw [Claims.ZKMal.sessOf_flock hD.setup]
  dsimp only
  rw [hh] <;> rfl

/-- **The deployed loop's release requests the loop's openings** (`Loop.reqOf`). -/
theorem LoopModel.Deployed.req {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts) :
    ∀ j, M.L.req M.x j = Loop.reqOf M.x j := fun j => by
  rw [hD.rel]
  rfl

open Claims.ZKMal in
/-- **The deployed loop's frames take the coins after the `Hello`**: the round after it is rep 0's first, at the
frame's start (`Loop.freshOf` of no coins). -/
theorem LoopModel.Deployed.frameAnswer_one {T : ZKMal.CoinLayer.TableData}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} (hD : LoopModel.Deployed M L₀ real s0 spec sts)
    (hlt : 0 < M.τ.length) : (M.L.frameAnswer M.x M.ρ (M.hist 1)).isOk = true := by
  have h1 : M.hist 1 = [((M.τ[0]'hlt).1, msgOf (M.τ[0]'hlt).2)] := M.hist_snoc hlt
  rw [h1, hD.rel]
  show (Loop.freshOf M.x []).isOk = true
  unfold Loop.freshOf
  rw [hD.setup] <;> rfl

/-- **`LoopJ` is kept at a frame round of the deployed loop**: at the loop's own tables on `src`
(`LoopModel.Deployed` at `Loop.realOf src`), the loop handed `src`'s bytes (`hsrc`) whose level 0 are sampled from the
tape (`Loop.Level0Src`) at a statement the masked session proves (`hlive`), where `RoundFacts` holds at each request at
the slot as the loop answers it (`hA`, `roundFacts_loop`'s): `loopKept_round`, the round's own rep one round on by
`repJ_round`, the loop's tables being `RealTabs` (`Loop.realTabs_realOf`) and the model's samples at each rep the table
its level 0 gives once the session's entry is the setup's and its level 0 the ones the loop drew (`realAt_h0`). -/
theorem loopKept_round_deployed {M : LoopModel Claims.Flock.Wit
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
    (hS : Loop.Level0Src src M.x M.w) (hlive : ZKMal.Masked.Live M.x)
    {sid : Nat} {nonce : ByteArray} {stream : String} {n : Nat}
    (hA : ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat) (dom : ByteArray)
      (v : Claims.Msg), slot st pl = some (.round stream n) → LoopFacts sid spec seen st pl σ →
      LoopJ M sid spec sts seen st σ → σ.c < M.τ.length → M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
      InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) →
      RoundFacts M sid spec seen stream n i r dom v σ) :
    LoopKept M sid spec sts nonce (LoopJ M sid spec sts) (.round stream n) :=
  loopKept_round hD.names hD.setup hD.req hA
    (repJ_round ZKMal.Masked.dataE hD (Loop.realTabs_realOf src M.x M.w M.ρ hD.setup)
      fun _ hK hB => realAt_h0 hsrc hD.setup hK hD.names hB hS hlive)

end Claims.ZKServer.Relay
