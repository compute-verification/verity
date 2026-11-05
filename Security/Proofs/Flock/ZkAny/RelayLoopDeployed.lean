module

public import Proofs.Flock.ZkAny.RelayLoopRep
public import Proofs.Flock.ZkAny.RelayLoopL0
public import Proofs.Flock.SessionLoop

/-!
# `LoopJ` kept at a frame round of the deployed loop

* **`loopKept_round_deployed`**: `loopKept_round` at the loop's own tables on `src` (`Loop.realOf src`), its `hcur`
  being `repJ_round` with the loop's tables `RealTabs` (`Loop.realTabs_realOf`) and the model's draws the tables the
  loop's level 0 give (`realAt_h0`), given `RoundFacts` at each request at the slot as the loop answers it (`hA`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed)
open _root_.Flock (V SessionSpec Stmt)

/-- **`LoopJ` is kept at a frame round of the deployed loop**: at the loop's own tables on `src`
(`LoopModel.Deployed` at `Loop.realOf src`), the loop handed `src`'s bytes (`hsrc`) whose level 0 are drawn from the
tape (`Loop.Level0Src`) at a statement the masked session proves (`hlive`), where `RoundFacts` holds at each request at
the slot as the loop answers it (`hA`, `roundFacts_loop`'s): `loopKept_round`, the round's own rep one round on by
`repJ_round`, the loop's tables being `RealTabs` (`Loop.realTabs_realOf`) and the model's draws at each rep the table
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
      (∃ o, e.round = .ok o) → RoundFacts M sid spec seen stream n i r dom v σ) :
    LoopKept M sid spec sts nonce (LoopJ M sid spec sts) (.round stream n) :=
  loopKept_round hD.names hA (repJ_round ZKMal.Masked.dataE hD (Loop.realTabs_realOf src M.x M.w M.ρ hD.setup)
    fun _ hK hB => realAt_h0 hsrc hD.setup hK hD.names hB hS hlive)

end Claims.ZKServer.Relay
