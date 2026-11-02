import Proofs.Flock.Claims.ZK.OwnHistAny
import Proofs.Flock.SessionLoop.TakenVals
import Proofs.Flock.SessionLoop.FinishIsLoop

/-!
# The session loop's live rounds at the worker's own histories

`LiveRoundEq` at the session loop (`LeanRelease.liveRoundEq_ofLoopLink`, from `LiveSched` and `LiveInner`)
holds for the honest worker. Its inner rounds take `Loop.TakesFinalC` and `Loop.TakesTau`, premises on the worker's
own rounds, which no theorem gives at every live history: `live` reads the worker's values at each prefix, never the
history's messages, and the finish reads the values the frame took off the history's released rounds (`seenRounds`,
`maskedOf`, `Loop.valsOf`). At the histories the deferred release meets, its own (`ZkServer.OwnHist`), those messages
are the frame's releases of the worker's rounds. So each live equation is asked there only, as `FinalRoundEq` already
is:

* `Loop.TakesValsOwn`: `Loop.TakesVals` at the own histories (`Loop.TakesVals.own`).
* `LeanRelease.LiveSchedOwn`, `LeanRelease.LiveInnerOwn`: `LiveSched` and `LiveInner` at the own histories
  (`LiveSched.own`, `LiveInner.own`).
* **`LeanRelease.liveRoundEqOwn_ofLoopLink`**: at `SessionLoopZKDeployed`'s release, for every worker, the two give
  `LiveRoundEqOwn`.
* **`Loop.finishIs_own`**: `FinishIs` at `SessionLoopZKDeployed`'s premises from `LiveRoundEqOwn` in place of
  `LiveRoundEq`: the `Finish` reads each frame round on the own history it ends (`msg_of_liveRoundEq` at its prefixes).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

/-- **The worker's free words are `vs`'s values at its own histories**: `TakesVals`'s equation after every history it
reads whose messages are the deferred release's own (`ZkServer.OwnHist`). -/
def TakesValsOwn (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (m : _root_.Flock.Firewall.Frame.Masker)
    (out : ByteArray) (s : _root_.Flock.Firewall.Frame.Masker) (fr : Array ℕ) (j : Fin (jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m →
    _root_.Flock.Firewall.Frame.roundFree (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false } ⟨v.toArray⟩
      (L.coins x (ops.length + 1)) = .ok (out, s, fr) →
    L.rep x (ops.length + 1) = 2 * j.val + r.val →
    ∀ k < fr.size,
      (_root_.Flock.Firewall.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]! =
        vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
          (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
          (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]!

/-- `TakesVals` gives its equation at the own histories. -/
theorem TakesVals.own {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals}
    {L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {wk : L.toSrv.Worker} {x : Msg} {w : Flock.Wit} (hT : TakesVals vs L wk x w) : TakesValsOwn vs L wk x w :=
  fun ν t κ m₀ ρ₀ ops v m out s fr j r h0 hlt hl hok _ hv ha hf hr =>
    hT ν t κ m₀ ρ₀ ops v m out s fr j r h0 hlt hl hok hv ha hf hr

end Loop

namespace LeanRelease

/-- **`LiveSched` at the worker's own histories**: its equation after every history it reads whose messages are the
deferred release's own (`ZkServer.OwnHist`, placed as `FinalRoundEq` places it). -/
def LiveSchedOwn (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    (ofLoopLink L₀ src real).toSrv.OwnHist (ofLoopLink L₀ src real).fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`LiveInner` at the worker's own histories**: its equation after every history it reads whose messages are the
deferred release's own (`ZkServer.OwnHist`, placed as `FinalRoundEq` places it). -/
def LiveInnerOwn (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    (ofLoopLink L₀ src real).toSrv.OwnHist (ofLoopLink L₀ src real).fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

section Own

variable {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals}
  {L₀ : LeanRelease Flock.Wit
    ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
  {src : Msg → Flock.Wit →
    (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
    ℕ → V (Array F128 × ByteArray × Array ByteArray)}
  {real : Msg → Flock.Wit →
    (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
    ℕ → V (Tab × Array ByteArray)}
  {wk : (ofLoopLink L₀ src real).toSrv.Worker} {x : Msg} {w : Flock.Wit}

/-- `LiveSched` gives its equation at the own histories. -/
theorem LiveSched.own (hS : LiveSched vs L₀ src real wk x w) : LiveSchedOwn vs L₀ src real wk x w :=
  fun ν t κ m₀ ρ₀ ops v r h0 hlt hl hok href _ hv hsc => hS ν t κ m₀ ρ₀ ops v r h0 hlt hl hok href hv hsc

/-- `LiveInner` gives its equation at the own histories. -/
theorem LiveInner.own (hI : LiveInner vs L₀ src real wk x w) : LiveInnerOwn vs L₀ src real wk x w :=
  fun ν t κ m₀ ρ₀ ops v h0 hlt hl hok href _ hv hin => hI ν t κ m₀ ρ₀ ops v h0 hlt hl hok href hv hin

end Own

/-- **`LiveRoundEq` for every worker at its own histories, from the scheduled rounds and the inner proof's**: at
`SessionLoopZKDeployed`'s release, wherever the loop schedules its own round (`LiveSchedOwn`) and at the inner proof's
rounds (`LiveInnerOwn`), each at the worker's own histories, at every worker `wk`. -/
theorem liveRoundEqOwn_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hS : LiveSchedOwn vs L₀ src real wk x w) (hI : LiveInnerOwn vs L₀ src real wk x w) :
    LiveRoundEqOwn (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w := by
  intro ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hown hv
  rcases hsc : Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) with _ | r
  · exact hI ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hown hv
      ((Loop.schedOf_eq_none_iff (by omega) (by simpa using hpos)).mp hsc)
  · exact hS ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hown hv hsc

end LeanRelease

namespace Loop

/-- **`FinishIs` at `SessionLoopZKDeployed`'s premises from the live rounds at the worker's own histories**: `finishIs`
with `LiveRoundEqOwn` in place of `LiveRoundEq`. The `Finish` reads each frame round on the own history it ends, and
each prefix of an own history is own (`ZkServer.OwnHist.take`). -/
theorem finishIs_own (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w) (hSrc : Loop.Src L₀ src x w)
    (hE : LeanRelease.LiveRoundEqOwn finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) :
    LeanRelease.FinishIs finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w := by
  sorry

end Loop

end Claims.ZKServer
