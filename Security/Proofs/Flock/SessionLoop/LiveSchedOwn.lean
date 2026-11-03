import Proofs.Flock.SessionLoop.RoundPiop

/-!
# `LiveSched` for any worker, at its own histories

`LeanRelease.LiveSched` asks its equation after every live history, but `live` checks the worker's values at each
prefix, never the history's messages: only `ZkServer.OwnHist` ties each message to the deferred server's frame of the
worker's round there. The deferred server meets no other history (`ZkServer.RefDOwn`, `split_deferred`'s `hown`), so the
equation is asked there only, as `FinalRoundEq` asks its own:

* **`Loop.TakesValsAt`**: `Loop.TakesVals` after one history: each word the frame frees in the worker's round is `vs`'s
  value there. `TakesVals` is it after every history (`Loop.takesVals_iff`).
* **`Loop.TakesValsOwn`**: it after every history of the worker's own. A worker whose rounds read the history's messages
  can meet it and not `TakesVals`; a worker that meets `TakesVals` meets it (`Loop.takesValsOwn_of_takesVals`), the honest
  worker `Loop.valsWk honestVals` among them (`Loop.takesValsOwn_honestVals`).
* **`LeanRelease.LiveSchedOwn`**: `LiveSched` with `OwnHist` among its hypotheses, in `FinalRoundEq`'s place; `LiveSched`
  gives it (`LeanRelease.liveSchedOwn_of_liveSched`).
* **`LeanRelease.liveSchedAt_ofLoopLink`**: `LiveSched`'s equation after one history (`LeanRelease.LiveSchedAt`;
  `LiveSched` is it after every history, `LeanRelease.liveSched_iff`), from `TakesValsAt` at that history alone, at
  `SessionLoopZKDeployed`'s premises on the session, the statement and the loop's inputs, where the rows hold at `vs`
  (`Loop.RowsOK`).
* **`LeanRelease.liveSchedOwn_ofLoopLink`**: for every worker `wk` and every `vs` it takes at its own histories
  (`TakesValsOwn`), `LiveSchedOwn` holds: `liveSchedAt_ofLoopLink` at each. `LeanRelease.liveSchedOwn_honestVals`: its
  premises on the worker hold at the honest worker and values.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- **The worker's free words are `vs`'s values after one history** `(m₀, ρ₀) :: ops` at the tape `(ν, t)` and the check
coins `κ`: `TakesVals`'s conclusion there, wherever `LiveRoundEq` reads the history (`0 < ops.length < refK x` openings,
live, whose openings pass under the tape's nonce), the frame takes the stream's coins and the worker's round is rep
stream `2j + r`'s. -/
def TakesValsAt (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) : Prop :=
  ∀ (v : Msg) (m : _root_.Flock.Firewall.Frame.Masker) (out : ByteArray) (s : _root_.Flock.Firewall.Frame.Masker)
    (fr : Array ℕ) (j : Fin (Masked.jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
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

/-- `TakesVals` is `TakesValsAt` after every history. -/
theorem takesVals_iff (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) :
    TakesVals vs L wk x w ↔ ∀ ν t κ m₀ ρ₀ ops, TakesValsAt vs L wk x w ν t κ m₀ ρ₀ ops :=
  Iff.rfl

/-- **The worker's free words are `vs`'s values at its own histories**: `TakesValsAt` after every history each of whose
messages is the deferred server's frame of the worker's round before it (`ZkServer.OwnHist`), the only histories the
deferred server meets. -/
def TakesValsOwn (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    L.toSrv.OwnHist L.fillOf wk x w (ν, t) κ ((m₀, ρ₀) :: ops) → TakesValsAt vs L wk x w ν t κ m₀ ρ₀ ops

theorem takesValsOwn_of_takesVals {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals}
    {L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {wk : L.toSrv.Worker} {x : Msg} {w : Flock.Wit} (h : TakesVals vs L wk x w) : TakesValsOwn vs L wk x w :=
  fun ν t κ m₀ ρ₀ ops _ => h ν t κ m₀ ρ₀ ops

/-- **The honest worker takes the honest values at its own histories**: `valsWk honestVals` at the session loop
(`takes_honestVals_ofLoopLink`). -/
theorem takesValsOwn_honestVals
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) :
    TakesValsOwn (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)
      (valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (LeanRelease.ofLoopLink L₀ src real)) x w :=
  takesValsOwn_of_takesVals (takes_honestVals_ofLoopLink L₀ src real).1

end Loop

namespace LeanRelease

/-- **`LiveSched` at the worker's own histories**: `LiveSched`'s equation after every history it reads at which the loop
schedules its own round `r`, each of whose messages is the deferred server's frame of the worker's round before it
(`ZkServer.OwnHist`), in `FinalRoundEq`'s place. -/
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

theorem liveSchedOwn_of_liveSched {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals}
    {L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray)}
    {wk : (ofLoopLink L₀ src real).toSrv.Worker} {x : Msg} {w : Flock.Wit} (h : LiveSched vs L₀ src real wk x w) :
    LiveSchedOwn vs L₀ src real wk x w :=
  fun ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href _ hv hsc => h ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hv hsc

/-- **`LiveSched`'s equation after one history** `(m₀, ρ₀) :: ops` at the tape `(ν, t)` and the check coins `κ`. -/
def LiveSchedAt (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) : Prop :=
  ∀ (v : Msg) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- `LiveSched` is `LiveSchedAt` after every history. -/
theorem liveSched_iff (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) :
    LiveSched vs L₀ src real wk x w ↔ ∀ ν t κ m₀ ρ₀ ops, LiveSchedAt vs L₀ src real wk x w ν t κ m₀ ρ₀ ops :=
  Iff.rfl

/-- **`LiveSched` after one history, from the worker's free words there**: at `SessionLoopZKDeployed`'s release and
premises (as in `liveSchedOwn_ofLoopLink`), where the rows hold at `vs`, after any history at which the worker's free
words are `vs`'s (`Loop.TakesValsAt`), `LiveSched`'s equation holds there. The release at round `n` reads the worker's
round `n`; the frame's state before it reads the history only through `live` and the coins. -/
theorem liveSchedAt_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (hRows : Loop.RowsOK vs x w)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) :
    LiveSchedAt vs L₀ src real wk x w ν t κ m₀ ρ₀ ops := by
  sorry

/-- **`LiveSched` for any worker, at its own histories**: at `SessionLoopZKDeployed`'s release (`ofLoopLink` at the
loop's own tables, `real = Loop.realOf src`), at its premises on the session (`Tables`, `TapeOK`), the statement (in the
masked verifier's scope, under the relation, served by the loop) and the loop's inputs (`Loop.Src`), every worker `wk`
whose free words are `vs`'s values at its own histories (`Loop.TakesValsOwn`), where the rows hold at `vs`
(`Loop.RowsOK`), meets `LiveSchedOwn` at `vs`: `liveSchedAt_ofLoopLink` at each own history. -/
theorem liveSchedOwn_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (hRows : Loop.RowsOK vs x w) (hTV : Loop.TakesValsOwn vs (ofLoopLink L₀ src real) wk x w) :
    LiveSchedOwn vs L₀ src real wk x w :=
  fun ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hown hv hsc =>
    liveSchedAt_ofLoopLink vs L₀ src real hreal wk hTab hTape hR hSv hSc hS hRows ν t κ m₀ ρ₀ ops
      (hTV ν t κ m₀ ρ₀ ops hown) v r hpos hlt hl hok href hv hsc

/-- **At the honest worker and values**: `liveSchedOwn_ofLoopLink` at `valsWk honestVals`, whose premises on the worker
hold there (`Loop.takesValsOwn_honestVals`, `Loop.rowsOK_honestVals`). -/
theorem liveSchedOwn_honestVals
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) {x : Msg} {w : Flock.Wit}
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w) :
    LiveSchedOwn (CoinLayer.session Loop.loopNr Masked.dataE).honestVals L₀ src real
      (Loop.valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals (ofLoopLink L₀ src real)) x w :=
  liveSchedOwn_ofLoopLink _ L₀ src real hreal _ hTab hTape hR hSv hSc hS (Loop.rowsOK_honestVals hR)
    (Loop.takesValsOwn_honestVals L₀ src real x w)

end LeanRelease

end Claims.ZKServer
