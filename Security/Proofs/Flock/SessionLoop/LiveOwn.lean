import Proofs.Flock.Claims.ZK.OwnHistAny
import Proofs.Flock.SessionLoop.LiveSchedOwn
import Proofs.Flock.SessionLoop.LiveInnerOwn
import Proofs.Flock.SessionLoop.FinishIsLoop

/-!
# `LiveRoundEq` for any worker, at its own histories

`LeanRelease.LiveRoundEq` asks its equation after every live history, but `live` checks the worker's values at each
prefix, never the history's messages, and the loop's finish reads the values the frame took off the history's released
rounds (`seenRounds`, `maskedOf`, `Loop.valsOf`). At the histories the deferred release meets, its own
(`ZkServer.OwnHist`), those messages are the frame's releases of the worker's rounds, so the equation is asked there
only (`LeanRelease.LiveRoundEqOwn`), as `FinalRoundEq` asks its own:

* **`LeanRelease.liveRoundEqOwn_ofLoopLink`**: at `SessionLoopZKDeployed`'s release, for every worker, the scheduled
  rounds' equation (`LiveSchedOwn`) and the inner proof's (`LiveInnerOwn`) give `LiveRoundEqOwn`.
* **`LeanRelease.liveRoundEqOwn_withInner`**: at that release with the inner proof's rounds the loop's finish's
  (`withInner` at `Loop.finOf`), at `SessionLoopZKDeployed`'s premises on the session, the statement and the loop's
  inputs, `LiveRoundEqOwn` holds at every worker taking some admissible `vs`'s values at its own histories
  (`Loop.TakesValsOwn`), `vs` being admissible when it is causal (`SwCausal`), its round-1c values fix its `final_c`
  (`Loop.FinalCFixed`) and its rows hold (`Loop.RowsOK`). It takes neither `Loop.TakesFinalC` nor `Loop.TakesTau`. A
  worker taking no admissible `vs`'s values is outside it. `liveRoundEqOwn_honestVals`: its premises on the worker and
  the values hold at the honest worker `Loop.valsWk honestVals`.

`Loop.finishIs` takes `LiveRoundEqOwn`: the `Finish` reads each frame round on the history it ends, a prefix of the own
history `FinishIs` is asked on, which is own (`ZkServer.OwnHist.take`, `msg_of_liveRoundEq`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace LeanRelease

/-- **`LiveRoundEq` for every worker at its own histories, from the scheduled rounds and the inner proof's**: at
`SessionLoopZKDeployed`'s release, wherever the loop schedules its own round (`LiveSchedOwn`) and at the inner proof's
rounds (`LiveInnerOwn`), each at the worker's own histories, at every worker `wk`. -/
theorem liveRoundEqOwn_ofLoopLink (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hS : LiveSchedOwn vs L₀ src real wk x w) (hI : LiveInnerOwn vs L₀ src real wk x w) :
    LiveRoundEqOwn Loop.finS vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w := by
  intro ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hown hv
  rcases hsc : Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) with _ | r
  · exact hI ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hown hv
      ((Loop.schedOf_eq_none_iff (by omega) (by simpa using hpos)).mp hsc)
  · exact hS ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hown hv hsc

/-- **`LiveRoundEq` for every worker, at its own histories**: at `SessionLoopZKDeployed`'s release with the inner
proof's rounds the loop's finish's (`withInner` at `Loop.finOf`, at `L₁`'s frame draws), on the loop's own tables
(`Loop.realOf`) and inputs (`Loop.Src`), at a statement in the masked scope with a witness that the loop serves,
`LiveRoundEqOwn` holds at every worker taking some admissible `vs`'s values at its own histories (`Loop.TakesValsOwn`):
one that is causal (`SwCausal`), whose `final_c` its round-1c values fix (`Loop.FinalCFixed`) and whose rows hold
(`Loop.RowsOK`). `liveRoundEqOwn_ofLoopLink` at `liveSchedOwn_ofLoopLink` and `liveInnerOwn_withInner`. -/
theorem liveRoundEqOwn_withInner (vs : Loop.finS.SwVals) (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (x : Msg) (w : Flock.Wit) (hTab : Loop.finS.Tables) (hTape : Loop.finS.TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (hcaus : Loop.finS.SwCausal vs) (hC : Loop.FinalCFixed vs x w) (hrows : Loop.RowsOK vs x w)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hV : Loop.TakesValsOwn vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real)
      wk x w) :
    LiveRoundEqOwn Loop.finS vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real)
      CoinOpen.shaH wk x w :=
  liveRoundEqOwn_ofLoopLink vs _ src real wk x w
    (liveSchedOwn_ofLoopLink vs _ src real hreal wk hTab hTape hR hSv hsc hSrc hrows hV)
    (liveInnerOwn_withInner vs L₁ src real hreal x w hR hSv hsc hSrc hcaus wk hV hC hrows)

/-- **At the honest worker and values**: `liveRoundEqOwn_withInner` at `Loop.valsWk honestVals`, whose premises on the
worker and the values hold there (`Loop.takesValsOwn_honestVals`, `RefSession.swCausal_vals`,
`Loop.finalCFixed_honestVals`, `Loop.rowsOK_honestVals`). -/
theorem liveRoundEqOwn_honestVals (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (x : Msg) (w : Flock.Wit) (hTab : Loop.finS.Tables) (hTape : Loop.finS.TapeOK) (hcz : Loop.finS.Causal)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w) :
    LiveRoundEqOwn Loop.finS Loop.finS.honestVals
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) CoinOpen.shaH
      (Loop.valsWk Loop.finS.honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real)) x w :=
  liveRoundEqOwn_withInner _ L₁ src real hreal x w hTab hTape hR hSv hsc hSrc (RefSession.swCausal_vals _ hcz)
    (Loop.finalCFixed_honestVals x w) (Loop.rowsOK_honestVals hR) _ (Loop.takesValsOwn_honestVals _ src real x w)

end LeanRelease

end Claims.ZKServer
