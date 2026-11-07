module

public import Proofs.Flock.Claims.ZK.OwnHistAny
public import Proofs.Flock.SessionLoop.LiveSchedOwn
public import Proofs.Flock.SessionLoop.LiveInnerOwn
public import Proofs.Flock.SessionLoop.FinishIsLoop
public import Proofs.Flock.SessionLoop.VsOfTakes
public import Proofs.Flock.SessionLoop.SwCausalExe

/-!
# `LiveRoundEq` for any worker, at its own histories

`LeanRelease.LiveRoundEq` asks its equation after every live history, but `live` checks the worker's values at each
prefix, never the history's messages, and the loop's finish reads the values the frame took off the history's released
rounds (`seenRounds`, `maskedOf`, `Loop.valsOf`). At the histories the deferred release meets, its own
(`ZkServer.OwnHist`), those messages are the frame's releases of the worker's rounds, so the equation is asked there
only (`LeanRelease.LiveRoundEqOwn`), as `FinalRoundEq` asks its own:

* **`LeanRelease.liveRoundEqOwn_ofLoopLink`**: at `SessionLoopZKDeployed`'s release, for every worker, the scheduled
  rounds' equation (`LiveSchedOwn`) and the inner proof's (`LiveInnerOwn`) give `LiveRoundEqOwn` at a statement that
  is no V\* part's.
* **`LeanRelease.liveRoundEqOwn_withInner`**: at that release with the inner proof's rounds the loop's finish's
  (`withInner` at `Loop.finOf`), at `SessionLoopZKDeployed`'s premises on the session, the statement and the loop's
  inputs, `LiveRoundEqOwn` holds at every worker taking some admissible `vs`'s values at its own histories
  (`Loop.TakesValsOwn`), `vs` being admissible when it is causal on the coins `Loop.SwOk` admits (`SwCausalOn`), its
  round-1c values fix its `final_c` (`Loop.FinalCFixed`), its level-0 words are the ZK server's (`Loop.L0Fixed`) and its
  rows hold (`Loop.RowsOK`). It takes neither `Loop.TakesFinalC` nor `Loop.TakesTau`. A worker taking no admissible
  `vs`'s values is outside it. `liveRoundEqOwn_honestVals`: its premises on the worker and the values hold at the
  honest worker `Loop.valsWk honestVals`.
* **`LeanRelease.liveRoundEqOwn_vsOf`**: they hold at `Loop.vsOf bx` and the worker `Loop.phased bx post`, for every
  algebraic-phase worker `bx` and every `post`, wherever `bx`'s served values' rows hold (`Loop.vsServed`).

`Loop.finishIsOn` takes `LiveRoundEqOwn`: the `Finish` reads each frame round on the history it ends, a prefix of the own
history `FinishIs` is asked on, which is own (`ZkServer.OwnHist.take`, `msg_of_liveRoundEq`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace LeanRelease

/-- **`LiveRoundEq` for every worker at its own histories, from the scheduled rounds and the inner proof's**: at
`SessionLoopZKDeployed`'s release, wherever the loop schedules its own round (`LiveSchedOwn`) and at the inner proof's
rounds (`LiveInnerOwn`), each at the worker's own histories, at every worker `wk`, at a statement that is no V\*
part's. -/
theorem liveRoundEqOwn_ofLoopLink (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hd : RecDeploy.decodeRec x = none)
    (hS : LiveSchedOwn vs L₀ src real wk x w) (hI : LiveInnerOwn vs L₀ src real wk x w) :
    LiveRoundEqOwn Loop.finS vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w := by
  intro ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hown hv
  rcases hsc : Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) with _ | r
  · exact hI ν t κ m₀ ρ₀ ops v hpos hlt hl hok href hown hv
      ((Loop.schedOf_eq_none_iff hd (by omega) (by simpa using hpos)).mp hsc)
  · exact hS ν t κ m₀ ρ₀ ops v r hpos hlt hl hok href hown hv hsc

/-- **`LiveRoundEq` for every worker, at its own histories**: at `SessionLoopZKDeployed`'s release with the inner
proof's rounds the loop's finish's (`withInner` at `Loop.finOf`, at `L₁`'s frame samples), on the loop's own tables
(`Loop.realOf`) and inputs (`Loop.Src`), at a statement in the masked scope with a witness that the loop serves,
`LiveRoundEqOwn` holds at every worker taking some admissible `vs`'s values at its own histories (`Loop.TakesValsOwn`):
one causal on the coins `Loop.SwOk` admits (`SwCausalOn`), whose `final_c` its round-1c values fix
(`Loop.FinalCFixed`), whose level-0 words are
the ZK server's (`Loop.L0Fixed`) and whose rows hold (`Loop.RowsOK`). `liveRoundEqOwn_ofLoopLink` at
`liveSchedOwn_ofLoopLink` and `liveInnerOwn_withInner`. -/
theorem liveRoundEqOwn_withInner (vs : Loop.finS.SwVals) (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (x : Msg) (w : Flock.Wit) (hTab : Loop.finS.Tables) (hTape : Loop.finS.TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (hcaus : Loop.finS.SwCausalOn (Loop.SwOk Masked.dataE) vs) (hC : Loop.FinalCFixed vs x w)
    (hL0 : Loop.L0Fixed vs x w) (hrows : Loop.RowsOK vs x w)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hV : Loop.TakesValsOwn vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real)
      wk x w) :
    LiveRoundEqOwn Loop.finS vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real)
      CoinOpen.shaH wk x w :=
  liveRoundEqOwn_ofLoopLink vs _ src real wk x w (by obtain ⟨_, hp, -⟩ := hsc; exact decodeRec_none_of_parse hp)
    (liveSchedOwn_ofLoopLink vs _ src real hreal wk hTab hTape hR hSv hsc hSrc hrows hL0 hV)
    (liveInnerOwn_withInner vs L₁ src real hreal x w hTab hTape hR hSv hsc hSrc hcaus wk hV hC hL0 hrows)

/-- **At the honest worker and values**: `liveRoundEqOwn_withInner` at `Loop.valsWk honestVals`, whose premises on the
worker and the values hold there (`Loop.takesValsOwn_honestVals`, `RefSession.swCausal_vals` through
`Loop.swCausalOn_swOk`, `Loop.finalCFixed_honestVals`, `Loop.l0Fixed_honestVals`, `Loop.rowsOK_honestVals`). -/
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
  liveRoundEqOwn_withInner _ L₁ src real hreal x w hTab hTape hR hSv hsc hSrc
    (Loop.swCausalOn_swOk Masked.dataE _ (RefSession.swCausal_vals _ hcz)) (Loop.finalCFixed_honestVals x w)
    (Loop.l0Fixed_honestVals x w (Masked.live_of hsc hR) hR) (Loop.rowsOK_honestVals hR) _
    (Loop.takesValsOwn_honestVals _ src real x w)

/-- **At every algebraic-phase worker, through `vsOf`**: `liveRoundEqOwn_withInner` at `Loop.vsOf bx` and the worker
`Loop.phased bx post`, which runs `bx`'s algebraic phase and then `post`, at a witness where `bx`'s served values'
rows hold (`Loop.RowsOK (Loop.vsServed bx)`). `vsOf bx` is causal on the coins `Loop.SwOk` admits
(`Loop.swCausal_oneTape_vsOf`), fixes its `final_c` (`Loop.finalCFixed_vsOf`) and its level-0 words
(`Loop.l0Fixed_vsOf`), its rows hold (`Loop.rowsOK_vsOf`), and the worker takes its values (`Loop.takesVals_vsOf`). -/
theorem liveRoundEqOwn_vsOf (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (x : Msg) (w : Flock.Wit) (hTab : Loop.finS.Tables) (hTape : Loop.finS.TapeOK)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (bx : Loop.AlgWorker (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real))
    (hRo : Loop.RowsOK (Loop.vsServed _ bx) x w)
    (post : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker) :
    LiveRoundEqOwn Loop.finS (Loop.vsOf _ bx)
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) CoinOpen.shaH
      (Loop.phased _ bx post) x w :=
  liveRoundEqOwn_withInner (Loop.vsOf _ bx) L₁ src real hreal x w hTab hTape hR hSv hsc hSrc
    (Loop.swCausal_oneTape_vsOf L₁ src real _ hreal bx) (Loop.finalCFixed_vsOf _ bx x w)
    (Loop.l0Fixed_vsOf _ src real bx x w (Masked.live_of hsc hR) hR) (Loop.rowsOK_vsOf _ bx hR) _
    (Loop.takesValsOwn_of_takesVals (Loop.takesVals_vsOf L₁ src real _ hreal x w hSv bx hRo post))

end LeanRelease

end Claims.ZKServer
