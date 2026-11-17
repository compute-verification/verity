module

public import Proofs.Flock.Claims.ZK.OwnHistAny
public import Proofs.Flock.SessionLoop.LiveSchedOwn
public import Proofs.Flock.SessionLoop.LiveInnerOwn
public import Proofs.Flock.SessionLoop.FinishIsLoop
public import Proofs.Flock.SessionLoop.VsOfTakes
public import Proofs.Flock.SessionLoop.SwCausalExe

/-! `LiveRoundEq` for any worker at its own histories, without `Loop.TakesFinalC` or `Loop.TakesTau`
(`LeanRelease.liveRoundEqOwn_withInner`); a worker taking no admissible `vs`'s values is outside it. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.ZkServer.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace LeanRelease

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
