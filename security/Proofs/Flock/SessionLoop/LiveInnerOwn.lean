module

public import Proofs.Flock.SessionLoop.FinInnerOwn
public import Proofs.Flock.SessionLoop.L0FixedHonest

/-!
# `LiveInner` at every worker taking some `vs`'s values, on its own histories

`LeanRelease.liveInner_withInner` proves `LiveInner` at `withInner` from two premises over the worker, `Loop.TakesFinalC`
and `Loop.TakesTau`: at every live history, the worker's `final_c'` and `τ` commitment are `vs`'s. Neither follows from
`Loop.TakesVals`, `SwCausal` and `Loop.FinalCFixed` at every live history, because `live` checks the worker's values and
not the history's messages, while the loop's finish reads its request off the history. On the deferred release's own
runs (`ZkServer.OwnHist`) the history's messages are the frame's releases of the worker's own rounds, the switched
session's where `LiveSchedOwn` holds, so the values the loop reads back are `vs`'s, and there both follow from the worker
taking `vs`'s values at its own histories (`Loop.TakesValsOwn`, which gives `LiveSchedOwn`:
`LeanRelease.liveSchedOwn_ofLoopLink`):

* **`LeanRelease.LiveInnerOwn`**: `LiveInner`'s equation on the histories satisfying `OwnHist`, in the place
  `FinalRoundEq` and `LiveRefine` take it. `LeanRelease.liveInnerOwn_of_liveInner`: `LiveInner` gives it.
* **`LeanRelease.finOf_finalC_own`** and **`LeanRelease.finOf_tau_own`**: at the worker's own histories, where
  `LiveSchedOwn` holds, the round 0 the loop's finish hands the frame (`Loop.finOf`, in `FinInner`) is the inner proof's
  label and `vs`'s `final_c` under the configuration's pad, and its round 2 `vs`'s `τ` commitment (`Loop.vsTau`): the
  premises `liveInner_withInner_at` takes at a rep's inner rounds. Each from the parts in `FinInnerParts`, and
  `finInner_returns_own` (`FinInnerOwn`): the finish returns there.
* **`LeanRelease.liveInnerOwn_withInner`**: `LiveInnerOwn` at `withInner` on the finish's rounds (`Loop.finOf`), at every
  worker taking some `vs`'s values at its own histories (`Loop.TakesValsOwn`), where `vs` meets `SwCausalOn` at
  `Loop.SwOk`, `Loop.FinalCFixed`, `Loop.L0Fixed` and `Loop.RowsOK`; a worker taking no `vs`'s values is outside it.
  Its premises on the session (`Tables`, `TapeOK`), the statement, the tables and the loop's inputs are
  `SessionLoopZKDeployed`'s own (`Loop.realOf`, `Claims.Flock.Scope`, the witness, `Loop.Serves`, `Loop.Src`). It takes
  neither `TakesFinalC` nor `TakesTau`.
  `LeanRelease.liveInnerOwn_honestVals`: at the honest values and `Loop.valsWk`, every premise over the worker and the
  values holds.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Hashes TMW)
open _root_.Flock.Firewall.Frame (INNER_ROUNDS labelFrame wordBytes words)
open _root_.Flock.Firewall.Finish (Tab RepIn Asks)
open _root_.Flock.Transcript (header)
open FlockSoundness.Refine (colBytes)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`


namespace LeanRelease

/-- **`LiveInner` on the worker's own histories**: `LiveInner`'s equation after every history it reads that is the
deferred release's own run (`ZkServer.OwnHist`), in the place `FinalRoundEq` and `LiveRefine` take it. -/
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

/-- `LiveInner` at every history gives it on the worker's own. -/
theorem liveInnerOwn_of_liveInner {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals}
    {L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)}
    {src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray)}
    {wk : (ofLoopLink L₀ src real).toSrv.Worker} {x : Msg} {w : Flock.Wit}
    (hI : LiveInner vs L₀ src real wk x w) : LiveInnerOwn vs L₀ src real wk x w :=
  fun ν t κ m₀ ρ₀ ops v hpos hlt hl hok hr _ hv hin => hI ν t κ m₀ ρ₀ ops v hpos hlt hl hok hr hv hin

/-- **The finish's round 0 at the worker's own history**: at a history `LiveInnerOwn` reads whose next round is rep
`2j + r`'s inner round 0, the round the finish hands the frame (`Loop.finOf`) is the inner proof's label and `vs`'s
`final_c` masked by the configuration's pad there. The loop's request carries the worker's values, `vs`'s where
`LiveSchedOwn` holds; the finish accepts it at `vs`'s `final_c` alone (`consWhy`'s `zerocheck c_eval`,
`Loop.FinalCFixed`), returns there (`Loop.finInner_returns_own`), and masks it with the rep's pads, the
configuration's. From the parts in `FinInnerParts` and `FinInnerOwn`. -/
theorem finOf_finalC_own (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (hcaus : (CoinLayer.session Loop.loopNr Masked.dataE).SwCausalOn (Loop.SwOk Masked.dataE) vs)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hSch : LiveSchedOwn vs (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real wk x w)
    (hC : Loop.FinalCFixed vs x w)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (v : Msg) (h0 : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.live wk x w (ν, t) κ
      ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true)
    (hr : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).refusesCoins x (ν, t)
      ((m₀, ρ₀) :: ops) = false)
    (hown : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.OwnHist
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).fillOf wk x w (ν, t) κ
      ((m₀, ρ₀) :: ops))
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false))
    (hI : Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2))
    (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hrep : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).rep x (ops.length + 1) =
      2 * j.val + r.val)
    (hslot : ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0)) :
    (Loop.finOf real L₁.draws x w (ν, t) ((m₀, ρ₀) :: ops)).getD 0 .empty =
      Loop.inner0 (Loop.vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
          (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
          ((ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).cfg x (ν, t)
            ((m₀, ρ₀) :: ops)).lay.fc +
        ((ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).cfg x (ν, t)
            ((m₀, ρ₀) :: ops)).h.getD
          ((ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).cfg x (ν, t)
            ((m₀, ρ₀) :: ops)).lay.fc 0).toBytes := by
  have hH : Loop.InnerHist (Loop.relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r 0 :=
    ⟨h0, hlt, hl, hok, hr, hown, ⟨v, hv⟩, hI, hrep, hslot⟩
  obtain ⟨s0, spec, sts, tr, hi, q₀, hs, hrt, hq, hlayT, hlay, hcfgh, hfin⟩ :=
    Loop.innerHist_fin L₁ src real hreal x w hR hSv hsc hSrc wk (by decide) hH
  obtain ⟨hsz, -, hpads⟩ :=
    Loop.innerReq_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch (by decide) hH hs hrt hq
  obtain ⟨fc', tc, hfi⟩ :=
    Loop.finInner_returns_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch hC (by decide) hH hs hrt hq
  rw [hfin _ fc' tc hfi fun a' ha' =>
    Loop.finInner_accepts_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch hC (by decide) hH hs hrt hq hlayT
      a' ha']
  show Loop.inner0 fc'.toBytes = _
  have hfv : (Masked.layE x j).fc < q₀.vals.size := by
    rw [hsz]
    simp only [_root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc]
    omega
  have hfh : (Masked.layE x j).fc <
      (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h.size := by
    rw [FlockComplete.padsOf_size]
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw, _root_.Flock.Zk.PadLayout.hab,
      _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc]
    omega
  have hfc := Loop.finInner_fc hfi
  rw [hlayT, Loop.askOf_draws, Loop.askOf_vals_fc _ _ hfv] at hfc
  have hph : (_root_.Flock.Firewall.Finish.padsOf q₀.draws).h =
      (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h :=
    congrArg (fun p => p.h) hpads
  rw [hlay, hcfgh, FlockSoundness.Refine.getD_of_lt hfh, ← hph, ← hfc]

/-- **The finish's round 2 at the worker's own history**: at a history `LiveInnerOwn` reads whose next round is rep
`2j + r`'s inner round 2, the round the finish hands the frame (`Loop.finOf`) carries `vs`'s `τ` commitment
(`Loop.vsTau`). The finish's request carries `vs`'s values (`LiveSchedOwn`), on which it accepts at `vs`'s
`final_c` (`finOf_finalC_own`), and its inner proof commits to `τ` from the request's values, coins and salts as the
switched source does where the rows hold (`Loop.RowsOK`). From the parts in `FinInnerParts` and `FinInnerOwn`. -/
theorem finOf_tau_own (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (hcaus : (CoinLayer.session Loop.loopNr Masked.dataE).SwCausalOn (Loop.SwOk Masked.dataE) vs)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hSch : LiveSchedOwn vs (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real wk x w)
    (hC : Loop.FinalCFixed vs x w) (hrows : Loop.RowsOK vs x w)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (v : Msg) (h0 : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.live wk x w (ν, t) κ
      ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true)
    (hr : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).refusesCoins x (ν, t)
      ((m₀, ρ₀) :: ops) = false)
    (hown : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.OwnHist
      (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).fillOf wk x w (ν, t) κ
      ((m₀, ρ₀) :: ops))
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false))
    (hI : Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2))
    (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hrep : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).rep x (ops.length + 1) =
      2 * j.val + r.val)
    (hslot : ops.length = Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 2)) :
    (Loop.finOf real L₁.draws x w (ν, t) ((m₀, ρ₀) :: ops)).getD 2 .empty =
      Loop.inner2 (Loop.vsTau vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r) := by
  have hH : Loop.InnerHist (Loop.relIn L₁ src real) wk real x w ν t κ m₀ ρ₀ ops j r 2 :=
    ⟨h0, hlt, hl, hok, hr, hown, ⟨v, hv⟩, hI, hrep, hslot⟩
  obtain ⟨s0, spec, sts, tr, hi, q₀, hs, hrt, hq, hlayT, -, -, hfin⟩ :=
    Loop.innerHist_fin L₁ src real hreal x w hR hSv hsc hSrc wk (by decide) hH
  obtain ⟨fc', tc, hfi⟩ :=
    Loop.finInner_returns_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch hC (by decide) hH hs hrt hq
  rw [hfin _ fc' tc hfi fun a' ha' =>
    Loop.finInner_accepts_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch hC (by decide) hH hs hrt hq hlayT
      a' ha']
  exact congrArg Loop.inner2
    (Loop.finInner_tau_own L₁ src real vs hreal x w hR hSv hsc hSrc hcaus wk hSch hC hrows hH hs hrt hq hfi)

/-- **`LiveInner` at every worker taking some `vs`'s values, on its own histories**: at `withInner` on the loop's
finish's rounds (`Loop.finOf`, at `L₁`'s frame samples), at a statement in the masked scope with a witness that the loop
serves, on the loop's own tables (`Loop.realOf`) and inputs (`Loop.Src`), at the session's `Tables` and `TapeOK`,
`LiveInnerOwn` holds at every worker whose free words are `vs`'s at its own histories (`Loop.TakesValsOwn`), for every
`vs` causal at the coins `Loop.SwOk` admits (`SwCausalOn`), whose `final_c` its round-1c values fix
(`Loop.FinalCFixed`), whose level-0 words are the clear Ligerito's (`Loop.L0Fixed`) and whose rows hold (`Loop.RowsOK`):
there the worker's scheduled rounds are the switched session's (`liveSchedOwn_ofLoopLink`). -/
theorem liveInnerOwn_withInner (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit)
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (hcaus : (CoinLayer.session Loop.loopNr Masked.dataE).SwCausalOn (Loop.SwOk Masked.dataE) vs)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hV : Loop.TakesValsOwn vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) wk x w)
    (hC : Loop.FinalCFixed vs x w) (hL0 : Loop.L0Fixed vs x w) (hrows : Loop.RowsOK vs x w) :
    LiveInnerOwn vs (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real wk x w := by
  have hSch := liveSchedOwn_ofLoopLink vs _ src real hreal wk hTab hTape hR hSv hsc hSrc hrows hL0 hV
  intro ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hown hv hI
  have hR' : ∀ (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ) := by
    intro ρ s0 spec sts hs
    subst hreal
    exact Loop.realTabs_of src x w ρ hs
  refine liveInner_withInner_at vs L₁ src real (Loop.finOf real L₁.draws) x w (Masked.live_of hsc hR) hSv hR'
    hSrc.2 wk hrows ν t κ m₀ ρ₀ ops v h0 hlt hl hok hr hv hI ?_ ?_
  · intro w' j r _ _ hrep hslot hv0
    rw [hv0, finOf_finalC_own vs L₁ src real hreal x w hR hSv hsc hSrc hcaus wk hSch hC ν t κ m₀ ρ₀ ops v h0 hlt hl
      hok hr hown hv hI j r hrep hslot, Loop.wordBytes_words_inner0, Loop.finalCOf_inner0]
    exact FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self _ _
  · intro m w' j r _ _ _ hvs hrep hslot hv2
    have key : ∀ T : ByteArray, T.size = 64 → Loop.inner2 (wordBytes (words (Loop.inner2 T) 16 4)) = Loop.inner2 T :=
      fun T hT => by rw [Loop.wordBytes_words_inner2 hT]
    rw [finOf_tau_own vs L₁ src real hreal x w hR hSv hsc hSrc hcaus wk hSch hC hrows ν t κ m₀ ρ₀ ops v h0 hlt hl hok
      hr hown hv hI j r hrep hslot, key] at hv2
    · exact Loop.inner2_inj (hvs.symm.trans hv2)
    · exact Loop.vsTau_size _ _ _ _ _ _ _ _

/-- **At the honest values**: the worker that sends the scheduled rounds with them (`Loop.valsWk`) meets `TakesValsOwn`
(`Loop.takesValsOwn_honestVals`), and they are causal at the session's causality (`RefSession.swCausal_vals`), hence at
the coins `Loop.SwOk` admits (`Loop.swCausalOn_swOk`), fix their `final_c` (`Loop.finalCFixed_honestVals`), their
level-0 words (`Loop.l0Fixed_honestVals`) and their rows (`Loop.rowsOK_honestVals`), so `liveInnerOwn_withInner` holds
there with no premise over the worker or the values. -/
theorem liveInnerOwn_honestVals
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit)
    (hTab : (CoinLayer.session Loop.loopNr Masked.dataE).Tables)
    (hTape : (CoinLayer.session Loop.loopNr Masked.dataE).TapeOK)
    (hR : Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (hc : (CoinLayer.session Loop.loopNr Masked.dataE).Causal) :
    LiveInnerOwn (CoinLayer.session Loop.loopNr Masked.dataE).honestVals
      (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real
      (Loop.valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real)) x w :=
  liveInnerOwn_withInner _ L₁ src real hreal x w hTab hTape hR hSv hsc hSrc
    (Loop.swCausalOn_swOk Masked.dataE _ (RefSession.swCausal_vals _ hc)) _
    (Loop.takesValsOwn_honestVals _ src real x w) (Loop.finalCFixed_honestVals x w)
    (Loop.l0Fixed_honestVals x w (Masked.live_of hsc hR) hR) (Loop.rowsOK_honestVals hR)

end LeanRelease

end Claims.ZKServer
