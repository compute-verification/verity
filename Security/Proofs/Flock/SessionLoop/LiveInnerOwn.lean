import Proofs.Flock.SessionLoop.LiveInnerAt
import Proofs.Flock.SessionLoop.FinalC

/-!
# `LiveInner` at every worker, on its own histories

`LeanRelease.liveInner_withInner` proves `LiveInner` at `withInner` from two premises over the worker, `Loop.TakesFinalC`
and `Loop.TakesTau`: at every live history, the worker's `final_c'` and `τ` commitment are `vs`'s. Neither follows from
`Loop.TakesVals`, `SwCausal` and `Loop.FinalCFixed` at every live history, because `live` checks the worker's values and
not the history's messages, while the loop's finish reads its request off the history. On the deferred release's own
runs (`ZkServer.OwnHist`) the history's messages are the frame's releases of the worker's own rounds, so the values the
loop reads back are the worker's, and there both follow:

* **`LeanRelease.LiveInnerOwn`**: `LiveInner`'s equation on the histories satisfying `OwnHist`, in the place
  `FinalRoundEq` and `LiveRefine` take it. `LeanRelease.liveInnerOwn_of_liveInner`: `LiveInner` gives it.
* **`Loop.finOf`**: the rounds the loop's finish hands a rep's frame for its inner proof (`Finish.innerLive`): its
  `repCons` on the loop's request (`repInOf`: the values the frame took and the coins the stream answered), refused where
  `consWhy` refuses, then `innerP` on `repCons`'s proof and constraints. Its `final_c` slot is the one at which the finish
  accepts the request: the passed values fix it (`consWhy`'s `zerocheck c_eval`), and at another the finish computes no
  round and the frame stands in and stops the rep. `Loop.finInner` is the run, `Loop.asksPad` its coins.
* **`LeanRelease.liveInnerOwn_withInner`**: `LiveInnerOwn` at `withInner` on the finish's rounds (`Loop.finOf`), at every
  worker that meets `Loop.TakesVals`, for every `vs` meeting `SwCausal`, `Loop.FinalCFixed` and `Loop.RowsOK`, at
  `SessionLoopZKDeployed`'s own premises on the statement, the tables and the loop's inputs (`Loop.realOf`,
  `Claims.Flock.Scope`, the witness, `Loop.Serves`, `Loop.Src`). It takes neither `TakesFinalC` nor `TakesTau`.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Hashes TMW)
open _root_.Flock.Firewall.Frame (INNER_ROUNDS)
open _root_.Flock.Firewall.Finish (Tab RepIn Asks)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

/-- **A rep's rounds answered by the coins `C`, then by ones**: round `k` (counted in the state) by `C[k]` while `C`
lasts, as `Finish.asksOf C`; past it by ones, as many as its squeeze header counts. -/
def asksPad (C : Array (Array F128)) : Asks (StateT ℕ V) where
  link _ _ := throw "finish: a link message inside a rep"
  round c := do
    let k ← get
    let n := (_root_.Flock.u64OfLE c (c.size - 8)).toNat
    let cs := C.getD k (Array.replicate n 1)
    if cs.size != n then throw s!"finish: round {k} squeezes {n} coins where the rep's coins have {cs.size}"
    set (k + 1)
    return fun i => cs[i]!

/-- **The finish's live inner proof of a rep** on the table `t`, as rep `rep` of a stream whose transcript starts at
`dom`, at the request `q` whose first `n` rounds of coins are the rep's scheduled rounds' (`Finish.innerLive`, its rounds
answered by `q.coins` and past them by ones, `asksPad`): its pads commitment, `repCons`, refused where `consWhy`
refuses, then `innerP` on `repCons`'s proof and constraints. Returns the masked `final_c` its round 0 absorbs and the
`τ` commitment its round 2 absorbs. -/
def finInner (t : Tab) (rep : ℕ) (dom : ByteArray) (n : ℕ) (q : RepIn) : V (F128 × ByteArray) := do
  let pads := _root_.Flock.Firewall.Finish.padsOf q.draws
  let ic ← (_root_.Flock.Firewall.Finish.innerCommitP (Hashes.ofSha512 _root_.Flock.Sha512.hash) pads
    (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) : V _root_.Flock.Firewall.Finish.InnerCommit)
  let hs : Hashes (StateT ℕ V) := Hashes.ofSha512 _root_.Flock.Sha512.hash
  let commit : _root_.Flock.Level → Array F128 → (ℕ → ByteArray) → StateT ℕ V _root_.Flock.Firewall.Finish.Commit :=
    fun lv g salt => pure (_root_.Flock.Firewall.Finish.commitFw lv g salt)
  let salt : ℕ → ℕ → ByteArray := fun l =>
    _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)
  let prog : TMW (StateT ℕ V) (F128 × ByteArray) := do
    let (p, _, cons) ← _root_.Flock.Firewall.Finish.repCons (asksPad q.coins) commit t.st t.sch t.tPad rep dom t.g
      t.cm0 (_root_.Flock.Firewall.Finish.claimsOf q.xs) pads ic q.vals t.extra t.lanePad salt
    let k ← getThe ℕ
    if let some why := _root_.Flock.Firewall.Finish.consWhy n k cons pads.h then throw why
    let ip ← _root_.Flock.Firewall.Finish.innerP (asksPad q.coins) hs
      (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) pads ic p.zerocheck.finalC cons
    return (p.zerocheck.finalC, ip.tauCommit)
  let (r, _) ← (prog.run (_root_.Flock.Transcript.init #[] dom)).run 0
  return r

open Classical in
/-- **The rounds the loop's finish hands a rep's frame for its inner proof** on the history `h` at `x`, the witness `w`
and the tape `ρ`, on the rep tables and salts `real` and the frame draws `draws`: where the history's next round is rep
stream `k`'s round `i` (`CoinLayer.locate`, as `Loop.innerOf` places it), the finish's live inner proof (`finInner`) on
rep stream `k`'s table and salts and the loop's request for it (`repInOf`: the values the frame took and the coins of the
stream's first `i` rounds), at the `final_c` slot at which the finish accepts the request, if one does (the passed values
fix it: `consWhy`'s `zerocheck c_eval`): round 0 the inner proof's label and `final_c'`, round 2 `τ`'s commitment
(`Frame.innerSched` reads nothing else of them). None where it accepts none: the finish computes no round there. -/
noncomputable def finOf {R : Type} (real : Msg → Flock.Wit → R → ℕ → V (Tab × Array ByteArray))
    (draws : R → ℕ → ByteArray) (x : Msg) (w : Flock.Wit) (ρ : R) (h : List (Msg × Msg)) : Array ByteArray :=
  match FlockDeploy.setupOf true x, h.tail.length with
  | some (_, spec, sts), q + 1 =>
    let ki := CoinLayer.locate (roundsOf spec sts) 0 q
    match real x w ρ ki.1 with
    | .ok (t, hi) =>
      match repInOf x h spec sts ki.1 (draws ρ ki.1) hi with
      | .ok q₀ =>
        let dom := (spec.domain spec.tables[ki.1 / 2]! (ki.1 % 2)).toUTF8
        let nS := (roundsOf spec sts).getD ki.1 0 - INNER_ROUNDS
        let ask : F128 → RepIn := fun a =>
          { q₀ with vals := q₀.vals.set! (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc a,
                    coins := q₀.coins.extract 0 ki.2 }
        if hA : ∃ a, (finInner t (ki.1 % 2) dom nS (ask a)).isOk = true then
          match finInner t (ki.1 % 2) dom nS (ask hA.choose) with
          | .ok (fc, tc) => #[inner0 fc.toBytes, .empty, inner2 tc, .empty]
          | .error _ => #[]
        else #[]
      | .error _ => #[]
    | .error _ => #[]
  | _, _ => #[]

end Loop

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

/-- **`LiveInner` at every worker, on its own histories**: at `withInner` on the loop's finish's rounds (`Loop.finOf`, at
`L₁`'s frame draws), at a statement in the masked scope with a witness that the loop serves, on the loop's own tables
(`Loop.realOf`) and inputs (`Loop.Src`), `LiveInnerOwn` holds at every worker whose free words are `vs`'s
(`Loop.TakesVals`), for every `vs` that is causal (`SwCausal`), whose `final_c` its round-1c values fix
(`Loop.FinalCFixed`) and whose rows hold (`Loop.RowsOK`). -/
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
    (hR : Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (hcaus : (CoinLayer.session Loop.loopNr Masked.dataE).SwCausal vs)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hV : Loop.TakesVals vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) wk x w)
    (hC : Loop.FinalCFixed vs x w) (hrows : Loop.RowsOK vs x w) :
    LiveInnerOwn vs (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real wk x w := by
  sorry

end LeanRelease

end Claims.ZKServer
