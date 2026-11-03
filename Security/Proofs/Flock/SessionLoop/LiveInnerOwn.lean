import Proofs.Flock.SessionLoop.LiveInnerAt
import Proofs.Flock.SessionLoop.LiveSchedOwn
import Proofs.Flock.SessionLoop.FinalC
import Proofs.Flock.Claims.ZK.FinishLive

/-!
# `LiveInner` at every worker taking some `vs`'s values, on its own histories

`LeanRelease.liveInner_withInner` proves `LiveInner` at `withInner` from two premises over the worker, `Loop.TakesFinalC`
and `Loop.TakesTau`: at every live history, the worker's `final_c'` and `τ` commitment are `vs`'s. Neither follows from
`Loop.TakesVals`, `SwCausal` and `Loop.FinalCFixed` at every live history, because `live` checks the worker's values and
not the history's messages, while the loop's finish reads its request off the history. On the deferred release's own
runs (`ZkServer.OwnHist`) the history's messages are the frame's releases of the worker's own rounds, so the values the
loop reads back are the worker's, and there both follow from the worker taking `vs`'s values at its own histories
(`Loop.TakesValsOwn`):

* **`LeanRelease.LiveInnerOwn`**: `LiveInner`'s equation on the histories satisfying `OwnHist`, in the place
  `FinalRoundEq` and `LiveRefine` take it. `LeanRelease.liveInnerOwn_of_liveInner`: `LiveInner` gives it.
* **`Loop.finOf`**: the rounds the loop's finish hands a rep's frame for its inner proof (`Finish.innerLive`): its
  `repCons` on the loop's request (`repInOf`: the values the frame took and the coins the stream answered), refused where
  `consWhy` refuses, then `innerP` on `repCons`'s proof and constraints. Its `final_c` slot is the one at which the finish
  accepts the request: the passed values fix it (`consWhy`'s `zerocheck c_eval`), and at another the finish computes no
  round and the frame stands in and stops the rep. `Loop.finInner` is the run, `Loop.asksPad` its coins.
* **`LeanRelease.finOf_finalC_own`** and **`LeanRelease.finOf_tau_own`**: at the worker's own histories, the finish's
  round 0 is the inner proof's label and `vs`'s `final_c` under the configuration's pad, and its round 2 `vs`'s `τ`
  commitment (`Loop.vsTau`): the premises `liveInner_withInner_at` takes at a rep's inner rounds.
* **`LeanRelease.liveInnerOwn_withInner`**: `LiveInnerOwn` at `withInner` on the finish's rounds (`Loop.finOf`), at every
  worker taking some `vs`'s values at its own histories (`Loop.TakesValsOwn`), where `vs` meets `SwCausal`,
  `Loop.FinalCFixed` and `Loop.RowsOK`; a worker taking no `vs`'s values is outside it. Its premises on the statement,
  the tables and the loop's inputs are `SessionLoopZKDeployed`'s own (`Loop.realOf`, `Claims.Flock.Scope`, the witness,
  `Loop.Serves`, `Loop.Src`). It takes neither `TakesFinalC` nor `TakesTau`.
  `LeanRelease.liveInnerOwn_honestVals`: at the honest values and `Loop.valsWk`, every premise over the worker and the
  values holds.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Hashes TMW)
open _root_.Flock.Firewall.Frame (INNER_ROUNDS labelFrame wordBytes words)
open _root_.Flock.Firewall.Finish (Tab RepIn Asks)
open _root_.Flock.Transcript (header)
open FlockSoundness.Refine (colBytes)

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

/-- **A run as one that returns alike within `n` rounds**: a run from round `k` that returns `a` at round `k'` takes no
fewer rounds, and where `k' ≤ n`, the other run returns the same. -/
def padSim (n : ℕ) : _root_.Flock.With.Sim (StateT ℕ V) (StateT ℕ V) where
  R x y := ∀ k a k', x.run k = .ok (a, k') → k ≤ k' ∧ (k' ≤ n → y.run k = .ok (a, k'))
  pure _ _ _ _ h := by
    cases h
    exact ⟨Nat.le_refl _, fun _ => rfl⟩
  bind {α β x y f g} hxy hfg k b k'' h := by
    rw [StateT.run_bind] at h
    rcases hx : x.run k with _ | ⟨a, k1⟩
    · rw [hx] at h; cases h
    · rw [hx] at h
      obtain ⟨hk1, hy⟩ := hxy k a k1 hx
      obtain ⟨hk2, hg⟩ := hfg a k1 b k'' h
      refine ⟨hk1.trans hk2, fun hle => ?_⟩
      rw [StateT.run_bind, hy (hk2.trans hle)]
      exact hg hle
  throw _ _ _ _ h := by cases h
  lift v _ _ _ h := by
    cases v with
    | error e => cases h
    | ok a0 =>
      cases h
      exact ⟨Nat.le_refl _, fun _ => rfl⟩

/-- `asksOf C'` and `asksPad C` answer each round below `C'.size` alike, where `C'` is a prefix of `C`. -/
theorem asksPad_rel (C' C : Array (Array F128)) (hpre : ∀ k < C'.size, C[k]? = C'[k]?) :
    _root_.Flock.Firewall.Finish.AsksRel (padSim C'.size) (_root_.Flock.Firewall.Finish.asksOf C').toC
      (asksPad C).toC where
  link _ _ _ _ _ h := by cases h
  round c k a k' h := by
    change ((_root_.Flock.Firewall.Finish.asksOf C').round c).run k = _ at h
    dsimp only [_root_.Flock.Firewall.Finish.asksOf] at h
    rw [StateT.run_bind, StateT.run_get, pure_bind] at h
    rcases hk' : C'[k]? with _ | cs
    · rw [hk'] at h; cases h
    · rw [hk'] at h
      have hk : C[k]? = some cs := by
        rw [hpre k (Array.getElem?_eq_some_iff.1 hk').1, hk']
      dsimp only at h
      split at h
      · rw [StateT.run_bind] at h
        cases h
      · rename_i hn
        rw [StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure] at h
        cases h
        refine ⟨Nat.le_succ k, fun _ => ?_⟩
        have hg : C.getD k (Array.replicate (_root_.Flock.u64OfLE c (c.size - 8)).toNat 1) = cs := by
          rw [Array.getD_eq_getD_getElem?, hk]
          rfl
        have hn' := Bool.eq_false_iff.mpr hn
        show ((asksPad C).round c).run k = _
        simp only [asksPad, StateT.run_bind, StateT.run_get, StateT.run_set, StateT.run_pure, pure_bind, hg, hn',
          Bool.false_eq_true, ↓reduceIte]

/-- **`repCons` answered past its coins by ones** returns as `repCons` answered by a prefix `C'` of them, where that
returns within `C'`. -/
theorem repCons_asksPad (C' C : Array (Array F128)) (hpre : ∀ k < C'.size, C[k]? = C'[k]?) (st : _root_.Flock.Setup)
    (hm : st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (sch : _root_.Flock.Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (g : Array F128) (cm0 : _) (extras : Array _root_.Flock.Claim) (pads : _) (ic : _)
    (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (tr : _root_.Flock.Transcript)
    {r : _} {tr' : _root_.Flock.Transcript} {k' : ℕ}
    (h : ((_root_.Flock.Firewall.Finish.repCons (_root_.Flock.Firewall.Finish.asksOf C')
      (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run tr).run 0 = .ok ((r, tr'), k'))
    (hk : k' ≤ C'.size) :
    ((_root_.Flock.Firewall.Finish.repCons (asksPad C)
      (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run tr).run 0 = .ok ((r, tr'), k') :=
  ((_root_.Flock.Firewall.Finish.repCons_rel (padSim C'.size) (asksPad_rel C' C hpre) st hm sch tPad rep dom g cm0 extras pads ic
    vals extra lanePad salt tr 0 (r, tr') k' h).2 hk)

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
  let ((r, _), _) ← (prog.run (_root_.Flock.Transcript.init #[] dom)).run 0
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

/-- Round 0's `final_c'`, read as the frame reads it and sent back as bytes, is the word's bytes. -/
theorem wordBytes_words_inner0 (a : F128) :
    wordBytes (words (inner0 a.toBytes) ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1) =
      a.toBytes := by
  have h1 : words (inner0 a.toBytes) ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1 = #[a] := by
    have e := finalCOf_inner0 a
    simp only [finalCOf, Array.toArray_toList] at e
    apply Array.ext
    · simp [words_size]
    · intro i h1 h2
      obtain rfl : i = 0 := by simpa using h2
      rw [← getElem!_pos _ 0 h1]
      exact e
  rw [h1]
  apply ByteArray.ext
  apply Array.ext'
  rw [wordBytes_data]
  simp [colBytes]

/-- Round 2's `τ` commitment, read as the frame reads it (four words past its header) and sent back as bytes, is the
commitment. -/
theorem wordBytes_words_inner2 {t : ByteArray} (ht : t.size = 64) : wordBytes (words (inner2 t) 16 4) = t := by
  obtain ⟨P, hP, hc⟩ := colBytes_read 4 t.data.toList (by rw [Array.length_toList]; exact ht)
  have hd : ∀ i (hi : i < 4), (inner2 t).data.toList.drop (16 + 16 * i) =
      (P[i]'(by omega)).toBytes.data.toList ++ (colBytes (P.drop (i + 1)) ++ (header 4 1 1).data.toList) := by
    intro i hi
    have hs : P = P.take i ++ P[i]'(by omega) :: P.drop (i + 1) := by
      conv_lhs => rw [← List.take_append_drop i P]
      rw [List.drop_eq_getElem_cons (by omega)]
    have hl : ((header 5 0 64).data.toList ++ colBytes (P.take i)).length = 16 + 16 * i := by
      rw [List.length_append, FlockSoundness.Refine.Frame.header_length, FlockSoundness.Refine.Frame.colBytes_length,
        List.length_take]
      omega
    have e : (inner2 t).data.toList = ((header 5 0 64).data.toList ++ colBytes (P.take i)) ++
        ((P[i]'(by omega)).toBytes.data.toList ++ (colBytes (P.drop (i + 1)) ++ (header 4 1 1).data.toList)) := by
      rw [inner2, ByteArray.toList_data_append, ByteArray.toList_data_append, ← hc]
      conv_lhs => rw [hs]
      simp only [colBytes, List.flatMap_append, List.flatMap_cons, List.append_assoc]
    rw [e, List.drop_left' hl]
  have hw : words (inner2 t) 16 4 = P.toArray := by
    apply Array.ext
    · rw [words_size, List.size_toArray, hP]
    · intro i h1 h2
      have hi : i < 4 := by rw [words_size] at h1; exact h1
      have e1 : (words (inner2 t) 16 4)[i]'h1 = wd (inner2 t).data.toList (16 + 16 * i) := by
        unfold wd
        have hz : 0 < (words (ByteArray.mk (inner2 t).data.toList.toArray) (16 + 16 * i) 1).size := by
          rw [words_size]
          omega
        rw [getElem!_pos _ 0 hz]
        simp only [words, Array.getElem_map, Array.getElem_range, Nat.mul_zero, Nat.add_zero]
      rw [e1, wd_drop (hd i hi)]
      simp
  rw [hw]
  apply ByteArray.ext
  apply Array.ext'
  rw [wordBytes_data]
  simpa using hc

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

/-- **The finish's round 0 at the worker's own history**: at a history `LiveInnerOwn` reads whose next round is rep
`2j + r`'s inner round 0, the round the finish hands the frame (`Loop.finOf`) is the inner proof's label and `vs`'s
`final_c` masked by the configuration's pad there. The loop's request carries the worker's values, `vs`'s at its own
histories (`Loop.TakesValsOwn`); the finish accepts it at `vs`'s `final_c` alone (`consWhy`'s `zerocheck c_eval`,
`Loop.FinalCFixed`) and masks it with the rep's pads, the configuration's. -/
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
    (hcaus : (CoinLayer.session Loop.loopNr Masked.dataE).SwCausal vs)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hV : Loop.TakesValsOwn vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) wk x w)
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
  sorry

/-- **The finish's round 2 at the worker's own history**: at a history `LiveInnerOwn` reads whose next round is rep
`2j + r`'s inner round 2, the round the finish hands the frame (`Loop.finOf`) carries `vs`'s `τ` commitment
(`Loop.vsTau`). The finish's request carries `vs`'s values (`Loop.TakesValsOwn`), on which it accepts at `vs`'s
`final_c` (`finOf_finalC_own`), and its inner proof commits to `τ` from the request's values, coins and salts as the
switched source does where the rows hold (`Loop.RowsOK`). -/
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
    (hcaus : (CoinLayer.session Loop.loopNr Masked.dataE).SwCausal vs)
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real).toSrv.Worker)
    (hV : Loop.TakesValsOwn vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) wk x w)
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
  sorry

/-- **`LiveInner` at every worker taking some `vs`'s values, on its own histories**: at `withInner` on the loop's
finish's rounds (`Loop.finOf`, at `L₁`'s frame draws), at a statement in the masked scope with a witness that the loop
serves, on the loop's own tables (`Loop.realOf`) and inputs (`Loop.Src`), `LiveInnerOwn` holds at every worker whose free
words are `vs`'s at its own histories (`Loop.TakesValsOwn`), for every `vs` that is causal (`SwCausal`), whose `final_c`
its round-1c values fix (`Loop.FinalCFixed`) and whose rows hold (`Loop.RowsOK`). -/
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
    (hV : Loop.TakesValsOwn vs (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real) wk x w)
    (hC : Loop.FinalCFixed vs x w) (hrows : Loop.RowsOK vs x w) :
    LiveInnerOwn vs (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real wk x w := by
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
    rw [hv0, finOf_finalC_own vs L₁ src real hreal x w hR hSv hsc hSrc hcaus wk hV hC ν t κ m₀ ρ₀ ops v h0 hlt hl hok
      hr hown hv hI j r hrep hslot, Loop.wordBytes_words_inner0, Loop.finalCOf_inner0]
    exact FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self _ _
  · intro m w' j r _ _ _ hvs hrep hslot hv2
    have key : ∀ T : ByteArray, T.size = 64 → Loop.inner2 (wordBytes (words (Loop.inner2 T) 16 4)) = Loop.inner2 T :=
      fun T hT => by rw [Loop.wordBytes_words_inner2 hT]
    rw [finOf_tau_own vs L₁ src real hreal x w hR hSv hsc hSrc hcaus wk hV hC hrows ν t κ m₀ ρ₀ ops v h0 hlt hl hok
      hr hown hv hI j r hrep hslot, key] at hv2
    · exact Loop.inner2_inj (hvs.symm.trans hv2)
    · exact Loop.vsTau_size _ _ _ _ _ _ _ _

/-- **At the honest values**: the worker that sends the scheduled rounds with them (`Loop.valsWk`) meets `TakesValsOwn`
(`Loop.takesValsOwn_honestVals`), and they are causal at the session's causality (`RefSession.swCausal_vals`), fix
their `final_c` (`Loop.finalCFixed_honestVals`) and hold their rows (`Loop.rowsOK_honestVals`), so
`liveInnerOwn_withInner` holds there with no premise over the worker or the values. -/
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
    (hR : Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src x w)
    (hc : (CoinLayer.session Loop.loopNr Masked.dataE).Causal) :
    LiveInnerOwn (CoinLayer.session Loop.loopNr Masked.dataE).honestVals
      (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real
      (Loop.valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (Loop.finOf real L₁.draws)) src real)) x w :=
  liveInnerOwn_withInner _ L₁ src real hreal x w hR hSv hsc hSrc (RefSession.swCausal_vals _ hc) _
    (Loop.takesValsOwn_honestVals _ src real x w) (Loop.finalCFixed_honestVals x w) (Loop.rowsOK_honestVals hR)

end LeanRelease

end Claims.ZKServer
