module

public import Proofs.Flock.SessionLoop.LiveInnerAt
public import Proofs.Flock.SessionLoop.LiveSchedOwn
public import Proofs.Flock.SessionLoop.FinalC
public import Proofs.Flock.Claims.ZK.FinishLive

/-!
# The loop's finish at a rep's inner rounds

The rounds the loop's finish hands a rep's frame for its inner proof (`Finish.innerLive`), as `withInner` reads them:

* **`Loop.asksPad`**: a rep's rounds answered by the request's coins, then by ones. `Loop.asksPad_rel` and
  `Loop.repCons_asksPad`: `repCons` answered by a prefix of them (`Finish.asksOf`) returns as `repCons` answered so,
  where it returns within the prefix (`Loop.padSim`).
* **`Loop.finInner`**: the finish's live inner proof of a rep: its pads commitment, `repCons`, refused where `consWhy`
  refuses, then `innerP` on `repCons`'s proof and constraints. Returns the masked `final_c` its round 0 absorbs and the
  `τ` commitment its round 2 absorbs.
* **`Loop.finOf`**: at a history whose next round is a rep's inner round, `finInner` on the rep's table and the loop's
  request (`repInOf`), at the `final_c` at which the finish accepts the request (`Loop.askOf`), if one does: the passed
  values fix it (`consWhy`'s `zerocheck c_eval`), and at another the finish computes no round and the frame stands in
  and stops the rep.
* `Loop.wordBytes_words_inner0`, `Loop.wordBytes_words_inner2`: rounds 0's and 2's words, read as the frame reads them.
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

namespace Loop

/-- **A rep's rounds answered by the coins `C`, then by ones**: round `k` (counted in the state) by `C[k]` while `C`
lasts, as `Finish.asksOf C`; past it by the constant `1`, as a cut answered by ones past `C` (`coinsPad`). -/
def asksPad (C : Array (Array F128)) : Asks (StateT ℕ V) where
  link _ _ := throw "finish: a link message inside a rep"
  round c := do
    let k ← get
    match C[k]? with
    | some cs =>
      let n := (_root_.Flock.u64OfLE c (c.size - 8)).toNat
      if cs.size != n then throw s!"finish: round {k} squeezes {n} coins where the rep's coins have {cs.size}"
      set (k + 1)
      return fun i => cs[i]!
    | none =>
      set (k + 1)
      return fun _ => 1

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
        have hn' := Bool.eq_false_iff.mpr hn
        show ((asksPad C).round c).run k = _
        simp only [asksPad, StateT.run_bind, StateT.run_get, pure_bind, hk, hn', Bool.false_eq_true, ↓reduceIte,
          StateT.run_set, StateT.run_pure]
        rfl

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

/-- Rep stream `k`'s transcript domain, as the loop's `Finish` starts its rep (`proofsOf`). -/
def domOf (spec : SessionSpec) (k : ℕ) : ByteArray := (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8

/-- Rep stream `k`'s scheduled rounds: its rounds less the inner proof's. -/
def nSched (spec : SessionSpec) (sts : Array Stmt) (k : ℕ) : ℕ := (roundsOf spec sts).getD k 0 - INNER_ROUNDS

/-- **The loop's request at `final_c = a`**, its coins cut to the first `i` rounds: `q₀` with its value at `final_c`'s
pad `fc` set to `a`. -/
def askOf (q₀ : RepIn) (fc i : ℕ) (a : F128) : RepIn :=
  { q₀ with vals := q₀.vals.set! fc a, coins := q₀.coins.extract 0 i }

theorem askOf_vals_fc {q₀ : RepIn} {fc : ℕ} (i : ℕ) (a : F128) (h : fc < q₀.vals.size) :
    (askOf q₀ fc i a).vals[fc]! = a :=
  Array.getElem!_set!_self q₀.vals fc a h

theorem askOf_draws (q₀ : RepIn) (fc i : ℕ) (a : F128) : (askOf q₀ fc i a).draws = q₀.draws := rfl

open Classical in
/-- **The rounds the loop's finish hands a rep's frame for its inner proof** on the history `h` at `x`, the witness `w`
and the tape `ρ`, on the rep tables and salts `real` and the frame samples `draws`: where the history's next round is rep
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
        let dom := domOf spec ki.1
        let nS := nSched spec sts ki.1
        let ask := askOf q₀ (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc ki.2
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

end Claims.ZKServer
