module

public import Proofs.Flock.SessionLoop.Phase
public import Proofs.Flock.SessionLoop.FinalC
public import Proofs.Flock.SessionLoop.Inner
public import Proofs.Flock.SessionLoop.FreeWords

/-!
# The values a pad-blind worker supplies

`Loop.TakesVals vs L wk x w` ties a worker to values: each word the frame frees in the worker's round is `vs`'s value at
that pad index. A worker of the algebraic phase (`Loop.AlgWorker`, run as `Loop.phased`) reads the nonce, what the
workers see of the tape (`RefSession.seenOf`) and the answers read as coins, so the words it frees are a function of
what `RefSession.SwVals` reads, once each pad index has one round that frees it. This file builds that function:

* `Loop.FreedAt L wk x w ν s cs j r i a`: after a history `TakesVals` reads, whose answers after `Hello` open the coins
  `cs` and whose tape the workers see as `s`, the frame frees pad `i` of rep stream `2j + r` in `wk`'s round as the
  word `a`.
* `Loop.freeWord L bx`: the word `bx`, as a pad-blind worker (`blind (bwOf bx)`), frees at pad `i`, at the history
  whose answers open the coins' first `n` (`FreedAt`), where there is one.
* **`Loop.vsOf L bx`**: at a statement the loop serves (`Loop.Serves`) and a witness at which its rows hold
  (`Loop.RowsOK`), `Loop.vsServed L bx`: at each pad index, `freeWord`'s word where the frame frees one there; at
  `final_c`'s index the c_eval of the round-1c values at the coins' zerocheck point (`zcE`), as the ZK server's finish
  computes it; at every other index the table's own value (`Table.vals`). Elsewhere the tables' own values
  (`honestVals`). The choice reads only the statement and the witness. `vsOf` is a proof device: no guarantee's
  statement reads it.
* **`Loop.finalCFixed_vsOf`**: `vsOf bx`'s `final_c` is its round-1c values' c_eval (`Loop.FinalCFixed`), at every
  coins: by construction where `vsOf` is `vsServed` (no round frees `final_c`'s pad), and as the honest values'
  elsewhere.
* **`Loop.rowsOK_vsOf`**: `RowsOK (vsOf bx) x w` at every `bx` and every witness under the relation, so the rows are no
  condition on the worker.

`Loop.takesVals_vsOf` (`SessionLoop/VsOfTakes`) proves that `phased bx post` meets `TakesVals` at `vsOf bx`: `bwOf`
hands `bx` the answers after `Hello`'s (`List.tail?`), the coins `TakesVals` reads `vs` at, and never `Hello`'s answer.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

section VsOf

variable (L : LeanRelease Flock.Wit
  ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))

/-- **The frame frees pad `i` of rep stream `2j + r` as the word `a`** in `wk`'s round after a history `TakesVals`
reads (`0 < ops.length < refK x` openings, live, whose openings pass under the nonce `ν`), whose answers after `Hello`
open the coins `cs` and whose tape the workers see as `s`: the frame takes the stream's coins (`answered`), frees words
in `wk`'s round there (`Frame.roundFree`), and its `k`-th, at the absorbed order's place `m.pos + k`, is pad `i`, as
`a`. -/
def FreedAt (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z) (cs : List FlockDeploy.Coin)
    (j : Fin (jOf x)) (r : Fin 2) (i : ℕ) (a : F128) : Prop :=
  ∃ (t : (CoinLayer.session loopNr Masked.dataE).Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (v : Msg) (m : _root_.Flock.Firewall.Frame.Masker) (out : ByteArray) (s' : _root_.Flock.Firewall.Frame.Masker)
    (fr : Array ℕ) (k : ℕ),
    (CoinLayer.session loopNr Masked.dataE).seenOf x t = s ∧
    ops.map (fun p => (CoinLayer.session loopNr Masked.dataE).decOf p.2) = cs ∧
    0 < ops.length ∧ ops.length < refK x ∧ L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true ∧
    ((allOk ((CoinLayer.session loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true ∧
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) ∧
    L.answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m ∧
    _root_.Flock.Firewall.Frame.roundFree (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false }
      ⟨v.toArray⟩ (L.coins x (ops.length + 1)) = .ok (out, s', fr) ∧
    L.rep x (ops.length + 1) = 2 * j.val + r.val ∧ k < fr.size ∧
    (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]! = i ∧
    (_root_.Flock.Firewall.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]! = a

open Classical in
/-- **The word `bx` frees at pad `i` of rep stream `2j + r`**, at the coins `c` and what the workers see `s`: where for
some `n < refK x` the frame frees pad `i` in the pad-blind worker `blind (bwOf bx)`'s round after a history whose
answers after `Hello` open `c`'s first `n` coins (`FreedAt`), a word it frees there; none where no round frees it. -/
noncomputable def freeWord (bx : AlgWorker L) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (c : Fin (refK x) → FlockDeploy.Coin) (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z)
    (j : Fin (jOf x)) (r : Fin 2) (i : ℕ) : Option F128 :=
  if h : ∃ n a, n < refK x ∧ FreedAt L (blind L (bwOf L bx)) x w ν s ((List.ofFn c).take n) j r i a then
    some (Classical.choose (Classical.choose_spec h))
  else none

/-- **`vsOf`'s value at pad `i` other than `final_c`'s**: `freeWord`'s where a round frees pad `i`, else the table's own
value at the table witness `wt` and the base `(u, V)`; `0` from `n_E` on. -/
noncomputable def vsOfWord (bx : AlgWorker L) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (c : Fin (refK x) → FlockDeploy.Coin) (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z)
    (j : Fin (jOf x)) (r : Fin 2) (wt : FlockSoundness.Model.Witness ((CoinLayer.session loopNr Masked.dataE).C x j).S.m)
    (u : (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.Mask)
    (V : (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.Vis) (i : ℕ) : F128 :=
  match freeWord L bx x w ν c s j r i with
  | some a => a
  | none =>
    if hi : i < (((CoinLayer.session loopNr Masked.dataE).C x j).T
        ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.nE then
      FlockLevel3.GF128.val ((((CoinLayer.session loopNr Masked.dataE).C x j).T
        ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.vals r wt u V ⟨i, hi⟩)
    else 0

/-- **The values a pad-blind worker of the algebraic phase supplies where the loop serves the statement**: at each pad
index the frame frees, the word `bx` frees there at the history whose answers open the coins' prefix (`freeWord`); at
`final_c`'s index the c_eval `Σ_i λ_i(z)·r1c_i` of the round-1c values at the rep's zerocheck point (`zcE`), as the
ZK server's finish computes it (`Flock.dot`, `Flock.lagrange`); at every other index the table's own value
(`vsOfWord`). -/
noncomputable def vsServed (bx : AlgWorker L) : (CoinLayer.session loopNr Masked.dataE).SwVals :=
  fun x w ν c s j r wt u V i =>
    if i.val = (layE x j).fc then
      (_root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
        ((Array.range 64).map fun k => vsOfWord L bx x w ν c s j r wt u V ((layE x j).r1c k)) : FlockLevel3.GF128)
    else (vsOfWord L bx x w ν c s j r wt u V i.val : FlockLevel3.GF128)

open Classical in
/-- **The values a pad-blind worker of the algebraic phase supplies**: `vsServed`'s at a statement the loop serves
(`Serves`) and a witness at which `vsServed`'s rows hold (`RowsOK`), and the tables' own (`honestVals`) elsewhere: at a
statement whose `Hello` the loop refuses, where no round frees a pad, and where the worker's words break the rows. The
choice reads only the statement and the witness, never the coins. A proof device: no guarantee's statement reads it. -/
noncomputable def vsOf (bx : AlgWorker L) : (CoinLayer.session loopNr Masked.dataE).SwVals :=
  fun x w => if Serves x ∧ RowsOK (vsServed L bx) x w then vsServed L bx x w
    else (CoinLayer.session loopNr Masked.dataE).honestVals x w

theorem vsOf_of_serves (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} (h : Serves x)
    (hr : RowsOK (vsServed L bx) x w) : vsOf L bx x w = vsServed L bx x w :=
  ite_eq_left ⟨h, hr⟩

theorem vsOf_of_not (bx : AlgWorker L) {x : Msg} {w : Flock.Wit} (h : ¬ (Serves x ∧ RowsOK (vsServed L bx) x w)) :
    vsOf L bx x w = (CoinLayer.session loopNr Masked.dataE).honestVals x w :=
  ite_eq_right h

theorem vsOf_of_not_serves (bx : AlgWorker L) {x : Msg} (h : ¬ Serves x) :
    vsOf L bx x = (CoinLayer.session loopNr Masked.dataE).honestVals x :=
  funext fun _ => vsOf_of_not L bx fun h' => h h'.1

/-- `RowsOK` reads `vs` only at its statement and witness. -/
theorem rowsOK_congr {vs vs' : (CoinLayer.session loopNr Masked.dataE).SwVals} {x : Msg} {w : Flock.Wit}
    (h : vs x w = vs' x w) : RowsOK vs x w ↔ RowsOK vs' x w := by
  unfold RowsOK
  rw [h]

/-- **`vsOf`'s rows hold** at every witness under the relation, at every `bx`: `vsServed`'s where they hold there, and
the honest values' (`rowsOK_honestVals`) where `vsOf` is them. -/
theorem rowsOK_vsOf (bx : AlgWorker L) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) :
    RowsOK (vsOf L bx) x w := by
  by_cases hg : Serves x ∧ RowsOK (vsServed L bx) x w
  · exact (rowsOK_congr (vsOf_of_serves L bx hg.1 hg.2)).2 hg.2
  · exact (rowsOK_congr (vsOf_of_not L bx hg)).2 (rowsOK_honestVals hR)

/-- **`vsOf`'s word below `n_E`** at a statement the loop serves and a witness at which `vsServed`'s rows hold: the
c_eval of its round-1c words at `final_c`'s index, `vsOfWord` elsewhere, at the tape's base and what the workers see of
the tape. -/
theorem vsWord_vsOf (bx : AlgWorker L) {x : Msg} (hSv : Serves x) (w : Flock.Wit) (hRo : RowsOK (vsServed L bx) x w)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {i : ℕ}
    (hi : i < (tabC PE x j).z.nE) :
    vsWord (vsOf L bx) x w ν c τ j r i =
      if i = (layE x j).fc then
        _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
          ((Array.range 64).map fun k => vsOfWord L bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r
            (wtOf PE x w j) (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 ((layE x j).r1c k))
      else vsOfWord L bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r (wtOf PE x w j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 i := by
  unfold vsWord
  rw [dite_eq_left hi, vsOf_of_serves L bx hSv hRo]
  rfl

/-- **`vsOf`'s `final_c` is its round-1c values' c_eval**, at every coins: where `vsOf` is `vsServed` it sets it so,
and the round-1c indices are below `final_c`'s; elsewhere `vsOf` is the honest values, whose `final_c` is
(`finalCFixed_honestVals`). -/
theorem finalCFixed_vsOf (bx : AlgWorker L) (x : Msg) (w : Flock.Wit) : FinalCFixed (vsOf L bx) x w := by
  by_cases hg : ¬ (Serves x ∧ RowsOK (vsServed L bx) x w)
  · have e : ∀ ν c τ j r i, vsWord (vsOf L bx) x w ν c τ j r i =
        vsWord (CoinLayer.session loopNr Masked.dataE).honestVals x w ν c τ j r i := fun _ _ _ _ _ _ => by
      unfold vsWord
      rw [vsOf_of_not L bx hg]
    intro ν c τ j ho r
    simp only [e]
    exact finalCFixed_honestVals x w ν c τ j ho r
  obtain ⟨hSv, hRo⟩ := not_not.mp hg
  intro ν c τ j _ r
  have hE : (tabC PE x j).z.nE = (layE x j).hab := by
    show (layE x j).len - 4 = (layE x j).hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlt : ∀ i < 64, (layE x j).r1c i < (layE x j).fc := fun i _ => by
    simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.r1c, _root_.Flock.Zk.PadLayout.fc,
      _root_.Flock.Zk.PadLayout.fa]
    omega
  have hfc : (layE x j).fc < (tabC PE x j).z.nE := by rw [hE]; exact lt_hab (layE x j) le_rfl
  have hr1c : ∀ i < 64, (layE x j).r1c i < (tabC PE x j).z.nE := fun i hi => by
    rw [hE]; exact lt_hab (layE x j) (hlt i hi).le
  rw [vsWord_vsOf L bx hSv w hRo ν c τ j r hfc, ite_eq_left rfl]
  refine congrArg₂ _root_.Flock.dot rfl (Array.ext (by simp) fun i h1 _ => ?_)
  have hi : i < 64 := by simpa using h1
  simp only [Array.getElem_map, Array.getElem_range]
  rw [vsWord_vsOf L bx hSv w hRo ν c τ j r (hr1c i hi), ite_eq_right (Nat.ne_of_lt (hlt i hi))]

end VsOf

end Loop

end Claims.ZKServer
