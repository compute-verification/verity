module

public import Proofs.Flock.SessionLoop.Phase
public import Proofs.Flock.SessionLoop.FinalC
public import Proofs.Flock.SessionLoop.Inner
public import Proofs.Flock.SessionLoop.FreeWords

/-! The values a pad-blind worker supplies (`Loop.vsOf`), whose `final_c` is fixed and whose rows hold. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128)
open _root_.Flock.ZkServer.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

section VsOf

variable (L : LeanRelease Flock.Wit
  ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))

def FreedAt (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z) (cs : List FlockDeploy.Coin)
    (j : Fin (jOf x)) (r : Fin 2) (i : ℕ) (a : F128) : Prop :=
  ∃ (t : (CoinLayer.session loopNr Masked.dataE).Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (v : Msg) (m : _root_.Flock.ZkServer.Frame.Masker) (out : ByteArray) (s' : _root_.Flock.ZkServer.Frame.Masker)
    (fr : Array ℕ) (k : ℕ),
    (CoinLayer.session loopNr Masked.dataE).seenOf x t = s ∧
    ops.map (fun p => (CoinLayer.session loopNr Masked.dataE).decOf p.2) = cs ∧
    0 < ops.length ∧ ops.length < refK x ∧ L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true ∧
    ((allOk ((CoinLayer.session loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true ∧
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) ∧
    L.answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m ∧
    _root_.Flock.ZkServer.Frame.roundFree (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false }
      ⟨v.toArray⟩ (L.coins x (ops.length + 1)) = .ok (out, s', fr) ∧
    L.rep x (ops.length + 1) = 2 * j.val + r.val ∧ k < fr.size ∧
    (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]! = i ∧
    (_root_.Flock.ZkServer.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]! = a

open Classical in
noncomputable def freeWord (bx : AlgWorker L) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (c : Fin (refK x) → FlockDeploy.Coin) (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z)
    (j : Fin (jOf x)) (r : Fin 2) (i : ℕ) : Option F128 :=
  if h : ∃ n a, n < refK x ∧ FreedAt L (blind L (bwOf L bx)) x w ν s ((List.ofFn c).take n) j r i a then
    some (Classical.choose (Classical.choose_spec h))
  else none

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

noncomputable def vsServed (bx : AlgWorker L) : (CoinLayer.session loopNr Masked.dataE).SwVals :=
  fun x w ν c s j r wt u V i =>
    if i.val = (layE x j).fc then
      (_root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
        ((Array.range 64).map fun k => vsOfWord L bx x w ν c s j r wt u V ((layE x j).r1c k)) : FlockLevel3.GF128)
    else (vsOfWord L bx x w ν c s j r wt u V i.val : FlockLevel3.GF128)

open Classical in
/-- Reads only the statement and the witness, never the coins. A proof device: no guarantee's statement reads it. -/
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

theorem rowsOK_congr {vs vs' : (CoinLayer.session loopNr Masked.dataE).SwVals} {x : Msg} {w : Flock.Wit}
    (h : vs x w = vs' x w) : RowsOK vs x w ↔ RowsOK vs' x w := by
  unfold RowsOK
  rw [h]

theorem rowsOK_vsOf (bx : AlgWorker L) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) :
    RowsOK (vsOf L bx) x w := by
  by_cases hg : Serves x ∧ RowsOK (vsServed L bx) x w
  · exact (rowsOK_congr (vsOf_of_serves L bx hg.1 hg.2)).2 hg.2
  · exact (rowsOK_congr (vsOf_of_not L bx hg)).2 (rowsOK_honestVals hR)

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
